import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare the data
data = pd.read_csv("supermarket_sales - Sheet1.csv")

# Preprocess data
X = data.drop(columns=['Total', 'cogs', 'gross margin percentage'])
y = data[['Total', 'cogs', 'gross margin percentage']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
num_features = ['Unit price', 'Quantity', 'Tax 5%']
cat_features = ['Branch', 'City', 'Customer type', 'Gender', 'Product line']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

# Fit and transform the training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Multi-output regression model
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train_transformed, y_train)

# Streamlit app title
st.title("Supermarket Sales Prediction App")

# Input fields for features
st.header("Input Features")
unit_price = st.number_input("Unit Price", min_value=0.0, format="%.2f")
quantity = st.number_input("Quantity", min_value=1)
tax = st.number_input("Tax (5%)", min_value=0.0, format="%.2f")
branch = st.selectbox("Branch", options=["A", "B", "C"])
city = st.selectbox("City", options=["Yangon", "Mandalay", "Naypyitaw"])
customer_type = st.selectbox("Customer Type", options=["Member", "Normal"])
gender = st.selectbox("Gender", options=["Male", "Female"])
product_line = st.selectbox("Product Line", options=["Health and beauty", "Electronic accessories", "Home and lifestyle", "Sports and travel", "Food and beverages", "Fashion accessories"])

# Button for prediction
if st.button("Predict"):
    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Unit price': [unit_price],
        'Quantity': [quantity],
        'Tax 5%': [tax],
        'Branch': [branch],
        'City': [city],
        'Customer type': [customer_type],
        'Gender': [gender],
        'Product line': [product_line]
    })

    # Preprocess the input data
    input_transformed = preprocessor.transform(input_data)

    # Make predictions
    y_pred = model.predict(input_transformed)

    # Display results
    st.subheader("Predicted Results")
    st.write(f"Total: {y_pred[0][0]:.2f}")
    st.write(f"Cogs: {y_pred[0][1]:.2f}")
    st.write(f"Gross Margin Percentage: {y_pred[0][2]:.2f}")
