import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Create synthetic dataset
data = {
    "square_footage": [1000, 1200, 1500, 1700, 2000, 2500, 1800, 1300, 2200, 1600],
    "bedrooms":        [2,    3,    3,    4,    4,    5,    4,    3,    4,    3],
    "bathrooms":       [1,    2,    2,    2,    3,    3,    2,    1,    3,    2],
    "price":           [150000, 200000, 250000, 300000, 350000, 450000, 320000, 210000, 380000, 270000]
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[["square_footage", "bedrooms", "bathrooms"]]
y = df["price"]

# Step 3: Train/test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Show model coefficients
print("Model coefficients:")
print(f"Square Footage weight: {model.coef_[0]:.2f}")
print(f"Bedrooms weight:       {model.coef_[1]:.2f}")
print(f"Bathrooms weight:      {model.coef_[2]:.2f}")
print(f"Intercept:             {model.intercept_:.2f}")

# Step 6: Make prediction
def predict_price(sqft, beds, baths):
    features = pd.DataFrame([[sqft, beds, baths]], columns=["square_footage", "bedrooms", "bathrooms"])
    predicted_price = model.predict(features)[0]
    return predicted_price

# Example usage
print("\n--- Price Prediction ---")
sqft_input = int(input("Enter square footage: "))
beds_input = int(input("Enter number of bedrooms: "))
baths_input = int(input("Enter number of bathrooms: "))

predicted = predict_price(sqft_input, beds_input, baths_input)
print(f"Estimated house price: ${predicted:,.2f}")

