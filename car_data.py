
# STEP 1: IMPORT NECESSARY LIBRARIES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


try:
    car_data = pd.read_csv(r"C:\Users\anita\OneDrive\Desktop\car data.csv")
except FileNotFoundError:
    print("Error: Dataset file not found. Please replace 'car data.csv' with your file name.")
    exit()

# Display the first 5 rows to get an overview of the data
print("# --- First 5 Rows of the Dataset ---")
print(car_data.head())

# Get a summary of the dataset (columns, data types, non-null values)
print("\n# --- Dataset Information ---")
car_data.info()

# Check for missing values in each column
print("\n# --- Missing Values Count ---")
print(car_data.isnull().sum())

# Get statistical summary for numerical columns
print("\n# --- Statistical Summary ---")
print(car_data.describe())

# STEP 3: DATA PREPROCESSING & VISUALIZATION
print("\n# --- Visualizing Data ---")

# Plotting the distribution of the target variable 'Selling_Price'
plt.figure(figsize=(10, 6))
sns.histplot(car_data['Selling_Price'], kde=True, bins=30)
plt.title('Distribution of Car Selling Prices')
plt.xlabel('Selling Price (in Lakhs)')
plt.ylabel('Frequency')
plt.show()

# Visualizing relationships between numerical features using a correlation heatmap
plt.figure(figsize=(12, 8))
# We select only numeric columns for correlation calculation
numeric_cols = car_data.select_dtypes(include=np.number)
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# STEP 4: FEATURE ENGINEERING (CONVERTING CATEGORICAL DATA)
car_data_processed = pd.get_dummies(car_data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# We can also create a new feature like 'Car_Age'
current_year = 2024 # You can update this to the current year
car_data_processed['Car_Age'] = current_year - car_data_processed['Year']

# We no longer need 'Car_Name' and 'Year' for the model
car_data_processed.drop(['Car_Name', 'Year'], axis=1, inplace=True)

print("\n# --- Processed Data (First 5 Rows) ---")
print(car_data_processed.head())

# STEP 5: SPLITTING DATA AND TRAINING THE MODEL
# Define features (X) and the target (y)
X = car_data_processed.drop('Selling_Price', axis=1)
y = car_data_processed['Selling_Price']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n# --- Data Split ---")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Initialize the model - RandomForestRegressor is a powerful and popular choice
model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1)

# Train the model on the training data
print("\n# --- Training the Model... ---")
model.fit(X_train, y_train)
print("# --- Model Training Complete! ---")

# STEP 6: EVALUATING THE MODEL

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n# --- Model Performance Evaluation ---")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# STEP 7: VISUALIZING THE PREDICTIONS

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2) # Perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted Car Prices")
plt.grid(True)
plt.show()
