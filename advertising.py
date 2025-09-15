
# STEP 1: IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# STEP 2: LOAD AND INSPECT THE DATA
file_path = r"C:\Users\anita\OneDrive\Desktop\advertising.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()

print("## First 5 rows of the dataset:")
print(df.head())

print("\n## Dataset Information:")
df.info()

print("\n## Missing Values Count:")
print(df.isnull().sum())

print("\n## Statistical Summary:")
print(df.describe())

# STEP 3: VISUALIZE THE DATA (EXPLORATORY DATA ANALYSIS)

print("\n## Visualizing relationships between advertising channels and sales...")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df, x='TV', y='Sales', ax=axs[0])
axs[0].set_title('TV Advertising vs. Sales')
sns.scatterplot(data=df, x='Radio', y='Sales', ax=axs[1])
axs[1].set_title('Radio Advertising vs. Sales')
sns.scatterplot(data=df, x='Newspaper', y='Sales', ax=axs[2])
axs[2].set_title('Newspaper Advertising vs. Sales')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# STEP 4: PREPARE DATA FOR MODELING

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: TRAIN THE LINEAR REGRESSION MODEL

model = LinearRegression()
print("\n## Training the Linear Regression model...")
model.fit(X_train, y_train)
print("## Model training complete!")


# STEP 6: EVALUATE THE MODEL

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n## Model Performance Evaluation:")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# STEP 7: VISUALIZE THE PREDICTIONS

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.grid(True)
plt.show()
