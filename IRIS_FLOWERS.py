
# STEP 1: IMPORT NECESSARY LIBRARIES

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# STEP 2: LOAD AND EXPLORE THE DATASET FROM YOUR CSV FILE

file_path = r"C:\Users\anita\OneDrive\Desktop\Iris.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

print("## First 5 rows of your dataset:")
print(df.head())

print("\n## Dataset Information:")
df.info()


# STEP 3: VISUALIZE THE DATA

print("\n## Visualizing feature relationships...")
sns.pairplot(df, hue='Species', markers=["o", "s", "D"])
plt.suptitle('Pair Plot of Iris Dataset Features', y=1.02)
plt.show()


# STEP 4: DEFINE FEATURES (X) AND TARGET (y)
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n## Data Split:")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# STEP 5: CHOOSE AND TRAIN THE MODEL

model = KNeighborsClassifier(n_neighbors=3)

print("\n## Training the K-Nearest Neighbors model...")
model.fit(X_train, y_train)
print("## Model training complete!")


# STEP 6: EVALUATE THE MODEL
y_pred = model.predict(X_test)

class_names = y.unique()
class_names.sort()

accuracy = accuracy_score(y_test, y_pred)
print(f"\n## Model Accuracy: {accuracy:.2f}")

print("\n## Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n## Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=class_names)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
