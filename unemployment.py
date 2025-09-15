# STEP 1: IMPORT NECESSARY LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# STEP 2: LOAD THE DATASET
file_path = r"C:\Users\anita\OneDrive\Desktop\unemployment.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()

# STEP 3: CLEAN AND PREPARE THE DATA

print("## First 5 rows of the dataset:")
print(df.head())

print("\n## Dataset Information:")
df.info()

df.columns = ['Region', 'Date', 'Frequency', 'Estimated Unemployment Rate (%)',
              'Estimated Employed', 'Estimated Labour Participation Rate (%)', 'Area']

# Drop rows with any missing values to prevent conversion errors
df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Extract month and year for analysis
df['Month_int'] = df['Date'].dt.month
df['Month_int'] = df['Month_int'].astype(int) # <-- ADD THIS LINE TO FIX THE ERROR
df['Month'] = df['Month_int'].apply(lambda x: calendar.month_abbr[x])
df['Year'] = df['Date'].dt.year

# STEP 4: PERFORM EXPLORATORY DATA ANALYSIS (EDA)
print("\n## Statistical Summary:")
print(df[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].describe())

state_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()
state_unemployment = state_unemployment.sort_values('Estimated Unemployment Rate (%)', ascending=False)

print("\n## Average Unemployment Rate by State/Region:")
print(state_unemployment)

# STEP 5: VISUALIZE THE DATA
print("\n## Generating visualizations...")

sns.set(style="whitegrid")

plt.figure(figsize=(14, 8))
sns.barplot(x='Estimated Unemployment Rate (%)', y='Region', data=state_unemployment)
plt.title('Average Unemployment Rate by State/Region', fontsize=16)
plt.xlabel('Average Estimated Unemployment Rate (%)', fontsize=12)
plt.ylabel('State / Region', fontsize=12)
plt.tight_layout()
plt.show()

area_unemployment = df.groupby('Area')['Estimated Unemployment Rate (%)'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Area', y='Estimated Unemployment Rate (%)', data=area_unemployment)
plt.title('Average Unemployment Rate by Area', fontsize=16)
plt.xlabel('Area', fontsize=12)
plt.ylabel('Average Estimated Unemployment Rate (%)', fontsize=12)
plt.show()

unemployment_over_time = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=unemployment_over_time)
plt.title('National Average Unemployment Rate Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Estimated Unemployment Rate (%)', fontsize=12)
plt.show()
