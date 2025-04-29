import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"/Users/mac/Downloads/INC 5000 Companies 2019 2.csv")

print(f"\n Shape of dataset:",df.shape)
print(f"\n First 5 rows of a dataset:",df.head())
print(f"\n Datatypes used in dataset:",df.info())
print(f"\n Statistical Summary of the dataset: \n",df.describe())
print(f"\n No.of missing values: \n",df.isnull().sum())
print(f"Total Missing values in the dataset:",df.isnull().sum().sum())
print(df.fillna(df['metro'].mode()[0],inplace=True))
print(f"Updated Missing values: \n",df.isnull().sum())

# Histogram
plt.figure(figsize=(8,6))
sns.histplot(df["yrs_on_list"], kde=True, bins=30, color="pink")
plt.title("Years on the list")
plt.ylabel("Count")
plt.xlabel("Number of years")
plt.show()

# Box plot
plt.figure(figsize=(8,6))
sns.boxplot(data=df.head(20), x="industry",y="revenue",palette='viridis')
plt.title("Revenue by Industry")
plt.xlabel("Industry")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# Heat map
plt.figure(figsize=(8,6))
corr_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, cmap="cividis")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot
df["workers"] = pd.to_numeric(df["workers"], errors="coerce")
df["founded"] = pd.to_numeric(df["founded"], errors="coerce")
df = df.dropna(subset=["workers", "founded"])

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="workers", y="founded", alpha=0.8, legend=True)
plt.ylabel("founded")
plt.xlabel("workers")
plt.title("Workers vs Founded Year by State")
plt.show()

# Replace 'Billion' and 'Million' with numbers, then convert to float
revenue_str = df['revenue'].astype(str)
revenue_str = revenue_str.str.replace('$', '', regex=False)
revenue_str = revenue_str.str.replace(' Billion', '000', regex=False)
revenue_str = revenue_str.str.replace(' Million', '', regex=False)

# Convert to numeric, errors='coerce' will turn invalid strings into NaN
df['revenue_clean'] = pd.to_numeric(revenue_str, errors='coerce')

# Get top 10 companies by revenue
top10 = df.nlargest(10, 'revenue_clean')

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top10, x='name', y='revenue_clean', hue='industry', dodge=False)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Company")
plt.ylabel("Revenue (Million USD)")
plt.title("Top 10 INC 5000 Companies by Revenue (2019)")
plt.legend(title="Industry", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.legend(title="Industry", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
