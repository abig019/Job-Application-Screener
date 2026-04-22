import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import os
# seaborn = beautiful charts with less code than matplotlib

df = pd.read_csv('Job_Placement_Data.csv')
# WHY? First thing always — load and look at your data

print("── Shape ───────────────────────────")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print()

print("── First 5 rows ────────────────────")
print(df.head())
print()

print("── Column names ────────────────────")
print(df.columns.tolist())
print()

print("── Data types ──────────────────────")
print(df.dtypes)
print()

print("── Missing values ──────────────────")
print(df.isnull().sum())
# WHY? Real datasets always have missing values.
# isnull() finds them, .sum() counts them per column.
# This is the first thing you check with real data.
# You never had this problem with your synthetic Focus Score data!
print()


# ── STEP 4: VISUALISE THE DATA ────────────────────────────

plt.figure(figsize=(6, 4))
df['status'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Placed vs Not Placed')
plt.xlabel('Status')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('target_distribution.png')
plt.show()
print("✅ Chart saved as target_distribution.png")