"""
Task 3: Customer Churn Data Preparation & EDA
Dataset: Telco Customer Churn (IBM sample dataset, ~7,043 rows)

Pipeline:
1. Data Understanding
2. Data Cleaning
3. Feature Engineering
4. EDA & Visualization
5. Deliverables (cleaned CSV + insight summary)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/Telco-Customer-Churn.csv"
VISUALS_DIR = "visuals"
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# 1. DATA UNDERSTANDING
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("1. DATA UNDERSTANDING")
print("=" * 70)
print(f"Shape: {df.shape}")
print("\n--- df.info() ---")
df.info()
print("\n--- df.describe() ---")
print(df.describe(include="all").T)

# Identify "strange" values - TotalCharges has blank strings " " instead of NaN
blank_total_charges = (df["TotalCharges"].str.strip() == "").sum()
print(f"\nBlank/whitespace TotalCharges values found: {blank_total_charges}")

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. DATA CLEANING")
print("=" * 70)

# Fix TotalCharges: blank strings -> NaN -> float
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

n_missing = df["TotalCharges"].isna().sum()
print(f"TotalCharges converted to float. Missing values: {n_missing}")

# These rows correspond to customers with tenure == 0 (brand-new customers,
# no bill yet). Impute with 0 rather than dropping, since they are valid
# zero-tenure customers.
df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0

# Drop exact duplicate customerIDs if any
dupes = df["customerID"].duplicated().sum()
print(f"Duplicate customerIDs: {dupes}")
df = df.drop_duplicates(subset="customerID")

# SeniorCitizen is 0/1 int -> make it Yes/No for consistency with other flags
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

print(f"Shape after cleaning: {df.shape}")

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. FEATURE ENGINEERING")
print("=" * 70)


def tenure_bucket(t):
    if t <= 12:
        return "0-12"
    elif t <= 24:
        return "13-24"
    elif t <= 48:
        return "25-48"
    elif t <= 60:
        return "49-60"
    else:
        return "61-72"


df["TenureGroup"] = df["tenure"].apply(tenure_bucket)

# AvgMonthlySpend = TotalCharges / tenure (guard against divide-by-zero for
# brand new customers by falling back to their current MonthlyCharges)
df["AvgMonthlySpend"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
)


def spend_group(x):
    if x < 35:
        return "Low"
    elif x < 70:
        return "Medium"
    else:
        return "High"


df["MonthlySpendGroup"] = df["AvgMonthlySpend"].apply(spend_group)

# Convert binary Yes/No columns to 1/0
binary_cols = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "Churn", "SeniorCitizen",
]
for col in binary_cols:
    df[col + "Flag"] = df[col].map({"Yes": 1, "No": 0})

# One-hot encode key categorical columns
categorical_to_encode = ["Contract", "InternetService", "PaymentMethod"]
df_encoded = pd.get_dummies(df, columns=categorical_to_encode, drop_first=False)

print("New columns added: TenureGroup, AvgMonthlySpend, MonthlySpendGroup, "
      "*Flag columns")
print(f"One-hot encoded shape: {df_encoded.shape}")

# ---------------------------------------------------------------------------
# 4. EDA & VISUALIZATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. EDA & VISUALIZATION")
print("=" * 70)

churn_rate = df["ChurnFlag"].mean() * 100
print(f"Overall churn rate: {churn_rate:.2f}%")

# --- Countplot for Churn ---
plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
ax.set_title("Customer Churn Count")
for c in ax.containers:
    ax.bar_label(c)
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/01_churn_countplot.png")
plt.close()

# --- Barplot: Contract vs Churn ---
plt.figure(figsize=(6, 4))
contract_churn = (
    df.groupby("Contract")["ChurnFlag"].mean().mul(100).sort_values()
)
ax = contract_churn.plot(kind="bar", color="#4C72B0")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Contract Type")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/02_contract_vs_churn.png")
plt.close()

# --- Heatmap of correlations ---
numeric_for_corr = df_encoded.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 9))
sns.heatmap(numeric_for_corr.corr(), cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Correlation Heatmap (Numeric + Encoded Features)")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/03_correlation_heatmap.png")
plt.close()

# --- Boxplot: MonthlyCharges by Churn ---
plt.figure(figsize=(5, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn",
            palette="Set3", legend=False)
plt.title("Monthly Charges by Churn Status")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/04_monthlycharges_boxplot.png")
plt.close()

# --- Bonus: pairplot on a small subset (keep it light) ---
subset_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
pair = sns.pairplot(df[subset_cols], hue="Churn", palette="husl", height=2.2)
pair.savefig(f"{VISUALS_DIR}/05_pairplot.png")
plt.close()

print(f"Saved 5 visuals to {VISUALS_DIR}/")

# ---------------------------------------------------------------------------
# 5. DELIVERABLES
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. DELIVERABLES")
print("=" * 70)

df.to_csv(f"{OUTPUT_DIR}/cleaned_telco_churn.csv", index=False)
print(f"Cleaned CSV saved to {OUTPUT_DIR}/cleaned_telco_churn.csv")

insights = f"""
TOP 5 INSIGHTS - Customer Churn Analysis
==========================================
1. Overall churn rate is {churn_rate:.1f}% ({df['ChurnFlag'].sum()} of {len(df)} customers).

2. Contract type is the strongest churn driver: month-to-month customers
   churn at {contract_churn.get('Month-to-month', 0):.1f}%, vs
   {contract_churn.get('One year', 0):.1f}% for one-year and
   {contract_churn.get('Two year', 0):.1f}% for two-year contracts.

3. Customers with higher MonthlyCharges churn more often - the churned
   group has a noticeably higher median monthly bill than the retained
   group (see boxplot), suggesting price sensitivity or dissatisfaction
   with premium add-ons (fiber, streaming bundles).

4. New customers (TenureGroup 0-12 months) churn at a much higher rate
   than long-tenured customers, indicating the first year is the highest-
   risk window - onboarding and early engagement matter most.

5. Payment method correlates with churn: customers paying via electronic
   check churn more than those on automatic bank transfer / credit card,
   possibly reflecting a less "sticky" payment relationship.
"""
print(insights)

with open(f"{OUTPUT_DIR}/insight_summary.txt", "w") as f:
    f.write(insights)

print("Insight summary saved to output/insight_summary.txt")
print("\nDone.")
