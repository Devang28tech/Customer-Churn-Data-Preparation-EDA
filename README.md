Customer Churn Data Preparation & EDA

## Objective
Prepare telecom customer data for modeling through cleaning, feature
engineering, and exploratory data analysis (EDA), and surface the key
drivers of customer churn.

## Dataset
**Telco Customer Churn** (IBM sample dataset)
- **Rows:** 7,043 customers
- **Columns:** 21 (demographics, subscribed services, billing, churn label)
- Source CSV mirrored from IBM's public sample-data repository. Original
  distribution: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## What This Project Does

### 1. Data Understanding
- Loaded the dataset and inspected shape, dtypes, and summary statistics via `.info()` / `.describe()`
- Identified `TotalCharges` was stored as text with 11 blank (`" "`) values instead of `NaN`

### 2. Data Cleaning
- Converted `TotalCharges` to numeric, coercing blanks to `NaN`
- Imputed the resulting missing `TotalCharges` with `0` — these all correspond to brand-new customers with `tenure == 0`, so a zero bill is correct, not missing data
- Checked for and confirmed no duplicate `customerID` rows
- Normalized `SeniorCitizen` from `0/1` to `No/Yes` for consistency with the other Yes/No service columns

### 3. Feature Engineering
- `TenureGroup` — bucketed tenure into `0-12`, `13-24`, `25-48`, `49-60`, `61-72` months
- `AvgMonthlySpend` — `TotalCharges / tenure` (falls back to current `MonthlyCharges` for zero-tenure customers)
- `MonthlySpendGroup` — `Low` / `Medium` / `High` spend buckets
- `*Flag` columns — binary 0/1 versions of `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `SeniorCitizen`, and `Churn`
- One-hot encoded `Contract`, `InternetService`, and `PaymentMethod` for the correlation heatmap

### 4. EDA & Visualization
All charts are in [`visuals/`](visuals/):

| File | Chart |
|---|---|
| `01_churn_countplot.png` | Overall churn count |
| `02_contract_vs_churn.png` | Churn rate by contract type |
| `03_correlation_heatmap.png` | Correlation heatmap across numeric + encoded features |
| `04_monthlycharges_boxplot.png` | Monthly charges distribution by churn status |
| `05_pairplot.png` | Pairplot of tenure / charges / churn |

## Top 5 Insights

1. **Overall churn rate is 26.5%** (1,869 of 7,043 customers).
2. **Contract type is the strongest churn driver** — month-to-month customers churn at **42.7%**, vs **11.3%** for one-year and **2.8%** for two-year contracts.
3. **Higher monthly charges correlate with higher churn** — the churned group has a noticeably higher median monthly bill, hinting at price sensitivity around premium add-ons (fiber, streaming bundles).
4. **The first year is the highest-risk window** — customers in the `0-12` month tenure bucket churn far more than long-tenured customers, pointing to onboarding/early engagement as the key lever.
5. **Payment method matters** — electronic-check payers churn more than customers on automatic bank transfer or credit card, suggesting a less "sticky" payment relationship.

## Deliverables
- Executed notebook: [`notebooks/customer_churn_eda.ipynb`](notebooks/customer_churn_eda.ipynb)
- Standalone script: [`churn_analysis.py`](churn_analysis.py)
- Cleaned dataset: [`output/cleaned_telco_churn.csv`](output/cleaned_telco_churn.csv)
- Insight summary: [`output/insight_summary.txt`](output/insight_summary.txt)

## Run It

```bash
pip install -r ../requirements.txt
python churn_analysis.py
```
