# Telco Customer Churn Analysis

## Live Report

The full Quarto report are available at:`https://daiqwer.github.io/telco-customer-churn-analysis/`

## Executive Summary

The analysis shows that churn is **not randomly distributed** across the customer base. It is concentrated in a set of high-risk conditions tied to **low commitment**, **early lifecycle**, and **value sensitivity**.

### Key findings
- **Overall churn rate:** 26.5%
- **Tenure below 12 months:** 47.4% churn
- **Month-to-month contracts:** 42.7% churn
- **Electronic check users:** 45.3% churn
- **Fiber optic customers:** 41.9% churn

### Business implication
The strongest retention opportunities lie in:
1. early lifecycle retention,
2. contract conversion,
3. automatic payment adoption,
4. service experience review for fiber optic customers.

---

## Dataset

- **Source:** IBM Telco Customer Churn dataset from Kaggle
- **Rows:** 7,043
- **Columns:** 21 raw fields + engineered analytical features
- **Target:** `Churn`

The dataset is a **customer-level snapshot**, which makes it suitable for segmentation, EDA, and predictive prioritization. It is not an event-level log, so findings should be interpreted as **associations**, not causal proof.

---

## Repository Structure

```text
telco-customer-churn-analysis/
├── README.md
├── requirements.txt
├── _quarto.yml
├── index.qmd
├── data/
│   └── README.md
├── src/
│   └── telco_churn_analysis.py
├── assets/
│   └── charts/
│       ├── 01_churn_overview.png
│       ├── 02_tenure_analysis.png
│       ├── 03_monthly_charges.png
│       ├── 04_internet_service.png
│       ├── 05_contract_type.png
│       ├── 06_addon_services.png
│       ├── 07_payment_method.png
│       ├── 08_demographics.png
│       ├── 09_multivariate_heatmap.png
│       ├── 10_churn_drivers.png
│       ├── 11_model_performance.png
│       └── 12_risk_score_distribution.png
└── .github/
    └── workflows/
        └── publish-quarto.yml
```

---

## Methodology

The project follows a standard data analysis workflow:

1. **Business framing**  
   Define the churn problem in retention terms.

2. **Data understanding and validation**  
   Inspect dataset structure and establish the churn baseline.

3. **Data cleaning**  
   Correct invalid data types, resolve missing values, and validate duplicates.

4. **Feature engineering**  
   Create lifecycle, pricing, and service-depth features.

5. **Exploratory data analysis**  
   Compare customer segments against the 26.5% churn baseline.

6. **Predictive modeling**  
   Use Logistic Regression and Random Forest as prioritization tools.

7. **Recommendation design**  
   Convert insights into retention actions.

---

## Data Quality Summary

The main cleaning issue was `TotalCharges`, which was stored as text.

### Cleaning output
- `TotalCharges` converted to numeric
- **11 missing values** created after coercion
- Those rows correspond to customers with **tenure = 0**
- Missing values filled with **0**, which is consistent with newly activated customers
- **Duplicate rows:** 0
- **Remaining nulls after cleaning:** None

This preserves legitimate new customers instead of removing them from the analysis.

---

## Feature Engineering Summary

To improve interpretability, the analysis creates:
- `tenure_group`
- `charge_band`
- `num_addon_services`
- `is_new_customer`
- `has_internet`
- `is_high_value`

### Engineered feature highlights
- **New (0–12 months):** 2,186 customers
- **Loyal (49–72 months):** 2,239 customers
- **Very new customers (≤ 3 months):** 1,062
- **Average add-on services:** 2.04

These features make it easier to analyze churn through customer lifecycle and service engagement rather than raw columns alone.

---

## Exploratory Analysis

### 1) Churn Overview

![Churn Overview](assets/charts/01_churn_overview.png)

**Interpretation:**  
The baseline churn rate is **26.5%**. Churn is immediately concentrated in **new customers** and **month-to-month contracts**, suggesting that retention risk is highest where commitment is lowest.

### 2) Tenure and Lifecycle Risk

![Tenure Analysis](assets/charts/02_tenure_analysis.png)

**Key lifecycle churn rates**
- **New (0–12m):** 47%
- **Growing (13–24m):** 29%
- **Mature (25–48m):** 20%
- **Loyal (49–72m):** 10%

**Interpretation:**  
The first year is the main retention risk window. Customers churn early, then stabilize sharply over time.

### 3) Monthly Charges and Value Sensitivity

![Monthly Charges](assets/charts/03_monthly_charges.png)

**Key charge-band churn rates**
- **Low (<$35):** 11%
- **Mid ($35–65):** 23%
- **High ($65–90):** 36%
- **Premium (>$90):** 33%

**Interpretation:**  
Churn increases as charges move above the mid-range band, but the relationship is not purely linear. The pattern points more to a **value perception issue** than to price alone.

### 4) Internet Service Type

![Internet Service](assets/charts/04_internet_service.png)

**Key churn rates**
- **No internet:** 7%
- **DSL:** 19%
- **Fiber optic:** 42%

**Interpretation:**  
Fiber optic is a high-risk segment despite likely being a higher-value product. This suggests a gap in service experience, pricing fit, or competitive pressure.

### 5) Contract Type

![Contract Type](assets/charts/05_contract_type.png)

**Key churn rates**
- **Month-to-month:** 43%
- **One year:** 11%
- **Two year:** 3%

**Interpretation:**  
Contract structure is one of the strongest churn drivers in the dataset. Long-term contracts materially reduce churn.

### 6) Add-on Services and Protective Features

![Add-on Services](assets/charts/06_addon_services.png)

**Interpretation:**  
Service depth has a **non-linear** relationship with churn. Customers with 1–3 add-on services churn much more than deeply embedded customers with 5–6 services.  
At the same time, customers without **Online Security** or **Tech Support** churn at around **42%**, showing that certain protective services are associated with greater retention.

### 7) Payment Method

![Payment Method](assets/charts/07_payment_method.png)

**Key churn rates**
- **Electronic check:** 45%
- **Mailed check:** 19%
- **Bank transfer:** 17%
- **Credit card:** 15%

**Interpretation:**  
Payment method acts as a proxy for commitment and friction. Electronic check users are the most churn-prone, especially when combined with month-to-month contracts.

### 8) Demographics

![Demographics](assets/charts/08_demographics.png)

**Key churn rates**
- **Senior customers:** 42%
- **Non-seniors:** 24%
- **No partner:** 33%
- **With partner:** 20%
- **No dependents:** 31%
- **With dependents:** 15%

**Interpretation:**  
Demographics provide context for message targeting and service design, but they are weaker levers than contract, tenure, and payment structure.

### 9) Multi-Variable Risk Map

![Risk Map](assets/charts/09_multivariate_heatmap.png)

**Highest-risk intersection**
- **Month-to-month + New (0–12m): 51% churn**

**Interpretation:**  
The most important finding is not a single variable. It is the **interaction** of early lifecycle and low commitment.

### 10) Churn Driver Summary

![Churn Drivers](assets/charts/10_churn_drivers.png)

**Largest positive deviations from the 26.5% baseline**
- **Tenure < 12 months:** +21pp
- **Electronic check:** +19pp
- **Month-to-month:** +16pp
- **Fiber optic:** +15pp
- **No online security:** +15pp
- **Senior citizen:** +15pp
- **No tech support:** +15pp

**Interpretation:**  
The analysis points to three primary churn dimensions:
1. commitment,
2. lifecycle stage,
3. service value and support fit.

---

## Modeling Results

### Model performance

| Model | ROC-AUC | Accuracy | Precision (Churn) | Recall (Churn) |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.8393** | 0.74 | 0.51 | **0.80** |
| Random Forest | 0.8374 | **0.77** | **0.55** | 0.69 |

![Model Performance](assets/charts/11_model_performance.png)

**Interpretation:**  
Both models perform similarly, which suggests the churn signal is stable rather than model-specific.

- **Logistic Regression** captures more churned customers
- **Random Forest** produces fewer false positives

### Risk score distribution

![Risk Score Distribution](assets/charts/12_risk_score_distribution.png)

**Interpretation:**  
The model is strong enough to support **risk-based intervention prioritization**, even if it is not used as a standalone decision engine.

---

## Recommendations

### Priority 1 — Contract conversion
**Why:** Month-to-month churn is **42.7%**  
**Action:** Offer incentives to migrate early-tenure customers into 1-year plans.

### Priority 2 — Early lifecycle retention
**Why:** Tenure below 12 months churn is **47.4%**  
**Action:** Trigger retention workflows during the first 30 / 60 / 90 / 180 days.

### Priority 3 — Autopay adoption
**Why:** Electronic check churn is **45.3%**  
**Action:** Offer convenience and pricing nudges toward automatic payment methods.

### Priority 4 — Fiber optic review
**Why:** Fiber optic churn is **41.9%**  
**Action:** Audit service quality, support, and plan positioning for fiber customers.

### Priority 5 — Support-service bundling
**Why:** Lack of Online Security and Tech Support aligns with high churn  
**Action:** Test protective-service bundles for new or mid-risk customers.

---

## Decision Framework

| Initiative | Expected Impact | Feasibility | Priority |
|---|---|---|---|
| Contract conversion | High | High | 1 |
| Autopay adoption | High | High | 1 |
| Early lifecycle retention | High | Medium | 2 |
| Fiber optic service review | High | Medium | 2 |
| Support-service bundling | Medium | Medium | 3 |
| Demographic personalization | Medium | Medium | 3 |

This prevents the analysis from ending in a list of disconnected insights.  
The aim is to identify **where action should start first**.

---

## Analytical Limits

- The dataset is a **customer snapshot**, not a behavioral timeline
- Findings reflect **association**, not causal proof
- Variables such as support ticket history, outage frequency, NPS, or acquisition channel are not present
- Revenue impact cannot be estimated precisely without margin and customer lifetime assumptions

These constraints do not reduce the value of the analysis; they define the boundary of what can be concluded responsibly.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download the CSV from Kaggle and place it here:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Run the analysis
```bash
python src/telco_churn_analysis.py
```

Or pass a custom dataset path:

```bash
python src/telco_churn_analysis.py --input /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Charts will be saved to:

```text
assets/charts/
```

---

## Full Report with Quarto and GitHub Pages

This repository includes:
- `index.qmd` — the full report source
- `_quarto.yml` — project configuration
- `.github/workflows/publish-quarto.yml` — automated render and deploy workflow

### To publish
1. Push the repository to GitHub
2. Go to **Settings → Pages**
3. Set **Build and deployment** to **GitHub Actions**
4. Commit and push to `main`

The workflow will render the Quarto report and deploy it to GitHub Pages.

---

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Quarto
- GitHub Pages

---

## Why this public project structure works

GitHub does not reliably surface notebook outputs in the way a report should be read.  
This repository solves that by separating responsibilities:

- **README.md** → curated case study
- **src/telco_churn_analysis.py** → reproducible source code
- **assets/charts/** → static visuals for GitHub rendering
- **index.qmd + GitHub Pages** → full published report

That makes the project easier to review for both recruiters and technical readers.
