"""
=============================================================================
  TELCO CUSTOMER CHURN — DATA ANALYSIS & STORYTELLING
  Senior Data Analyst Portfolio Project
=============================================================================

Dataset  : IBM Telco Customer Churn  (7,043 rows × 21 columns)
Metric   : Churn (Yes / No)
Goal     : Understand WHY customers churn, WHICH segments are most at risk,
           and WHAT actions the business should take.
Author   : Data Analyst
=============================================================================
"""

# ─── 0. IMPORTS ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # non-interactive backend

# ─── GLOBAL STYLE ────────────────────────────────────────────────────────────
PALETTE_CHURN = {'Yes': '#E74C3C', 'No': '#2ECC71'}
CHURN_COLORS  = ['#2ECC71', '#E74C3C']
BG_COLOR      = '#FAFAFA'
ACCENT        = '#2C3E50'
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
CHARTS_DIR = ROOT_DIR / 'assets' / 'charts'
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor':   BG_COLOR,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.labelcolor':  ACCENT,
    'xtick.color':      ACCENT,
    'ytick.color':      ACCENT,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

def save_fig(name, tight=True):
    path = CHARTS_DIR / f'{name}.png'
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f'  [saved] {name}.png -> {path}')


parser = argparse.ArgumentParser(description='Run Telco Customer Churn analysis and save charts to assets/charts.')
parser.add_argument('--input', type=str, default=str(DEFAULT_DATA_PATH),
                    help='Path to the Telco Customer Churn CSV file')
args = parser.parse_args()
DATA_PATH = Path(args.input)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Download the Kaggle CSV and place it under "
        f"{DEFAULT_DATA_PATH} or pass a custom path with --input."
    )

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — DATA LOADING & OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART 2 — DATA LOADING & OVERVIEW')
print('='*60)

df_raw = pd.read_csv(DATA_PATH)

print(f'\nShape        : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns')
print(f'Memory usage : {df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB')
print('\n--- Data Types ---')
print(df_raw.dtypes.to_string())

# Group variables by business meaning
DEMOGRAPHIC   = ['gender','SeniorCitizen','Partner','Dependents']
SERVICE_CORE  = ['PhoneService','MultipleLines','InternetService']
SERVICE_ADDON = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                 'TechSupport','StreamingTV','StreamingMovies']
ACCOUNT       = ['tenure','Contract','PaperlessBilling']
PAYMENT       = ['PaymentMethod','MonthlyCharges','TotalCharges']
TARGET        = 'Churn'

print('\n--- Churn Distribution ---')
churn_counts = df_raw['Churn'].value_counts()
churn_pct    = df_raw['Churn'].value_counts(normalize=True) * 100
for label in ['No','Yes']:
    print(f'  {label}: {churn_counts[label]:,} ({churn_pct[label]:.1f}%)')


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART 3 — DATA CLEANING')
print('='*60)

df = df_raw.copy()

# 3.1 Fix TotalCharges (stored as string in raw data)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
n_missing_tc = df['TotalCharges'].isna().sum()
print(f'\nTotalCharges NaN after coerce : {n_missing_tc}')
# All missing are new customers (tenure=0), fill with 0
df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = 0
print('  → Filled with 0 (all are tenure=0 new customers)')

# 3.2 Check duplicates
n_dup = df.duplicated().sum()
print(f'Duplicate rows : {n_dup}')

# 3.3 Check remaining nulls
remaining_null = df.isnull().sum()
print('\nRemaining nulls:')
print(remaining_null[remaining_null > 0] if remaining_null.any() else '  None ✓')

# 3.4 Encode target
df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)

# 3.5 SeniorCitizen is 0/1 int, keep as-is but make a label for display
df['SeniorCitizen_label'] = df['SeniorCitizen'].map({0:'Non-Senior', 1:'Senior'})

print('\n✓ Cleaning complete. Working dataset shape:', df.shape)


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART 4 — FEATURE ENGINEERING')
print('='*60)

# 4.1 Tenure groups — translate raw months into customer lifecycle stage
tenure_bins   = [0, 12, 24, 48, 72]
tenure_labels = ['New (0-12m)', 'Growing (13-24m)', 'Mature (25-48m)', 'Loyal (49-72m)']
df['tenure_group'] = pd.cut(df['tenure'], bins=tenure_bins,
                             labels=tenure_labels, include_lowest=True)

# 4.2 Monthly charge bands
charge_bins   = [0, 35, 65, 90, 200]
charge_labels = ['Low (<$35)', 'Mid ($35-65)', 'High ($65-90)', 'Premium (>$90)']
df['charge_band'] = pd.cut(df['MonthlyCharges'], bins=charge_bins,
                            labels=charge_labels)

# 4.3 Number of add-on services subscribed (security, backup, protection, etc.)
addon_cols = SERVICE_ADDON
df['num_addon_services'] = df[addon_cols].apply(
    lambda row: sum(v == 'Yes' for v in row), axis=1
)

# 4.4 Is new customer (tenure <= 3 months)
df['is_new_customer'] = (df['tenure'] <= 3).astype(int)

# 4.5 Has any internet service
df['has_internet'] = (df['InternetService'] != 'No').astype(int)

# 4.6 Revenue flag: above-median monthly charges
median_charge = df['MonthlyCharges'].median()
df['is_high_value'] = (df['MonthlyCharges'] > median_charge).astype(int)

print(f'\ntenure_group distribution:')
print(df['tenure_group'].value_counts().sort_index().to_string())
print(f'\ncharge_band distribution:')
print(df['charge_band'].value_counts().sort_index().to_string())
print(f'\nAvg add-on services: {df["num_addon_services"].mean():.2f}')
print(f'New customers (<=3m): {df["is_new_customer"].sum():,} ({df["is_new_customer"].mean()*100:.1f}%)')


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART 5 — EDA')
print('='*60)

churn_rate_overall = df['Churn_binary'].mean()
print(f'\nOverall churn rate: {churn_rate_overall*100:.1f}%')

# ─── CHART 1: Churn Overview KPI Banner ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Telco Customer Churn — High-Level Overview', fontsize=15,
             fontweight='bold', color=ACCENT, y=1.02)

# KPI 1: Churn donut
sizes = [churn_counts['No'], churn_counts['Yes']]
wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
axes[0].pie(sizes, labels=['Retained', 'Churned'], colors=CHURN_COLORS,
            autopct='%1.1f%%', startangle=90, wedgeprops=wedge_props,
            textprops={'fontsize': 11})
axes[0].set_title('Overall Churn Rate', fontweight='bold')

# KPI 2: Churn by tenure group
tg_churn = df.groupby('tenure_group', observed=True)['Churn_binary'].mean() * 100
bars = axes[1].bar(tg_churn.index, tg_churn.values,
                   color=['#E74C3C','#E67E22','#3498DB','#2ECC71'], edgecolor='white')
axes[1].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.2, label='Overall avg')
axes[1].set_title('Churn Rate by Customer Lifecycle', fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, 75)
for bar, val in zip(bars, tg_churn.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
                 f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].tick_params(axis='x', rotation=15)

# KPI 3: Churn by Contract
ct_churn = df.groupby('Contract')['Churn_binary'].mean() * 100
colors_ct = ['#E74C3C', '#F39C12', '#2ECC71']
bars2 = axes[2].bar(ct_churn.index, ct_churn.values, color=colors_ct, edgecolor='white')
axes[2].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.2)
axes[2].set_title('Churn Rate by Contract Type', fontweight='bold')
axes[2].set_ylabel('Churn Rate (%)')
axes[2].set_ylim(0, 75)
for bar, val in zip(bars2, ct_churn.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
                 f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
axes[2].tick_params(axis='x', rotation=10)

save_fig('01_churn_overview')

# ─── CHART 2: Tenure Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Customer Tenure: How Long Before They Leave?',
             fontsize=14, fontweight='bold', color=ACCENT)

# KDE by churn
for label, color in PALETTE_CHURN.items():
    subset = df[df['Churn'] == label]['tenure']
    axes[0].hist(subset, bins=30, alpha=0.5, color=color, label=f'Churn={label}',
                 density=True, edgecolor='white')
    subset.plot.kde(ax=axes[0], color=color, linewidth=2)
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Density')
axes[0].set_title('Tenure Distribution — Churned vs Retained')
axes[0].legend()
axes[0].axvline(df[df['Churn']=='Yes']['tenure'].median(), color='#E74C3C',
                linestyle=':', linewidth=1.5, label='Median churn tenure')

# Churn rate by tenure group with count annotations
tg = df.groupby('tenure_group', observed=True).agg(
    churn_rate=('Churn_binary','mean'),
    count=('Churn_binary','count')
).reset_index()
tg['churn_pct'] = tg['churn_rate'] * 100

bar_colors = ['#E74C3C' if r > churn_rate_overall else '#3498DB' for r in tg['churn_rate']]
bars = axes[1].bar(tg['tenure_group'].astype(str), tg['churn_pct'],
                   color=bar_colors, edgecolor='white', width=0.6)
axes[1].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5,
                label=f'Avg {churn_rate_overall*100:.0f}%')
axes[1].set_title('Churn Rate by Lifecycle Stage')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, 80)
axes[1].legend()
for bar, row in zip(bars, tg.itertuples()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
                 f'{row.churn_pct:.0f}%\n(n={row.count:,})', ha='center', fontsize=9)
axes[1].tick_params(axis='x', rotation=10)

save_fig('02_tenure_analysis')

# ─── CHART 3: Monthly Charges vs Churn ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Monthly Charges: Does Price Drive Churn?',
             fontsize=14, fontweight='bold', color=ACCENT)

# Box plot
df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[0],
           boxprops=dict(color=ACCENT),
           whiskerprops=dict(color=ACCENT),
           capprops=dict(color=ACCENT),
           medianprops=dict(color='#E74C3C', linewidth=2),
           flierprops=dict(marker='o', alpha=0.3, markersize=3))
axes[0].set_title('Monthly Charges by Churn Status')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Monthly Charges ($)')
plt.sca(axes[0])
plt.title('Monthly Charges by Churn Status')
fig.suptitle('Monthly Charges: Does Price Drive Churn?',
             fontsize=14, fontweight='bold', color=ACCENT)

# Churn rate by charge band
cb = df.groupby('charge_band', observed=True)['Churn_binary'].agg(['mean','count']).reset_index()
cb['pct'] = cb['mean'] * 100
clrs = ['#2ECC71','#F39C12','#E74C3C','#8E44AD']
bars = axes[1].bar(cb['charge_band'].astype(str), cb['pct'], color=clrs, edgecolor='white')
axes[1].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5)
axes[1].set_title('Churn Rate by Charge Band')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, 80)
for bar, row in zip(bars, cb.itertuples()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
                 f'{row.pct:.0f}%\n(n={row.count:,})', ha='center', fontsize=9)

save_fig('03_monthly_charges')

# ─── CHART 4: Internet Service ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Internet Service: The Fiber Optic Paradox',
             fontsize=14, fontweight='bold', color=ACCENT)

# Count stacked bar by internet service
is_ct = df.groupby(['InternetService','Churn']).size().unstack().fillna(0)
is_ct_pct = is_ct.div(is_ct.sum(axis=1), axis=0) * 100
is_ct_pct.plot(kind='bar', stacked=True, ax=axes[0],
               color=CHURN_COLORS, edgecolor='white', width=0.6)
axes[0].set_title('Customer Composition by Internet Service')
axes[0].set_xlabel('')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(['Retained','Churned'], loc='upper right')
axes[0].tick_params(axis='x', rotation=0)

# Churn rate comparison
is_churn = df.groupby('InternetService')['Churn_binary'].mean() * 100
bars = axes[1].barh(is_churn.index, is_churn.values,
                    color=['#E74C3C','#3498DB','#2ECC71'], edgecolor='white')
axes[1].axvline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5,
                label=f'Avg {churn_rate_overall*100:.0f}%')
axes[1].set_title('Churn Rate by Internet Service Type')
axes[1].set_xlabel('Churn Rate (%)')
axes[1].legend()
for bar, val in zip(bars, is_churn.values):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}%', va='center', fontweight='bold')

save_fig('04_internet_service')

# ─── CHART 5: Contract Type Deep Dive ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Contract Type: The #1 Predictor of Churn',
             fontsize=14, fontweight='bold', color=ACCENT)

# Churn counts by contract
ct_ct = df.groupby(['Contract','Churn']).size().unstack().fillna(0)
ct_ct.plot(kind='bar', ax=axes[0], color=CHURN_COLORS, edgecolor='white', width=0.6)
axes[0].set_title('Churned vs Retained by Contract')
axes[0].set_xlabel('')
axes[0].set_ylabel('Number of Customers')
axes[0].legend(['Retained','Churned'])
axes[0].tick_params(axis='x', rotation=10)

# Churn rate + tenure median inside contract
ct_stats = df.groupby('Contract').agg(
    churn_rate=('Churn_binary','mean'),
    median_tenure=('tenure','median'),
    n=('Churn_binary','count')
).reset_index()
ct_stats['churn_pct'] = ct_stats['churn_rate'] * 100

colors_ct = ['#E74C3C','#F39C12','#2ECC71']
bars = axes[1].bar(ct_stats['Contract'], ct_stats['churn_pct'],
                   color=colors_ct, edgecolor='white', width=0.5)
axes[1].set_ylim(0, 80)
axes[1].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5)
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_title('Churn Rate & Median Tenure by Contract')
for bar, row in zip(bars, ct_stats.itertuples()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
                 f'{row.churn_pct:.0f}%\nmedian {row.median_tenure:.0f}m',
                 ha='center', fontsize=9, fontweight='bold')
axes[1].tick_params(axis='x', rotation=10)

save_fig('05_contract_type')

# ─── CHART 6: Add-on Services & Security ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Support Services: Churn Shield or Missed Opportunity?',
             fontsize=14, fontweight='bold', color=ACCENT)

# Churn rate by number of add-on services
addon_churn = df.groupby('num_addon_services')['Churn_binary'].mean() * 100
addon_n     = df.groupby('num_addon_services')['Churn_binary'].count()
clrs_addon  = ['#E74C3C' if v > churn_rate_overall*100 else '#2ECC71'
               for v in addon_churn.values]
axes[0].bar(addon_churn.index, addon_churn.values, color=clrs_addon, edgecolor='white')
axes[0].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5,
                label=f'Avg {churn_rate_overall*100:.0f}%')
axes[0].set_xlabel('Number of Add-on Services')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].set_title('Churn Rate vs Add-on Service Depth')
axes[0].legend()
for x, (rate, n) in enumerate(zip(addon_churn.values, addon_n.values)):
    axes[0].text(x, rate+1.5, f'{rate:.0f}%', ha='center', fontsize=9)

# Churn rate for key protective services
protect_services = {
    'OnlineSecurity': 'Online Security',
    'TechSupport':    'Tech Support',
    'OnlineBackup':   'Online Backup',
    'DeviceProtection':'Device Protection'
}
records = []
for col, label in protect_services.items():
    for val in ['Yes','No']:
        mask = df[col] == val
        rate = df.loc[mask, 'Churn_binary'].mean() * 100
        n    = mask.sum()
        records.append({'Service': label, 'Has Service': val, 'Churn Rate': rate, 'n': n})
prot_df = pd.DataFrame(records)

prot_pivot = prot_df.pivot(index='Service', columns='Has Service', values='Churn Rate')
prot_pivot[['No','Yes']].plot(kind='bar', ax=axes[1],
                               color=['#E74C3C','#2ECC71'], edgecolor='white', width=0.6)
axes[1].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5)
axes[1].set_title('Churn Rate: With vs Without Protective Services')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_xlabel('')
axes[1].set_ylim(0, 80)
axes[1].legend(['Without Service','With Service'])
axes[1].tick_params(axis='x', rotation=15)

save_fig('06_addon_services')

# ─── CHART 7: Payment Method ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Payment Method: Friction, Commitment & Churn',
             fontsize=14, fontweight='bold', color=ACCENT)

pm_churn = df.groupby('PaymentMethod')['Churn_binary'].agg(['mean','count']).reset_index()
pm_churn['pct'] = pm_churn['mean'] * 100
pm_churn = pm_churn.sort_values('pct', ascending=False)

short_labels = {
    'Electronic check': 'Elec. Check',
    'Mailed check': 'Mailed Check',
    'Bank transfer (automatic)': 'Bank Transfer',
    'Credit card (automatic)': 'Credit Card'
}
pm_churn['label'] = pm_churn['PaymentMethod'].map(short_labels)

clrs_pm = ['#E74C3C' if v > churn_rate_overall*100 else '#2ECC71' for v in pm_churn['pct']]
bars = axes[0].bar(pm_churn['label'], pm_churn['pct'], color=clrs_pm, edgecolor='white')
axes[0].axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5)
axes[0].set_title('Churn Rate by Payment Method')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].set_ylim(0, 60)
for bar, row in zip(bars, pm_churn.itertuples()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.2,
                 f'{row.pct:.0f}%', ha='center', fontsize=10, fontweight='bold')
axes[0].tick_params(axis='x', rotation=10)

# Payment × Contract heatmap
pm_ct_pivot = df.groupby(['PaymentMethod','Contract'])['Churn_binary'].mean() * 100
pm_ct_pivot = pm_ct_pivot.unstack()
pm_ct_pivot.index = [short_labels.get(i, i) for i in pm_ct_pivot.index]
sns.heatmap(pm_ct_pivot, ax=axes[1], annot=True, fmt='.0f', cmap='RdYlGn_r',
            linewidths=0.5, cbar_kws={'label': 'Churn Rate (%)'})
axes[1].set_title('Churn Rate: Payment × Contract Heatmap')
axes[1].set_xlabel('Contract Type')
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', rotation=0)

save_fig('07_payment_method')

# ─── CHART 8: Demographic Analysis ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Customer Demographics: Who Is Most at Risk?',
             fontsize=14, fontweight='bold', color=ACCENT)

for ax, col, title in zip(
    axes,
    ['SeniorCitizen_label','Partner','Dependents'],
    ['Senior vs Non-Senior','Has Partner','Has Dependents']
):
    demo_churn = df.groupby(col)['Churn_binary'].mean() * 100
    demo_n     = df.groupby(col)['Churn_binary'].count()
    clrs = ['#E74C3C' if v > churn_rate_overall*100 else '#2ECC71' for v in demo_churn.values]
    ax.bar(demo_churn.index, demo_churn.values, color=clrs, edgecolor='white', width=0.5)
    ax.axhline(churn_rate_overall*100, color='gray', linestyle='--', linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 60)
    for x, (rate, n) in enumerate(zip(demo_churn.values, demo_n.values)):
        ax.text(x, rate+1.5, f'{rate:.0f}%\n(n={n:,})', ha='center', fontsize=9)

save_fig('08_demographics')

# ─── CHART 9: Multivariate — High-Risk Segment Heatmap ───────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('High-Risk Segment Map: Contract × Internet × Tenure',
             fontsize=14, fontweight='bold', color=ACCENT)

heat_data = df.groupby(['Contract','tenure_group'], observed=True)['Churn_binary'].mean() * 100
heat_pivot = heat_data.unstack()
sns.heatmap(heat_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
            linewidths=0.8, ax=ax, cbar_kws={'label': 'Churn Rate (%)'},
            vmin=0, vmax=80)
ax.set_title('Churn Rate (%) — Contract Type × Lifecycle Stage', pad=10)
ax.set_xlabel('Customer Lifecycle Stage')
ax.set_ylabel('Contract Type')
ax.tick_params(axis='x', rotation=10)
ax.tick_params(axis='y', rotation=0)

save_fig('09_multivariate_heatmap')

# ─── CHART 10: Summary — Top Churn Drivers ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

# Compute churn rate for each meaningful segment vs baseline
def churn_delta(mask, label):
    rate = df.loc[mask, 'Churn_binary'].mean() * 100
    return {'Segment': label, 'Churn Rate (%)': rate,
            'Delta vs Avg (pp)': rate - churn_rate_overall*100}

segments = [
    churn_delta(df['Contract'] == 'Month-to-month',          'Contract: Month-to-month'),
    churn_delta(df['tenure'] <= 12,                          'Tenure: < 12 months'),
    churn_delta(df['InternetService'] == 'Fiber optic',      'Internet: Fiber Optic'),
    churn_delta(df['PaymentMethod'] == 'Electronic check',   'Payment: Electronic Check'),
    churn_delta(df['OnlineSecurity'] == 'No',                'No Online Security'),
    churn_delta(df['TechSupport'] == 'No',                   'No Tech Support'),
    churn_delta(df['num_addon_services'] == 0,               '0 Add-on Services'),
    churn_delta(df['SeniorCitizen'] == 1,                    'Senior Citizen'),
    churn_delta(df['MonthlyCharges'] > 75,                   'Monthly Charges > $75'),
    churn_delta(df['Dependents'] == 'No',                    'No Dependents'),
]
seg_df = pd.DataFrame(segments).sort_values('Delta vs Avg (pp)', ascending=True)

colors = ['#E74C3C' if d > 0 else '#2ECC71' for d in seg_df['Delta vs Avg (pp)']]
bars = ax.barh(seg_df['Segment'], seg_df['Delta vs Avg (pp)'], color=colors, edgecolor='white')
ax.axvline(0, color=ACCENT, linewidth=1)
ax.set_xlabel('Churn Rate Difference vs Overall Average (percentage points)')
ax.set_title(f'Key Risk Drivers — Deviation from Overall Churn Rate ({churn_rate_overall*100:.0f}%)',
             fontweight='bold')
for bar, row in zip(bars, seg_df.itertuples()):
    x_pos = row._3 + 0.3 if row._3 > 0 else row._3 - 0.3
    ha = 'left' if row._3 > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{row._2:.0f}% ({row._3:+.0f}pp)', va='center', fontsize=9, fontweight='bold')

save_fig('10_churn_drivers')

print('\n✓ All EDA charts saved.')


# ═══════════════════════════════════════════════════════════════════════════
# PART 7 — PREDICTIVE MODELING
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART 7 — PREDICTIVE MODELING')
print('='*60)

# Prepare features
model_df = df.copy()

# Encode categoricals
cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
            'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies','Contract',
            'PaperlessBilling','PaymentMethod','Churn']

le = LabelEncoder()
for col in cat_cols:
    model_df[col] = le.fit_transform(model_df[col].astype(str))

FEATURE_COLS = ['gender','SeniorCitizen','Partner','Dependents','tenure',
                'PhoneService','MultipleLines','InternetService',
                'OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies',
                'Contract','PaperlessBilling','PaymentMethod',
                'MonthlyCharges','TotalCharges',
                'num_addon_services','is_high_value']

X = model_df[FEATURE_COLS]
y = model_df['Churn_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f'\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}')
print(f'Train churn rate: {y_train.mean()*100:.1f}%')
print(f'Test  churn rate: {y_test.mean()*100:.1f}%')

# ── Model 1: Logistic Regression ─────────────────────────────────────────────
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
lr_pred  = lr_pipe.predict(X_test)
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f'\nLogistic Regression AUC: {lr_auc:.4f}')
print(classification_report(y_test, lr_pred, target_names=['Retained','Churned']))

# ── Model 2: Random Forest ────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred  = rf.predict(X_test)
rf_auc   = roc_auc_score(y_test, rf_proba)
print(f'\nRandom Forest AUC: {rf_auc:.4f}')
print(classification_report(y_test, rf_pred, target_names=['Retained','Churned']))

# ── Model Evaluation Charts ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Performance — Logistic Regression vs Random Forest',
             fontsize=14, fontweight='bold', color=ACCENT)

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
axes[0].plot(fpr_lr, tpr_lr, color='#3498DB', linewidth=2,
             label=f'Logistic Reg. (AUC={lr_auc:.3f})')
axes[0].plot(fpr_rf, tpr_rf, color='#E74C3C', linewidth=2,
             label=f'Random Forest (AUC={rf_auc:.3f})')
axes[0].plot([0,1],[0,1], 'k--', linewidth=1, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend(fontsize=9)

# Confusion Matrix — Random Forest
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Retained','Churned'])
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title('Confusion Matrix — Random Forest')

# Feature Importance
fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
axes[2].barh(fi.index, fi.values, color='#3498DB', edgecolor='white')
axes[2].set_title('Top 15 Feature Importances (RF)')
axes[2].set_xlabel('Importance Score')

save_fig('11_model_performance')

# ── Churn Risk Score Distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(rf_proba[y_test==0], bins=40, alpha=0.6, color='#2ECC71',
        label='Retained (actual)', density=True)
ax.hist(rf_proba[y_test==1], bins=40, alpha=0.6, color='#E74C3C',
        label='Churned (actual)', density=True)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision threshold 0.5')
ax.set_xlabel('Predicted Churn Probability')
ax.set_ylabel('Density')
ax.set_title('Churn Risk Score Distribution — Random Forest', fontweight='bold')
ax.legend()
save_fig('12_risk_score_distribution')

print('\n✓ Modeling complete.')
print('\n' + '='*60)
print(f'ANALYSIS COMPLETE — All charts saved to {CHARTS_DIR}')
print('='*60)

# ── Key Metrics Summary ────────────────────────────────────────────────────────
print('\n--- KEY METRICS SUMMARY ---')
print(f'Overall Churn Rate      : {churn_rate_overall*100:.1f}%')
print(f'Month-to-month Churn    : {df[df.Contract=="Month-to-month"]["Churn_binary"].mean()*100:.1f}%')
print(f'Fiber Optic Churn       : {df[df.InternetService=="Fiber optic"]["Churn_binary"].mean()*100:.1f}%')
print(f'Electronic Check Churn  : {df[df.PaymentMethod=="Electronic check"]["Churn_binary"].mean()*100:.1f}%')
print(f'New Customer (<12m) Churn: {df[df.tenure<=12]["Churn_binary"].mean()*100:.1f}%')
print(f'0 Add-on Services Churn : {df[df.num_addon_services==0]["Churn_binary"].mean()*100:.1f}%')
print(f'Random Forest AUC       : {rf_auc:.4f}')
