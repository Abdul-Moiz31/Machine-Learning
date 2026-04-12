# Insurance cost project (Day 1)

**Notebook:** [`insurance.ipynb`](insurance.ipynb)  
**Data:** [`insurance-checkpoint.csv`](insurance-checkpoint.csv) — keep the notebook in this folder so `pd.read_csv("insurance-checkpoint.csv")` works.

**Target for analysis:** **`charges`** (medical insurance cost). The notebook prepares data for later modeling: EDA, cleaning, encodings, feature engineering, correlation and chi-square screens, scaling, and a modeling-ready **`final_df`**.

---

## What this notebook does (flow)

1. Load data and explore with tables and plots.  
2. Clean: duplicates, map **`sex`** / **`smoker`** to numeric flags, rename to **`is_male`** / **`is_smoker`**.  
3. One-hot **`region`** and later **`bmi_category`**.  
4. Bin BMI with **`pd.cut`**, then dummy-encode bands.  
5. **`StandardScaler`** on **`age`**, **`bmi`**, **`children`**.  
6. **Pearson** correlation of features with **`charges`**.  
7. **Chi-square:** **`pd.qcut`** on **`charges`** into quartile bins, crosstab each categorical feature vs bins.  
8. Build **`final_df`** with selected columns.

---

## Concepts and definitions (this project)

### Python environment

The Jupyter **kernel** uses one Python interpreter. Packages must be installed for **that** interpreter or imports fail (`ModuleNotFoundError`). Use `import sys; print(sys.executable)` and install with that environment’s `pip`.

**Libraries:** **NumPy**, **pandas**, **Matplotlib**, **Seaborn**, **scikit-learn** (`StandardScaler`), **SciPy** (`pearsonr`, `chi2_contingency`).

### Exploratory data analysis (EDA)

**EDA** means understanding the data *before* heavy modeling: mistakes in types, duplicates, missing values, and rough relationships.

- **`shape`**, **`head`**, **`info`**, **`describe`**, **`isnull().sum()`**  
- **`value_counts()`** on categories  
- Plots: count plots, box plots, histograms, **correlation heatmap** on numeric columns only  

The heatmap shows **linear** pairwise links among numerics; it does not encode categorics until you transform them.

### Data cleaning

**Cleaning** makes the table consistent: **`drop_duplicates`**, map binary text columns to **0/1**, keep names clear (**`is_male`**, **`is_smoker`**). The numeric coding is arbitrary but must stay consistent.

### One-hot encoding (`get_dummies`) and `drop_first`

**Nominal** categories (e.g. **`region`**) have no natural order. **`pd.get_dummies(..., drop_first=True)`** creates 0/1 columns and drops one **reference** level per source column so dummy columns are not linearly dependent with an intercept (**multicollinearity**). The dropped level is implied when all related dummies are 0.

### Feature engineering: BMI bands (`pd.cut`)

**Feature engineering** creates new inputs from old ones. **`bmi_category`** uses **`pd.cut`** with clinical-style BMI edges (underweight / normal / overweight / obesity). That can capture **non-linear** or threshold effects vs using BMI alone. Then **`get_dummies`** on **`bmi_category`** with **`drop_first=True`**.

### Standardization (`StandardScaler`)

**StandardScaler** rescales columns to roughly **mean 0** and **std 1**. Here it is applied to **`age`**, **`bmi`**, **`children`**. In production, **`fit`** only on **training** data, then **`transform`** train and test (avoid leakage). This notebook fits on the full table for simplicity.

### Pearson correlation

**Pearson *r*** measures **linear** association between two numeric series (−1 to 1). **`scipy.stats.pearsonr`** also returns a **p-value** (interpret carefully on messy data). Comparing many features to **`charges`** helps screen linear drivers; **correlation ≠ causation**. Do not read too much into correlating **`charges`** with itself.

### Chi-square and binned charges

**Chi-square test of independence** asks if two **categorical** variables are associated. Build a **contingency table** with **`pd.crosstab`**, then **`chi2_contingency`**.

This notebook bins **`charges`** with **`pd.qcut(..., q=4)`** (**quartiles**, roughly equal counts). Each categorical feature is cross-tabbed with **`charges_bin`**. Compare **p-value** to **α = 0.05** for **exploratory** screening only; many tests ⇒ **multiple testing** caveats.

### `final_df`

A explicit slice of columns (features + **`charges`**) used as the modeling view—bookkeeping so you do not pass stray or raw-text columns into a model.

---

Workspace learning log: [`../../README.md`](../../README.md).  
Sibling projects: see [`../README.md`](../README.md) for the list.
