# Heart disease — practice project

**Notebook:** [`heart.ipynb`](heart.ipynb)  
**Data:** [`heart.csv`](heart.csv)

Clinical-style tabular data with a binary outcome **`HeartDisease`** (0/1). This README matches the **current** notebook in this folder: EDA, treating some zeros as missing values, encoding, and scaling.

**Educational use only**—not for diagnosis.

---

## What this notebook does (flow)

1. **Load** `heart.csv` and inspect **`columns`**, **`shape`**, **`info`**, **`describe`**, **duplicate** counts, **nulls**, and **`HeartDisease`** counts (with a simple bar plot).  
2. **Histograms** for several numeric fields (via a small plotting helper).  
3. **Imputation / cleaning:** **`Cholesterol == 0`** and **`RestingBP == 0`** are treated as invalid or missing and replaced with the **mean** of non-zero values in that column (a modeling choice—zeros may mean “unknown” in this dataset).  
4. **Plots:** **`countplot`** with **`hue=HeartDisease`** for **`Sex`**, **`ChestPainType`**, **`FastingBS`**; **`boxplot`** / **`violinplot`** for numeric vs disease; **correlation heatmap** on numeric columns.  
5. **Optional:** a cell installs **`sheryanalysis`** (`pip install sheryanalysis==0.1.0`) and runs `import sheryanalysis as sh` for automated profiling—requires network; skip if you do not want extra dependencies.  
6. **Encoding:** **`pd.get_dummies(df, drop_first=True)`** on the frame (all default **object/category** columns become dummies; numeric columns like **`HeartDisease`** stay as-is if not object-typed).  
7. **Cast** encoded columns to **int** where appropriate.  
8. **`StandardScaler`** on **`Age`**, **`RestingBP`**, **`Cholesterol`**, **`MaxHR`**, **`Oldpeak`** (continuous-style fields).

---

## Concepts and definitions (this project)

### Exploratory data analysis (EDA)

**EDA** means exploring the table before modeling: **shape**, **dtypes**, **summaries**, **missingness**, **duplicates**, and **plots**. Here EDA also motivates **cleaning** (e.g. suspicious zeros in **`Cholesterol`** / **`RestingBP`**).

### Treating zeros as missing (imputation)

**Imputation** fills missing or invalid values. Replacing **0** with the **mean of non-zero** values assumes **0** is not a real measurement in those columns. That assumption should be validated from data documentation; otherwise you risk **bias**. Alternatives: mark as missing and use dedicated imputers, or use models that handle missing data.

### Correlation heatmap

The **heatmap** shows **Pearson**-style **linear** correlations **between numeric columns** only. It is descriptive; it does not replace multivariate modeling or causal reasoning.

### `pd.get_dummies` on the full DataFrame

**One-hot encoding** turns categorical columns into **0/1** columns. **`drop_first=True`** drops one reference category **per** encoded column to reduce **multicollinearity** when you later use models with an intercept. Encoding **the whole frame at once** is convenient; verify which columns pandas treats as categorical (**`object`** / **`category`**) so you do not accidentally expand columns you did not intend to.

### `StandardScaler`

**Standardization** rescales selected columns to roughly **mean 0** and **std 1**. That can help algorithms that are sensitive to feature scale. **Best practice:** **`fit`** on **training** data only, then **`transform`** train and test. This notebook fits on the **full** prepared table for simplicity (**possible leakage** if you later split for evaluation).

### Binary target **`HeartDisease`**

**0/1** labels define a **binary** outcome (e.g. presence vs absence of disease in this dataset’s encoding). Any model or metric you add next should respect whether the problem is **classification** and whether classes are **balanced**.

---

Sibling projects: [`../README.md`](../README.md).
