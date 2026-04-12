# Housing prices — practice project (synthetic)

**Notebook:** [`housing.ipynb`](housing.ipynb)  
**Data:** [`housing-sample.csv`](housing-sample.csv)

The CSV is **synthetic** (made-up listings with a simple price formula). It is **not** real market data—safe to share and good for repeating a full prep pipeline without PII concerns.

**Target:** **`price`**. Workflow mirrors the insurance project: EDA → encodings → age buckets → scaling → Pearson vs price → chi-square vs binned price → **`final_df`**.

---

## What this notebook does (flow)

1. Load CSV, EDA (shape, types, nulls, plots).  
2. Copy to **`df_cleaned`**, **`drop_duplicates`**, map **`garage`** to **`has_garage`**.  
3. **`get_dummies`** on **`neighborhood`** with **`drop_first=True`**.  
4. **`pd.cut`** on **`age_years`** into **`age_bracket`** (`new` / `mature` / `older`), then **`get_dummies`** with **`drop_first=True`** (reference **`new`**), leaving **`age_bracket_mature`** and **`age_bracket_older`**.  
5. **`StandardScaler`** on **`sqft_living`**, **`bedrooms`**, **`age_years`**.  
6. **Pearson** vs **`price`**.  
7. **`pd.qcut`** on **`price`** → **`price_bin`**; chi-square each categorical vs bins.  
8. **`final_df`** (drops **`price_bin`** for modeling).

---

## Concepts and definitions (this project)

### Why synthetic data

You control the story: no licensing or privacy issues, and you can still practice **EDA**, **encoding**, and **statistics** end to end.

### Exploratory data analysis (EDA)

Same idea as insurance: **`head`**, **`info`**, **`describe`**, **`isnull`**, plots (histograms, box plots, heatmap on **numeric** columns). Builds intuition before changing the table.

### `get_dummies` and `drop_first` on **`neighborhood`**

**Nominal** areas (north / east / south) become 0/1 columns. With **`drop_first=True`**, pandas drops one reference category. In this dataset the dropped dummy is **`neighborhood_east`** (alphabetical ordering of level names), so you keep **`neighborhood_north`** and **`neighborhood_south`**. **Always check `df_cleaned.columns` after encoding**—reference levels depend on pandas’ ordering.

### Age brackets (`pd.cut`)

**Feature engineering:** **`pd.cut`** maps **`age_years`** into **`age_bracket`** with labels **`new`**, **`mature`**, **`older`** (bins `[-1, 15, 40, 100]`). Dummy-encode with **`drop_first=True`**: the first label in **category order** is **`new`**, so it is dropped and you keep **`age_bracket_mature`** and **`age_bracket_older`** (baseline = **new** when both dummies are 0).

### `StandardScaler`

Applied to selected **continuous** numerics (**`sqft_living`**, **`bedrooms`**, **`age_years`**). **Dummy** columns stay 0/1. Fitting on the full table before a train/test split is a **simplification**; production workflows **`fit`** on train only.

### Pearson correlation with **`price`**

**Pearson *r*** measures **linear** association. Sorting |*r*| helps rank features vs **`price`**. **Correlation is not causation.** Exclude **`price`** from the “driver” list if you list it only as the target.

### Chi-square vs **`price_bin`**

**`pd.qcut(price, q=4)`** creates **four frequency-based bins** (quartiles). **`pd.crosstab`** + **`chi2_contingency`** test association between each **categorical** (or dummy) feature and **`price_bin`**. **α = 0.05** is a rough exploratory rule; many tests increase false-positive risk.

### `final_df`

Chosen columns for the next modeling step: scaled numerics, dummies, and **`price`**.

---

Sibling projects: [`../README.md`](../README.md).
