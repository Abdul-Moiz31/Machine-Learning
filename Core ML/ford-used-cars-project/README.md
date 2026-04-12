# Ford used cars — price regression

**Notebook:** [`ford.ipynb`](ford.ipynb)  
**Data:** [`ford.csv`](ford.csv) — about **18k** rows: **`model`**, **`year`**, **`price`**, **`transmission`**, **`mileage`**, **`fuelType`**, **`tax`**, **`mpg`**, **`engineSize`**.

**Target:** **`price`** (continuous). This project goes through **EDA → encoding → scaling → train/test split → linear regression → error metrics** on held-out data.

---

## What this notebook does (flow)

1. EDA: **`head`**, **`shape`**, **`info`**, **`describe`**, null checks, histogram of **`price`**, correlation **heatmap** (numeric columns only), box/scatter plots vs **`price`**.  
2. **`X`**, **`y`**: drop **`price`** from features.  
3. **`pd.get_dummies`** on **`model`**, **`fuelType`**, **`transmission`** with **`drop_first=True`**, **`astype(int)`**.  
4. **`StandardScaler`** on **`mileage`**, **`engineSize`**, **`tax`**, **`mpg`**, **`year`** only (not on dummies).  
5. **`train_test_split`** (e.g. 33% test, fixed **`random_state`**).  
6. **`LinearRegression`**: **`fit`** on train, **`predict`** on test.  
7. **MSE**, **RMSE**, **MAE**, **R²**, **adjusted R²**.

---

## Concepts and definitions (this project)

### Supervised regression

**Supervised learning:** each row has features **X** and a known label **y**. Here **y** is **`price`**, a **continuous** number, so the task is **regression**: learn a mapping **X → price**. Success is judged by how small prediction errors are (in the same units as price).

### Feature matrix and target

**`X = df.drop("price", axis=1)`** and **`y = df["price"]`**. Features must be things you would have at prediction time—no sneaking **`price`** (or derivatives) back into **X**.

### Exploratory data analysis (EDA)

**EDA** builds intuition and catches data issues before modeling: distributions (histogram), **linear** relationships among numerics (heatmap), and category vs **`price`** (box plots). Categorical columns only enter the heatmap **after** encoding.

### One-hot encoding (`get_dummies`)

**`model`**, **`fuelType`**, **`transmission`** are **nominal** (unordered). **`get_dummies`** expands them into many **0/1** columns. **`drop_first=True`** removes one reference level **per original column** to reduce **multicollinearity** with an intercept in **linear regression**. The matrix becomes **wide** (many model dummies)—expected for this dataset.

### Why not `LabelEncoder` for these columns here

**`LabelEncoder`** turns categories into **0, 1, 2, …**, which implies an **order** that does not exist for car **model** names. **Ordinary least squares** could then treat “Focus as halfway between Fiesta and Kuga,” which is meaningless. **One-hot** is the standard approach for **nominal** inputs in linear models.

### `StandardScaler` on numerics only

**StandardScaler** shifts/scales columns to roughly **mean 0**, **std 1**. Apply it to **continuous** fields (**`mileage`**, **`engineSize`**, **`tax`**, **`mpg`**, **`year`**). **Do not** scale 0/1 dummies—that would distort their meaning.

**Leakage note:** in production, **`fit`** the scaler on **`X_train`** only, then **`transform`** train and test. Fitting on all of **X** before splitting, as in this practice notebook, lets test statistics influence scaling slightly.

### Train–test split

**`train_test_split`** holds out random rows for **evaluation** so you do not score the model on the same data used to **fit** coefficients. **`test_size`** sets the held-out fraction; **`random_state`** makes the split **reproducible**.

### Linear regression (OLS)

**`LinearRegression`** fits **ordinary least squares**: coefficients (and intercept) that **minimize the sum of squared residuals** on the training set. The prediction is a **linear combination** of features (each dummy and each scaled numeric has a weight). Fast and interpretable; assumes rough **linearity** and can be sensitive to **outliers** and **correlated** predictors.

### Regression metrics (on the test set)

- **MSE / RMSE:** **MSE** is mean squared error; **RMSE** is its square root—in **same units as `price`**, so easy to read as “typical” error magnitude (still sensitive to large outliers).  
- **MAE:** mean **absolute** error; robust in spirit to occasional huge errors.  
- **R²:** fraction of variance in **test** **`y`** explained by predictions (1 = perfect on that sample; negative = worse than predicting the mean).  
- **Adjusted R²:** **R²** penalized for **extra predictors**, so adding useless columns does not automatically inflate “fit.”

Together: **MAE / RMSE** answer “how far off in pounds?” and **R² / adjusted R²** answer “how much variance did we capture?”

---

Sibling projects: [`../README.md`](../README.md).
