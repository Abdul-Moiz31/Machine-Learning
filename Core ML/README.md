# Core ML

This folder holds small projects where we practice machine learning workflows end to end. Each project lives in its own subfolder with its data and notebook.

## Projects

| Project | Folder | Goal |
|--------|--------|------|
| Insurance charges (Day 1) | [`insurance-cost-project/`](insurance-cost-project/) | Explore, clean, and prepare insurance data so we can later predict or explain **medical charges**. |
| Housing prices (practice, synthetic) | [`housing-prices-project/`](housing-prices-project/) | Same workflow on **made-up** listings: target **`price`**, features like sqft, bedrooms, age, neighborhood, garage. |
| Heart disease (tabular) | [`heart-disease-project/`](heart-disease-project/) | Same workflow on clinical-style rows; binary target **`HeartDisease`**, chi-square vs outcome (not binned continuous target). |

---

## Housing sample project (second notebook)

The notebook [`housing-prices-project/housing.ipynb`](housing-prices-project/housing.ipynb) reuses the **same ideas** as the insurance piece: EDA, cleaning, **`get_dummies`** with **`drop_first=True`**, **`pd.cut`** buckets (here **house age** instead of BMI), **`StandardScaler`** on selected numerics, **Pearson** correlation with the target, **chi-square** vs **`pd.qcut`** price bins, and a trimmed **`final_df`**.

Because the data are **synthetic**, you can share the repo freely and still get believable plots. One teaching detail: after `get_dummies(..., drop_first=True)`, the column names depend on **alphabetical** category order—this project drops **`neighborhood_east`** as the reference level, so the dummy columns are **`neighborhood_north`** and **`neighborhood_south`**. Checking `df.columns` after encoding is a good habit.

For deep explanations of EDA, scaling, correlation, and chi-square, read the sections below (they were written with the insurance notebook in mind; the **meaning** is the same when you swap `charges` for `price`).

---

## Heart disease project (third notebook)

Notebook: [`heart-disease-project/heart.ipynb`](heart-disease-project/heart.ipynb).

Same preprocessing habits: **`get_dummies(..., drop_first=True)`** on **`ChestPainType`**, **`RestingECG`**, **`ST_Slope`** (reference levels **ASY**, **LVH**, **Down** in default alphabetical ordering—verify on **`columns`**), binary maps for **`Sex`** and **`ExerciseAngina`**, **`pd.cut`** age bands, **`StandardScaler`** on continuous vitals.

**Chi-square twist:** the target **`HeartDisease`** is already **binary**, so crosstabs use **feature × HeartDisease** instead of “binned continuous target × feature” like the insurance notebook. Pearson *r* with a 0/1 target is still a valid **linear** association screen, but interpretation stays cautious.

---

## Concepts from the insurance project

The notes below match what we do in [`insurance-cost-project/insurance.ipynb`](insurance-cost-project/insurance.ipynb). They apply the same way to [`housing-prices-project/housing.ipynb`](housing-prices-project/housing.ipynb) (swap **`charges`** for **`price`**) and mostly to [`heart-disease-project/heart.ipynb`](heart-disease-project/heart.ipynb) (binary **`HeartDisease`**, chi-square as described above). Read here, then trace the steps in whichever notebook you are using.

---

### Python environment and the libraries we use

When you run a Jupyter notebook, a **kernel** starts one specific Python interpreter. Every `import` looks for packages in *that* interpreter’s environment. If you install a library in the terminal but your notebook uses a different Python (for example a virtualenv you forgot to select), you will see `ModuleNotFoundError` even though “you already installed it.” The fix is always to align the kernel with the environment where you installed the packages, or install into the interpreter shown by `import sys; print(sys.executable)`.

In this project we use several libraries together:

- **NumPy** provides efficient n-dimensional arrays and vectorized math. Many other libraries (including pandas and scikit-learn) build on it under the hood.
- **pandas** gives us the **DataFrame**, a table with named columns and row indices. It is the main tool for loading CSVs, inspecting data, cleaning, merging, and selecting columns for modeling.
- **Matplotlib** is a low-level plotting library; **Seaborn** sits on top of it and makes statistical plots (distributions, box plots, heatmaps) with less boilerplate.
- **scikit-learn** is the standard toolkit for machine learning in Python. In this notebook we only use **preprocessing** (`StandardScaler`); later projects will use estimators, pipelines, and train/test splits.
- **SciPy** adds scientific routines. We use **`scipy.stats`** for **Pearson correlation** and the **chi-square test**, which are classical statistical tools rather than ML models.

Loading the dataset is one line once pandas is available:

```python
import pandas as pd

df = pd.read_csv("insurance-checkpoint.csv")
# Housing practice project (same folder as the notebook):
# df = pd.read_csv("housing-sample.csv")
# Heart project (same folder as that notebook):
# df = pd.read_csv("heart.csv")
```

---

### Exploratory data analysis (EDA)

**Exploratory data analysis** means systematically understanding what you have *before* you commit to a model. The goal is not to “prove” anything yet; it is to avoid silent mistakes (wrong types, duplicate rows, skewed targets, columns that need encoding) and to build intuition about which variables might matter for charges.

We typically check:

- **Shape** (`df.shape`): how many rows and columns. Row count affects how much you can trust fine-grained patterns.
- **A sample of rows** (`df.head()`): do values look plausible?
- **Column types and non-null counts** (`df.info()`): are numerics stored as numbers? Are there missing values? Object columns are often text categories that models cannot use without encoding.
- **Summaries** (`df.describe()` for numeric columns): means, spreads, min/max. Extreme values or odd scales show up here.
- **Missing values** (`df.isnull().sum()`): if something is often missing, imputation or dropping rows becomes part of the pipeline.
- **Category frequencies** (`value_counts()` on `sex`, `smoker`, `region`, etc.): rare categories or severe imbalance can affect both plots and models.

Visual tools used in the notebook include **count plots** for categorical counts, **box plots** for numeric columns (median, quartiles, outliers), a **histogram** of BMI to see the shape of the distribution, and a **correlation heatmap** for numeric columns only. The heatmap shows *linear* pairwise relationships; it does not replace thinking about causality or about categorical variables until they are encoded.

---

### Data cleaning

**Data cleaning** means preparing the table so that downstream steps (statistics, ML) operate on consistent, intentional representations of reality.

**Removing duplicates:** If the same person or row appears twice by mistake, any model or correlation can be biased. `drop_duplicates()` keeps one copy of identical rows and drops the rest. We use `inplace=True` to modify the working copy `df_cleaned` in place; many teams prefer `df_cleaned = df_cleaned.drop_duplicates()` for clearer reassignment.

**Encoding binary categories as numbers:** Algorithms and many statistical routines expect numbers. Columns like `sex` and `smoker` have two levels. We **map** them to 0 and 1 and rename to **`is_male`** and **`is_smoker`** so the column names read as yes/no questions (1 = male, 1 = smoker, depending on how you defined the map). The actual 0/1 choice is arbitrary for many models, but you must stay consistent and document it.

Cleaning is not “making data pretty”; it is making sure each column’s meaning and type match what you think you are measuring.

---

### One-hot encoding (dummy variables) and `drop_first`

**Nominal categories** (like `region`: northeast, northwest, …) have no natural ordering. A single numeric code (0,1,2,3) would wrongly imply that “northwest is between northeast and southeast.” **One-hot encoding** fixes that by creating one new **binary column per category** (or per category minus one, see below). Each row has a 1 in exactly one column (or all zeros if we dropped the reference category).

In pandas, `pd.get_dummies(..., columns=["region"], drop_first=True)` does two important things:

1. It replaces the `region` column with columns such as `region_northwest`, `region_southeast`, `region_southwest` (if northeast is the dropped **reference** level).
2. **`drop_first=True`** removes one category’s column on purpose. If you kept all four dummies plus an intercept in a linear model, the columns would be **linearly dependent** (multicollinearity): the model cannot uniquely identify coefficients. Dropping one level is a standard fix; the dropped level is implicit when all dummy columns are 0.

The same idea applies when we later dummy-encode **`bmi_category`** after binning BMI.

---

### Feature engineering: BMI categories with `pd.cut`

**Feature engineering** means defining new input columns from existing ones so the model (or the analysis) can capture patterns the raw columns do not express directly.

**BMI** (body mass index) is a single continuous number. For this project we also create **`bmi_category`** by **binning** BMI into bands that follow common clinical cutoffs: underweight, normal, overweight, obesity. `pd.cut` assigns each row to an interval based on the edges you pass in `bins=` and the `labels=` you assign to those intervals.

Why bother? Some relationships with charges might be **non-linear** or **threshold-like** (risk jumping past a BMI band). A model that only sees raw BMI assumes a smooth trend; category dummies let the data speak differently per band. Whether that helps depends on the model and the signal; engineering is a hypothesis you validate later.

After cutting, we again use **`get_dummies`** with **`drop_first=True`** on `bmi_category` so we do not over-parameterize the category set.

---

### Standardization and `StandardScaler`

**Standardization** (z-score scaling) transforms each numeric feature to have **mean 0** and **standard deviation 1** (approximately), using the mean and std estimated from the data you **fit** on. The formula per value is: subtract the feature’s mean, divide by its standard deviation.

Why it matters: features on very different scales (age in decades vs charges in thousands) can dominate distance-based or regularized models, or slow optimization. Putting selected columns on a common scale makes their coefficients or contributions more comparable in magnitude.

`StandardScaler` in scikit-learn implements this. **`fit_transform`** on a DataFrame column subset learns the mean and scale from those columns and returns the transformed array. In the Day 1 notebook we fit on the **entire** cleaned table for simplicity.

In a proper **train/test** workflow you should call **`fit`** only on **training** data, then **`transform`** both train and test with the same statistics. Otherwise information from the test set leaks into the scaling and your evaluation looks better than it should. When we add modeling with a split, we will move scaling inside that pattern (often with a **Pipeline**).

---

### Pearson correlation

The **Pearson correlation coefficient** *r* measures how strongly two **numeric** variables move together in a **straight-line** way. It ranges from −1 to +1. Values near 0 mean little linear association; +1 or −1 mean perfect positive or negative linear alignment. Pearson does **not** capture curved relationships well and is sensitive to outliers.

`scipy.stats.pearsonr(x, y)` returns two numbers: the correlation *r* and a **p-value** for a test of the null hypothesis “correlation is zero” under standard assumptions (roughly: paired samples, approximate normality for inference—interpret p-values with care on real messy data).

In the project we compute *r* between many columns and **`charges`** (including dummy variables, which are 0/1 numeric). Including **`charges`** as both a feature and the “target” in the same list will give a trivial correlation of 1 with itself; for interpretation you normally exclude the target from the “driver” list or ignore that row. Sorting the results helps you see which engineered features line up most linearly with charges in this dataset—**correlation is not causation**, but it is a useful first screen.

---

### Chi-square test of independence and charge quartiles

The **chi-square test of independence** asks whether two **categorical** variables are related: knowing one variable’s category changes the distribution of the other’s categories, compared to what you would expect if they were independent.

To run it you need a **contingency table**: counts of how many rows fall into each combination of categories. `pd.crosstab(row_feature, column_feature)` builds that table. **`chi2_contingency`** then compares observed counts to **expected** counts under independence and produces a **chi-square statistic**, **degrees of freedom**, **expected frequencies**, and a **p-value**.

In the notebook we do not test “raw” continuous charges against categories directly (that is a different kind of problem). Instead we turn **`charges`** into **`charges_bin`** with **`pd.qcut(..., q=4)`**, which splits the charge distribution into **four groups with roughly equal counts** (quartile bins). Each row gets a bin label 0–3. Then for each categorical feature we cross-tabulate feature × `charges_bin` and run the chi-square test.

We compare the p-value to a chosen **significance level** **α = 0.05**. A small p-value means the data would be surprising if the variable and charge bin were independent; we treat that as evidence of association for **exploratory feature screening**. A large p-value does not “prove” independence; it only means we did not detect strong evidence in this table. Also, with many tests, **multiple testing** becomes an issue if you treat every threshold rigidly—here we use it as a learning exercise and a rough keep/drop hint, not as final science.

---

### Building `final_df`

After exploration and preprocessing, it helps to freeze a **rectangular table** you intend to use for the next step (for example training a regression model). **`final_df`** is that slice: a chosen list of **feature** columns plus the **target** **`charges`**. Everything not in the list is left out on purpose (for instance some BMI category dummies might be dropped from modeling after chi-square screening, or you might simplify the first model).

This step is mostly **bookkeeping**: explicit column names document the modeling view of the data and avoid accidentally passing raw text columns or temporary columns into `.fit()`.

```python
final_df = df_cleaned[
    [
        "age", "bmi", "children", "is_male", "is_smoker", "charges",
        "region_northwest", "region_southeast", "region_southwest",
    ]
]
```

---

*As you add more days or projects, append sections here or link to a new project README.*
