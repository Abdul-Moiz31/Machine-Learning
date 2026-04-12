# ML learning notes (course workspace)

This repo is a personal workspace for learning machine learning. The file you care about day to day is this README: we **update it as we go** so it stays a short log of what you practiced and what stuck.

## Day 1 — April 11, 2026

**What we did**

- Set up a Python environment with **NumPy**, **pandas**, **Matplotlib**, **Seaborn**, and **scikit-learn** (and hit the usual `ModuleNotFoundError` until packages matched the Jupyter kernel).
- Opened a real tabular dataset: **medical insurance** costs (`insurance-checkpoint.csv`).
- Walked through **EDA** (shape, dtypes, missing values, summaries, plots).
- **Cleaned** the data: duplicates, encodings for sex and smoker, **one-hot** / dummy variables for region and BMI category.
- **Feature engineering**: BMI buckets (underweight / normal / overweight / obesity) with `pd.cut`.
- Measured **Pearson correlation** with `charges` and ran **chi-square** tests on categorical features vs binned charges.
- Applied **standard scaling** to selected numeric columns and built a **`final_df`** for the next steps (modeling later).

**Where the hands-on work lives**

- Insurance — notebook and data: [`Core ML/insurance-cost-project/`](Core%20ML/insurance-cost-project/)
- Housing (synthetic) — same workflow, target `price`: [`Core ML/housing-prices-project/`](Core%20ML/housing-prices-project/)
- Heart disease — binary target `HeartDisease`: [`Core ML/heart-disease-project/`](Core%20ML/heart-disease-project/)
- Ford used cars — regression to **`price`**: [`Core ML/ford-used-cars-project/`](Core%20ML/ford-used-cars-project/)
- Concept write-ups live in **each project’s `README.md`** (see [`Core ML/README.md`](Core%20ML/README.md) for the index).

**Mistakes that taught something**

- Imports fail if the kernel’s environment does not have the package — fix with `pip install` (or conda) **for that same interpreter**.
- `sort_values(by=...)` needs a real **column** name, not a row index label.
- `NameError: df_cleaned` means earlier cells were not run (or the kernel was restarted) — run from the top or “Run all.”

## Second practice project — synthetic housing

- Added [`Core ML/housing-prices-project/`](Core%20ML/housing-prices-project/): **`housing.ipynb`** + **`housing-sample.csv`** (fake listings), same pipeline as insurance for extra repetition.

## Third practice project — heart disease tabular set

- Added [`Core ML/heart-disease-project/`](Core%20ML/heart-disease-project/): **`heart.ipynb`** + **`heart.csv`** — see that folder’s **[`README.md`](Core%20ML/heart-disease-project/README.md)** for the current pipeline and definitions (notebook may evolve).

## Fourth practice project — Ford used car prices

- Added [`Core ML/ford-used-cars-project/`](Core%20ML/ford-used-cars-project/): **`ford.ipynb`** + **`ford.csv`** — definitions in **[`ford-used-cars-project/README.md`](Core%20ML/ford-used-cars-project/README.md)**.

---

*Next sessions: append a new “Day N” section here with date, bullets, and links.*
