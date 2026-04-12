# Core ML

Hands-on notebooks plus data in **per-project folders**. Definitions, explanations, and “what this notebook does” live in **each project’s `README.md`**, not here—open the folder you care about first.

---

## Projects

| Project | Folder | Notebook | What to read |
|--------|--------|----------|----------------|
| Insurance charges | [`insurance-cost-project/`](insurance-cost-project/) | `insurance.ipynb` | [`insurance-cost-project/README.md`](insurance-cost-project/README.md) |
| Housing (synthetic) | [`housing-prices-project/`](housing-prices-project/) | `housing.ipynb` | [`housing-prices-project/README.md`](housing-prices-project/README.md) |
| Heart disease | [`heart-disease-project/`](heart-disease-project/) | `heart.ipynb` | [`heart-disease-project/README.md`](heart-disease-project/README.md) |
| Ford used cars | [`ford-used-cars-project/`](ford-used-cars-project/) | `ford.ipynb` | [`ford-used-cars-project/README.md`](ford-used-cars-project/README.md) |

---

## Shared setup (short)

- Use a **virtual environment** and install from the repo root:  
  `pip install -r requirements.txt`
- In Jupyter / Cursor, pick the **same** Python as that venv so `import` works (`ModuleNotFoundError` usually means kernel ≠ env).
- Open each notebook **from its project folder** (or keep CSV paths relative to that folder), e.g. `pd.read_csv("insurance-checkpoint.csv")` next to `insurance.ipynb`.

---

## Workspace log

Day-by-day notes: [`../README.md`](../README.md).

---

*Add new projects by creating a subfolder + `README.md` there, then add a row to the table above.*
