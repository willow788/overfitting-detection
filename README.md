# Overfitting Detection (Learning Curve Demo)

A small, notebook-first project that demonstrates a simple way to **detect overfitting (or underfitting)** by comparing **training vs. test accuracy** as the training set size grows (a basic *learning curve*).

The demo uses scikit-learn’s **Breast Cancer Wisconsin (Diagnostic)** dataset and trains a `RandomForestClassifier`, then reports the **generalization gap**:

\[
\text{gap} = \text{train\_accuracy} - \text{test\_accuracy}
\]

## What this repository contains

- `Jupyter Notebook/overfitting.ipynb` — the main notebook (data load → train/test split → train models on increasing fractions of the training data → plot learning curve → compute overfitting gap).
- `Output learning curves/` — saved learning-curve plots produced by the notebook:
  - `1.png`
  - `2.png`

## How it works (high level)

Inside the notebook:

1. **Load data**
   - `sklearn.datasets.load_breast_cancer()`
   - Features are placed into a pandas `DataFrame` `x`, with labels `y`.

2. **Train/test split**
   - `train_test_split(test_size=0.2, random_state=42)`

3. **Train on increasing dataset sizes**
   - `train_sizes = np.linspace(0.1, 1.0, 10)`
   - For each fraction `frac`, the notebook trains on the first `int(len(x_train) * frac)` samples.

4. **Model**
   - `RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3)`

5. **Evaluate & plot learning curve**
   - Collects accuracy on the subset used for training and on the fixed test set.
   - Plots `train_accuracy` and `test_accuracy` vs. training fraction.

6. **Simple overfitting/underfitting heuristic**

At the end:

- `gap = train_scores[-1] - test_scores[-1]`

The notebook prints one of the following based on the final gap:

- `gap > 0.1`  → **overfitting**
- `gap < 0.02` → **may be underfitted or well regularized**
- otherwise    → **generalizing well**

> Note: These thresholds are a simple heuristic for demonstration. In real projects, you’d typically compare cross-validated metrics, consider class imbalance/variance, and look at multiple metrics (AUC, F1, etc.).

## Getting started

### Prerequisites

- **Python 3.11** (the notebook metadata indicates Python `3.11.8`).
- Jupyter (Notebook or JupyterLab).

### Install dependencies

This repository does not currently include a `requirements.txt` or `environment.yml`. You can install the core dependencies manually:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

pip install -U pip
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Run the notebook

```bash
jupyter notebook
```

Then open:

- `Jupyter Notebook/overfitting.ipynb`

Run all cells to reproduce the learning curve and the printed summary.

## Outputs

The notebook plots a learning curve (train vs. test accuracy) and prints the final train/test accuracy and gap.

Saved plot examples are in:

- `Output learning curves/1.png`
- `Output learning curves/2.png`

## Notes / limitations

- The notebook uses a **single** train/test split (no cross-validation).
- It uses **accuracy** only.
- The subset selection is the *first N rows* of `x_train`; shuffling or using stratified sampling per fraction could make the curve more stable.

## Suggested next steps (optional)

If you want to extend this repo:

- Add `requirements.txt` (or `environment.yml`) for reproducibility.
- Use `sklearn.model_selection.learning_curve` to compute learning curves with cross-validation.
- Add alternative models (Logistic Regression, deeper trees) and compare gaps.
- Track additional metrics (ROC AUC, F1) and add confusion matrices.

## License

No license file is currently included in this repository. If you intend others to use, modify, or distribute this code, consider adding an explicit open-source license (e.g., MIT, Apache-2.0, GPL-3.0).