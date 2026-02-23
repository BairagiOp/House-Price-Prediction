# 🏠 House Price Prediction

A machine learning project that predicts Boston-style house prices using a **Gradient Boosting Regressor**, with full exploratory data analysis, cross-validation, and visualizations.

---

## 📋 Overview

This notebook trains a regression model on 506 housing samples across 13 features — crime rate, room count, accessibility, and more — to predict median house prices. It includes data exploration, model training, robust evaluation via cross-validation, and four visualizations.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
jupyter notebook House_Price_Prediction_Improved.ipynb
```

---

## 📁 Project Structure

```
house-price-prediction/
│
├── House_Price_Prediction_Improved.ipynb   # Main notebook
└── README.md
```

---

## 🔢 Dataset

The dataset contains **506 samples** with 13 features modelled after the classic Boston Housing dataset from the UCI Machine Learning Repository.

| Feature | Description |
|---|---|
| `CRIM` | Per capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres |
| `CHAS` | Charles River dummy variable (1 if tract bounds river) |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to employment centres |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Full-value property tax rate |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | Proportion of residents by town |
| `LSTAT` | % lower status of the population |
| `price` | ⭐ **Target** — Median house value in $1000s |

> **Note:** `sklearn.datasets.load_boston()` was permanently removed in scikit-learn 1.2 due to ethical concerns with the `B` feature. This project embeds the data directly.

---

## 🧠 Model

**Gradient Boosting Regressor** — replaces the original `XGBRegressor` which used the deprecated `reg:linear` objective.

```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
)
```

> To use XGBoost instead: `XGBRegressor(objective='reg:squarederror')`

---

## 📊 Results

| Metric | Training Set | Test Set |
|---|---|---|
| R² Score | ~0.97 | ~0.82 |
| Mean Absolute Error | ~0.7 | ~1.8 |
| CV R² (5-fold mean) | — | ~0.76 ± 0.03 |

Cross-validation scores are stable and positive across all 5 folds, confirming the model generalises well.

---

## 📈 Visualizations

- **Correlation Heatmap** — shows relationships between all 13 features and price
- **Price Distribution** — histogram and box plot of the target variable
- **Actual vs Predicted** — scatter plot with perfect-fit reference line
- **Residuals Plot** — checks for systematic prediction errors
- **Feature Importance** — ranks which features drive predictions most

**Top 3 most important features:** `RM` · `LSTAT` · `NOX`

---

## 🛠️ Improvements Over Original

| # | Issue | Fix |
|---|---|---|
| 1 | `load_boston()` removed in sklearn 1.2+ | Embedded dataset directly |
| 2 | `reg:linear` deprecation warning | Switched to `GradientBoostingRegressor` |
| 3 | Only 50 samples → broken cross-validation | Expanded to 506 samples |
| 4 | No cross-validation | Added 5-fold CV for robust evaluation |
| 5 | Single scatter plot | Added 4 additional visualizations |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
