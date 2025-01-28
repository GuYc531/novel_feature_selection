# novel feature selection
This repository introduces an innovative approach to feature selection, enabling robust, efficient, and scalable identification of the most relevant features in your dataset, for tree based models.


# Feature Selection Framework for Tree-Based Models

This repository provides a framework for performing feature selection using tree-based machine learning models. It enables users to evaluate the importance of features in a dataset while optimizing model performance. The implementation includes utility functions for computing feature importance, handling quantiles, and plotting results.

---

## Features

- **Feature Importance by Percentile**: Iteratively evaluates the impact of feature importance percentiles on model performance.
- **Cross-Validation Support**: Incorporates flexible cross-validation policies for model evaluation.
- **Visualization**: Generates bar plots to visualize feature importance and model performance across quantiles.
- **Robust Input Validation**: Ensures proper inputs for models, metrics, and datasets.
- **Support for Multiple Models**: Compatible with tree-based models like `RandomForestClassifier`, `DecisionTreeClassifier`, and `XGBClassifier`.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GuYc531/novel_feature_selection.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### There is main.py file you can use but here is basic example:

### 1. Compute Feature Importance by Percentile

```python
from feature_selection import compute_feature_importance_by_percentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# Example data
df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
y = pd.Series([0, 1, 0])
model = RandomForestClassifier()
cv = StratifiedKFold(n_splits=3)

# Compute feature importance and scores metric
x_axis_labels, scores, final_features_importance_df = compute_feature_importance_by_percentile(
    cv=cv,
    model=model,
    df=df,
    y=y,
    quantiles_number=3,
    metric='f1'
)
```

---

### 2. Plot Feature Importance Across Quantiles

```python
from feature_selection import plot_feature_importance_across_quantiles

# Plot the results
plot_feature_importance_across_quantiles(
    quantiles_number=3,
    scores=scores,
    x_axis_labels=x_axis_labels,
    metric='f1',
    model=model,
    save_figure_in_path=False
)
```

---

## Functionality Highlights

### `compute_feature_importance_by_percentile`
Evaluates the performance of a machine learning model by filtering features based on importance percentiles. Outputs:
- X-axis labels (feature counts and percentiles)
- Performance scores (mean/std for train/test/validation splits)
- A DataFrame of feature importance values

### `plot_feature_importance_across_quantiles`
Generates a bar plot showing the relationship between feature importance percentiles and model performance metrics.

## Supported Models
- `RandomForestClassifier` (sklearn)
- `DecisionTreeClassifier` (sklearn)
- `XGBClassifier` (XGBoost)

---

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- tqdm

---

## Contribution
Contributions are welcome! Feel free to submit issues or pull requests for improvements.

---

## License
This project is licensed under the MIT License.

---

## Author
Created by Guy Cohen. Feel free to connect on [LinkedIn](https://www.linkedin.com/in/guy-cohen-a17a5a160/) or check out my [Medium articles](https://medium.com/@guycohen_958).
