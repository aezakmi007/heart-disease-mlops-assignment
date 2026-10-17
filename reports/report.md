## 3. Data Acquisition & Exploratory Data Analysis

### 3.1 Dataset
- **Source:** UCI Machine Learning Repository – Heart Disease (Cleveland subset)
- **Size:** [303] rows × [14] columns
- **Target:** Original `num` column (0–4) converted to binary `target`
  (0 = no disease, 1 = disease present), since the task is risk prediction.

### 3.2 Data Acquisition
The dataset is downloaded with `src/download_data.py`, which fetches the raw file
from UCI, assigns column names (the source has no header), treats `?` as missing,
and saves it to `data/raw/heart_raw.csv`.

Run: `python src/download_data.py`

### 3.3 Data Cleaning
| Issue | Column(s) | Count | Action taken | Reason |
|---|---|---|---|---|
| Missing values | ca | [4] | Filled with mode | Categorical, very few rows |
| Missing values | thal | [2] | Filled with mode | Categorical, very few rows |
| Duplicates | all | [0] | Dropped | Avoid data leakage |
| Multi-class target | num | — | Binarised to `target` | Binary risk prediction |

Encoding and scaling are deliberately **not** done here; they are part of the
scikit-learn pipeline (Section 4) so training and inference use identical steps.

### 3.4 EDA Findings

**Class balance** (Fig: class_balance.png)
[No disease: ___ (___%), Disease: ___ (___%)] → [roughly balanced / imbalanced],
so [accuracy is a reasonable metric, but we also report recall and ROC-AUC].

**Numeric features** (Fig: numeric_histograms.png, boxplots.png)
- [Observation about thalach – e.g., patients with disease tend to have lower/higher max heart rate]
- [Observation about oldpeak]
- [Observation about age]
- [Outliers seen in chol/trestbps? Kept or removed, and why]

**Categorical features** (Fig: categorical_vs_target.png)
- [Which chest-pain type (cp) is most associated with disease]
- [Observation about exang, ca, thal]

**Correlations** (Fig: correlation_heatmap.png)
- Strongest positive correlation with target: [feature (value)], [feature (value)]
- Strongest negative correlation with target: [feature (value)]
- [Any features strongly correlated with each other? → multicollinearity note]

### 3.5 Key Takeaways for Modelling
- [e.g., Features X, Y, Z look most predictive]
- [e.g., Different scales on numeric features → scaling needed]
- [e.g., Categorical codes like cp/thal are nominal → one-hot encoding needed]