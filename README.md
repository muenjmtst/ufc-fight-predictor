# UFC Fight Predictor

**Machine Learning project for predicting UFC fight outcomes using historical fighter statistics.**

This project explores whether pre-fight fighter attributes, recorded fight history, and performance statistics can be used to predict the winner of a UFC fight.

The project covers exploratory data analysis, feature engineering, chronological validation, hyperparameter tuning, model comparison, and probability-based classification.

**Project Status:** Model training and evaluation completed with corrected date-based splits. FastAPI and Docker integration planned. Verification of historical feature availability remains pending.

## Final Results

The final selected model is **Logistic Regression with L2 regularization**.

Selected configuration:

- `C=0.01`
- `solver="lbfgs"`
- `class_weight=None`
- `max_iter=1000`

The model was retrained on **5,726 training fights** and evaluated on **1,443 chronologically held-out UFC fights**.

- Last training date: **2023-05-20**
- First test date: **2023-06-03**

| Metric | Test Score |
|---|---:|
| Accuracy | **64.73%** |
| Balanced Accuracy | **65.21%** |
| ROC-AUC | **0.703590** |
| Log Loss | **0.626085** |
| Classification Threshold | **0.60** |

The final model outperformed an always-Red baseline of **56.27% accuracy**.

The classification threshold was selected using validation data to improve recall balance between Red and Blue fighters.

## Dataset

The original dataset contains **7,177 UFC fight records with 118 columns**.

After excluding **5 draws and 3 no-contests**, **7,169 fights** remained for binary classification.

**Target variable:**

- `1` — Red fighter wins
- `0` — Blue fighter wins

The dataset was sorted chronologically. Training and evaluation partitions keep complete fight dates together.

## Feature Engineering

The model uses **46 input features**, including fighter attributes, recorded historical statistics, and engineered differences between Red and Blue fighters.

Examples include:

- Fighter age, height, and reach
- Recorded wins and losses
- Current winning and losing streaks
- Total rounds fought and title fight experience
- Significant striking, takedown, and submission statistics
- Fighter stances and fight-level attributes
- Age, reach, height, streak, and experience differences
- Recorded win-rate differences
- Zero recorded-experience indicators

Recorded counts should not be assumed to represent complete professional careers. Zero recorded-experience indicators identify fighters with no recorded wins, losses, or draws in the supplied data; they do not necessarily establish an actual debut.

Invalid zero reach values are replaced with missing values before calculating reach differences.

Missing numerical values are handled using mean or median imputation. Numerical features are scaled, and categorical features are imputed and one-hot encoded.

Preprocessing is included inside scikit-learn Pipelines so that imputation, scaling, and encoding are fitted within each training fold.

## Models Evaluated

The following models were trained and compared:

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost
- Soft Voting Ensemble

Hyperparameters for individual models were tuned using `GridSearchCV` and chronological cross-validation.

**Log Loss** was the primary selection metric for models producing probability predictions. SVC was tuned using **ROC-AUC** from decision scores; probability-based Log Loss was not evaluated for SVC.

Accuracy, Balanced Accuracy, Precision, Recall, and F1-score were used for additional evaluation.

### Validation Results

The following results were obtained after correcting the inner chronological split, using **1,150 validation fights**:

| Model | Accuracy | ROC-AUC | Log Loss ↓ |
|---|---:|---:|---:|
| Logistic Regression | 60.96% | 0.6375 | **0.652326** |
| SVC | 58.87% | 0.6370 | Not evaluated |
| Regularized Decision Tree | 58.17% | 0.5939 | 0.673318 |
| Random Forest | 60.61% | 0.6378 | 0.656120 |
| Extra Trees | 60.52% | 0.6263 | 0.661653 |
| XGBoost | **62.00%** | 0.6393 | 0.656893 |
| Soft Voting | 60.52% | **0.6417** | 0.653417 |

Classification metrics in this table use the models' default prediction rules. Logistic Regression threshold tuning was performed separately.

Logistic Regression achieved the lowest validation Log Loss and was selected as the final model.

XGBoost achieved the highest validation Accuracy, while Soft Voting achieved the highest validation ROC-AUC.

The Log Loss difference between Logistic Regression and Soft Voting was small and does not establish statistical significance.

## Validation Strategy

The project uses chronological evaluation with complete fight dates kept together.

| Partition | Fights | Date Range |
|---|---:|---|
| Inner training | 4,576 | 2010-03-21 to 2021-02-13 |
| Validation | 1,150 | 2021-02-20 to 2023-05-20 |
| Final test | 1,443 | 2023-06-03 to 2026-03-28 |

- Inner training and validation are used for model comparison and threshold selection.
- Cross-validation uses **four expanding folds based on unique dates** within the inner training partition.
- All four CV folds satisfy `max(train date) < min(validation date)`.
- Preprocessing is fitted inside the model pipelines.
- After model and threshold selection, the final pipeline is retrained on all **5,726 outer training fights**.
- The test partition is excluded from model fitting and threshold selection.

Earlier EDA used a row-based split that included nine fights from the current test boundary date. Therefore, the test set was not completely isolated from all project exploration.

## Classification Threshold

Logistic Regression outputs predicted probabilities for Red and Blue victories.

Thresholds **0.50, 0.55, 0.60, 0.65, and 0.70** were compared on validation data.

Threshold **0.60** achieved the highest validation balanced accuracy among these candidates: **59.32%**.

The standard and validation-selected thresholds were then evaluated on the test set:

| Metric | Threshold 0.50 | Threshold 0.60 |
|---|---:|---:|
| Test Accuracy | 64.10% | **64.73%** |
| Test Balanced Accuracy | 61.29% | **65.21%** |

Threshold 0.60 improved test balanced accuracy by **3.93 percentage points** compared with threshold 0.50.

The threshold changes classification decisions, not predicted probabilities. ROC-AUC and Log Loss therefore remain unchanged.

The selected decision rule is:

- Predict **Red** when `P(Red) >= 0.60`.
- Predict **Blue** otherwise.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- Matplotlib and Seaborn
- Jupyter Notebook
- joblib

**Planned:** FastAPI, Pydantic, Docker, and MLflow.

## Repository Structure

Current and planned project components:

- `data/` — historical UFC data
- `notebooks/01_eda.ipynb` — exploratory data analysis
- `notebooks/02_model_training.ipynb` — feature engineering, training, comparison, and evaluation
- `models/ufc_logistic_regression.joblib` — serialized trained model bundle
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
- `src/` — planned reusable feature engineering and prediction code

## Current Usage

The ML workflow is currently implemented in Jupyter notebooks.

The selected model has been serialized using `joblib`. The bundle includes:

- The fitted preprocessing pipeline and Logistic Regression model
- The selected classification threshold
- The expected feature columns

To make predictions for new fights, the same feature engineering operations used during training must first be applied.

Use `predict_proba` and apply the saved threshold to obtain the selected classification decision. The model's default `predict` method does not apply the saved 0.60 threshold.

The bundle expects prepared fighter features. Predicting from two fighter names requires a separate process for retrieving historical statistics.

A standalone prediction API is not yet implemented.

## Roadmap

- [x] Exploratory Data Analysis
- [x] Data cleaning and feature engineering
- [x] Date-separated train/test split
- [x] Date-separated inner train/validation split
- [x] Date-based cross-validation
- [x] Model training and hyperparameter tuning
- [x] Model comparison
- [x] Soft Voting experiment
- [x] Classification threshold selection
- [x] Final test evaluation
- [x] Model serialization and reload check
- [ ] Verify dataset provenance and pre-fight availability of historical features
- [ ] Reusable feature engineering and inference code
- [ ] Historical fighter lookup for predictions by name
- [ ] FastAPI prediction endpoint
- [ ] Automated inference and API tests
- [ ] Docker containerization
- [ ] MLflow integration

## Limitations

UFC fight prediction is inherently uncertain. Historical statistics cannot capture every factor influencing a fight, such as injuries, preparation, tactical changes, or unexpected events.

Date overlap between training and evaluation partitions has been corrected. However, chronological splitting and pipeline-based preprocessing do not prove the absence of data leakage.

The source's historical aggregation logic and feature timestamps still require verification to establish that each input was available before its fight. Some recorded fighter histories also contain inconsistent count transitions that require source investigation.

Earlier EDA included nine fights from the current test boundary date, limiting the independence of the final test evaluation.

The current approach may be influenced by differences in Red and Blue corner assignment. Repeated fighter appearances also mean that individual fight observations are not fully independent.

The model has not yet been validated as a live prediction service.

Prediction probabilities should not be interpreted as guarantees.

## Future Improvements

Potential improvements include:

- Elo-based fighter ratings
- Recent fight performance
- Opponent strength
- Time since the previous fight
- Probability calibration
- Fighter-order symmetry testing
- Evaluation on a fresh future holdout

---

**Purpose:** Educational machine learning and MLOps portfolio project.