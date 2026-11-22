# UFC Fight Predictor

**Machine Learning project for predicting UFC fight outcomes using historical fighter statistics.**

This project explores whether pre-fight fighter attributes, career records, and performance statistics can be used to predict the winner of a UFC fight.

The project covers exploratory data analysis, feature engineering, chronological validation, hyperparameter tuning, model comparison, and probability-based classification.

**Project Status:** ML research and model evaluation completed. FastAPI and Docker integration planned.

## Final Results

The final selected model is **Logistic Regression with L2 regularization**.

Evaluated on 1,434 chronologically held-out UFC fights:

| Metric                   | Test Score |
| ------------------------ | ---------: |
| Accuracy                 | **64.78%** |
| Balanced Accuracy        | **65.28%** |
| ROC-AUC                  | **0.7036** |
| Log Loss                 | **0.6262** |
| Classification Threshold |   **0.60** |

The final model outperformed an always-Red baseline of 56.28% accuracy.

The classification threshold was selected using validation data to improve recall balance between Red and Blue fighters.

## Dataset

The original dataset contains 7,177 UFC fight records with 118 columns.

After excluding draws and no-contests, 7,169 fights remained for binary classification.

**Target variable:**

- `1` — Red fighter wins
- `0` — Blue fighter wins

The dataset was sorted chronologically to simulate predicting future fights from historical information.

## Feature Engineering

The model uses fighter attributes, historical performance statistics, and engineered differences between Red and Blue fighters.

Examples include:

- Fighter age, height, and reach
- Career wins and losses
- Current winning and losing streaks
- Total rounds fought and title fight experience
- Significant striking and takedown statistics
- Age, reach, height, and experience differences
- Career win-rate differences
- UFC debut indicators

Missing values are handled using imputation, numerical features are scaled, and categorical features are one-hot encoded.

Preprocessing is included inside scikit-learn Pipelines to reduce data leakage during cross-validation.

## Models Evaluated

The following machine learning algorithms were trained and compared:

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost
- Soft Voting Ensemble

Hyperparameters were optimized using `GridSearchCV` and chronological cross-validation.

**Log Loss** was the primary model-selection metric, while Accuracy, Balanced Accuracy, ROC-AUC, Precision, Recall, and F1-score were used for additional evaluation.

## Validation Strategy

Unlike random train/test splitting, this project uses chronological evaluation.

- Training and testing data are separated by fight date.
- A separate chronological validation period is used for model comparison.
- Cross-validation is performed using date-based splits, ensuring fights from the same date are not divided across training and validation folds.
- The final test set is held out until model selection and threshold tuning are completed.

## Classification Threshold

Logistic Regression outputs predicted probabilities for Red and Blue victories.

Two classification thresholds were evaluated:

| Metric                 | Threshold 0.50 | Threshold 0.60 |
| ---------------------- | -------------: | -------------: |
| Test Accuracy          |         64.23% |     **64.78%** |
| Test Balanced Accuracy |         61.37% |     **65.28%** |

A threshold of 0.60 was selected on validation data to improve balanced classification performance.

The threshold changes the classification decision, not the predicted probabilities.

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
- `notebooks/` — EDA, feature engineering, training, and evaluation
- `models/` — serialized trained model (local; to be committed)
- `requirements.txt` — project dependencies (to be completed)
- `src/` — planned reusable feature engineering and prediction code
- `README.md` — project documentation

## Current Usage

The ML workflow is currently implemented in Jupyter notebooks.

The selected trained model has been serialized using `joblib`. It includes the preprocessing pipeline, trained Logistic Regression model, classification threshold, and expected feature columns.

To make predictions for new fights, the same feature engineering operations used during training must first be applied.

A standalone prediction API is not yet implemented.

## Roadmap

- [x] Exploratory Data Analysis
- [x] Data cleaning and feature engineering
- [x] Chronological train/validation/test splitting
- [x] Model training and hyperparameter tuning
- [x] Model comparison
- [x] Soft Voting experiment
- [x] Classification threshold optimization
- [x] Final test evaluation
- [x] Model serialization
- [ ] Reusable inference pipeline
- [ ] FastAPI prediction endpoint
- [ ] Automated API tests
- [ ] Docker containerization
- [ ] MLflow integration

## Limitations

UFC fight prediction is inherently uncertain. Historical statistics cannot capture every factor influencing a fight, such as injuries, preparation, tactical changes, or unexpected events.

The model was evaluated on historical fights and has not yet been validated as a live prediction service.

The current approach may also be influenced by differences in how Red and Blue fighters are assigned to corners.

Prediction probabilities should not be interpreted as guarantees.

## Future Improvements

Potential improvements include Elo-based fighter ratings, recent fight performance, opponent strength, time since the previous fight, probability calibration, and fighter-order symmetry testing.

---

**Purpose:** Educational machine learning and MLOps portfolio project.
