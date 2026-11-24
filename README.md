# UFC Fight Predictor

**Machine Learning project for predicting UFC fight outcomes using historical fighter statistics.**

This project explores whether pre-fight fighter attributes, recorded fight history, and performance statistics can be used to predict the winner of a UFC fight.

The project covers exploratory data analysis, feature engineering, chronological validation, hyperparameter tuning, model comparison, and probability-based classification.

**Project Status:** Model training and evaluation completed with corrected date-based splits. FastAPI endpoints and Docker containerization implemented and manually verified on 2026-11-24. Verification of historical feature availability remains pending.

## Final Results

The retained deployment model is **Logistic Regression with L2 regularization**.

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

## Historical Model Experiments

Logistic Regression, SVC, Decision Tree, Random Forest, Extra Trees, XGBoost, and Soft Voting were explored during earlier experiments, including hyperparameter searches.

After correcting the chronological validation split, the retained Logistic Regression configuration was re-evaluated without a new GridSearch. On 1,150 validation fights, it achieved Accuracy **60.96%**, ROC-AUC **0.637528**, and Log Loss **0.652326** at threshold 0.50.

The other models were not rerun on the corrected split. Their earlier scores therefore do not establish a current ranking against the retained model. Original experiments are preserved in [the historical notebook](notebooks/archive/02_model_training_before_split_audit.ipynb); see [the training audit](docs/training_audit.md) for evaluation corrections.

## Validation Strategy

The project uses chronological evaluation with complete fight dates kept together.

| Partition | Fights | Date Range |
|---|---:|---|
| Inner training | 4,576 | 2010-03-21 to 2021-02-13 |
| Validation | 1,150 | 2021-02-20 to 2023-05-20 |
| Final test | 1,443 | 2023-06-03 to 2026-03-28 |

- Inner training and validation are used for evaluation of the retained configuration and threshold selection. Earlier model comparisons used superseded validation.

- Cross-validation uses **four expanding folds based on unique dates** within the inner training partition.

- All four CV folds satisfy `max(train date) < min(validation date)`.

- Preprocessing is fitted inside the model pipelines.

- After validation and threshold selection, the final pipeline is retrained on all **5,726 outer training fights**.

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

- XGBoost (historical experiments only)

- Matplotlib and Seaborn

- Jupyter Notebook

- joblib

- FastAPI and Pydantic

- Uvicorn

- Docker

**Planned:** MLflow.

## Repository Structure

ML files available in this checkout:

- `data/` — historical UFC data

- `notebooks/01_eda.ipynb` — exploratory data analysis

- `notebooks/02_model_training.ipynb` — feature engineering, retained-model training, and corrected evaluation

- `models/ufc_logistic_regression.joblib` — serialized trained model bundle

- `requirements.txt` — Python dependencies

- `README.md` — project documentation

- `models/training_metrics.json` — saved evaluation metrics and runtime metadata

- `docs/training_audit.md` — training and validation audit

The tested API/container workflow additionally uses `app/main.py`, `Dockerfile`, and `.dockerignore`. These files are not included in this local ML snapshot; the API and Docker instructions below require the working copy containing them.

## Training and Model Artifact

Training and evaluation are implemented in Jupyter notebooks. Running the saved model through the API does not require retraining or opening Jupyter.

The selected model has been serialized using `joblib`. The bundle includes:

- The fitted preprocessing pipeline and Logistic Regression model

- The selected classification threshold

- The expected feature columns

To make predictions for new fights, the same feature engineering operations used during training must first be applied.

Use `predict_proba` and apply the saved threshold to obtain the selected classification decision. The model's default `predict` method does not apply the saved 0.60 threshold.

The bundle expects prepared fighter features. Predicting from two fighter names requires a separate process for retrieving historical statistics.

## Run with Docker

Prerequisites: Docker installed and running, and the working project copy containing `app/main.py`, `Dockerfile`, `.dockerignore`, the API dependencies in `requirements.txt`, and `models/ufc_logistic_regression.joblib`. The saved model must be included in the image.

Run from that project's root:

```bash
docker build -t ufc-predictor:local .
docker run --rm --name ufc-api -p 8000:8000 ufc-predictor:local
```

Open [Swagger UI](http://localhost:8000/docs) to explore and send API requests. Check the running service from another terminal:

```bash
curl http://localhost:8000/health
```

The checked health response is `{"status":"ok"}`. This confirms that the service responds; it does not independently establish model quality.

Press `Ctrl+C` in the container terminal to stop the service. `--rm` removes the stopped container; the image remains available for another run. Port 8000 on the host must be available.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | General service information |
| GET | `/health` | Service response check |
| GET | `/model_info` | Information about the loaded model |
| POST | `/compare` | Educational comparison of two fighter objects; returns their names and absolute age difference, without ML inference |
| POST | `/predict` | Winner prediction from 46 prepared input features |

See [Swagger UI](http://localhost:8000/docs) or [the OpenAPI schema](http://localhost:8000/openapi.json) for request schemas. `/compare` accepts nested `fighter_a` and `fighter_b` objects. `/predict` accepts one flat object containing the prepared model features, including engineered differences. It does not retrieve statistics by fighter name or calculate these differences for the caller.

Height and reach inputs are in centimeters, ages are in years, and percentage features such as `R_avg_SIG_STR_pct` use fractions (for example, `0.39`, not `39`). JSON keys `R_win_by_KO/TKO` and `B_win_by_KO/TKO` include a slash.

## Example Prediction

The following request and response were manually checked through the API. Save the request as `request.json`:

```json
{
  "age_diff": 1,
  "reach_diff": 2.54,
  "height_diff": -2.54,
  "win_streak_diff": -4,
  "lose_streak_diff": 1,
  "wins_diff": 3,
  "losses_diff": 3,
  "experience_diff": 6,
  "win_rate_diff": -0.3,
  "R_is_debutant": 0,
  "B_is_debutant": 0,
  "R_age": 30,
  "R_Height_cms": 162.56,
  "R_Reach_cms": 175.26,
  "R_wins": 7,
  "R_losses": 3,
  "R_current_win_streak": 0,
  "R_current_lose_streak": 1,
  "R_total_rounds_fought": 25,
  "R_total_title_bouts": 1,
  "R_avg_SIG_STR_pct": 0.39,
  "R_avg_TD_landed": 0.61,
  "R_avg_TD_pct": 0.33,
  "R_avg_SUB_ATT": 0.0,
  "R_win_by_KO/TKO": 2,
  "R_win_by_Submission": 0,
  "R_Stance": "Orthodox",
  "B_age": 29,
  "B_Height_cms": 165.1,
  "B_Reach_cms": 172.72,
  "B_wins": 4,
  "B_losses": 0,
  "B_current_win_streak": 4,
  "B_current_lose_streak": 0,
  "B_total_rounds_fought": 8,
  "B_total_title_bouts": 0,
  "B_avg_SIG_STR_pct": 0.39,
  "B_avg_TD_landed": 1.71,
  "B_avg_TD_pct": 0.33,
  "B_avg_SUB_ATT": 0.7,
  "B_win_by_KO/TKO": 1,
  "B_win_by_Submission": 2,
  "B_Stance": "Orthodox",
  "weight_class": "Flyweight",
  "title_bout": 0,
  "no_of_rounds": 5
}
```

Send it while the container is running:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

Example response (last decimal places may vary with the runtime):

```json
{
  "predicted_winner": "Blue",
  "red_win_probability": 0.4568581885684729,
  "blue_win_probability": 0.5431418114315272,
  "threshold": 0.6
}
```

The decision uses the saved threshold: Red when `P(Red) >= 0.60`, otherwise Blue. It does not simply select the larger probability; for example, `P(Red) = 0.58` still produces Blue. The threshold does not modify the probabilities.

## Verification

Manual checks returned HTTP 200 for `/`, `/health`, `/model_info`, `/compare`, and `/predict` inside Docker. For the sample above, API probabilities matched notebook inference to floating-point precision. These checks establish basic service operation and sample inference parity; they are not an automated regression suite or validation on future fights.

## Roadmap

- [x] Exploratory Data Analysis

- [x] Data cleaning and feature engineering

- [x] Date-separated train/test split

- [x] Date-separated inner train/validation split

- [x] Date-based cross-validation

- [x] Historical model training and hyperparameter searches

- [x] Historical model comparison (not rerun on the corrected split)

- [x] Soft Voting experiment

- [x] Classification threshold selection

- [x] Final test evaluation

- [x] Model serialization and reload check

- [ ] Verify dataset provenance and pre-fight availability of historical features

- [ ] Reusable feature engineering and inference code

- [ ] Historical fighter lookup for predictions by name

- [x] FastAPI prediction endpoint and manual API checks

- [ ] Automated inference and API tests

- [x] Docker containerization and manual container checks

- [ ] MLflow integration

## Limitations

UFC fight prediction is inherently uncertain. Historical statistics cannot capture every factor influencing a fight, such as injuries, preparation, tactical changes, or unexpected events.

Date overlap between training and evaluation partitions has been corrected. However, chronological splitting and pipeline-based preprocessing do not prove the absence of data leakage.

The source's historical aggregation logic and feature timestamps still require verification to establish that each input was available before its fight. Some recorded fighter histories also contain inconsistent count transitions that require source investigation.

Earlier EDA included nine fights from the current test boundary date, limiting the independence of the final test evaluation.

The current approach may be influenced by differences in Red and Blue corner assignment. Repeated fighter appearances also mean that individual fight observations are not fully independent.

The API has been manually verified locally and inside Docker. Predictive performance and probability calibration on new future fights have not been validated.

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