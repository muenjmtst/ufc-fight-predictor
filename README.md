# UFC Fight Predictor

**An end-to-end machine learning project for predicting UFC fight outcomes using historical fighter statistics.**

**Tech stack:** Python · scikit-learn · pandas · FastAPI · Pydantic · Streamlit · Docker · MLflow

**Status:** Model training and evaluation completed. FastAPI inference, Streamlit UI, Docker deployment, and MLflow experiment tracking/model registration have been implemented and manually tested.

> **Educational project.** Predictions are estimates, not guarantees. The availability of every historical input feature *before* its corresponding fight has not yet been fully verified.

## Overview

UFC Fight Predictor estimates whether the Red or Blue corner will win a UFC fight using fighter attributes, recorded fight history, performance statistics, and engineered differences between opponents.

The project covers exploratory data analysis (EDA), feature engineering, chronological validation, model comparison, hyperparameter tuning, probability-based classification, API development, a user interface, containerization, and model lifecycle tracking.

The deployed Logistic Regression pipeline expects **46 prepared input features** and returns probabilities for both corners, plus a classification decision based on a saved **0.60 threshold**.

## System Architecture

```text
User
  │ enters fighter information
  ▼
Streamlit UI (localhost:8501)
  │ calculates fight-level features
  │ POST /predict (JSON)
  ▼
FastAPI (localhost:8000)
  │ Pydantic validation → pandas DataFrame
  ▼
Saved scikit-learn Pipeline (.joblib)
  │ imputation → scaling / one-hot encoding → Logistic Regression
  ▼
FastAPI JSON response → Streamlit prediction & probabilities
```

- **Streamlit (frontend):** Collects information about Red and Blue corner fighters, calculates derived features, sends a JSON request, and displays the prediction.
- **FastAPI (backend):** Validates the request with Pydantic, selects the expected columns, invokes the saved pipeline, and returns the predicted winner and probabilities.
- **ML pipeline:** The deployed preprocessing and Logistic Regression estimator are loaded from a serialized model bundle. EDA, model comparison, and hyperparameter tuning happened earlier during development; they do **not** run on every prediction.
- **Docker:** Packages the **FastAPI backend** and its dependencies in an isolated Linux-based container. Streamlit is launched separately in the current setup.
- **MLflow:** Tracks experiment runs, parameters, metrics, and model artifacts, and records registered model versions. The live API currently loads a local `.joblib` file rather than serving directly from MLflow Registry.

## Model Performance

The retained model is **Logistic Regression with L2 regularization**, implemented as a scikit-learn Pipeline.

It was fitted on **5,726 fights** and evaluated on **1,443 chronologically held-out fights**.

| Metric | Final test result |
|---|---:|
| Accuracy | **64.73%** |
| Balanced accuracy | **65.21%** |
| ROC-AUC | **0.703590** |
| Log loss | **0.626085** |
| Selected classification threshold | **0.60** |

The model exceeded an always-Red prediction baseline of **56.27% accuracy**.

### Selected hyperparameters

| Parameter | Value |
|---|---|
| Estimator | Logistic Regression |
| Regularization | L2 |
| `C` | `0.01` |
| Solver | `lbfgs` |
| Class weight | `None` |
| Maximum iterations | `1000` |

The decision threshold was selected using validation data, not by optimizing on the final test set.

## Dataset

The original dataset includes **7,177 UFC fight records** and **118 columns**. After removing **5 draws and 3 no-contests**, **7,169 fights** remained for binary classification.

**Target encoding:** `1` = Red corner wins; `0` = Blue corner wins.

### Chronological partitions

| Partition | Fights | Dates |
|---|---:|---|
| Inner training | 4,576 | 2010-03-21 to 2021-02-13 |
| Validation | 1,150 | 2021-02-20 to 2023-05-20 |
| Final test | 1,443 | 2023-06-03 to 2026-03-28 |

The inner training and validation partitions together make up the **5,726-fight outer training set**. The final model was refitted on that outer training set after selecting the configuration and threshold.

## Feature Engineering & Preprocessing

The deployed model uses **46 features** from several groups:

| Group | Examples |
|---|---|
| Fighter attributes | Age, height, reach, stance, weight class |
| Recorded fight history | Wins, losses, current streaks, rounds fought, title bouts |
| Performance statistics | Significant strike accuracy, takedowns, submission attempts, KO/TKO and submission wins |
| Engineered differences | Age, reach, height, streak, experience, and win-rate differences |
| Special indicators | Zero recorded-experience indicators, title bout, scheduled rounds |

Differences are computed consistently in the **Red − Blue** direction. When calculating recorded experience and win rate in Streamlit, wins, losses **and draws** are included. A win rate with zero recorded fights is undefined, so its difference is passed as a missing value rather than an artificial zero.

The saved Pipeline handles missing numerical values with mean/median imputation, scales numerical features using `StandardScaler`, and processes categorical values using most-frequent imputation and `OneHotEncoder`. Preprocessing is fitted inside training folds rather than on validation data.

Recorded wins and losses should not automatically be interpreted as full professional career histories; a zero recorded-experience indicator does not necessarily mean a real UFC debut. Invalid zero reach values were treated as missing during feature preparation.

## Model Experiments & Validation

The following algorithms were explored during development:

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost
- Soft Voting Ensemble

Historical experiments used hyperparameter searches and chronological cross-validation. **Log loss** was a primary metric for probability-producing models, with accuracy, balanced accuracy, ROC-AUC, precision, recall, and F1 used for additional evaluation.

After correcting the chronological split, the retained Logistic Regression configuration was re-evaluated on **1,150 validation fights**:

| Validation metric | Result |
|---|---:|
| Accuracy at threshold 0.50 | 60.96% |
| ROC-AUC | 0.637528 |
| Log loss | 0.652326 |

The other models were **not all re-evaluated on the corrected split**. Their earlier scores cannot establish a definitive final ranking under the revised validation protocol.

### Validation protocol

- Complete fight dates were kept together in each partition.
- The inner training data used **four expanding, date-based cross-validation folds**.
- Every fold satisfied `max(training_date) < min(validation_date)`.
- Validation was used for the retained model's evaluation and classification-threshold selection.
- The final outer training set was used to fit the saved pipeline; the later test period was used for final evaluation.

**Caveat:** Earlier EDA included nine fights from the current test boundary date. The final test evaluation was therefore not entirely isolated from *all* earlier project exploration.

### Classification threshold

Validation thresholds **0.50, 0.55, 0.60, 0.65, and 0.70** were compared. Threshold **0.60** gave the highest validation balanced accuracy among these candidates (**59.32%**).

| Final test metric | Threshold 0.50 | Threshold 0.60 |
|---|---:|---:|
| Accuracy | 64.10% | **64.73%** |
| Balanced accuracy | 61.29% | **65.21%** |

**Decision rule:** predict **Red** if `P(Red) >= 0.60`; otherwise predict **Blue**. This is not the same as always choosing the corner with the higher probability. Changing the threshold changes predicted *classes*, not the underlying probabilities, ROC-AUC, or log loss.

## Technology Stack

| Purpose | Technologies |
|---|---|
| Language and data | Python, pandas, NumPy |
| ML and evaluation | scikit-learn; XGBoost in earlier experiments |
| Notebooks and visualizations | Jupyter, Matplotlib, Seaborn |
| Serialization | joblib |
| REST API | FastAPI, Pydantic, Uvicorn |
| Interactive frontend | Streamlit, requests |
| Containerization | Docker |
| Experiment tracking and model registry | MLflow, skops |

## Repository Structure

```text
ml-ufc/
├── app/
│   └── main.py                   # FastAPI application
├── data/                         # Dataset / saved evaluation data
├── models/
│   └── ufc_logistic_regression.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── streamlit_app.py              # UI and feature construction
├── train.py                      # MLflow logging / registration
├── requirements.txt
├── requirements-dev.txt          # Development environment snapshot
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

MLflow's local `mlflow.db` and `mlruns/` are generated runtime data and are not intended to be committed. Ensure the saved model file and any evaluation data needed by `train.py` are available in the cloned project.

## Getting Started

### Option A — Run FastAPI with Docker (recommended for prediction)

**Prerequisites:** Git, Docker Desktop or Docker Engine, and a local Python environment with Streamlit dependencies if you want the graphical interface.

Clone the project:

```bash
git clone https://github.com/muenjmtst/ufc-fight-predictor.git
cd ufc-fight-predictor
```

Build and run the backend:

```bash
docker build -t ufc-predictor:local .
docker run --rm --name ufc-api -p 8000:8000 ufc-predictor:local
```

FastAPI: `http://localhost:8000`
Interactive API docs: `http://localhost:8000/docs`

To start the **Streamlit UI separately**, open another terminal with the project's Python environment activated and run:

```bash
streamlit run streamlit_app.py
```

Streamlit UI: `http://localhost:8501`

The Streamlit process sends HTTP requests to FastAPI on port `8000`. It does not run inside the current API container.

### Option B — Run FastAPI and Streamlit locally (for development)

Use a Python environment with the dependencies in `requirements.txt`. **Python 3.13 is recommended** to match the Docker runtime and the saved model's scikit-learn environment (`scikit-learn==1.9.0`). If another Python/scikit-learn version is used, serialized-model compatibility warnings or errors may occur.

For an existing environment, activate it. For a new one with Python 3.13 installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate      # macOS / Linux
python -m pip install -r requirements.txt
```

Then start the services in **separate terminals**, from the repository root:

**Terminal 1 — FastAPI**

```bash
uvicorn app.main:app --reload
```

**Terminal 2 — Streamlit**

```bash
streamlit run streamlit_app.py
```

Do **not** run Docker's API container and local Uvicorn at the same time on the same host port `8000`.

### Quick API health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok"}
```

Press `Ctrl+C` in the respective terminal to stop a service. `docker run --rm` removes the container after it stops, not the built image.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | General application information |
| `GET` | `/health` | Backend status check |
| `GET` | `/model_info` | Model type, threshold, feature count, and load status |
| `POST` | `/compare` | Simple comparison of two fighter objects (names and age difference; **no ML**) |
| `POST` | `/predict` | Predicted corner and probabilities from 46 prepared input features |

Example `GET /model_info` response:

```json
{
  "model_type": "LogisticRegression",
  "threshold": 0.6,
  "n_features": 46,
  "status": "loaded"
}
```

`POST /predict` accepts a **flat JSON object** with prepared features, not merely two fighter names. Streamlit assists with entering inputs and calculating several derived features, but it does **not** automatically retrieve verified historical records. The FastAPI service itself expects the prepared values.

## Example Prediction

The following input was manually checked through FastAPI and Docker, and its predicted probabilities matched notebook inference to floating-point precision.

### Request — `POST /predict`

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

### Response

```json
{
  "predicted_winner": "Blue",
  "red_win_probability": 0.4568581885684729,
  "blue_win_probability": 0.5431418114315272,
  "threshold": 0.6
}
```

The model estimates **45.69%** for Red and **54.31%** for Blue. The selected label is **Blue**, because `P(Red) < 0.60`.

**Input conventions:** age in years; height and reach in centimeters; percentage-based features as fractions (for example, `0.39` rather than `39`). Keys such as `R_win_by_KO/TKO` include a literal slash. A missing `win_rate_diff` may be passed as JSON `null` when a fighter has no recorded fights; the preprocessing pipeline imputes that missing value.

## MLflow — Experiment Tracking and Model Registry

MLflow is used to log the retained model's parameters, evaluation metrics, and artifacts and to register model versions. It is **not necessary for routine predictions** from the local FastAPI model file.

From the project root, in a Python environment containing MLflow:

**1. Record an evaluation run and register the model** (only when you intend to create a new run/version):

```bash
python train.py
```

The current `train.py` evaluates an **already trained model** from `models/ufc_logistic_regression.joblib`; it does not refit the Logistic Regression estimator. It logs the saved classification threshold, test metrics, model artifacts, and a registered model version.

**2. Open the MLflow UI:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Visit `http://localhost:5000`, select **Model training**, and open experiment **`ufc-fight-prediction`**. Within a run, inspect **Parameters**, **Metrics**, and **Artifacts**. In **Registered Models**, find **`ufc-fight-prediction`** and its versions.

The local audit confirmed a registered **version 2**, but **Registry version numbers are local to the tracking database**: cloning this Git repository does not recreate version 2 automatically. Running `train.py` against a new tracking database will normally create a new experiment/run and start the registered model's versions from 1.

**Important serving detail:** The registered scikit-learn estimator's standard `predict()` does not automatically apply the application's custom threshold `0.60`. For reproducible inference, the complete original `.joblib` bundle also retains the threshold and feature-column list; the API uses these values when serving predictions.

`mlflow.db` and `mlruns/` are local outputs and should remain excluded from Git.

## Verification

Manually verified scenarios include:

- Successful local and Docker API responses for `/`, `/health`, `/model_info`, and `/predict`.
- FastAPI probabilities matching notebook inference for the provided 46-feature sample.
- Streamlit submitting input to FastAPI and displaying both probabilities and the predicted winner.
- A fighter with **zero recorded fights**, for whom an undefined `win_rate_diff` is represented as a missing value and processed by the pipeline.
- Streamlit displaying a backend-offline error message when FastAPI is stopped.
- MLflow recording experiment runs and registering a model version.

These are manual and audit checks, **not a complete automated regression-test suite**. A clean installation from GitHub should still be validated before submission or use on another machine.

## Roadmap

### Completed

- [x] Exploratory data analysis, cleaning, and feature engineering
- [x] Chronological splits and cross-validation
- [x] Model experiments, retained configuration, and hyperparameter tuning
- [x] Classification threshold selection and final evaluation
- [x] Model serialization with joblib
- [x] FastAPI endpoints and Pydantic input schema
- [x] Docker API containerization and manual inference checks
- [x] Streamlit frontend and HTTP integration
- [x] MLflow experiment tracking and model registration
- [x] Missing-value handling for an undefined win-rate difference

### Planned / Not yet fully verified

- [ ] Verify dataset provenance and pre-fight historical feature availability
- [ ] Extract shared, reusable feature engineering for training and inference
- [ ] Retrieve historical fighter statistics by name
- [ ] Automated API, frontend, and inference tests
- [ ] More comprehensive domain validation for all input features
- [ ] Probability calibration and evaluation on fresh future fights
- [ ] Fully portable MLflow-based model serving with the custom threshold contract

## Limitations

1. **Potential historical leakage:** The input source's aggregation logic and feature timestamps have not been fully audited to confirm that each value existed before the corresponding fight. Chronological splitting and pipeline-based preprocessing alone do not prove leakage-free evaluation.
2. **Early EDA overlap:** Nine fights from the current test boundary date appeared in earlier exploratory work, reducing the independence of the final evaluation from the entire analysis process.
3. **Recorded histories:** Some win/loss histories are incomplete or inconsistent; zero recorded experience does not necessarily imply a real debut.
4. **Corner assignment bias:** Results may reflect systematic differences in how Red and Blue corners are assigned.
5. **Repeated observations:** Fighters appear in multiple fights, so fight records are not fully independent.
6. **Manual inputs:** The user must provide fighter statistics; automatic historical lookup and fully validated feature generation are not yet implemented. Incorrect or mutually inconsistent inputs can make predictions less reliable.
7. **Limited operational tests:** The API and Streamlit have been manually tested, but the project does not yet have a comprehensive automated test suite or live-future evaluation.
8. **Uncertain outcomes:** Injuries, preparation, tactics, and other factors are not fully represented. Probabilities are model estimates, **not guarantees**.

## Future Improvements

- Elo-based fighter ratings and opponent-strength features
- Recent-form and time-since-last-fight features
- Shared feature generation and automatic fighter lookup
- Fighter-order symmetry testing
- Probability calibration and fresh future holdout evaluation
- Automated tests and continuous integration

---

**Purpose:** An educational machine learning and MLOps portfolio project.
