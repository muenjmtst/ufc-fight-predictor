# UFC Fight Predictor

**An end-to-end Machine Learning project for predicting UFC fight outcomes using historical fighter statistics.**

Python · Scikit-learn · FastAPI · Docker · Machine Learning

**Project Status:** ML training and evaluation completed. FastAPI inference API implemented, containerized with Docker, and manually tested.

## Overview

UFC Fight Predictor is a machine learning project that estimates the outcome of a UFC fight based on fighter characteristics, recorded fight history, and performance statistics.

The project covers the complete machine learning workflow:

- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Chronological train/validation/test splitting
- Hyperparameter tuning and model comparison
- Probability-based classification
- Model serialization and inference
- REST API development using FastAPI
- Containerization using Docker

The deployed model accepts 46 prepared fight features and returns predicted probabilities for Red and Blue corner victories.

**Important:** The project is intended for educational and portfolio purposes. Verification that every historical input feature was available before its corresponding fight remains incomplete.

---

## Model Performance

The final deployed model is **Logistic Regression with L2 regularization**, trained using a scikit-learn Pipeline.

### Final Test Results

The model was trained on **5,726 fights** and evaluated on **1,443 chronologically held-out fights**.

| Metric | Test Score |
|---|---:|
| Accuracy | **64.73%** |
| Balanced Accuracy | **65.21%** |
| ROC-AUC | **0.703590** |
| Log Loss | **0.626085** |
| Classification Threshold | **0.60** |

The model outperformed the always-Red baseline of **56.27% accuracy**.

### Selected Hyperparameters

| Parameter | Value |
|---|---|
| Model | Logistic Regression |
| Regularization | L2 |
| C | 0.01 |
| Solver | lbfgs |
| Class Weight | None |
| Max Iterations | 1000 |

The classification threshold was selected using validation data rather than the final test set.

---

## Dataset

The original dataset contains **7,177 UFC fight records with 118 columns**.

After excluding 5 draws and 3 no-contests, **7,169 fights** remained for binary classification.

### Target Variable

| Class | Meaning |
|---|---|
| `1` | Red fighter wins |
| `0` | Blue fighter wins |

The data was sorted chronologically to evaluate the model on later fights rather than using a random train/test split.

### Dataset Split

| Partition | Number of Fights | Date Range |
|---|---:|---|
| Inner Training | 4,576 | 2010-03-21 to 2021-02-13 |
| Validation | 1,150 | 2021-02-20 to 2023-05-20 |
| Final Test | 1,443 | 2023-06-03 to 2026-03-28 |

The final model was retrained on all **5,726 outer training fights** after validation and threshold selection.

---

## Feature Engineering

The deployed model uses **46 input features**, including fighter attributes, historical performance statistics, and engineered differences between opponents.

### Main Feature Groups

**Fighter Attributes**
- Age, height and reach
- Fighter stance
- Weight class

**Fight History**
- Recorded wins and losses
- Current winning and losing streaks
- Total rounds fought
- Title fight experience

**Performance Statistics**
- Significant strike accuracy
- Takedown statistics
- Submission attempts
- Recorded KO/TKO and submission wins

**Engineered Features**
- Age difference
- Reach and height differences
- Winning and losing streak differences
- Experience difference
- Recorded win-rate difference
- Zero recorded-experience indicators

Differences between fighters are calculated consistently in the **Red − Blue** direction.

### Data Preprocessing

The preprocessing pipeline includes:

1. Missing value imputation using mean, median, or most-frequent strategies.
2. StandardScaler for numerical features.
3. OneHotEncoder for categorical features.
4. Logistic Regression for binary classification.

All preprocessing steps are included inside the scikit-learn Pipeline.

This prevents preprocessing statistics from being fitted on validation data during cross-validation, although it does not establish that the underlying dataset is free from historical data leakage.

**Data note:** Recorded fight histories may not represent complete professional careers. Zero recorded-experience indicators do not necessarily mean that a fighter is making their professional or UFC debut. Invalid zero reach values were treated as missing during feature preparation.

---

## Model Experiments

Several machine learning algorithms were explored:

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost
- Soft Voting Ensemble

Hyperparameter searches were conducted using GridSearchCV and chronological cross-validation.

**Log Loss** was the primary selection metric for models producing probability predictions. Accuracy, Balanced Accuracy, ROC-AUC, Precision, Recall, and F1-score were also considered.

### Corrected Validation

Following correction of the chronological validation split, the retained Logistic Regression configuration was re-evaluated on 1,150 validation fights.

| Metric | Validation Score |
|---|---:|
| Accuracy at threshold 0.50 | 60.96% |
| ROC-AUC | 0.637528 |
| Log Loss | 0.652326 |

The other models were not all retrained and compared again on the corrected split. Therefore, the earlier experiments do not establish a definitive ranking of all candidate algorithms under the final validation protocol.

The retained Logistic Regression configuration was used for the final deployment.

---

## Validation Strategy

Chronological evaluation was used to better approximate predictions on future fights.

The validation process included:

- Date-separated training and evaluation partitions.
- Complete fight dates kept together.
- Four expanding chronological cross-validation folds within the inner training partition.
- Model evaluation and threshold selection using validation data.
- Final retraining on the full outer training set.
- Final test evaluation after model and threshold selection.

Every cross-validation fold maintained the relationship:

`max(training_date) < min(validation_date)`

**Limitation:** Earlier exploratory analysis included nine fights from the current test boundary date. The final test evaluation was not completely isolated from all earlier project exploration.

### Classification Threshold

The model produces probabilities for Red and Blue victories.

The following classification thresholds were evaluated on validation data:

`0.50`, `0.55`, `0.60`, `0.65`, `0.70`

The selected threshold of **0.60** achieved the highest validation balanced accuracy among these candidates: **59.32%**.

| Test Metric | Threshold 0.50 | Threshold 0.60 |
|---|---:|---:|
| Accuracy | 64.10% | **64.73%** |
| Balanced Accuracy | 61.29% | **65.21%** |

The selected decision rule is:

- Predict **Red** when `P(Red) >= 0.60`.
- Predict **Blue** otherwise.

The threshold changes the classification decision but does not change the predicted probabilities.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Historical Experiments | XGBoost |
| Visualization | Matplotlib, Seaborn |
| Development | Jupyter Notebook |
| Model Serialization | joblib |
| API Framework | FastAPI, Pydantic |
| ASGI Server | Uvicorn |
| Containerization | Docker |

---

## Project Structure

```text
ml-ufc/
├── app/
│   └── main.py
├── data/
├── models/
│   └── ufc_logistic_regression.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

**Main Components**

- `app/main.py` — FastAPI application and prediction endpoints.
- `data/` — historical UFC dataset.
- `models/` — saved machine learning pipeline and model metadata.
- `notebooks/` — exploratory analysis, training and evaluation.
- `Dockerfile` — instructions for building the Docker image.
- `.dockerignore` — files excluded from the Docker build context.
- `requirements.txt` — Python dependencies required to run the application.

---

## Getting Started

### Prerequisites

- Git
- Docker Desktop or Docker Engine

The Docker image includes the Python runtime and required application dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/muenjmtst/ufc-fight-predictor.git
cd ufc-fight-predictor
```

### 2. Build the Docker Image

```bash
docker build -t ufc-predictor:local .
```

This builds the application image containing FastAPI, its dependencies and the serialized machine learning model.

### 3. Run the Container

```bash
docker run --rm --name ufc-api -p 8000:8000 ufc-predictor:local
```

The API will be accessible at:

**http://localhost:8000**

### 4. Open API Documentation

FastAPI provides interactive Swagger documentation at:

**http://localhost:8000/docs**

You can inspect available endpoints, enter JSON requests and test predictions directly in the browser.

### 5. Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

Press `Ctrl+C` in the terminal running Docker to stop the container.

---

## API Endpoints

The application exposes the following REST API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns general application information |
| GET | `/health` | Checks whether the API responds |
| GET | `/model_info` | Returns model metadata |
| POST | `/compare` | Compares basic fighter attributes |
| POST | `/predict` | Predicts a UFC fight outcome |

### GET /model_info

Returns information about the loaded ML model.

Example response:

```json
{
  "model_type": "LogisticRegression",
  "threshold": 0.6,
  "n_features": 46,
  "status": "loaded"
}
```

### POST /predict

Accepts a JSON object containing 46 prepared model features.

The request is validated using Pydantic, converted to a pandas DataFrame, and passed to the saved scikit-learn Pipeline.

The response contains:

- Predicted winning corner
- Probability of a Red victory
- Probability of a Blue victory
- Classification threshold

**Note:** The API expects prepared fight statistics, not fighter names. Historical lookup and automatic feature generation are not yet implemented.

---

## Example Prediction

Below is an example of a request tested against the deployed UFC prediction API.

### Request

**POST /predict**

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

**Prediction:** Blue Corner wins.

The model estimates:

- Red win probability: **45.69%**
- Blue win probability: **54.31%**

This prediction was manually checked through FastAPI inside a Docker container. The probabilities matched the original Jupyter Notebook inference to floating-point precision.

### Input Format

- Age inputs are measured in years.
- Height and reach are measured in centimeters.
- Percentage statistics are represented as fractions, such as `0.39` instead of `39`.
- Engineered differences must follow the same feature definitions used during model training.
- JSON field names must match the expected Pydantic schema.

---

## API and Docker Verification

The application was manually tested locally and inside Docker.

The following endpoints successfully returned HTTP 200:

- `GET /`
- `GET /health`
- `GET /model_info`
- `POST /compare`
- `POST /predict`

For the tested fight, predictions from the Docker container matched the probabilities produced in the training notebook.

These checks demonstrate that the inference service works for the tested inputs. They do not replace automated integration tests or establish predictive performance on new future fights.

---

## Roadmap

### Completed

- [x] Exploratory Data Analysis
- [x] Data cleaning and feature engineering
- [x] Chronological train/validation/test splitting
- [x] Date-based cross-validation
- [x] Model training and hyperparameter tuning
- [x] Historical model comparison
- [x] Classification threshold selection
- [x] Final test evaluation
- [x] Model serialization with joblib
- [x] FastAPI REST API
- [x] Pydantic input schemas
- [x] ML inference endpoint
- [x] Manual API testing
- [x] Docker containerization
- [x] Prediction verification inside Docker

### Planned

- [ ] Verify dataset provenance and historical feature availability
- [ ] Reusable automated feature engineering for inference
- [ ] Fighter lookup and prediction by name
- [ ] Automated API and inference tests
- [ ] Probability calibration
- [ ] MLflow experiment tracking
- [ ] Additional evaluation on future fights

---

## Limitations

UFC fight prediction is inherently uncertain. Historical statistics cannot capture every factor that influences a fight, including injuries, preparation, tactical changes and unexpected events.

Important limitations include:

1. **Historical feature availability:** The source dataset's aggregation logic and timestamps have not been fully verified to establish that every feature was available before the corresponding fight.

2. **Earlier exploratory analysis:** Nine fights from the final test boundary date appeared in an earlier EDA split, limiting the independence of the final evaluation from all project exploration.

3. **Historical fighter statistics:** Recorded wins, losses and other aggregated statistics may be incomplete or inconsistent.

4. **Corner assignment bias:** Predictions may be influenced by systematic differences between Red and Blue corner assignments.

5. **Repeated fighters:** Individual fight observations are not fully independent because fighters appear across multiple fights.

6. **Inference requirements:** The API requires 46 prepared features. It does not currently retrieve fighter statistics or construct the features automatically from fighter names.

7. **Future performance:** The deployed API has been manually tested, but predictive performance and probability calibration have not been validated on fresh future fights.

Predicted probabilities are statistical model outputs and should not be interpreted as guarantees of fight outcomes.

---

## Future Improvements

Potential extensions include:

- Elo-based fighter ratings
- Recent fight performance and form
- Opponent strength
- Time since the previous fight
- Fighter-order symmetry testing
- Probability calibration
- Automatic feature generation
- Evaluation on a fresh future holdout

---

**Purpose:** Educational machine learning and MLOps portfolio project.