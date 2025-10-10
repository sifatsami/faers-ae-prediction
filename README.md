# FAERS Adverse Event Prediction

Predicting serious adverse drug events using the FDA FAERS dataset with end-to-end ML pipelines and a FastAPI interface.

---

## About

The **FAERS Adverse Event Prediction System** predicts whether a reported drug reaction is **serious or non-serious**, using data from the **FDA Adverse Event Reporting System (FAERS)**.

It includes:
- Data ingestion and preprocessing (OpenFDA → DuckDB → Parquet)
- Model training with Logistic Regression, Random Forest, Gradient Boosting
- Explainability using SHAP
- Real-time predictions via FastAPI

---

## Structure

```
faers-ae-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_ingest.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_validation.ipynb
│   └── 05_api_tests.ipynb
├── src/
│   ├── data/
│   │   └── fetch_openfda.py
│   ├── etl/
│   │   └── normalize.py
│   ├── models/
│   │   ├── train_models.py
│   │   └── evaluate.py
│   └── api/
│       └── app.py
├── models/
│   ├── logreg_baseline.joblib
│   ├── rf_model.joblib
│   ├── gb_model.joblib
│   └── logreg_features.json
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1️. Clone the Repository
```bash
git clone https://github.com/<sifatsami>/faers-ae-prediction.git
cd faers-ae-prediction
```

### 2️. Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3️. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️. Add OpenFDA API Key
Copy `.env.example` → `.env` and add:
```
OPENFDA_API_KEY=your_key_here
```

### 5. Fetch Data
```bash
python src/data/fetch_openfda.py
```

### 6. Run the API
```bash
uvicorn src.api.app:app --reload
```
Open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Example API Request

**POST** `/predict?model=logreg`
```json
{
  "age": 35,
  "sex": "male",
  "drug_name": "aspirin",
  "reaction": "death"
}
```

**Response**
```json
{
  "probability": 0.78
}
```

---

## Models Trained

| Model | Type | ROC-AUC | PR Score | Notes |
|--------|------|----------|-----------|--------|
| Logistic Regression | Linear | ~0.83 | Stable baseline |
| Random Forest | Ensemble | ~0.88 | Non-linear feature capture |
| Gradient Boosting | Ensemble | ~0.90 | Best performer overall |

---

## Explainability with SHAP

- `drug_name` → Most predictive feature  
- `reaction` type → Strongest correlation with seriousness  
- `age_bin_65+` → Higher risk  
- `sex_male` → Moderate effect  

---

## Deployment (Docker)
```bash
docker build -t faers-api .
docker run -p 8000:8000 faers-api
```

---

🔗 [LinkedIn](https://www.linkedin.com/in/sifat-sami/)
