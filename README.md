# SuperKart Sales Forecasting + Model Deployment

End-to-end regression pipeline to forecast product-store sales for a retail chain, packaged with a Flask REST API and a lightweight frontend for non-technical stakeholders. Covers the full ML lifecycle: EDA → feature engineering → model training → serialisation → API → frontend demo.

## Business Problem
Inaccurate demand forecasts lead to stockouts and overstock at SuperKart. The goal is to predict `Product_Store_Sales_Total` at the product-store level so procurement and inventory teams can plan accurately.

## Dataset
- `SuperKart.csv` — product features, store metadata, and historical sales
- Key features: `Product_Sugar_Content`, `Store_Size`, `Store_Location_City_Type`, `Product_MRP`, allocated display area, product type

## Approach
1. EDA — product and store level distributions; feature correlation analysis
2. Feature engineering — one-hot encoding via `ColumnTransformer`; scikit-learn `Pipeline` for reproducible preprocessing
3. Modeling — Decision Tree and XGBoost regressors (base + hyperparameter-tuned via RandomizedSearchCV)
4. Model serialisation — production pipeline saved with `joblib`
5. Deployment — Flask REST API (`/predict` endpoint, JSON in/out); HTML/JS frontend for demo queries; Dockerfiles for both services

## Results

| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|-----|------|
| Decision Tree | 615.9 | 485.4 | 0.668 | 18.7% |
| **Decision Tree (tuned) — selected** | **615.7** | **485.4** | **0.669** | **18.7%** |
| XGBoost | 615.9 | 485.4 | 0.668 | 18.7% |
| XGBoost (tuned) | 616.1 | 486.0 | 0.668 | 18.8% |

**Selected model:** Tuned Decision Tree — highest R² (0.669), RMSE 615.7. Models explain ~67% of variance in product-store sales; remaining variance is attributable to promotions, seasonality, and local events not in the dataset.

## Key Findings
- Store size, product MRP, and allocated display area are consistently top contributors to sales
- Hyperparameter tuning yields marginal gains — the base models already capture most explainable variance from available features
- Packaging the model behind a REST API enables procurement teams to query forecasts without touching the code

## Deployment Architecture
```
SuperKart.csv
     │
     ▼
ColumnTransformer (OneHotEncoder) + Decision Tree Pipeline
     │   serialised via joblib → xgb_tuned_model.joblib
     ▼
Flask API  ──  POST /predict  (JSON input → sales forecast)
     │
     ▼
HTML/JS Frontend  (form → forecast display)
```

## Running Locally
```bash
# Backend
cd backend_files
pip install -r requirements.txt
python app.py

# Frontend — open frontend_files/ in a browser
# Or use Docker: docker build + docker run for each service
```

## Recommendations
- Enrich features with promotional calendar, local weather, and holiday flags to push R² above 0.70
- Automate weekly forecast refreshes and connect API output to inventory planning workflows
- Monitor MAPE drift by store/product category; trigger retraining when it exceeds a defined threshold

## Technologies
Python · Pandas · NumPy · Scikit-learn · XGBoost · joblib · Flask · HTML/JS · Docker · Jupyter Notebook

## Code
- Main notebook: [`Project_7_ChinmayRozekar_SuperKart_Model_Deployment_Notebook.ipynb`](Project_7_ChinmayRozekar_SuperKart_Model_Deployment_Notebook.ipynb)
- Minimal version: [`Low_Code_SuperKart_Model_Deployment_Notebook.ipynb`](Low_Code_SuperKart_Model_Deployment_Notebook.ipynb)
- API: [`backend_files/app.py`](backend_files/app.py)
- Frontend: [`frontend_files/`](frontend_files/)

---
*Author: Chinmay Rozekar*
