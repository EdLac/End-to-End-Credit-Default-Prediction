# CreditSight – Credit Default Prediction App

Live Demo: https://creditsight.streamlit.app

## Overview

**CreditSight** is an end-to-end machine learning application designed to predict the risk of credit default in real time.

The project combines data preprocessing, model training, and deployment into a production-ready web application using Streamlit.

It enables users to input borrower information and instantly receive:
- Default prediction (0 / 1)
- Probability of default

## Project Architecture

```
project/
│
├── app/ # Streamlit application
│ └── app.py
│
├── models/ # Trained models
│ ├── LogisticRegression.pkl
│ └── LogisticRegression_columns.json
│ ├── RandomForest.pkl
│ └── RandomForest.json
│ ├── XGBoost.pkl
│ └── XGBoost.json
│
├── src/ # Core ML logic
│ ├── preprocess.py
│ ├── training.py
│ └── inference.py
│
├── notebooks/ # EDA & experiments
├── requirements.txt
├── README.md
```

## Features

- Real-time credit risk prediction
- Feature engineering (debt-to-income, employment group)
- Consistent training/inference pipeline
- Clean MLOps-style structure
- Cloud deployment via Streamlit Community Cloud

## Best Model

- Algorithm: **Logistic Regression**
- Task: Binary classification (default vs no default)
- Output:
  - Prediction (0 = no default, 1 = default)
  - Probability of default

## Data Processing

The model uses several engineered features:
- Debt-to-income ratio
- Employment group categorization
- One-hot encoding
- Column alignment between training and inference

## Deployment

The application is deployed using:

- **Streamlit Community Cloud**
- GitHub integration (main branch)
- `requirements.txt` for dependency management

## Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/End-to-End-Credit-Default-Prediction.git

cd End-to-End-Credit-Default-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Future Improvements
- Improve UI/UX
- Add API endpoint (FastAPI)
- Model monitoring

## Authors
- Édouard LACROIX
- Élodie NGIRABANZI
- Élise PRIGENT
- Imane MAIGA

## Acknowledgements
This project was developed as part of a Data Analytics / MLOps learning journey.
