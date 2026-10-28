"""Прогноз оттока с использованием сохранённых артефактов модуля 3.

Запуск: python predict_saved.py (после выполнения notebook.ipynb).
Для нового клиента используйте predict_churn(dict(...)).
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SERVICE_COLUMNS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]


def load_artifacts(model_dir=ROOT / 'models'):
    """Загрузить только файлы собственного обученного проекта."""
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / 'best_model.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')
    encoders = joblib.load(model_dir / 'label_encoders.pkl')
    columns = json.loads((model_dir / 'feature_cols.json').read_text(encoding='utf-8'))
    config = json.loads((model_dir / 'preprocess_config.json').read_text(encoding='utf-8'))
    return model, scaler, encoders, columns, config


def make_features(frame):
    """Воспроизвести Feature Engineering из раздела 5 ноутбука."""
    z = frame.drop(columns=['customerID', 'Churn'], errors='ignore').copy()
    z['TotalCharges'] = pd.to_numeric(z['TotalCharges'], errors='coerce')
    z['TenureYears'] = z['tenure'] / 12
    z['AvgMonthlyCharge'] = z['TotalCharges'] / (z['tenure'] + 1)
    z['IsNewClient'] = (z['tenure'] <= 6).astype(int)
    z['HasMultipleServices'] = z[SERVICE_COLUMNS].eq('Yes').sum(axis=1)
    return z


def predict_churn(customer, model_dir=ROOT / 'models'):
    """Вернуть бинарный прогноз и вероятность ухода клиента из исходных полей."""
    model, scaler, encoders, columns, config = load_artifacts(model_dir)
    raw = pd.DataFrame([customer])
    required = [c for c in columns if c not in {
        'TenureYears', 'AvgMonthlyCharge', 'IsNewClient', 'HasMultipleServices'
    }]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError('Не хватает исходных полей: ' + ', '.join(missing))
    z = make_features(raw).reindex(columns=columns)
    numeric_cols = config['num_cols']
    medians = config['num_medians']
    z[numeric_cols] = z[numeric_cols].apply(pd.to_numeric, errors='coerce')
    z[numeric_cols] = z[numeric_cols].fillna(medians)
    z[numeric_cols] = scaler.transform(z[numeric_cols])
    for col, le in encoders.items():
        category_map = {label: i for i, label in enumerate(le.classes_)}
        z[col] = z[col].fillna('Unknown').astype(str).map(category_map).fillna(-1).astype(int)
    z = z[columns].astype(float)
    prob = float(model.predict_proba(z)[0, 1])
    return {'prediction': int(prob >= 0.5), 'churn_probability': prob}


if __name__ == '__main__':
    # Пример берётся из исходной таблицы, а не создаётся искусственно.
    data_path = ROOT / 'data' / 'telco_churn.csv'
    if not data_path.exists():
        raise FileNotFoundError('Нет data/telco_churn.csv: сначала выполните notebook.ipynb')
    data = pd.read_csv(data_path)
    for _, customer in data.drop(columns=['Churn']).head(5).iterrows():
        answer = predict_churn(customer.to_dict())
        print(customer['customerID'], answer)
