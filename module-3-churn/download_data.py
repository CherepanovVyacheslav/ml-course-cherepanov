"""Получение исходного CSV для учебной работы."""
from pathlib import Path
from io import BytesIO
import requests
import pandas as pd

URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
path = Path(__file__).resolve().parent / 'data' / 'telco_churn.csv'
response = requests.get(URL, timeout=45)
response.raise_for_status()
frame = pd.read_csv(BytesIO(response.content))
assert frame.shape == (7043, 21) and 'Churn' in frame.columns
path.parent.mkdir(exist_ok=True)
path.write_bytes(response.content)
print(f'Данные сохранены: {path}, строк: {len(frame)}')
