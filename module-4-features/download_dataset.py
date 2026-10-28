"""Скачивание оригинального набора Online Shoppers Purchasing Intention."""
from pathlib import Path
from io import BytesIO
import pandas as pd
import requests

URLS = [
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv',
    'https://raw.githubusercontent.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/master/online_shoppers_intention.csv',
]
TARGET = Path('data/online_shoppers_intention.csv')
TARGET.parent.mkdir(parents=True, exist_ok=True)

def get_dataset():
    last_error = None
    for url in URLS:
        try:
            response = requests.get(url, timeout=35)
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content))
            if frame.shape != (12330, 18):
                raise ValueError(f'Неожиданный размер: {frame.shape}')
            if frame['Revenue'].astype(str).str.lower().value_counts().to_dict() != {'false': 10422, 'true': 1908}:
                raise ValueError('Не совпадает количество классов')
            TARGET.write_bytes(response.content)
            print(f'Датасет сохранён: {TARGET} | источник: {url}')
            return
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            print('Источник недоступен:', url, str(exc)[:120])
    raise RuntimeError('Не получилось скачать датасет. Загрузите CSV вручную с https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset и поместите в data/online_shoppers_intention.csv') from last_error

if __name__ == '__main__':
    get_dataset()
