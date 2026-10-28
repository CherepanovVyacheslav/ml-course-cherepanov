# 🚢 Module 1 — Titanic: EDA + бинарная классификация

**Автор:** Черепанов В.С., ПКТб-23-1  
**Дата:** 2026-21-10

## 📊 Результаты

	accuracy	precision	recall	f1	roc_auc	
model						
logistic_regression	0.8045	0.7833	0.6812	0.7287	0.8489
decision_tree	0.7821	0.7419	0.6667	0.7023	0.8132
random_forest	0.7933	0.7759	0.6522	0.7087	0.8401


**Время обучения:** 0.0246 сек (LR),0.0069 сек (DT), 0.1854 (RF)


## 🚀 Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL =  "https://raw.githubusercontent.com/"
    "CherepanovVyacheslav/ml-course-cherepanov/"
    "main/module-1-titanic"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/lr_model.pkl").content))
