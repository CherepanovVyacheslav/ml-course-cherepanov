# Модуль 4 — Инженерия признаков: Online Shoppers

**Автор:** Черепанов Вячеслав  
**Дисциплина:** Машинное обучение и ИИ  
**Учебный год:** 2025/2026  
**Дата:** 28.10.2026

## Задача

Прогнозирование факта покупки по характеристикам сессии пользователя сайта. В работе создаются 13 производных признаков, исследуются распределения и сравниваются Random Forest, XGBoost, LightGBM, CatBoost. Для LightGBM выполняется подбор параметров Optuna (30 испытаний, 5-fold CV), интерпретация SHAP и сохранение моделей.

## Как запустить

```bash
cd module-4-features
python -m pip install -r requirements.txt
python download_dataset.py
jupyter notebook notebook.ipynb
```

В ноутбуке выполнить **Kernel → Restart & Run All**. Автоматически появятся `models/*.pkl`, `models/*.json`, `examples/*.png`. Если интернет недоступен, скачать `online_shoppers_intention.csv` с [UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) и положить в `data/`.

## Результаты

Таблица метрик выводится в разделе 7 и сохраняется в `models/metrics.json` после выполнения; числа не подставляются до запуска на исходном CSV.

## Публикация

После выполнения загрузить **все файлы** (включая `.pkl`) в `main` общего репозитория `ml-course-cherepanov`. Проверить раздел 12 ноутбука, который скачивает `best_model.pkl` по raw-ссылке и сравнивает три предсказания.

## Состав

`notebook.ipynb` — отчёт из 14 разделов; `data/` — источник; `models/` — модели и метаданные; `examples/` — графики; `download_dataset.py` — воспроизводимая загрузка; `requirements.txt` — зависимости.
