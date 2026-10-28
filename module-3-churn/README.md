# Модуль 3. Ансамблевые методы: Customer Churn

Практическая работа по дисциплине «Машинное обучение и ИИ». Датасет — [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7043 клиента.

## Содержание

`notebook.ipynb` — отчёт из 14 разделов с EDA, четырьмя ансамблями, метриками, Optuna (30 испытаний), SHAP и сохранением моделей.

`data/` — CSV и описание; `models/` — сохранённые объекты после выполнения; `examples/` — графики после выполнения.

## Как запустить

В папке `module-3-churn` выполните:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

В ноутбуке выберите **Restart & Run All**. Если интернет недоступен, скачайте `WA_Fn-UseC_-Telco-Customer-Churn.csv` с Kaggle и сохраните под именем `data/telco_churn.csv`. Данные также доступны [из IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data).

После запуска ноутбук создаёт файлы `.pkl`, `feature_cols.json`, `preprocess_config.json`, `metrics.json` и графики `.png`.

**Реальные метрики:** смотрите `models/metrics.json`, созданный после запуска; значения не задаются заранее.

Для проверки `predict_saved.py` сначала выполните ноутбук, затем `python predict_saved.py`.

## Публикация

Добавьте папку в существующий репозиторий `ml-course-cherepanov` рядом с `module-1-titanic` и `module-2-house-prices`. Не создавайте новый репозиторий. Проверьте, что `.pkl` опубликованы в `main`, а в разделе 12 ноутбука установите `CHECK_GITHUB=True`. Для загрузки используются прямые raw-ссылки, а не ссылки на HTML-страницу GitHub.

## Замечание

Набор является учебным снимком данных. Результаты нельзя автоматически переносить на будущие периоды и трактовать как причинные эффекты бизнес-мер.
