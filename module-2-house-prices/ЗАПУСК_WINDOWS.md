# Запуск на Windows (Python 3.11)

Откройте PowerShell в папке `module-2-house-prices` и выполните:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m notebook notebook.ipynb
```

В Jupyter выберите **Run → Run All Cells**. При ошибке отсутствия `seaborn` выполните `%pip install seaborn` в новой ячейке и перезапустите kernel.

Если доступ к GitHub заблокирован, скачайте исходные CSV из Kaggle в папку `data/` и повторите выполнение. После появления моделей и графиков загрузите папку `module-2-house-prices` в свой GitHub-репозиторий. В разделе 11 используется **прямой** адрес `raw.githubusercontent.com/.../main/module-2-house-prices/models/best_model.pkl`, а не `github.com/.../main/...`.

Нужно заполнить отчество, группу и ФИО преподавателя в первой ячейке ноутбука. После публикации повторите проверку GitHub в разделе 11.
