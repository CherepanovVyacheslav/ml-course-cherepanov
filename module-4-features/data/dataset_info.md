# Online Shoppers Purchasing Intention Dataset

**Официальный источник:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset (DOI: 10.24432/C5F88Q).  
**Лицензия:** CC BY 4.0.  
**Объём:** 12 330 сессий, 17 исходных признаков, 1 целевая переменная `Revenue`.  
**Классы:** False — 10 422 (84,5%), True — 1 908 (15,5%).

## Столбцы

- `Administrative`, `Informational`, `ProductRelated` — счётчики посещённых страниц.
- `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` — длительность просмотра.
- `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` — метрики поведения и близости к особым датам.
- `Month`, `Weekend` — признаки календаря.
- `OperatingSystems`, `Browser`, `Region`, `TrafficType` — технические категории (первоначально записаны целыми кодами).
- `VisitorType` — тип посетителя.
- `Revenue` — была ли покупка (True/False).

В исходном наборе нет пропусков. В проекте предусмотрено заполнение на случай иных версий CSV.
Загрузить файл можно командой `python download_dataset.py` из папки модуля.
