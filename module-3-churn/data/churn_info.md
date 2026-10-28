# Telco Customer Churn: сведения о данных

Источник: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Размер: 7043 наблюдений, 21 столбцов.

Целевая переменная: Churn (`Yes` — ушёл, `No` — остался).

## Баланс классов
Churn
No     5174
Yes    1869

## Доля классов (%)
Churn
No     73.46
Yes    26.54

## Пропуски
                  Пропусков  Доля, %
TotalCharges             11     0.16
gender                    0     0.00
SeniorCitizen             0     0.00
Partner                   0     0.00
customerID                0     0.00
Dependents                0     0.00
tenure                    0     0.00
MultipleLines             0     0.00
PhoneService              0     0.00
OnlineSecurity            0     0.00
OnlineBackup              0     0.00
DeviceProtection          0     0.00
InternetService           0     0.00
TechSupport               0     0.00
StreamingTV               0     0.00
Contract                  0     0.00
StreamingMovies           0     0.00
PaperlessBilling          0     0.00
PaymentMethod             0     0.00
MonthlyCharges            0     0.00
Churn                     0     0.00

## Типы полей
customerID              str
gender                  str
SeniorCitizen         int64
Partner                 str
Dependents              str
tenure                int64
PhoneService            str
MultipleLines           str
InternetService         str
OnlineSecurity          str
OnlineBackup            str
DeviceProtection        str
TechSupport             str
StreamingTV             str
StreamingMovies         str
Contract                str
PaperlessBilling        str
PaymentMethod           str
MonthlyCharges      float64
TotalCharges            str
Churn                   str
