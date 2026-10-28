@echo off
cd /d "%~dp0"
py -3.11 -m pip install -r requirements.txt
if errorlevel 1 (echo Ошибка установки библиотек & pause & exit /b 1)
py -3.11 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebook.ipynb
if errorlevel 1 (echo Ошибка выполнения. Проверьте файлы data/train.csv и data/test.csv & pause & exit /b 1)
echo Работа выполнена. Проверьте папки models и examples.
pause
