@echo off
REM Double-click to set up (first time only) and launch PranaWatch.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || (echo Python not found. Install Python 3.10+ from python.org & pause & exit /b 1)
  call .venv\Scripts\activate.bat
  echo Installing packages, this takes 2-4 minutes the first time...
  python -m pip install --upgrade pip >nul
  python -m pip install -r requirements.txt || (echo Install failed & pause & exit /b 1)
) else (
  call .venv\Scripts\activate.bat
  python -m pip install -q -r requirements.txt
)
echo.
echo Checking Gemini key...
python check_gemini.py
echo.
echo Starting PranaWatch at http://localhost:8501 ...
streamlit run app.py
pause
