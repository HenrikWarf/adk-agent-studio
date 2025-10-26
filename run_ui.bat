@echo off
REM Start the Agent Template UI
echo Starting Agent Template UI...
call venv\Scripts\activate.bat
streamlit run app.py

