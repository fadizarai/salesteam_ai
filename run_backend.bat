@echo off
echo ====================================================
echo Demarrage du Backend SalesTeam AI (FastAPI)...
echo ====================================================
call venv\Scripts\activate.bat
python -m uvicorn src.api.main:app --port 8000 --reload
pause
