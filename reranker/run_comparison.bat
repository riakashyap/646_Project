@REM Copyright:

@REM   Copyright © 2025 Ananya-Jha-code

@REM   You should have received a copy of the MIT license along with this file.
@REM   If not, see https://mit-license.org/

@echo off
REM Check if JAVA_HOME is set
if "%JAVA_HOME%"=="" (
    echo [ERROR] JAVA_HOME is not set.
    echo Please set JAVA_HOME to your JDK 21 installation path.
    echo Example: set JAVA_HOME="C:\Program Files\Java\jdk-21"
    pause
    exit /b 1
)


REM Activate Virtual Environment
call .\venv\Scripts\activate.bat

echo Starting Complete Comparison Run...

echo.
echo ==========================================
echo Step 1: Generating Training Data (Small Subset)
echo ==========================================
python src/setup_data.py

echo.
echo ==========================================
echo Step 2: Running Baseline: Pairwise Training
echo ==========================================
python -m reranker.run_trainer --model_type pairwise --save_dir output/pairwise_small --epochs 1 --batch_size 1

echo.
echo ==========================================
echo Step 3: Running New Method: E2Rank Training
echo ==========================================
python -m reranker.run_trainer --model_type e2rank --save_dir output/e2rank_small --epochs 1 --batch_size 1

echo.
echo ==========================================
echo Step 4: Running Evaluation
echo ==========================================
python evaluate_rerankers.py

echo.
echo ==========================================
echo Comparison Complete!
echo ==========================================
pause