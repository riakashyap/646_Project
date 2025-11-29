@REM Copyright:

@REM   Copyright © 2025 Ananya-Jha-code

@REM   You should have received a copy of the MIT license along with this file.
@REM   If not, see https://mit-license.org/

@echo off
REM Set JAVA_HOME to your JDK 21 installation
set JAVA_HOME="C:\Program Files\Microsoft\jdk-21.0.9.10-hotspot"

REM Activate Virtual Environment
call .\venv\Scripts\Activate.ps1

echo Starting Complete Comparison Run...

echo.
echo ==========================================
echo Step 1: Generating Training Data (Small Subset)
echo ==========================================
python src/prepare_training_data.py --limit 50

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
