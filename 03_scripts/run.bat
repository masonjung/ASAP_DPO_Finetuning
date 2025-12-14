@echo off
REM Windows batch script for easy training management (numbered layout)

echo =====================================
echo Llama-3.2-1B DPO Helper
echo =====================================
echo.

:menu
echo Choose an option:
echo.
echo 1. Start DPO Training
echo 2. Run Inference
echo 3. Merge LoRA
echo 4. Run Eval (A/B)
echo 5. Install Dependencies
echo 6. Exit
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto dpo
if "%choice%"=="2" goto inference
if "%choice%"=="3" goto merge
if "%choice%"=="4" goto eval
if "%choice%"=="5" goto install
if "%choice%"=="6" goto end

echo Invalid choice!
goto menu

:dpo
echo.
echo Starting DPO training...
echo.
python 02_src\train_dpo.py --config 00_configs\dpo.json
pause
goto menu

:inference
echo.
echo Starting inference...
python 02_src\run_inference.py --adapter_path 04_models\adapters\output_dpo
pause
goto menu

:merge
echo.
echo Merging LoRA adapters...
python 02_src\merge_lora.py --adapter_path 04_models\adapters\output_dpo --output_path 04_models\merged\merged_model_dpo
pause
goto menu

:eval
echo.
echo Running A/B eval...
python 02_src\eval\evaluate.py --prompts_path 01_data\eval\eval_prompts.jsonl --adapter_path 04_models\adapters\output_dpo
pause
goto menu

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pause
goto menu

:end
echo.
echo Goodbye!
