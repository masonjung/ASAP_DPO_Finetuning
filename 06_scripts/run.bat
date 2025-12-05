@echo off
REM Windows batch script for easy training management (numbered layout)

echo =====================================
echo Llama-3.2-1B Fine-Tuning Helper
echo =====================================
echo.

:menu
echo Choose an option:
echo.
echo 1. Start Training
echo 2. Run Inference
echo 3. Merge LoRA
echo 4. Run Eval (A/B)
echo 5. Install Dependencies
echo 6. Exit
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto inference
if "%choice%"=="3" goto merge
if "%choice%"=="4" goto eval
if "%choice%"=="5" goto install
if "%choice%"=="6" goto end

echo Invalid choice!
goto menu

:train
echo.
echo Starting training...
echo.
python 03_src\train_llama32_sft.py
pause
goto menu

:inference
echo.
echo Starting inference...
python 03_src\run_inference.py --adapter_path 04_models\adapters\output_llama32_sft
pause
goto menu

:merge
echo.
echo Merging LoRA adapters...
python 03_src\merge_lora.py --adapter_path 04_models\adapters\output_llama32_sft --output_path 04_models\merged\merged_model
pause
goto menu

:eval
echo.
echo Running A/B eval...
python 03_src\eval\evaluate.py --prompts_path 02_data\eval\eval_prompts.jsonl --adapter_path 04_models\adapters\output_llama32_sft
pause
goto menu

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pause
goto menu

:end
echo.
echo Goodbye!
