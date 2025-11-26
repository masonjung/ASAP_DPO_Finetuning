@echo off
REM Windows batch script for easy training management

echo =====================================
echo Llama-3.2-3B Fine-Tuning Helper
echo =====================================
echo.

:menu
echo Choose an option:
echo.
echo 1. Test Setup
echo 2. Validate Dataset
echo 3. Start Training
echo 4. Run Inference
echo 5. Merge LoRA
echo 6. Install Dependencies
echo 7. Exit
echo.

set /p choice="Enter choice (1-7): "

if "%choice%"=="1" goto test_setup
if "%choice%"=="2" goto validate_dataset
if "%choice%"=="3" goto train
if "%choice%"=="4" goto inference
if "%choice%"=="5" goto merge
if "%choice%"=="6" goto install
if "%choice%"=="7" goto end

echo Invalid choice!
goto menu

:test_setup
echo.
echo Testing setup...
python test_setup.py
pause
goto menu

:validate_dataset
echo.
set /p dataset_path="Enter dataset path (e.g., dataset/my_data.jsonl): "
python validate_dataset.py %dataset_path%
pause
goto menu

:train
echo.
echo Starting training...
echo This will take 40-90 minutes on RTX 4060
echo.
python train_llama32_sft.py
pause
goto menu

:inference
echo.
echo Starting inference...
python run_inference.py
pause
goto menu

:merge
echo.
echo Merging LoRA adapters...
python merge_lora.py
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
