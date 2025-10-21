@echo off
REM Script để chạy so sánh các phương pháp NeRF trên Windows
REM Usage: run_comparison.bat

echo 🔬 NEURAL RADIANCE FIELD COMPARISON
echo ==================================

REM Kiểm tra dataset
set DATASET_PATH=./data/nerf_llff_data/horns
if not exist "%DATASET_PATH%" (
    echo ❌ Dataset not found at: %DATASET_PATH%
    echo Please download LLFF dataset and place it in the correct location
    pause
    exit /b 1
)

echo ✅ Dataset found: %DATASET_PATH%

REM Tạo output directory
set OUTPUT_DIR=./comparison_results
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo 📁 Output directory: %OUTPUT_DIR%

REM Chạy so sánh với FrugalNeRF (hiện tại chỉ có model này)
echo.
echo 🚀 Running comparison with available models...

python compare_methods.py --dataset "%DATASET_PATH%" --output_dir "%OUTPUT_DIR%" --train_frames 20 42 --test_frames 0 8 16 24 32 40 48 56 --models FrugalNeRF

echo.
echo 🎉 Comparison completed!
echo 📊 Check results in: %OUTPUT_DIR%

pause
