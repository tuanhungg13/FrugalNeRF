#!/bin/bash

# Script để chạy so sánh các phương pháp NeRF
# Usage: bash run_comparison.sh

echo "🔬 NEURAL RADIANCE FIELD COMPARISON"
echo "=================================="

# Kiểm tra dataset
DATASET_PATH="./data/nerf_llff_data/horns"
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Dataset not found at: $DATASET_PATH"
    echo "Please download LLFF dataset and place it in the correct location"
    exit 1
fi

echo "✅ Dataset found: $DATASET_PATH"

# Tạo output directory
OUTPUT_DIR="./comparison_results"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Chạy so sánh với FrugalNeRF (hiện tại chỉ có model này)
echo ""
echo "🚀 Running comparison with available models..."

python compare_methods.py \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_frames 20 42 \
    --test_frames 0 8 16 24 32 40 48 56 \
    --models FrugalNeRF

echo ""
echo "🎉 Comparison completed!"
echo "📊 Check results in: $OUTPUT_DIR"

# Hiển thị kết quả nếu có
RESULTS_FILE="$OUTPUT_DIR/comparison_*/comparison_results.json"
if ls $RESULTS_FILE 1> /dev/null 2>&1; then
    echo ""
    echo "📋 Results summary:"
    cat $RESULTS_FILE | python -m json.tool
fi
