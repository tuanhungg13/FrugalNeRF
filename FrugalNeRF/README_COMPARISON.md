# So Sánh Các Phương Pháp NeRF

## Tổng Quan

Dự án hiện tại chỉ có **FrugalNeRF**, nhưng đã được chuẩn bị để so sánh với các phương pháp khác như TensoRF và SparseNeRF.

## Cách Sử Dụng

### 1. Chạy So Sánh Đơn Giản

**Trên Windows:**
```bash
run_comparison.bat
```

**Trên Linux/Mac:**
```bash
bash run_comparison.sh
```

### 2. Chạy So Sánh Tùy Chỉnh

```bash
python compare_methods.py \
    --dataset ./data/nerf_llff_data/horns \
    --output_dir ./comparison_results \
    --train_frames 20 42 \
    --test_frames 0 8 16 24 32 40 48 56 \
    --models FrugalNeRF
```

### 3. Thêm Model Mới

Để thêm model mới (ví dụ: SparseNeRF), bạn cần:

1. **Tạo config file** trong `configs/`:
   ```bash
   cp configs/sparsenerf_2v.txt configs/my_model_2v.txt
   ```

2. **Chỉnh sửa config** theo model của bạn

3. **Chạy so sánh**:
   ```bash
   python compare_methods.py --models FrugalNeRF MyModel
   ```

## Cấu Trúc Kết Quả

```
comparison_results/
├── comparison_20241201_143022/
│   ├── FrugalNeRF_experiment/
│   │   ├── checkpoints/
│   │   └── logs/
│   └── comparison_results.json
```

## Metrics Được Đo

- **PSNR** (Peak Signal-to-Noise Ratio) ↑ - Càng cao càng tốt
- **SSIM** (Structural Similarity Index) ↑ - Càng cao càng tốt  
- **LPIPS** (Learned Perceptual Image Patch Similarity) ↓ - Càng thấp càng tốt
- **Training Time** - Thời gian training

## Ví Dụ Kết Quả

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time |
|--------|--------|--------|---------|---------------|
| FrugalNeRF | 20.05 | 0.78 | 0.16 | 24 mins |

## Lưu Ý

1. **Dataset**: Cần có dataset LLFF trong `./data/nerf_llff_data/horns`
2. **GPU**: Cần GPU để training nhanh
3. **Memory**: Cần ít nhất 8GB RAM
4. **Time**: Mỗi model mất khoảng 20-60 phút để train

## Troubleshooting

### Lỗi "Dataset not found"
```bash
# Tải dataset LLFF
wget https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
# Giải nén vào ./data/nerf_llff_data/
```

### Lỗi "CUDA out of memory"
```bash
# Giảm batch size trong config
batch_size=2048  # thay vì 4096
```

### Lỗi "Config not found"
```bash
# Kiểm tra file config có tồn tại
ls configs/
# Tạo config mới nếu cần
```

## Mở Rộng

Để thêm model mới:

1. **Tạo model class** trong `models/`
2. **Tạo config file** trong `configs/`
3. **Cập nhật** `compare_methods.py` để hỗ trợ model mới
4. **Test** với dataset nhỏ trước

## Liên Hệ

Nếu có vấn đề, hãy tạo issue trên GitHub repository.
