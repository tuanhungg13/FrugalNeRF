#!/usr/bin/env python3
"""
Script để so sánh các phương pháp NeRF khác nhau
Hiện tại chỉ có FrugalNeRF, nhưng có thể mở rộng để thêm SparseNeRF, TensoRF baseline
"""

import os
import time
import json
import argparse
import subprocess
from datetime import datetime

def run_training(config_path, model_name, dataset_path, output_dir, train_frames, test_frames):
    """
    Chạy training cho một model cụ thể
    """
    print(f"\n🚀 Training {model_name}...")
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Train frames: {train_frames}")
    print(f"Test frames: {test_frames}")
    
    # Tạo output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Chuẩn bị command
    cmd = [
        "python", "train.py",
        "--config", config_path,
        "--datadir", dataset_path,
        "--basedir", model_output_dir,
        "--expname", f"{model_name}_experiment",
        "--train_frame_num"] + [str(f) for f in train_frames] + [
        "--test_frame_num"] + [str(f) for f in test_frames]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Chạy training và đo thời gian
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} training completed in {training_time/60:.1f} minutes")
            return {
                "model": model_name,
                "status": "success",
                "training_time": training_time,
                "output_dir": model_output_dir
            }
        else:
            print(f"❌ {model_name} training failed")
            print(f"Error: {result.stderr}")
            return {
                "model": model_name,
                "status": "failed",
                "error": result.stderr,
                "training_time": training_time
            }
    except subprocess.TimeoutExpired:
        training_time = time.time() - start_time
        print(f"⏰ {model_name} training timeout after {training_time/60:.1f} minutes")
        return {
            "model": model_name,
            "status": "timeout",
            "training_time": training_time
        }

def evaluate_model(model_output_dir, model_name, dataset_path, test_frames, config_path):
    """
    Đánh giá model đã train
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Tìm checkpoint mới nhất
    ckpt_dir = os.path.join(model_output_dir, f"{model_name}_experiment")
    if not os.path.exists(ckpt_dir):
        print(f"❌ No checkpoint directory found for {model_name}")
        return None
    
    # Tìm file checkpoint
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpt_files:
        print(f"❌ No checkpoint files found for {model_name}")
        return None
    
    latest_ckpt = max(ckpt_files, key=lambda x: os.path.getctime(os.path.join(ckpt_dir, x)))
    ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
    
    print(f"Using checkpoint: {ckpt_path}")
    
    # Chạy evaluation với config tương ứng
    cmd = [
        "python", "train.py",
        "--config", config_path,  # Sử dụng config tương ứng với model
        "--datadir", dataset_path,
        "--ckpt", ckpt_path,
        "--render_only", "1",
        "--render_test", "1",
        "--test_frame_num"] + [str(f) for f in test_frames]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            # Parse metrics từ output
            metrics = parse_metrics(result.stdout)
            
            # Nếu không parse được từ output, thử đọc từ file
            if not metrics:
                metrics = read_metrics_from_file(ckpt_dir)
            
            print(f"✅ {model_name} evaluation completed")
            print(f"PSNR: {metrics.get('psnr', 'N/A')}")
            print(f"SSIM: {metrics.get('ssim', 'N/A')}")
            print(f"LPIPS: {metrics.get('lpips', 'N/A')}")
            return metrics
        else:
            print(f"❌ {model_name} evaluation failed")
            print(f"Error: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} evaluation timeout")
        return None

def parse_metrics(output_text):
    """
    Parse metrics từ output text
    """
    metrics = {}
    lines = output_text.split('\n')
    
    print(f"🔍 Parsing metrics from output...")
    print(f"Output preview: {output_text[:500]}...")
    
    for line in lines:
        line = line.strip()
        # Tìm PSNR với format từ train.py: "======> {expname} test all psnr: {value} <========"
        if 'psnr:' in line.lower():
            try:
                import re
                # Format: "======> expname test all psnr: 25.3 <========"
                psnr_match = re.search(r'psnr:\s*(\d+\.?\d*)', line, re.IGNORECASE)
                if psnr_match:
                    psnr = float(psnr_match.group(1))
                    metrics['psnr'] = psnr
                    print(f"Found PSNR: {psnr}")
            except Exception as e:
                print(f"Error parsing PSNR: {e}")
                
        # Tìm SSIM
        elif 'ssim:' in line.lower():
            try:
                import re
                ssim_match = re.search(r'ssim:\s*(\d+\.?\d*)', line, re.IGNORECASE)
                if ssim_match:
                    ssim = float(ssim_match.group(1))
                    metrics['ssim'] = ssim
                    print(f"Found SSIM: {ssim}")
            except Exception as e:
                print(f"Error parsing SSIM: {e}")
                
        # Tìm LPIPS
        elif 'lpips:' in line.lower():
            try:
                import re
                lpips_match = re.search(r'lpips:\s*(\d+\.?\d*)', line, re.IGNORECASE)
                if lpips_match:
                    lpips = float(lpips_match.group(1))
                    metrics['lpips'] = lpips
                    print(f"Found LPIPS: {lpips}")
            except Exception as e:
                print(f"Error parsing LPIPS: {e}")
    
    print(f"Final metrics: {metrics}")
    return metrics

def read_metrics_from_file(ckpt_dir):
    """
    Đọc metrics từ file trong checkpoint directory
    """
    metrics = {}
    
    # Tìm file metrics có thể có
    possible_files = [
        'metrics.json',
        'results.json', 
        'evaluation.json',
        'test_results.json'
    ]
    
    for filename in possible_files:
        filepath = os.path.join(ckpt_dir, filename)
        if os.path.exists(filepath):
            try:
                import json
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if 'psnr' in data:
                        metrics['psnr'] = data['psnr']
                    if 'ssim' in data:
                        metrics['ssim'] = data['ssim']
                    if 'lpips' in data:
                        metrics['lpips'] = data['lpips']
                print(f"✅ Read metrics from {filename}")
                break
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return metrics

def create_comparison_table(results):
    """
    Tạo bảng so sánh kết quả
    """
    print("\n" + "="*80)
    print("📊 BẢNG SO SÁNH CÁC PHƯƠNG PHÁP")
    print("="*80)
    
    # Header
    print(f"{'Method':<15} {'PSNR ↑':<10} {'SSIM ↑':<10} {'LPIPS ↓':<10} {'Training Time':<15}")
    print("-" * 80)
    
    # Data rows
    for result in results:
        if result['status'] == 'success' and 'metrics' in result:
            method = result['model']
            psnr = result['metrics'].get('psnr', 'N/A')
            ssim = result['metrics'].get('ssim', 'N/A')
            lpips = result['metrics'].get('lpips', 'N/A')
            time_str = f"{result['training_time']/60:.1f} mins"
            
            print(f"{method:<15} {psnr:<10} {ssim:<10} {lpips:<10} {time_str:<15}")
        else:
            method = result['model']
            status = result['status']
            time_str = f"{result['training_time']/60:.1f} mins" if 'training_time' in result else 'N/A'
            print(f"{method:<15} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {time_str:<15}")
    
    print("="*80)

def save_results(results, output_file):
    """
    Lưu kết quả ra file JSON
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare different NeRF methods')
    parser.add_argument('--dataset', type=str, default='./data/nerf_llff_data/horns',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                        help='Output directory for results')
    parser.add_argument('--train_frames', nargs='+', type=int, default=[20, 42],
                        help='Training frame numbers')
    parser.add_argument('--test_frames', nargs='+', type=int, default=[0, 8, 16, 24, 32, 40, 48, 56],
                        help='Test frame numbers')
    parser.add_argument('--models', nargs='+', default=['FrugalNeRF'],
                        choices=['FrugalNeRF', 'TensoRF', 'SparseNeRF'],
                        help='Models to compare')
    
    args = parser.parse_args()
    
    # Tạo output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔬 NEURAL RADIANCE FIELD COMPARISON")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")
    print(f"Train frames: {args.train_frames}")
    print(f"Test frames: {args.test_frames}")
    
    results = []
    
    # Định nghĩa config cho từng model (sử dụng config RAM thấp)
    model_configs = {
        'FrugalNeRF': 'configs/llff_ultra_low_ram_2v.txt',
        'TensoRF': 'configs/tensorf_baseline_2v.txt',
        'SparseNeRF': 'configs/sparsenerf_2v.txt'
    }
    
    # Chạy training và evaluation cho từng model
    for model_name in args.models:
        if model_name not in model_configs:
            print(f"⚠️  Config not found for {model_name}, skipping...")
            continue
            
        config_path = model_configs[model_name]
        if not os.path.exists(config_path):
            print(f"⚠️  Config file {config_path} not found, skipping {model_name}...")
            continue
        
        # Training
        result = run_training(
            config_path=config_path,
            model_name=model_name,
            dataset_path=args.dataset,
            output_dir=output_dir,
            train_frames=args.train_frames,
            test_frames=args.test_frames
        )
        
        # Evaluation (chỉ nếu training thành công)
        if result['status'] == 'success':
            print(f"\n🔍 Starting evaluation for {model_name}...")
            metrics = evaluate_model(
                model_output_dir=result['output_dir'],
                model_name=model_name,
                dataset_path=args.dataset,
                test_frames=args.test_frames,
                config_path=config_path  # Truyền config path
            )
            if metrics:
                result['metrics'] = metrics
                print(f"✅ Metrics found for {model_name}: {metrics}")
            else:
                print(f"⚠️  No metrics found for {model_name}")
                result['metrics'] = {'psnr': 'N/A', 'ssim': 'N/A', 'lpips': 'N/A'}
        
        results.append(result)
    
    # Tạo bảng so sánh
    create_comparison_table(results)
    
    # Lưu kết quả
    results_file = os.path.join(output_dir, 'comparison_results.json')
    save_results(results, results_file)
    
    print(f"\n🎉 Comparison completed! Check results in: {output_dir}")

if __name__ == "__main__":
    main()
