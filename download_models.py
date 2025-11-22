#!/usr/bin/env python3
"""
批量下载 eval_ppl_comparison.sh 中使用的所有模型
使用 huggingface_hub 直接下载文件，不加载模型到内存
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download, scan_cache_dir

# 导入模型配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.config import cfgs

# 从 eval_ppl_comparison.sh 中提取的模型列表（未注释的模型）
MODELS_TO_DOWNLOAD = [
    # Group 1: Ultra-small models (<1B)
    "Qwen2.5-0.5B-Instruct",
    "SmolLM-135M",
    "SmolLM-360M",
    
    # Group 2: Small models (1B-2B)
    "Llama-3.2-1B",
    "Qwen2.5-1.5B-Instruct",
    "TinyLlama-1.1B",
    "SmolLM-1.7B",
    "Gemma-2B",
    "Gemma-2B-Instruct",
    
    # Group 3: Medium models (2B-4B)
    "Qwen2.5-3B-Instruct",
    "Llama-3.2-3B",
    "SmolLM-3B",
]


def check_model_cached(model_path: str) -> bool:
    """检查模型是否已经在 HuggingFace cache 中"""
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            # 检查 repo_id 是否匹配
            if model_path == repo.repo_id:
                return True
            # 检查是否是本地路径且存在
            if os.path.exists(model_path) and os.path.isdir(model_path):
                if os.path.exists(os.path.join(model_path, "config.json")):
                    return True
    except Exception as e:
        # 如果扫描失败，尝试直接检查本地路径
        if os.path.exists(model_path) and os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "config.json")):
                return True
    return False


def download_model(model_name: str, model_path: str):
    """下载单个模型（使用 snapshot_download，不加载到内存）"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"{'='*60}")
    
    # 检查是否已缓存
    if check_model_cached(model_path):
        print(f"  ✓ Model already cached, skipping...")
        return True
    
    try:
        # 如果是本地路径，跳过
        if os.path.exists(model_path) and os.path.isdir(model_path):
            print(f"  ✓ Model exists locally, skipping...")
            return True
        
        # 使用 snapshot_download 下载所有文件
        print(f"  → Downloading model files (this may take a while)...")
        snapshot_download(
            repo_id=model_path,
            local_files_only=False,
            resume_download=True,
        )
        print(f"  ✓ Model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("="*60)
    print("Batch Model Download Script")
    print("="*60)
    print(f"Total models to download: {len(MODELS_TO_DOWNLOAD)}")
    print("This script will download models to HuggingFace cache")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for model_name in MODELS_TO_DOWNLOAD:
        if model_name not in cfgs:
            print(f"\n✗ Model '{model_name}' not found in config.py, skipping...")
            fail_count += 1
            continue
        
        model_cfg = cfgs[model_name]
        model_path = model_cfg["model"]
        
        # 检查是否已缓存
        if check_model_cached(model_path):
            skipped_count += 1
            print(f"\n  ⊘ {model_name} already cached, skipping...")
            continue
        
        # 下载模型
        if download_model(model_name, model_path):
            success_count += 1
        else:
            fail_count += 1
    
    # 打印总结
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Already cached (skipped): {skipped_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(MODELS_TO_DOWNLOAD)}")
    print("="*60)
    print("\nNote: Models are cached in HuggingFace cache directory.")
    print("You can check cache location with: huggingface-cli scan-cache")


if __name__ == "__main__":
    main()
