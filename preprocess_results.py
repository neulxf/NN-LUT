#!/usr/bin/env python3
"""
预处理实验结果脚本
- 对于 ppl 开头的文件夹：提取 wiki ppl 值
- 对于 eval 开头的文件夹：提取评估表格数据
- 每个子文件夹生成一个 CSV 文件
"""

import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple

def extract_ppl_value(log_file: Path) -> float:
    """从 log 文件中提取最后的 wiki ppl 值"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找最后的 wiki ppl 行
    pattern = r'wiki ppl\s*:\s*([\d.]+)'
    matches = re.findall(pattern, content)
    
    if matches:
        return float(matches[-1])  # 返回最后一个匹配的值
    else:
        return None

def extract_eval_table(log_file: Path) -> List[Dict[str, str]]:
    """从 log 文件中提取评估表格数据"""
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到表格开始的行
    table_start = None
    for i, line in enumerate(lines):
        if '|' in line and 'Tasks' in line and 'Version' in line and 'Metric' in line:
            table_start = i
            break
    
    if table_start is None:
        return []
    
    # 解析表格数据
    results = []
    current_task = None
    
    # 从表头后两行开始（跳过分隔线）
    for i in range(table_start + 2, len(lines)):
        line = lines[i].strip()
        
        # 如果遇到空行或非表格行，停止
        if not line or not line.startswith('|'):
            break
        
        # 解析表格行
        # 格式: |task_name|version|filter|n-shot|metric|direction|value|±|stderr|
        parts = [p.strip() for p in line.split('|')[1:-1]]  # 去掉首尾空元素
        
        if len(parts) < 8:
            continue
        
        task = parts[0].strip()
        if task:  # 如果任务名称不为空，更新当前任务
            current_task = task
        
        version = parts[1].strip()
        filter_type = parts[2].strip()
        n_shot = parts[3].strip()
        metric = parts[4].strip()
        direction = parts[5].strip()
        value = parts[6].strip()
        stderr = parts[8].strip() if len(parts) > 8 else ''
        
        # 如果当前任务为空，使用上一个任务
        if not current_task:
            continue
        
        results.append({
            'task': current_task,
            'version': version,
            'filter': filter_type,
            'n_shot': n_shot,
            'metric': metric,
            'direction': direction,
            'value': value,
            'stderr': stderr
        })
    
    return results

def process_ppl_folder(folder_path: Path, output_dir: Path):
    """处理 ppl 文件夹，提取 wiki ppl 值"""
    results = []
    
    # 遍历所有 log 文件
    for log_file in sorted(folder_path.glob('*.log')):
        model_name = log_file.stem  # 文件名（不含扩展名）
        ppl_value = extract_ppl_value(log_file)
        
        if ppl_value is not None:
            results.append({
                'model': model_name,
                'wiki_ppl': ppl_value
            })
        else:
            print(f"Warning: Could not find wiki ppl in {log_file}")
    
    # 写入 CSV 文件
    if results:
        csv_file = output_dir / f"{folder_path.name}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'wiki_ppl'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Created {csv_file} with {len(results)} entries")

def process_eval_folder(folder_path: Path, output_dir: Path):
    """处理 eval 文件夹，提取评估表格数据"""
    all_results = []
    
    # 遍历所有 log 文件
    for log_file in sorted(folder_path.glob('*.log')):
        model_name = log_file.stem  # 文件名（不含扩展名）
        table_data = extract_eval_table(log_file)
        
        for row in table_data:
            row['model'] = model_name
            all_results.append(row)
        
        if not table_data:
            print(f"Warning: Could not find evaluation table in {log_file}")
    
    # 写入 CSV 文件
    if all_results:
        csv_file = output_dir / f"{folder_path.name}.csv"
        fieldnames = ['model', 'task', 'version', 'filter', 'n_shot', 'metric', 'direction', 'value', 'stderr']
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Created {csv_file} with {len(all_results)} entries")

def main():
    """主函数"""
    # 设置路径
    results_dir = Path('/home/lxf/workspace/NN-LUT/results')
    output_dir = Path('/home/lxf/workspace/NN-LUT/results_processed')
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 遍历 results 目录下的所有子文件夹
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # 处理 ppl 开头的文件夹
        if folder_name.startswith('ppl'):
            print(f"Processing PPL folder: {folder_name}")
            process_ppl_folder(folder, output_dir)
        
        # 处理 eval 开头的文件夹
        elif folder_name.startswith('eval'):
            print(f"Processing EVAL folder: {folder_name}")
            process_eval_folder(folder, output_dir)
        
        else:
            print(f"Skipping folder: {folder_name} (not ppl or eval)")

if __name__ == '__main__':
    main()