import pandas as pd
import os
from pathlib import Path

def process_eval_csv(input_file, output_file, exclude_fplut=True):
    """
    处理评估CSV文件，将相同model的条目合并成一行，task+metric作为列
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        exclude_fplut: 是否排除以_fplut结尾的模型，默认为True
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 如果启用，排除以_fplut结尾的模型
    if exclude_fplut:
        original_count = len(df)
        df = df[~df['model'].str.endswith('_fplut', na=False)]
        excluded_count = original_count - len(df)
        if excluded_count > 0:
            print(f"    排除 {excluded_count} 行以_fplut结尾的模型数据")
    
    # 移除不需要的列
    df_clean = df.drop(columns=['version', 'filter', 'n_shot', 'direction'], errors='ignore')
    
    # 创建透视表：model作为行，task+metric作为列
    # 使用 task_metric 作为列名
    df_clean['task_metric'] = df_clean['task'] + '_' + df_clean['metric']
    
    # 创建透视表
    pivot_df = df_clean.pivot_table(
        index='model',
        columns='task_metric',
        values='value',
        aggfunc='first'  # 如果有重复值，取第一个
    )
    
    # 重置索引，使model成为普通列
    pivot_df = pivot_df.reset_index()
    
    # 保存结果
    pivot_df.to_csv(output_file, index=False)
    
    return pivot_df

def batch_process_eval_files(input_dir, output_dir=None, exclude_fplut=True):
    """
    批量处理文件夹下所有以eval开头的CSV文件
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径（如果为None，则在input_dir下创建processed文件夹）
        exclude_fplut: 是否排除以_fplut结尾的模型，默认为True
    """
    input_path = Path(input_dir)
    
    # 确定输出文件夹
    if output_dir is None:
        output_path = input_path / 'eval_processed'
    else:
        output_path = Path(output_dir)
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有以eval开头的CSV文件
    eval_files = list(input_path.glob('eval*.csv'))
    
    if not eval_files:
        print(f"在 {input_dir} 中未找到以eval开头的CSV文件")
        return
    
    print(f"找到 {len(eval_files)} 个文件需要处理")
    print(f"输出目录: {output_path}")
    if exclude_fplut:
        print(f"将排除以_fplut结尾的模型")
    print()
    
    processed_count = 0
    failed_files = []
    
    for input_file in eval_files:
        try:
            # 生成输出文件名（保持原文件名，添加_processed后缀）
            output_file = output_path / f"{input_file.stem}_processed.csv"
            
            print(f"处理: {input_file.name} -> {output_file.name}")
            
            # 处理文件
            result_df = process_eval_csv(input_file, output_file, exclude_fplut=exclude_fplut)
            
            print(f"  ✓ 成功: {len(result_df)} 个模型, {len(result_df.columns) - 1} 个任务指标列")
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            failed_files.append(input_file.name)
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count}/{len(eval_files)} 个文件")
    
    if failed_files:
        print(f"失败的文件:")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == "__main__":
    import sys
    
    # 默认输入目录
    input_dir = "/home/lxf/workspace/NN-LUT/results_processed"
    
    # 可以从命令行参数获取
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # 可以通过命令行参数控制是否排除fplut模型
    exclude_fplut = True
    if len(sys.argv) > 3:
        exclude_fplut = sys.argv[3].lower() in ['true', '1', 'yes']
    
    batch_process_eval_files(input_dir, output_dir, exclude_fplut=exclude_fplut)