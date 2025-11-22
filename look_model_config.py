from transformers import AutoConfig
import pandas as pd
from tqdm import tqdm

# 定义要分析的模型列表
MODELS_TO_ANALYZE = [
    # Qwen系列
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B", 
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-Math-7B",
	"Qwen/Qwen2.5-3B-Instruct",
	"Qwen/Qwen2.5-3B",
    
    # LLaMA系列
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-3-8b",
    "meta-llama/Llama-3-70b",
	"meta-llama/Llama-3.2-3B",
	"meta-llama/Llama-3.2-1B",
	
    
    # GPT系列
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "openai-community/gpt2",
    
    # BERT系列
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased", 
    "bert-large-cased",
    
    # Gemma系列
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    
    # SmolLM系列（选择几个代表性模型）
    "HuggingFaceTB/SmolLM-135M",
    "HuggingFaceTB/SmolLM-360M", 
    "HuggingFaceTB/SmolLM-1.7B",
    "HuggingFaceTB/SmolLM3-3B",
    
    # TinyLLaMA系列
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
	"TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",

	# OLMo系列
	"allenai/OLMo-1B",
	"allenai/OLMo-1B-7B",

	# OpenELM系列
	"apple/OpenELM-1_1B",
	"apple/OpenELM-3B",

	# Pythia系列
	"EleutherAI/pythia-14m",
	"EleutherAI/pythia-70m",
	"EleutherAI/pythia-160m",
	"EleutherAI/pythia-410m",
	"EleutherAI/pythia-1b",
	"EleutherAI/pythia-1.4b",
	"EleutherAI/pythia-2.8b",

	# RedPajama系列
	"togethercomputer/RedPajama-INCITE-Base-3B-v1",

	# Phi-2系列
	"microsoft/phi-2",

	"facebook/opt-125m",
	"facebook/opt-350m",
	"facebook/opt-1.3b",
	"facebook/opt-2.7b",
	"facebook/opt-6.7b",
	"facebook/opt-13b",
	"facebook/opt-30b",
	"facebook/opt-66b",
	"facebook/opt-175b",
	"facebook/opt-330b",
	"facebook/opt-660b",
	"facebook/opt-1320b",
]

def get_model_info(model_name):
    """获取单个模型的配置信息"""
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        info = {
            'model_name': model_name,
            'num_hidden_layers': getattr(config, 'num_hidden_layers', 'N/A'),
            'hidden_size': getattr(config, 'hidden_size', 'N/A'),
            'num_attention_heads': getattr(config, 'num_attention_heads', 'N/A'),
            'intermediate_size': getattr(config, 'intermediate_size', 'N/A'),
            'vocab_size': getattr(config, 'vocab_size', 'N/A'),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', 'N/A'),
            'architecture': getattr(config, 'architectures', ['N/A'])[0] if getattr(config, 'architectures', None) else 'N/A'
        }
        
        # 对于 GPT-2 和 OPT 模型，如果 intermediate_size 不存在，使用默认比例 4x
        if info['intermediate_size'] == 'N/A' and info['hidden_size'] != 'N/A':
            architecture = info['architecture'].lower()
            if 'gpt2' in architecture or 'opt' in architecture:
                # GPT-2 和 OPT 模型通常使用 4x hidden_size 作为 FFN 维度
                info['intermediate_size'] = info['hidden_size'] * 4
                info['ffn_hidden_ratio'] = 4.0
            else:
                # 尝试其他可能的属性名
                info['intermediate_size'] = getattr(config, 'ffn_dim', 
                                                   getattr(config, 'ffn_hidden_size', 
                                                          getattr(config, 'd_ff', 'N/A')))
                if info['intermediate_size'] != 'N/A' and info['hidden_size'] != 'N/A':
                    info['ffn_hidden_ratio'] = round(info['intermediate_size'] / info['hidden_size'], 4)
                else:
                    info['ffn_hidden_ratio'] = 'N/A'
        # 计算FFN与隐藏维度的比例（如果还没有计算）
        elif info['intermediate_size'] != 'N/A' and info['hidden_size'] != 'N/A':
            info['ffn_hidden_ratio'] = round(info['intermediate_size'] / info['hidden_size'], 4)
        else:
            info['ffn_hidden_ratio'] = 'N/A'
            
        # 计算每个注意力头的维度
        if info['hidden_size'] != 'N/A' and info['num_attention_heads'] != 'N/A':
            info['head_dim'] = info['hidden_size'] // info['num_attention_heads']
        else:
            info['head_dim'] = 'N/A'
            
        return info
        
    except Exception as e:
        print(f"❌ 获取模型 {model_name} 信息失败: {e}")
        return {
            'model_name': model_name,
            'num_hidden_layers': 'Error',
            'hidden_size': 'Error', 
            'num_attention_heads': 'Error',
            'intermediate_size': 'Error',
            'vocab_size': 'Error',
            'max_position_embeddings': 'Error',
            'architecture': 'Error',
            'ffn_hidden_ratio': 'Error',
            'head_dim': 'Error'
        }

def analyze_models():
    """批量分析所有模型"""
    print("🚀 开始批量获取模型信息...")
    print("=" * 80)
    
    results = []
    
    # 使用进度条显示进度
    for model_name in tqdm(MODELS_TO_ANALYZE, desc="分析模型中"):
        model_info = get_model_info(model_name)
        results.append(model_info)
    
    # 转换为DataFrame便于分析
    df = pd.DataFrame(results)
    
    # 保存结果到CSV文件
    df.to_csv('model_analysis.csv', index=False)
    print(f"✅ 结果已保存到 model_analysis.csv")
    
    return df

def print_detailed_comparison(df):
    """打印详细的比较结果"""
    print("\n" + "="*100)
    print("📊 模型结构参数详细比较")
    print("="*100)
    
    # 按模型系列分组显示
    series_groups = {
        'Qwen': [m for m in MODELS_TO_ANALYZE if 'qwen' in m.lower()],
        'LLaMA': [m for m in MODELS_TO_ANALYZE if 'llama' in m.lower()],
        'GPT': [m for m in MODELS_TO_ANALYZE if 'gpt' in m.lower()],
        'BERT': [m for m in MODELS_TO_ANALYZE if 'bert' in m.lower()],
        'Gemma': [m for m in MODELS_TO_ANALYZE if 'gemma' in m.lower()],
        'SmolLM': [m for m in MODELS_TO_ANALYZE if 'smol' in m.lower()],
        'TinyLLaMA': [m for m in MODELS_TO_ANALYZE if 'tiny' in m.lower()]
    }
    
    for series_name, series_models in series_groups.items():
        if not series_models:
            continue
            
        print(f"\n🔍 {series_name} 系列:")
        print("-" * 80)
        
        series_df = df[df['model_name'].isin(series_models)]
        
        for _, row in series_df.iterrows():
            if row['num_hidden_layers'] == 'Error':
                print(f"   {row['model_name']}: 获取失败")
                continue
                
            print(f"   📁 {row['model_name']}")
            print(f"      层数: {row['num_hidden_layers']:>3} | "
                  f"隐藏维度: {row['hidden_size']:>5} | "
                  f"注意力头: {row['num_attention_heads']:>3} | "
                  f"头维度: {row['head_dim']:>3}")
            print(f"      FFN维度: {row['intermediate_size']:>6} | "
                  f"FFN/隐藏比例: {row['ffn_hidden_ratio']:>5} | "
                  f"词表大小: {row['vocab_size']:>6}")
            print(f"      最大长度: {row['max_position_embeddings']:>5} | "
                  f"架构: {row['architecture']}")

def print_summary_statistics(df):
    """打印统计摘要"""
    print("\n" + "="*100)
    print("📈 统计摘要")
    print("="*100)
    
    # 过滤掉错误的数据
    valid_df = df[df['num_hidden_layers'] != 'Error']
    
    if len(valid_df) == 0:
        print("❌ 没有有效数据可分析")
        return
    
    # 转换为数值类型
    numeric_cols = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 
                    'intermediate_size', 'vocab_size', 'max_position_embeddings', 'ffn_hidden_ratio']
    
    for col in numeric_cols:
        valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')
    
    stats = valid_df.describe()
    print(stats.round(2))

# 主执行函数
if __name__ == "__main__":
    # 执行分析
    df = analyze_models()
    
    # 显示详细比较
    print_detailed_comparison(df)
    
    # 显示统计摘要
    print_summary_statistics(df)
    
    # 保存美化后的文本报告
    with open('model_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("模型结构参数分析报告\n")
        f.write("="*50 + "\n")
        for _, row in df.iterrows():
            if row['num_hidden_layers'] != 'Error':
                f.write(f"\n模型: {row['model_name']}\n")
                f.write(f"  层数: {row['num_hidden_layers']}\n")
                f.write(f"  隐藏维度: {row['hidden_size']}\n")
                f.write(f"  注意力头数: {row['num_attention_heads']}\n")
                f.write(f"  FFN维度: {row['intermediate_size']}\n")
                f.write(f"  FFN/隐藏比例: {row['ffn_hidden_ratio']}\n")
                f.write(f"  词表大小: {row['vocab_size']}\n")
                f.write(f"  最大长度: {row['max_position_embeddings']}\n")
    
    print(f"\n✅ 详细报告已保存到 model_analysis_report.txt")