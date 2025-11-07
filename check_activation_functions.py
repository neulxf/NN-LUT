#!/usr/bin/env python3
"""
检测配置中所有模型使用的激活函数类型
"""
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from models.config import cfgs
import sys

def detect_activation_function(model_name, model_path):
    """检测模型使用的激活函数"""
    try:
        print(f"\n检测模型: {model_name}")
        print(f"  路径: {model_path}")
        
        # 先尝试加载config
        try:
            config = AutoConfig.from_pretrained(model_path)
            print(f"  模型类型: {config.model_type}")
            
            # 检查config中是否有激活函数信息
            if hasattr(config, 'hidden_act'):
                print(f"  配置中的激活函数: {config.hidden_act}")
                return config.hidden_act.lower()
        except Exception as e:
            print(f"  [WARN] 无法加载config: {e}")
        
        # 尝试加载模型的第一层来检测
        try:
            print(f"  尝试加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True
            )
            
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                if len(model.model.layers) > 0:
                    first_layer = model.model.layers[0]
                    if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'act_fn'):
                        act_fn = first_layer.mlp.act_fn
                        act_fn_str = str(type(act_fn)).lower()
                        act_fn_name = type(act_fn).__name__.lower()
                        
                        print(f"  激活函数类型: {type(act_fn).__name__}")
                        print(f"  激活函数字符串: {act_fn_str[:100]}...")
                        
                        if 'gelu' in act_fn_str or 'gelu' in act_fn_name:
                            return 'gelu'
                        elif 'silu' in act_fn_str or 'silu' in act_fn_name or 'swiglu' in act_fn_str:
                            return 'silu'
                        elif 'relu' in act_fn_str:
                            return 'relu'
                        else:
                            return f'unknown({act_fn_name})'
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [ERROR] 无法加载模型: {e}")
            return None
            
    except Exception as e:
        print(f"  [ERROR] 检测失败: {e}")
        return None

def main():
    print("=" * 80)
    print("检测配置中所有模型的激活函数类型")
    print("=" * 80)
    
    results = {
        'gelu': [],
        'silu': [],
        'unknown': [],
        'failed': []
    }
    
    # 检测所有模型
    for model_name, model_cfg in cfgs.items():
        if 'model' not in model_cfg:
            continue
        
        model_path = model_cfg['model']
        act_fn = detect_activation_function(model_name, model_path)
        
        if act_fn:
            if 'gelu' in act_fn.lower():
                results['gelu'].append(model_name)
            elif 'silu' in act_fn.lower():
                results['silu'].append(model_name)
            else:
                results['unknown'].append((model_name, act_fn))
        else:
            results['failed'].append(model_name)
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("检测结果汇总")
    print("=" * 80)
    
    print(f"\n使用 SiLU 的模型 ({len(results['silu'])}):")
    for model in results['silu']:
        print(f"  - {model}")
    
    print(f"\n使用 GELU 的模型 ({len(results['gelu'])}):")
    for model in results['gelu']:
        print(f"  - {model}")
    
    if results['unknown']:
        print(f"\n未知激活函数 ({len(results['unknown'])}):")
        for model, act_fn in results['unknown']:
            print(f"  - {model}: {act_fn}")
    
    if results['failed']:
        print(f"\n检测失败的模型 ({len(results['failed'])}):")
        for model in results['failed']:
            print(f"  - {model}")

if __name__ == "__main__":
    main()


