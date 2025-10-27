# fp16lut
fp16lut code

## 环境配置
```shell
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .

    conda install nvidia/label/cuda-12.4.0::cuda
    export CUDA_HOME=$CONDA_PREFIX
    MAX_JOBS=40 pip install flash-attn --no-build-isolation
```

## Eval Zero short
### 1. 原始模型
```shell
    export CUDA_VISIBLE_DEVICES=0
    python eval.py --model weights/Qwen2.5-7B-Instruct
    python eval.py --model weights/Qwen2.5-32B-Instruct
    python eval.py --model weights/llama3-8b-hf
    export CUDA_VISIBLE_DEVICES=0,1
    python eval.py --model weights/llama3-70b-hf
```

### 2. lut模型
```shell
    export CUDA_VISIBLE_DEVICES=0
    python eval.py --model_name Qwen2.5-7B-Instruct-fp16 --use_fplut
    python eval.py --model_name Qwen2.5-32B-Instruct-fp16 --use_fplut
    python eval.py --model llama3-8b-hf-fp16 --use_fplut
    export CUDA_VISIBLE_DEVICES=0,1
    python eval.py --model_name llama3-70b-hf-fp16 --use_fplut
```

## Eval PPL
### 1. 原始模型
```shell
    export CUDA_VISIBLE_DEVICES=5
    python eval_ppl.py --model_name Qwen2.5-7B-Instruct # 需要fp32才能跑出7.457的效果
    python eval_ppl.py --model_name Qwen2.5-32B-Instruct
```

### 2. lut模型
```shell
    export CUDA_VISIBLE_DEVICES=5
    python eval_ppl.py --model_name Qwen2.5-7B-Instruct-fp16 --use_fplut
    python eval_ppl.py --model_name Qwen2.5-32B-Instruct-fp16 --use_fplut
```

### 3. nnlut模型
#### 1. ppl
```shell
 python eval_ppl.py --model_name llama3-70b-hf --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 

  python eval_ppl.py --model_name llama3-8b-hf --use_nnlut --lut_silu_path nnlut_bench_ori/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench_ori/lut_details_exp_H256_sub.json 

 python eval_ppl.py --model_name llama3-70b-hf --use_nnlut --lut_silu_path nnlut_bench_ori/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench_ori/lut_details_exp_H256_sub.json 

 python eval_ppl.py --model_name Qwen2.5-7B-Instruct --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 

 python eval_ppl.py --model_name Qwen2.5-32B-Instruct --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H16_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H16_sub.json 


 python eval_ppl.py --model_name Qwen2.5-32B-Instruct --use_nnlut --lut_silu_path nnlut_bench_ori/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench_ori/lut_details_exp_H256_sub.json 

 python eval_ppl.py --model_name Qwen1.5-110B --use_nnlut --lut_silu_path nnlut_bench_ori/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench_ori/lut_details_exp_H256_sub.json 

```
#### 2. eval zero short
```shell
 python eval.py --model_name llama3-8b-hf --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 
python eval.py --model_name llama3-8b-hf --use_nnlut --lut_silu_path nnlut_bench_ori/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench_ori/lut_details_exp_H256_sub.json 
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2
python eval.py --model_name llama3-70b-hf --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
python eval.py --model_name Qwen2.5-7B-Instruct --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
python eval.py --model_name Qwen2.5-32B-Instruct --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 
```


### 统计
| **Model\Act** | **Silu** | **Nexp(Softmax)** | **Reciple** | **Rsqrt** |
|:-------------:|:--------:|:-----------------:|:-----------:|:---------:|
| qwen2.5 7B    | -77 80   |                   |             |           |
| qwen2.5 32B   | -73 130  |                   |             |           |
| qwen1.5 110B  |          |                   |             |           |
| qwen3 8B      |   -141 106       |                   |             |           |
| qwen3 30B-A3B     | -38 56.75          |                   |             |           |
| llama3 8B     | -38.75 26.125 |                   |             |           |
| llama3 70B    |   -38.25 27.45        |                   |             |           |