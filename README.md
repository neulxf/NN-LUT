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

python eval.py --model_name Qwen2.5-7B-Instruct-bf16 --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H32_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H32_sub.json --lut_gelu_path nnlut_bench/lut_details_gelu_H32_sub.json


result下的文件夹，如果表明 ori 的是用的 nnlut 的文件不同，表明 flut16 的是将 flut 从默认的 32entry改为了 16 entry

如果显卡内存太小，可以设置--seqlen 512这样可以减少显存占用，但是会影响精度，需要全部使用相同参数对比较好

eval_开头的脚本是批量执行实验的，ppl 是测量困惑度的，不带 ppl 的测量的是几个任务的精度度一类的信息，ppl 测量相对快些，但是几个任务的精确的全部测量耗时得一到两天, eval_ppl_comparison16.sh 和eval_comparison_flut16_ori.sh 用的是nnlut_branch 下的nnlut 的 16 版本；eval_ppl_comparison_ori2.sh 和 eval_comparison_ori2.sh用的是nnlut_branch_ori2 下的nnlut 的 16 版本，eval_ppl_comparison_flut16_ori.sh 和 eval_comparison_flut16_ori.sh用的是nnlut_branch_ori 下的nnlut 的 16 版本,同时需要将NN-LUT/models/fp_lut.py中的LUT 数量大概为 16（应该配置的Nnum_points=2，默认为 4）

最终的实验结果用的是基本是 nnlut_branch下的NN-LUT32配置，主要内容在results_processed_bk_new里，或者可以看论文实验用的 origin 文件中的数据，部分几个测试用的是 nnlut_branch_ori 和nnlut_branch_ori2下面的 NNLUT16配置代替的，

具体替换的数据来自这里：
对于 nnlut 来说，ori2以及 ori 中的16 位的测试，在下面相对原有的 32 位出现了不可预知的问题
qwen2.5_1.5b_nnlut,20331.947265625*
qwen2.5_3b_nnlut,3697547.75*
对于 ori还额外出现了
smollm_3b_nnlut,255907.390625*
smollm_360m_nnlut,275.5361022949219


新增数据来自 ori2 的结果：原来的对应结果可能因为 sequence 问题不一样，还是以 ori2 里的 baseline 为准

llama3_8b 6.135672092437744 6.174114227294922
qwen2.5_7b 7.457342624664307 389825.71875
qwen3_8b 9.715299606323242  700295.75

后来跑了几个小测试在，以随机数字为后缀的，主要是因为之前有的大模型的测试用的 sequence 大小不是 2048，进而补充的几个实验结果，最终我还有一个处理完的最终结果及包括更新必要数据为 ori和 ori2 中的结果，放在了画图的电脑的 Excel中，后面可以拷贝过来

eval_extra.sh的实验结果原因是因为我们部分数据用的 ori 中的结果，因此单独运行获得的对应的更完善的结果

preprocess_results.py用于将results 中的 log 文件，提取关键结果并保存到results_processed中

flaten_result.py用于将results_processed的 eval 结果进行平铺，因为之前一个模型几个下游任务是纵向排列的，这不是进行横向排列后对比，因此是用该脚本对齐格式进行进一步排版优化

look_model_config.py用于获得模型关键参数信息，比如维度

download_models.py主要下载几个大模型到本地

 总体上来说精度 flut 精度更高，误差不大，而 nnlut 的不确定很高，而且 nnlut_branch下的效果更好，其中的16 甚至比 32 还好有时

 