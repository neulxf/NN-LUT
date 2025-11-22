#!/usr/bin/env bash


export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/eval_extra3_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"


LUT_SILU="nnlut_bench_ori/lut_details_silu_H16_sub.json"
LUT_GELU="nnlut_bench_ori/lut_details_gelu_H16_sub.json"
LUT_EXP="nnlut_bench_ori/lut_details_exp_H16_sub.json"

# Qwen2.5-1.5B-Instruct
echo "  -> Qwen2.5-1.5B-Instruct - NN-LUT"
python eval.py --model_name Qwen2.5-1.5B-Instruct --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/qwen2.5_1.5b_nnlut.log"

# Qwen2.5-3B-Instruct
echo "  -> Qwen2.5-3B-Instruct - NN-LUT"
python eval.py --model_name Qwen2.5-3B-Instruct --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/qwen2.5_3b_nnlut.log"


LUT_SILU="nnlut_bench_ori/lut_details_silu_H16_sub.json"
LUT_GELU="nnlut_bench_ori/lut_details_gelu_H16_sub.json"
LUT_EXP="nnlut_bench_ori/lut_details_exp_H16_sub.json"


# SmolLM-3B
echo "  -> SmolLM-3B - NN-LUT"
python eval.py --model_name SmolLM-3B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/smollm_3b_nnlut.log"

