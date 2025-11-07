#!/usr/bin/env bash
# simple_comparison.sh - 简单的三组对比测试
# 1. Baseline (默认)
# 2. NN-LUT
# 3. FP-LUT

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/comparison_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# LUT 文件路径（使用 H32）
LUT_SILU="nnlut_bench/lut_details_silu_H32_sub.json"
LUT_EXP="nnlut_bench/lut_details_exp_H32_sub.json"
LUT_GELU="nnlut_bench/lut_details_gelu_H32_sub.json"

echo "=========================================="
echo "Simple Comparison Test"
echo "=========================================="
echo "Model: Qwen2.5-0.5B-Instruct"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

MODEL_NAME="Qwen2.5-0.5B-Instruct"

# ========== 1. Baseline (默认) ==========
echo "[1/3] Baseline Evaluation"
echo "----------------------------------------"
python eval.py --model_name "$MODEL_NAME" --batch_size 1 2>&1 | tee "$RESULTS_DIR/baseline.log"
echo ""

# ========== 2. NN-LUT ==========
echo "[2/3] NN-LUT Evaluation"
echo "----------------------------------------"
python eval.py --model_name "$MODEL_NAME" --use_nnlut \
    --lut_silu_path "$LUT_SILU" \
    --lut_exp_path "$LUT_EXP" \
    --batch_size 1 2>&1 | tee "$RESULTS_DIR/nnlut.log"
echo ""

# ========== 3. FP-LUT ==========
echo "[3/3] FP-LUT Evaluation"
echo "----------------------------------------"
python eval.py --model_name "$MODEL_NAME" --use_fplut --batch_size 1 2>&1 | tee "$RESULTS_DIR/fplut.log"
echo ""

echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files:"
ls -lh "$RESULTS_DIR"/*.log | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "To view results:"
echo "  Baseline:  cat $RESULTS_DIR/baseline.log"
echo "  NN-LUT:    cat $RESULTS_DIR/nnlut.log"
echo "  FP-LUT:    cat $RESULTS_DIR/fplut.log"


