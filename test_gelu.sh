#!/usr/bin/env bash
# test_gelu.sh - 测试 GELU 激活函数的模型
# 使用 Phi-2 模型（检测到使用 gelu_new）

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/gelu_test_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# LUT 文件路径（使用 H32）
LUT_SILU="nnlut_bench/lut_details_silu_H32_sub.json"
LUT_GELU="nnlut_bench/lut_details_gelu_H32_sub.json"
LUT_EXP="nnlut_bench/lut_details_exp_H32_sub.json"

echo "=========================================="
echo "GELU Activation Function Test"
echo "=========================================="
echo "Model:Gemma-2B (uses gelu_new)"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

MODEL_NAME="Gemma-2B"

# ========== Baseline ==========
echo "[Test 1/3] Baseline Evaluation"
echo "----------------------------------------"
python eval.py --model_name "$MODEL_NAME" --batch_size 1 2>&1 | tee "$RESULTS_DIR/baseline.log"
echo ""

# ========== NN-LUT with GELU ==========
echo "[Test 2/3] NN-LUT Evaluation (with GELU LUT)"
echo "----------------------------------------"
python eval.py --model_name "$MODEL_NAME" --use_nnlut \
    --lut_silu_path "$LUT_SILU" \
    --lut_gelu_path "$LUT_GELU" \
    --lut_exp_path "$LUT_EXP" \
    --batch_size 1 2>&1 | tee "$RESULTS_DIR/nnlut_gelu.log"
echo ""

# ========== FP-LUT with GELU ==========
echo "[Test 3/3] FP-LUT Evaluation (auto-detects GELU)"
echo "----------------------------------------"
python eval.py --model_name "$MODEL_NAME" --use_fplut --batch_size 1 2>&1 | tee "$RESULTS_DIR/fplut_gelu.log"
echo ""

echo "=========================================="
echo "GELU Test Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files:"
ls -lh "$RESULTS_DIR"/*.log | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Check the logs to verify:"
echo "  1. Detection message: '[INFO] Detected activation function: GELU'"
echo "  2. LUT usage: '[INFO] Using GELU LUT: ...'"
echo "  3. Evaluation results comparison"


