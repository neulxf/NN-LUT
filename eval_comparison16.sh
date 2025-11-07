#!/usr/bin/env bash
# eval_comparison.sh - 对比原始模型、FPLUT 和 NN-LUT 版本的 PPL 评估
# 包含所有小于4B的模型，统一使用 H32 LUT，并保存结果

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/eval16_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# LUT 文件路径（统一使用 H32）
LUT_SILU="nnlut_bench/lut_details_silu_H16_sub.json"
LUT_GELU="nnlut_bench/lut_details_gelu_H16_sub.json"
LUT_EXP="nnlut_bench/lut_details_exp_H16_sub.json"

echo "=========================================="
echo "PPL Evaluation Comparison"
echo "=========================================="
echo "All models use H32 LUT configuration"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ========== 超小模型 (<1B) ==========
echo "[Group 1] Ultra-small models (<1B) - SeqLen=default"

# Qwen2.5-0.5B-Instruct
echo "  -> Qwen2.5-0.5B-Instruct - Baseline"
python eval.py --model_name Qwen2.5-0.5B-Instruct  2>&1 | tee "$RESULTS_DIR/qwen2.5_0.5b_baseline.log"

echo "  -> Qwen2.5-0.5B-Instruct - FPLUT"
python eval.py --model_name Qwen2.5-0.5B-Instruct --use_fplut  2>&1 | tee "$RESULTS_DIR/qwen2.5_0.5b_fplut.log"

echo "  -> Qwen2.5-0.5B-Instruct - NN-LUT"
python eval.py --model_name Qwen2.5-0.5B-Instruct --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/qwen2.5_0.5b_nnlut.log"

# SmolLM-135M
echo "  -> SmolLM-135M - Baseline"
python eval.py --model_name SmolLM-135M  2>&1 | tee "$RESULTS_DIR/smollm_135m_baseline.log"

echo "  -> SmolLM-135M - FPLUT"
python eval.py --model_name SmolLM-135M --use_fplut  2>&1 | tee "$RESULTS_DIR/smollm_135m_fplut.log"

echo "  -> SmolLM-135M - NN-LUT"
python eval.py --model_name SmolLM-135M --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/smollm_135m_nnlut.log"

# SmolLM-360M
echo "  -> SmolLM-360M - Baseline"
python eval.py --model_name SmolLM-360M  2>&1 | tee "$RESULTS_DIR/smollm_360m_baseline.log"

echo "  -> SmolLM-360M - FPLUT"
python eval.py --model_name SmolLM-360M --use_fplut  2>&1 | tee "$RESULTS_DIR/smollm_360m_fplut.log"

echo "  -> SmolLM-360M - NN-LUT"
python eval.py --model_name SmolLM-360M --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/smollm_360m_nnlut.log"

# # Pythia-14M
# echo "  -> Pythia-14M - Baseline"
# python eval.py --model_name Pythia-14M  2>&1 | tee "$RESULTS_DIR/pythia_14m_baseline.log"
# echo "  -> Pythia-14M - FPLUT"
# python eval.py --model_name Pythia-14M --use_fplut  2>&1 | tee "$RESULTS_DIR/pythia_14m_fplut.log"
# echo "  -> Pythia-14M - NN-LUT"
# python eval.py --model_name Pythia-14M --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/pythia_14m_nnlut.log"

# # Pythia-70M
# echo "  -> Pythia-70M - Baseline"
# python eval.py --model_name Pythia-70M  2>&1 | tee "$RESULTS_DIR/pythia_70m_baseline.log"
# echo "  -> Pythia-70M - FPLUT"
# python eval.py --model_name Pythia-70M --use_fplut  2>&1 | tee "$RESULTS_DIR/pythia_70m_fplut.log"
# echo "  -> Pythia-70M - NN-LUT"
# python eval.py --model_name Pythia-70M --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/pythia_70m_nnlut.log"

# # Pythia-160M
# echo "  -> Pythia-160M - Baseline"
# python eval.py --model_name Pythia-160M  2>&1 | tee "$RESULTS_DIR/pythia_160m_baseline.log"
# echo "  -> Pythia-160M - FPLUT"
# python eval.py --model_name Pythia-160M --use_fplut  2>&1 | tee "$RESULTS_DIR/pythia_160m_fplut.log"
# echo "  -> Pythia-160M - NN-LUT"
# python eval.py --model_name Pythia-160M --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/pythia_160m_nnlut.log"

# # Pythia-410M
# echo "  -> Pythia-410M - Baseline"
# python eval.py --model_name Pythia-410M  2>&1 | tee "$RESULTS_DIR/pythia_410m_baseline.log"
# echo "  -> Pythia-410M - FPLUT"
# python eval.py --model_name Pythia-410M --use_fplut  2>&1 | tee "$RESULTS_DIR/pythia_410m_fplut.log"
# echo "  -> Pythia-410M - NN-LUT"
# python eval.py --model_name Pythia-410M --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/pythia_410m_nnlut.log"

# ========== 小模型 (1B-2B) ==========
echo ""
echo "[Group 2] Small models (1B-2B) - SeqLen=default"

# Llama-3.2-1B
echo "  -> Llama-3.2-1B - Baseline"
python eval.py --model_name Llama-3.2-1B  2>&1 | tee "$RESULTS_DIR/llama3.2_1b_baseline.log"

echo "  -> Llama-3.2-1B - FPLUT"
python eval.py --model_name Llama-3.2-1B --use_fplut  2>&1 | tee "$RESULTS_DIR/llama3.2_1b_fplut.log"

echo "  -> Llama-3.2-1B - NN-LUT"
python eval.py --model_name Llama-3.2-1B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/llama3.2_1b_nnlut.log"

# Qwen2.5-1.5B-Instruct
echo "  -> Qwen2.5-1.5B-Instruct - Baseline"
python eval.py --model_name Qwen2.5-1.5B-Instruct  2>&1 | tee "$RESULTS_DIR/qwen2.5_1.5b_baseline.log"

echo "  -> Qwen2.5-1.5B-Instruct - FPLUT"
python eval.py --model_name Qwen2.5-1.5B-Instruct --use_fplut  2>&1 | tee "$RESULTS_DIR/qwen2.5_1.5b_fplut.log"

echo "  -> Qwen2.5-1.5B-Instruct - NN-LUT"
python eval.py --model_name Qwen2.5-1.5B-Instruct --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/qwen2.5_1.5b_nnlut.log"

# TinyLlama-1.1B
echo "  -> TinyLlama-1.1B - Baseline"
python eval.py --model_name TinyLlama-1.1B  2>&1 | tee "$RESULTS_DIR/tinyllama_1.1b_baseline.log"

echo "  -> TinyLlama-1.1B - FPLUT"
python eval.py --model_name TinyLlama-1.1B --use_fplut  2>&1 | tee "$RESULTS_DIR/tinyllama_1.1b_fplut.log"

echo "  -> TinyLlama-1.1B - NN-LUT"
python eval.py --model_name TinyLlama-1.1B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/tinyllama_1.1b_nnlut.log"

# SmolLM-1.7B
echo "  -> SmolLM-1.7B - Baseline"
python eval.py --model_name SmolLM-1.7B  2>&1 | tee "$RESULTS_DIR/smollm_1.7b_baseline.log"

echo "  -> SmolLM-1.7B - FPLUT"
python eval.py --model_name SmolLM-1.7B --use_fplut  2>&1 | tee "$RESULTS_DIR/smollm_1.7b_fplut.log"

echo "  -> SmolLM-1.7B - NN-LUT"
python eval.py --model_name SmolLM-1.7B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/smollm_1.7b_nnlut.log"

# # Pythia-1B
# echo "  -> Pythia-1B - Baseline"
# python eval.py --model_name Pythia-1B  2>&1 | tee "$RESULTS_DIR/pythia_1b_baseline.log"
# echo "  -> Pythia-1B - FPLUT"
# python eval.py --model_name Pythia-1B --use_fplut  2>&1 | tee "$RESULTS_DIR/pythia_1b_fplut.log"
# echo "  -> Pythia-1B - NN-LUT"
# python eval.py --model_name Pythia-1B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/pythia_1b_nnlut.log"

# # Pythia-1.4B
# echo "  -> Pythia-1.4B - Baseline"
# python eval.py --model_name Pythia-1.4B  2>&1 | tee "$RESULTS_DIR/pythia_1.4b_baseline.log"
# echo "  -> Pythia-1.4B - FPLUT"
# python eval.py --model_name Pythia-1.4B --use_fplut  2>&1 | tee "$RESULTS_DIR/pythia_1.4b_fplut.log"
# echo "  -> Pythia-1.4B - NN-LUT"
# python eval.py --model_name Pythia-1.4B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/pythia_1.4b_nnlut.log"

# # OpenELM-1.1B
# echo "  -> OpenELM-1.1B - Baseline"
# python eval.py --model_name OpenELM-1.1B  2>&1 | tee "$RESULTS_DIR/openelm_1.1b_baseline.log"
# echo "  -> OpenELM-1.1B - FPLUT"
# python eval.py --model_name OpenELM-1.1B --use_fplut  2>&1 | tee "$RESULTS_DIR/openelm_1.1b_fplut.log"
# echo "  -> OpenELM-1.1B - NN-LUT"
# python eval.py --model_name OpenELM-1.1B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/openelm_1.1b_nnlut.log"

# # OLMo-1B
# echo "  -> OLMo-1B - Baseline"
# python eval.py --model_name OLMo-1B  2>&1 | tee "$RESULTS_DIR/olmo_1b_baseline.log"
# echo "  -> OLMo-1B - FPLUT"
# python eval.py --model_name OLMo-1B --use_fplut  2>&1 | tee "$RESULTS_DIR/olmo_1b_fplut.log"
# echo "  -> OLMo-1B - NN-LUT"
# python eval.py --model_name OLMo-1B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/olmo_1b_nnlut.log"

# Gemma-2B
echo "  -> Gemma-2B - Baseline"
python eval.py --model_name Gemma-2B  2>&1 | tee "$RESULTS_DIR/gemma_2b_baseline.log"

echo "  -> Gemma-2B - FPLUT"
python eval.py --model_name Gemma-2B --use_fplut  2>&1 | tee "$RESULTS_DIR/gemma_2b_fplut.log"

echo "  -> Gemma-2B - NN-LUT"
python eval.py --model_name Gemma-2B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/gemma_2b_nnlut.log"

# Gemma-2B-Instruct
echo "  -> Gemma-2B-Instruct - Baseline"
python eval.py --model_name Gemma-2B-Instruct  2>&1 | tee "$RESULTS_DIR/gemma_2b_instruct_baseline.log"

echo "  -> Gemma-2B-Instruct - FPLUT"
python eval.py --model_name Gemma-2B-Instruct --use_fplut  2>&1 | tee "$RESULTS_DIR/gemma_2b_instruct_fplut.log"

echo "  -> Gemma-2B-Instruct - NN-LUT"
python eval.py --model_name Gemma-2B-Instruct --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/gemma_2b_instruct_nnlut.log"

# ========== 中等模型 (2B-4B) ==========
echo ""
echo "[Group 3] Medium models (2B-4B) - SeqLen=default"

# # Phi-2
# echo "  -> Phi-2 - Baseline"
# python eval.py --model_name Phi-2  2>&1 | tee "$RESULTS_DIR/phi_2_baseline.log"
# echo "  -> Phi-2 - FPLUT"
# python eval.py --model_name Phi-2 --use_fplut  2>&1 | tee "$RESULTS_DIR/phi_2_fplut.log"
# echo "  -> Phi-2 - NN-LUT"
# python eval.py --model_name Phi-2 --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/phi_2_nnlut.log"

# # Pythia-2.8B
# echo "  -> Pythia-2.8B - Baseline"
# python eval.py --model_name Pythia-2.8B  2>&1 | tee "$RESULTS_DIR/pythia_2.8b_baseline.log"
# echo "  -> Pythia-2.8B - FPLUT"
# python eval.py --model_name Pythia-2.8B --use_fplut  2>&1 | tee "$RESULTS_DIR/pythia_2.8b_fplut.log"
# echo "  -> Pythia-2.8B - NN-LUT"
# python eval.py --model_name Pythia-2.8B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/pythia_2.8b_nnlut.log"

# Qwen2.5-3B-Instruct
echo "  -> Qwen2.5-3B-Instruct - Baseline"
python eval.py --model_name Qwen2.5-3B-Instruct  2>&1 | tee "$RESULTS_DIR/qwen2.5_3b_baseline.log"

echo "  -> Qwen2.5-3B-Instruct - FPLUT"
python eval.py --model_name Qwen2.5-3B-Instruct --use_fplut  2>&1 | tee "$RESULTS_DIR/qwen2.5_3b_fplut.log"

echo "  -> Qwen2.5-3B-Instruct - NN-LUT"
python eval.py --model_name Qwen2.5-3B-Instruct --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/qwen2.5_3b_nnlut.log"

# Llama-3.2-3B
echo "  -> Llama-3.2-3B - Baseline"
python eval.py --model_name Llama-3.2-3B  2>&1 | tee "$RESULTS_DIR/llama3.2_3b_baseline.log"

echo "  -> Llama-3.2-3B - FPLUT"
python eval.py --model_name Llama-3.2-3B --use_fplut  2>&1 | tee "$RESULTS_DIR/llama3.2_3b_fplut.log"

echo "  -> Llama-3.2-3B - NN-LUT"
python eval.py --model_name Llama-3.2-3B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/llama3.2_3b_nnlut.log"

# SmolLM-3B
echo "  -> SmolLM-3B - Baseline"
python eval.py --model_name SmolLM-3B  2>&1 | tee "$RESULTS_DIR/smollm_3b_baseline.log"

echo "  -> SmolLM-3B - FPLUT"
python eval.py --model_name SmolLM-3B --use_fplut  2>&1 | tee "$RESULTS_DIR/smollm_3b_fplut.log"

echo "  -> SmolLM-3B - NN-LUT"
python eval.py --model_name SmolLM-3B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/smollm_3b_nnlut.log"

# # StableLM-3B
# echo "  -> StableLM-3B - Baseline"
# python eval.py --model_name StableLM-3B  2>&1 | tee "$RESULTS_DIR/stablelm_3b_baseline.log"
# echo "  -> StableLM-3B - FPLUT"
# python eval.py --model_name StableLM-3B --use_fplut  2>&1 | tee "$RESULTS_DIR/stablelm_3b_fplut.log"
# echo "  -> StableLM-3B - NN-LUT"
# python eval.py --model_name StableLM-3B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/stablelm_3b_nnlut.log"

# # OpenELM-3B
# echo "  -> OpenELM-3B - Baseline"
# python eval.py --model_name OpenELM-3B  2>&1 | tee "$RESULTS_DIR/openelm_3b_baseline.log"
# echo "  -> OpenELM-3B - FPLUT"
# python eval.py --model_name OpenELM-3B --use_fplut  2>&1 | tee "$RESULTS_DIR/openelm_3b_fplut.log"
# echo "  -> OpenELM-3B - NN-LUT"
# python eval.py --model_name OpenELM-3B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/openelm_3b_nnlut.log"

# RedPajama-3B
# echo "  -> RedPajama-3B - Baseline"
# python eval.py --model_name RedPajama-3B  2>&1 | tee "$RESULTS_DIR/redpajama_3b_baseline.log"
# echo "  -> RedPajama-3B - FPLUT"
# python eval.py --model_name RedPajama-3B --use_fplut  2>&1 | tee "$RESULTS_DIR/redpajama_3b_fplut.log"
# echo "  -> RedPajama-3B - NN-LUT"
# python eval.py --model_name RedPajama-3B --use_nnlut --lut_silu_path "$LUT_SILU" --lut_gelu_path "$LUT_GELU" --lut_exp_path "$LUT_EXP"  2>&1 | tee "$RESULTS_DIR/redpajama_3b_nnlut.log"

echo ""
echo "=========================================="
echo "PPL Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "PPL Summary:"
for model_file in "$RESULTS_DIR"/*_baseline.log; do
    model_name=$(basename "$model_file" _baseline.log | sed 's/_/ /g')
    baseline_ppl=$(grep 'wiki ppl' "$model_file" 2>/dev/null | tail -1 | grep -o '[0-9]\+\.[0-9]\+' || echo "N/A")
    
    fplut_file="${model_file/_baseline.log/_fplut.log}"
    fplut_ppl=$(grep 'wiki ppl' "$fplut_file" 2>/dev/null | tail -1 | grep -o '[0-9]\+\.[0-9]\+' || echo "N/A")
    
    nnlut_file="${model_file/_baseline.log/_nnlut.log}"
    nnlut_ppl=$(grep 'wiki ppl' "$nnlut_file" 2>/dev/null | tail -1 | grep -o '[0-9]\+\.[0-9]\+' || echo "N/A")
    
    echo "  $model_name: Baseline=$baseline_ppl | FPLUT=$fplut_ppl | NN-LUT=$nnlut_ppl"
done
