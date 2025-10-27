export CUDA_VISIBLE_DEVICES=3

echo "======== Evaluate Qwen2.5-7B-Instruct w/ attn eager ========"
python eval.py --model_name Qwen2.5-7B-Instruct

echo "======== Evaluate Qwen2.5-7B-Instruct w/ attn eager + use_fplut ========"
python eval.py --model_name Qwen2.5-7B-Instruct--use_fplut