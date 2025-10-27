export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "======== Evaluate Qwen2.5-110B w/ attn eager ========"
python eval.py --model_name Qwen1.5-110B
echo "======== Evaluate Qwen2.5-110B w/ attn eager + use_fplut ========"
python eval.py --model_name Qwen1.5-110B --use_fplut
