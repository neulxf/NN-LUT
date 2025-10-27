export CUDA_VISIBLE_DEVICES=5

echo "======== Evaluate Qwen2.5-32B-Instruct w/ attn eager ========"
python eval.py --model weights/Qwen2.5-32B-Instruct --attn_implementation eager

echo "======== Evaluate Qwen2.5-32B-Instruct w/ attn eager + use_fplut ========"
python eval.py --model weights/Qwen2.5-32B-Instruct --attn_implementation eager --use_fplut
