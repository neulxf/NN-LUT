export CUDA_VISIBLE_DEVICES=3

echo "======== Evaluate Qwen2.5-7B-Instruct w/ attn eager ========"
python eval_ppl.py --model weights/Qwen2.5-7B-Instruct --attn_implementation eager

echo "======== Evaluate Qwen2.5-7B-Instruct w/ attn eager + use_fplut ========"
python eval_ppl.py --model weights/Qwen2.5-7B-Instruct --attn_implementation eager --use_fplut

echo "======== Evaluate llama3-8b-hf w/ attn eager ========"
python eval_ppl.py --model weights/llama3-8b-hf --attn_implementation eager

echo "======== Evaluate llama3-8b-hf w/ attn eager + use_fplut ========"
python eval_ppl.py --model weights/llama3-8b-hf --attn_implementation eager --use_fplut