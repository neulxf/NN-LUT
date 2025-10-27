export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "======== Evaluate Qwen2.5-110B-Instruct w/ attn eager ========"
python eval_ppl.py --model_name Qwen1.5-110B-Instruct
echo "======== Evaluate Qwen2.5-110B-Instruct w/ attn eager + use_fplut ========"
python eval_ppl.py --model_name Qwen1.5-110B-Instruct --use_fplut
python eval_ppl.py --model_name Qwen1.5-110B --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 

python eval.py --model_name Qwen1.5-110B --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 

