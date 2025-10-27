export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
python eval.py --model_name Qwen2.5-7B-Instruct --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H256_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H256_sub.json 

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
python eval.py --model_name Qwen2.5-32B-Instruct --use_nnlut --lut_silu_path nnlut_bench/lut_details_silu_H16_sub.json --lut_exp_path nnlut_bench/lut_details_exp_H16_sub.json 