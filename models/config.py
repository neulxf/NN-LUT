cfgs = dict()
cfgs["Qwen2.5-7B-Instruct"] = dict()
cfgs["Qwen2.5-7B-Instruct"]["model"] = "weights/Qwen2.5-7B-Instruct"
cfgs["Qwen2.5-7B-Instruct"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-7B-Instruct"]["torch_dtype"] = "float32"

cfgs["Qwen2.5-7B-Instruct-bf16"] = dict()
cfgs["Qwen2.5-7B-Instruct-bf16"]["model"] = "weights/Qwen2.5-7B-Instruct"
cfgs["Qwen2.5-7B-Instruct-bf16"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-7B-Instruct-bf16"]["torch_dtype"] = "bfloat16"

cfgs["Qwen2.5-32B-Instruct"] = dict()
cfgs["Qwen2.5-32B-Instruct"]["model"] = "weights/Qwen2.5-32B-Instruct"
cfgs["Qwen2.5-32B-Instruct"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-32B-Instruct"]["torch_dtype"] = "bfloat16"


cfgs["Qwen2.5-7B-Instruct-float"] = dict()
cfgs["Qwen2.5-7B-Instruct-float"]["model"] = "weights/Qwen2.5-7B-Instruct"
cfgs["Qwen2.5-7B-Instruct-float"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-7B-Instruct-float"]["torch_dtype"] = "float"

cfgs["Qwen2.5-32B-Instruct-float"] = dict()
cfgs["Qwen2.5-32B-Instruct-float"]["model"] = "weights/Qwen2.5-32B-Instruct"
cfgs["Qwen2.5-32B-Instruct-float"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-32B-Instruct-float"]["torch_dtype"] = "float"
cfgs["Qwen2.5-32B-Instruct-float"]["max_memory"] = {0: "70GB", 1: "70GB"}

cfgs["Qwen1.5-110B"] = dict()
cfgs["Qwen1.5-110B"]["model"] = "weights/Qwen1.5-110B"
cfgs["Qwen1.5-110B"]["attn_implementation"] = "eager"
cfgs["Qwen1.5-110B"]["torch_dtype"] = "bfloat16"
cfgs["Qwen1.5-110B"]["max_memory"] = {0: "65GB", 1: "65GB", 2: "65GB", 3: "65GB"}

cfgs["Qwen1.5-110B-float"] = dict()
cfgs["Qwen1.5-110B-float"]["model"] = "weights/Qwen1.5-110B"
cfgs["Qwen1.5-110B-float"]["attn_implementation"] = "eager"
cfgs["Qwen1.5-110B-float"]["torch_dtype"] = "float"
cfgs["Qwen1.5-110B-float"]["max_memory"] = {0: "60GB", 1: "60GB", 2: "60GB", 3: "60GB", 4: "60GB", 5: "60GB", 6: "60GB", 7: "60GB"}

cfgs["llama3-8b-hf"] = dict()
cfgs["llama3-8b-hf"]["model"] = "weights/llama3-8b-hf"
cfgs["llama3-8b-hf"]["attn_implementation"] = "eager"
cfgs["llama3-8b-hf"]["torch_dtype"] = "bfloat16"

cfgs["llama3-70b-hf"] = dict()
cfgs["llama3-70b-hf"]["model"] = "weights/llama3-70b-hf"
cfgs["llama3-70b-hf"]["attn_implementation"] = "eager"
cfgs["llama3-70b-hf"]["torch_dtype"] = "bfloat16"
cfgs["llama3-70b-hf"]["max_memory"] = {0: "70GB", 1: "70GB", 2: "70GB"}



cfgs["Qwen3-8B-bf16"] = dict()
cfgs["Qwen3-8B-bf16"]["model"] = "weights/Qwen3-8B"
cfgs["Qwen3-8B-bf16"]["attn_implementation"] = "eager"
cfgs["Qwen3-8B-bf16"]["torch_dtype"] = "bfloat16"

cfgs["Qwen3-30B-A3B-bf16"] = dict()
cfgs["Qwen3-30B-A3B-bf16"]["model"] = "weights/Qwen3-30B-A3B"
cfgs["Qwen3-30B-A3B-bf16"]["attn_implementation"] = "eager"
cfgs["Qwen3-30B-A3B-bf16"]["torch_dtype"] = "bfloat16"

cfgs["Qwen3-32B-bf16"] = dict()
cfgs["Qwen3-32B-bf16"]["model"] = "weights/Qwen3-32B"
cfgs["Qwen3-32B-bf16"]["attn_implementation"] = "eager"
cfgs["Qwen3-32B-bf16"]["torch_dtype"] = "bfloat16"

cfgs["Qwen3-8B-float"] = dict()
cfgs["Qwen3-8B-float"]["model"] = "weights/Qwen3-8B"
cfgs["Qwen3-8B-float"]["attn_implementation"] = "eager"
cfgs["Qwen3-8B-float"]["torch_dtype"] = "float"

# ========== Small Models (for 8GB GPU testing) ==========
# Qwen2.5-0.5B-Instruct (约 500MB, 适合快速测试)
cfgs["Qwen2.5-0.5B-Instruct"] = dict()
cfgs["Qwen2.5-0.5B-Instruct"]["model"] = "Qwen/Qwen2.5-0.5B-Instruct"
cfgs["Qwen2.5-0.5B-Instruct"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-0.5B-Instruct"]["torch_dtype"] = "bfloat16"

# Qwen2.5-1.5B-Instruct (约 1.5GB, 适合测试)
cfgs["Qwen2.5-1.5B-Instruct"] = dict()
cfgs["Qwen2.5-1.5B-Instruct"]["model"] = "Qwen/Qwen2.5-1.5B-Instruct"
cfgs["Qwen2.5-1.5B-Instruct"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-1.5B-Instruct"]["torch_dtype"] = "bfloat16"

# TinyLlama (1.1B, 约 2GB, 开源小模型)
cfgs["TinyLlama-1.1B"] = dict()
cfgs["TinyLlama-1.1B"]["model"] = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
cfgs["TinyLlama-1.1B"]["attn_implementation"] = "eager"
cfgs["TinyLlama-1.1B"]["torch_dtype"] = "bfloat16"

# Phi-2 (2.7B, 微软小模型，性能好)
cfgs["Phi-2"] = dict()
cfgs["Phi-2"]["model"] = "microsoft/phi-2"
cfgs["Phi-2"]["attn_implementation"] = "eager"
cfgs["Phi-2"]["torch_dtype"] = "bfloat16"

# Qwen2-1.5B-Instruct (备选)
cfgs["Qwen2-1.5B-Instruct"] = dict()
cfgs["Qwen2-1.5B-Instruct"]["model"] = "Qwen/Qwen2-1.5B-Instruct"
cfgs["Qwen2-1.5B-Instruct"]["attn_implementation"] = "eager"
cfgs["Qwen2-1.5B-Instruct"]["torch_dtype"] = "bfloat16"

# Qwen2.5-3B-Instruct (3B模型，性能好)
cfgs["Qwen2.5-3B-Instruct"] = dict()
cfgs["Qwen2.5-3B-Instruct"]["model"] = "Qwen/Qwen2.5-3B-Instruct"
cfgs["Qwen2.5-3B-Instruct"]["attn_implementation"] = "eager"
cfgs["Qwen2.5-3B-Instruct"]["torch_dtype"] = "bfloat16"

# LLaMA 3.2 系列 (Meta最新小模型)
cfgs["Llama-3.2-1B"] = dict()
cfgs["Llama-3.2-1B"]["model"] = "meta-llama/Llama-3.2-1B"
cfgs["Llama-3.2-1B"]["attn_implementation"] = "eager"
cfgs["Llama-3.2-1B"]["torch_dtype"] = "bfloat16"

cfgs["Llama-3.2-3B"] = dict()
cfgs["Llama-3.2-3B"]["model"] = "meta-llama/Llama-3.2-3B"
cfgs["Llama-3.2-3B"]["attn_implementation"] = "eager"
cfgs["Llama-3.2-3B"]["torch_dtype"] = "bfloat16"

# Mistral 7B (接近8B，但性能好)
cfgs["Mistral-7B-v0.1"] = dict()
cfgs["Mistral-7B-v0.1"]["model"] = "mistralai/Mistral-7B-v0.1"
cfgs["Mistral-7B-v0.1"]["attn_implementation"] = "eager"
cfgs["Mistral-7B-v0.1"]["torch_dtype"] = "bfloat16"

# ========== More Small Models (under 8B) ==========
# Gemma 系列 (Google)
cfgs["Gemma-2B"] = dict()
cfgs["Gemma-2B"]["model"] = "google/gemma-2b"
cfgs["Gemma-2B"]["attn_implementation"] = "eager"
cfgs["Gemma-2B"]["torch_dtype"] = "bfloat16"

cfgs["Gemma-2B-Instruct"] = dict()
cfgs["Gemma-2B-Instruct"]["model"] = "google/gemma-2b-it"
cfgs["Gemma-2B-Instruct"]["attn_implementation"] = "eager"
cfgs["Gemma-2B-Instruct"]["torch_dtype"] = "bfloat16"

cfgs["Gemma-7B"] = dict()
cfgs["Gemma-7B"]["model"] = "google/gemma-7b"
cfgs["Gemma-7B"]["attn_implementation"] = "eager"
cfgs["Gemma-7B"]["torch_dtype"] = "bfloat16"

# StableLM 系列 (Stability AI)
cfgs["StableLM-3B"] = dict()
cfgs["StableLM-3B"]["model"] = "stabilityai/stablelm-3b-4e1t"
cfgs["StableLM-3B"]["attn_implementation"] = "eager"
cfgs["StableLM-3B"]["torch_dtype"] = "bfloat16"

# OLMo 系列 (Allen AI)
cfgs["OLMo-1B"] = dict()
cfgs["OLMo-1B"]["model"] = "allenai/OLMo-1B"
cfgs["OLMo-1B"]["attn_implementation"] = "eager"
cfgs["OLMo-1B"]["torch_dtype"] = "bfloat16"

cfgs["OLMo-1B-7B"] = dict()
cfgs["OLMo-1B-7B"]["model"] = "allenai/OLMo-1B-7B"
cfgs["OLMo-1B-7B"]["attn_implementation"] = "eager"
cfgs["OLMo-1B-7B"]["torch_dtype"] = "bfloat16"

# OpenELM 系列 (Apple)
cfgs["OpenELM-1.1B"] = dict()
cfgs["OpenELM-1.1B"]["model"] = "apple/OpenELM-1_1B"
cfgs["OpenELM-1.1B"]["attn_implementation"] = "eager"
cfgs["OpenELM-1.1B"]["torch_dtype"] = "bfloat16"

cfgs["OpenELM-3B"] = dict()
cfgs["OpenELM-3B"]["model"] = "apple/OpenELM-3B"
cfgs["OpenELM-3B"]["attn_implementation"] = "eager"
cfgs["OpenELM-3B"]["torch_dtype"] = "bfloat16"

# SmolLM 系列 (HuggingFace)
# Note: SmolLM models may not be available on HuggingFace Hub
# If you need these models, download them manually or use alternative paths
cfgs["SmolLM-135M"] = dict()
cfgs["SmolLM-135M"]["model"] = "HuggingFaceTB/SmolLM-135M-Instruct"
cfgs["SmolLM-135M"]["attn_implementation"] = "eager"
cfgs["SmolLM-135M"]["torch_dtype"] = "bfloat16"

cfgs["SmolLM-360M"] = dict()
cfgs["SmolLM-360M"]["model"] = "HuggingFaceTB/SmolLM2-360M-Instruct"
cfgs["SmolLM-360M"]["attn_implementation"] = "eager"
cfgs["SmolLM-360M"]["torch_dtype"] = "bfloat16"

cfgs["SmolLM-1.7B"] = dict()
cfgs["SmolLM-1.7B"]["model"] = "HuggingFaceTB/SmolLM-1.7B-Instruct"
cfgs["SmolLM-1.7B"]["attn_implementation"] = "eager"
cfgs["SmolLM-1.7B"]["torch_dtype"] = "bfloat16"

cfgs["SmolLM-3B"] = dict()
cfgs["SmolLM-3B"]["model"] = "HuggingFaceTB/SmolLM3-3B"
cfgs["SmolLM-3B"]["attn_implementation"] = "eager"
cfgs["SmolLM-3B"]["torch_dtype"] = "bfloat16"

# Pythia 系列 (EleutherAI)

cfgs["Pythia-14M"] = dict()
cfgs["Pythia-14M"]["model"] = "EleutherAI/pythia-14m"
cfgs["Pythia-14M"]["attn_implementation"] = "eager"
cfgs["Pythia-14M"]["torch_dtype"] = "bfloat16"

cfgs["Pythia-70M"] = dict()
cfgs["Pythia-70M"]["model"] = "EleutherAI/pythia-70m"
cfgs["Pythia-70M"]["attn_implementation"] = "eager"
cfgs["Pythia-70M"]["torch_dtype"] = "bfloat16"

cfgs["Pythia-160M"] = dict()
cfgs["Pythia-160M"]["model"] = "EleutherAI/pythia-160m"
cfgs["Pythia-160M"]["attn_implementation"] = "eager"
cfgs["Pythia-160M"]["torch_dtype"] = "bfloat16"

cfgs["Pythia-410M"] = dict()
cfgs["Pythia-410M"]["model"] = "EleutherAI/pythia-410m"
cfgs["Pythia-410M"]["attn_implementation"] = "eager"
cfgs["Pythia-410M"]["torch_dtype"] = "bfloat16"

cfgs["Pythia-1B"] = dict()
cfgs["Pythia-1B"]["model"] = "EleutherAI/pythia-1b"
cfgs["Pythia-1B"]["attn_implementation"] = "eager"
cfgs["Pythia-1B"]["torch_dtype"] = "bfloat16"

cfgs["Pythia-1.4B"] = dict()
cfgs["Pythia-1.4B"]["model"] = "EleutherAI/pythia-1.4b"
cfgs["Pythia-1.4B"]["attn_implementation"] = "eager"
cfgs["Pythia-1.4B"]["torch_dtype"] = "bfloat16"

cfgs["Pythia-2.8B"] = dict()
cfgs["Pythia-2.8B"]["model"] = "EleutherAI/pythia-2.8b"
cfgs["Pythia-2.8B"]["attn_implementation"] = "eager"
cfgs["Pythia-2.8B"]["torch_dtype"] = "bfloat16"

# RedPajama 系列
cfgs["RedPajama-3B"] = dict()
cfgs["RedPajama-3B"]["model"] = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
cfgs["RedPajama-3B"]["attn_implementation"] = "eager"
cfgs["RedPajama-3B"]["torch_dtype"] = "bfloat16"

# LLaMA 2 系列 (7B)
cfgs["Llama-2-7B"] = dict()
cfgs["Llama-2-7B"]["model"] = "meta-llama/Llama-2-7b-hf"
cfgs["Llama-2-7B"]["attn_implementation"] = "eager"
cfgs["Llama-2-7B"]["torch_dtype"] = "bfloat16"

cfgs["Llama-2-7B-Chat"] = dict()
cfgs["Llama-2-7B-Chat"]["model"] = "meta-llama/Llama-2-7b-chat-hf"
cfgs["Llama-2-7B-Chat"]["attn_implementation"] = "eager"
cfgs["Llama-2-7B-Chat"]["torch_dtype"] = "bfloat16"

