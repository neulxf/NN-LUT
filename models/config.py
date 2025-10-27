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


