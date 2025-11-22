import torch
from tqdm import tqdm
import os
from torch import nn
import warnings
import types
import json
import torch
import torch.nn as nn
from transformers import AutoConfig

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.utils import (
    make_table,
)
from accelerate import infer_auto_device_map, dispatch_model
from models.fp_lut import FPLUT, FPSoftMax
from models.config import cfgs
from nnlut.activation import Nnlut, NnlutSoftmax


# ---------- range collectors ----------
silu_stats = {"min": float("inf"), "max": float("-inf")}
softmax_stats = {"min": float("inf"), "max": float("-inf")}

def _update_stats(stats_dict, tensor: torch.Tensor):
    if tensor.numel() == 0:
        return
    stats_dict["min"] = min(stats_dict["min"], tensor.min().item())
    stats_dict["max"] = max(stats_dict["max"], tensor.max().item())

def _silu_hook(module, inputs, output):
    _update_stats(silu_stats, inputs[0])


def eval_ppl_(model,test_loader,seqlen=-1,limit=-1):
    nlls = []
    nsamples = test_loader.numel() // seqlen
    # for i in tqdm(range(nsamples)):
    with tqdm(range(nsamples)) as pbar:
        pbar.set_description_str("evaling ppl")
        for i in pbar:
            batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device if model.device != torch.device("meta") else torch.device("cuda"))
            net_name = model.name.lower() if hasattr(model,"name") else type(model).__name__.lower()
            if "opt" in net_name:
                outputs = model.model.model.decoder(batch)
                hidden_states = outputs[0]
                logits = model.model.lm_head(hidden_states)
            elif "llama" in net_name or "mixtral" in net_name or "qwen" in net_name or "gemma" in net_name  or "smollm3" in net_name:
                outputs = model(batch)
                logits = outputs['logits'];outputs = None
            elif "falcon" in net_name:
                outputs = model.model.transformer(batch)
                hidden_states = outputs[0]
                logits = model.model.lm_head(hidden_states)
            elif "glm" in net_name:
                outputs = model(batch)
                logits = outputs['logits'];outputs = None
            shift_logits = logits[:, :-1, :]
            shift_labels = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            tmp_ppl =  torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen)).item()
            pbar.set_postfix_str(f"--{tmp_ppl:4.4}")
            if i == limit:
                break
    ppl = torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen))
    return ppl.item()



@torch.no_grad()
def eval_ppl(model,tokenizer,seqlen=2048,limit=-1,split="test",disk_file=None):
    from datasets import load_dataset,load_from_disk
    model.eval()
    tokenizer_name = tokenizer.__class__.__name__
    cached_loader = f"./cache/wikitext-2-raw-v1/{split}_{tokenizer_name}_loader.pt" 
    if os.path.exists(cached_loader):
        loader = torch.load(cached_loader, weights_only=False)
    else:
        os.makedirs("cache",exist_ok=True)
        if disk_file is not None:
            wiki_testdata = load_from_disk(disk_file)['test']
        else:
            wiki_testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",cache_dir="./cache",keep_in_memory=True)
        loader = tokenizer("\n\n".join(wiki_testdata["text"]), return_tensors="pt")
        os.makedirs("./cache/wikitext-2-raw-v1",exist_ok=True)
        torch.save(loader,cached_loader)
    wiki_ppl = eval_ppl_(model, loader.input_ids, seqlen, limit)
    print(f'wiki ppl : {wiki_ppl}')
    return wiki_ppl


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--use_nnlut", action="store_true", default=False)
    parser.add_argument("--lut_silu_path", type=str, default=None)
    parser.add_argument("--lut_gelu_path", type=str, default=None)
    parser.add_argument("--lut_exp_path", type=str, default=None)
    parser.add_argument("--use_fplut", action="store_true", default=False)
    parser.add_argument("--calc_range", action="store_true", default=False)
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length for PPL evaluation (default: 2048, reduce for smaller GPUs)")
    return parser.parse_args()

from typing import Callable, List, Optional, Tuple, Union

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    # _update_stats(softmax_stats, attn_weights)
    attn_weights = module.fp_softmax.forward(attn_weights)
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    LLaMA attention forward - compatible with both old and new transformers API.
    Handles both position_embeddings (new) and position_ids (old) formats.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Handle position embeddings (new API) or position_ids (old API)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    elif hasattr(self, 'rotary_emb') and position_ids is not None:
        # Use rotary_emb if available (old API)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    else:
        raise ValueError(
            "No position embedding information available. "
            "Either position_embeddings (new API) or rotary_emb with position_ids (old API) must be provided."
        )

    if past_key_value is not None:
        # Handle KV cache
        if hasattr(past_key_value, 'update'):
            cache_kwargs = {"sin": sin if position_embeddings is not None or (hasattr(self, 'rotary_emb') and position_ids is not None) else None, 
                          "cos": cos if position_embeddings is not None or (hasattr(self, 'rotary_emb') and position_ids is not None) else None, 
                          "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            # Legacy format
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        raise ValueError(f"Unsupported attention implementation: {self.config._attn_implementation}")

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    
    # LLaMA expects (attn_output, attn_weights) - always return 2 values
    # The cache handling is done internally by transformers
    return attn_output, attn_weights

def qwen2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Qwen2 attention forward - compatible with both old and new transformers API.
    Handles both position_embeddings (new) and position_ids (old) formats.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Handle position embeddings (new API) or position_ids (old API)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    elif hasattr(self, 'rotary_emb') and position_ids is not None:
        # Use rotary_emb if available (old API)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    else:
        raise ValueError(
            "No position embedding information available. "
            "Either position_embeddings (new API) or rotary_emb with position_ids (old API) must be provided."
        )

    if past_key_value is not None:
        # Handle KV cache
        if hasattr(past_key_value, 'update'):
            cache_kwargs = {"sin": sin if position_embeddings is not None else None, 
                          "cos": cos if position_embeddings is not None else None, 
                          "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            # Legacy format
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        raise ValueError(f"Unsupported attention implementation: {self.config._attn_implementation}")

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    
    # Qwen2 expects (attn_output, attn_weights) - always return 2 values
    # The cache handling is done internally by transformers
    return attn_output, attn_weights

def main(args): 
    # Get model config - must be explicitly defined
    if args.model_name not in cfgs:
        raise ValueError(
            f"Model '{args.model_name}' not found in configuration. "
            f"Please add it to models/config.py or use one of the configured models: {list(cfgs.keys())}"
        )
    model_cfg = cfgs[args.model_name]
    model_cfg["torch_dtype"] = "float32"
    # Convert string dtype to torch dtype if needed
    dtype_str = model_cfg["torch_dtype"]
    if isinstance(dtype_str, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_str.lower(), torch.bfloat16)
    else:
        torch_dtype = dtype_str
    
    print("=" * 60)
    print("[INFO] Loading Model")
    print("=" * 60)
    print(f"[INFO] Model Name: {args.model_name}")
    print(f"[INFO] Model Path: {model_cfg['model']}")
    print(f"[INFO] Attention Implementation: {model_cfg['attn_implementation']}")
    print(f"[INFO] Data Type: {model_cfg['torch_dtype']} ({torch_dtype})")
    if args.use_fplut:
        print("[INFO] Using FP-LUT for activation functions")
    elif args.use_nnlut:
        print(f"[INFO] Using NN-LUT for activation functions")
        if args.lut_silu_path:
            print(f"[INFO]   - SiLU LUT: {args.lut_silu_path}")
        if args.lut_gelu_path:
            print(f"[INFO]   - GELU LUT: {args.lut_gelu_path}")
        print(f"[INFO]   - Exp LUT: {args.lut_exp_path}")
    else:
        print("[INFO] Using standard activation functions (baseline)")
    print(f"[INFO] Sequence Length: {args.seqlen}")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model"], attn_implementation=model_cfg["attn_implementation"], torch_dtype=torch_dtype)
    no_split_module_classes = ['LlamaDecoderLayer','QuantDecoderLayer',"Qwen2DecoderLayer"]
    
    # Detect model architecture type
    model_type = model.config.model_type.lower()
    is_qwen2 = "qwen2" in model_type or ("qwen" in model_type and "qwen3" not in model_type)
    is_llama = "llama" in model_type
    
    # Print model configuration details
    print("\n" + "=" * 60)
    print("[INFO] Model Configuration")
    print("=" * 60)
    print(f"[INFO] Model Type: {model.config.model_type}")
    print(f"[INFO] Architecture: {model.config.architectures if hasattr(model.config, 'architectures') else 'N/A'}")
    print(f"[INFO] Hidden Size: {model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'N/A'}")
    print(f"[INFO] Number of Layers: {model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 'N/A'}")
    print(f"[INFO] Number of Attention Heads: {model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 'N/A'}")
    print(f"[INFO] Vocabulary Size: {model.config.vocab_size if hasattr(model.config, 'vocab_size') else 'N/A'}")
    print(f"[INFO] Max Position Embeddings: {model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 'N/A'}")
    
    # Print actual model dtype
    if hasattr(model, 'dtype'):
        print(f"[INFO] Model Loaded Dtype: {model.dtype}")
    elif hasattr(model, 'parameters') and next(model.parameters(), None) is not None:
        print(f"[INFO] Model Loaded Dtype: {next(model.parameters()).dtype}")
    else:
        print(f"[INFO] Model Loaded Dtype: {model_cfg.get('torch_dtype', 'unknown')}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"[INFO] Trainable Parameters: {trainable_params:,}")
    
    print("=" * 60 + "\n")
    
    # Select appropriate attention forward function
    if is_qwen2:
        attn_forward_func = qwen2_attn_forward
        print("[INFO] Detected Qwen2 architecture, using qwen2_attn_forward")
    elif is_llama:
        attn_forward_func = llama_attn_forward
        print("[INFO] Detected LLaMA architecture, using llama_attn_forward")
    else:
        attn_forward_func = llama_attn_forward
        print("[WARNING] Detected unknown architecture, using llama_attn_forward")
        # raise ValueError(
        #     f"Unknown model type '{model_type}'. "
        #     f"Supported architectures: qwen2, llama. "
        #     f"Please add support for '{model_type}' in the code."
        # )
    
    # Detect activation function type
    act_fn_type = None
    
    # Method 1: Check config.hidden_act first (most reliable)
    if hasattr(model.config, 'hidden_act'):
        hidden_act = model.config.hidden_act.lower()
        print(f"[DEBUG] Config hidden_act: {model.config.hidden_act}")
        if 'gelu' in hidden_act:
            act_fn_type = 'gelu'
        elif 'silu' in hidden_act:
            act_fn_type = 'silu'
    
    # Method 2: Check actual activation function object if config check failed
    if act_fn_type is None:
        if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            first_layer = model.model.layers[0]
            if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'act_fn'):
                act_fn = first_layer.mlp.act_fn
                act_fn_str = str(type(act_fn)).lower()
                act_fn_name = type(act_fn).__name__.lower()
                
                print(f"[DEBUG] Activation function class: {type(act_fn).__name__}")
                print(f"[DEBUG] Activation function string: {act_fn_str[:200]}")
                
                # Check for GELU variants
                if 'gelu' in act_fn_str or 'gelu' in act_fn_name:
                    act_fn_type = 'gelu'
                elif 'silu' in act_fn_str or 'silu' in act_fn_name or 'swiglu' in act_fn_str:
                    act_fn_type = 'silu'
                else:
                    # Try to check function name if it's a callable
                    if hasattr(act_fn, '__name__'):
                        fn_name = act_fn.__name__.lower()
                        if 'gelu' in fn_name:
                            act_fn_type = 'gelu'
                        elif 'silu' in fn_name or 'swiglu' in fn_name:
                            act_fn_type = 'silu'
    
    if act_fn_type is None:
        # Print debug information before raising error
        print("\n[DEBUG] Activation function detection failed:")
        print(f"  - Model has 'model.model.layers': {hasattr(model.model, 'layers')}")
        if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            first_layer = model.model.layers[0]
            print(f"  - First layer has 'mlp': {hasattr(first_layer, 'mlp')}")
            if hasattr(first_layer, 'mlp'):
                print(f"  - MLP has 'act_fn': {hasattr(first_layer.mlp, 'act_fn')}")
                if hasattr(first_layer.mlp, 'act_fn'):
                    act_fn = first_layer.mlp.act_fn
                    print(f"  - Activation function type: {type(act_fn)}")
                    print(f"  - Activation function name: {type(act_fn).__name__}")
                    print(f"  - Activation function string: {str(type(act_fn))}")
        
        raise ValueError(
            f"Could not detect activation function type for model '{args.model_name}'. "
            f"Please check the model architecture or manually specify the activation function type. "
            f"Config hidden_act: {getattr(model.config, 'hidden_act', 'N/A')}"
        )
    print(f"[INFO] Detected activation function: {act_fn_type.upper()}")
    
    if args.use_fplut:
        for layer in model.model.layers:
            if act_fn_type == 'gelu':
                layer.mlp.act_fn = FPLUT(func_name="gelu", function=layer.mlp.act_fn)
            else:
                layer.mlp.act_fn = FPLUT(func_name="silu", function=layer.mlp.act_fn)
            layer.self_attn.fp_softmax = FPSoftMax()
            layer.self_attn.forward = types.MethodType(attn_forward_func, layer.self_attn)
    elif args.use_nnlut:
        # Select appropriate LUT path based on activation function type
        if act_fn_type == 'gelu':
            if args.lut_gelu_path is None:
                raise ValueError(
                    f"Model uses GELU activation function but --lut_gelu_path not provided. "
                    f"Please provide the correct GELU LUT path using --lut_gelu_path."
                )
            lut_path = args.lut_gelu_path
            print(f"[INFO] Using GELU LUT: {lut_path}")
        else:
            if args.lut_silu_path is None:
                raise ValueError(
                    f"Model uses SiLU activation function but --lut_silu_path not provided. "
                    f"Please provide the correct SiLU LUT path using --lut_silu_path."
                )
            lut_path = args.lut_silu_path
            print(f"[INFO] Using SiLU LUT: {lut_path}")
        
        for layer in model.model.layers:
            lut_details = json.load(open(lut_path, "r"))
            load_keys = ["cut_points", "slopes", "biases"]
            lut_details = {k: torch.tensor(lut_details[k]).half().float() for k in load_keys}
            layer.mlp.act_fn = Nnlut(lut_details["cut_points"], lut_details["slopes"], lut_details["biases"])

            lut_details = json.load(open(args.lut_exp_path, "r"))
            load_keys = ["cut_points", "slopes", "biases"]
            lut_details = {k: torch.tensor(lut_details[k]).half().float() for k in load_keys}
            layer.self_attn.fp_softmax = NnlutSoftmax(lut_details["cut_points"], lut_details["slopes"], lut_details["biases"])
            layer.self_attn.forward = types.MethodType(attn_forward_func, layer.self_attn)
    device_map = infer_auto_device_map(model, no_split_module_classes=no_split_module_classes, max_memory=model_cfg.get("max_memory", None))
    dispatch_model(model, device_map)
    print(f"[INFO] Device map: {device_map}")
    
    if args.calc_range:
        # ---------- register range hooks ----------
        for lyr in model.model.layers:
            # --- SiLU / activation ---
            if isinstance(lyr.mlp.act_fn, FPLUT):
                lyr.mlp.act_fn.register_forward_hook(_silu_hook)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model"])
    
    print("\n" + "=" * 60)
    print("[INFO] Starting PPL Evaluation")
    print("=" * 60)
    print(f"[INFO] Dataset: WikiText")
    print(f"[INFO] Sequence Length: {args.seqlen}")
    print("=" * 60 + "\n")
    
    eval_ppl(model, tokenizer, seqlen=args.seqlen)
    if args.calc_range:
        print(f"[RANGE] SiLU input : min={silu_stats['min']:.6f}, max={silu_stats['max']:.6f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
