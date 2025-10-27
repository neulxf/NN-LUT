import warnings
import types

import torch
import torch.nn as nn
from transformers import AutoConfig
import json
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--use_fplut", action="store_true", default=False)
    parser.add_argument("--use_nnlut", action="store_true", default=False)
    parser.add_argument("--lut_silu_path", type=str, default=None)
    parser.add_argument("--lut_exp_path", type=str, default=None)
    parser.add_argument("--calc_range", action="store_true", default=False)
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
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
    return attn_output, attn_weights

def main(args): 
    model_cfg = cfgs[args.model_name]
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model"], attn_implementation=model_cfg["attn_implementation"], torch_dtype=model_cfg["torch_dtype"])
    no_split_module_classes = ['LlamaDecoderLayer','QuantDecoderLayer',"Qwen2DecoderLayer"]
    if args.use_fplut:
        for layer in model.model.layers:
            layer.mlp.act_fn = FPLUT(func_name="silu", function=layer.mlp.act_fn)
            layer.self_attn.fp_softmax = FPSoftMax()
            layer.self_attn.forward = types.MethodType(llama_attn_forward, layer.self_attn)
    elif args.use_nnlut:
        for layer in model.model.layers:
            lut_details = json.load(open(args.lut_silu_path, "r"))
            load_keys = ["cut_points", "slopes", "biases"]
            lut_details = {k: torch.tensor(lut_details[k]).half().float() for k in load_keys}
            layer.mlp.act_fn = Nnlut(lut_details["cut_points"], lut_details["slopes"], lut_details["biases"])

            lut_details = json.load(open(args.lut_exp_path, "r"))
            load_keys = ["cut_points", "slopes", "biases"]
            lut_details = {k: torch.tensor(lut_details[k]).half().float() for k in load_keys}
            layer.self_attn.fp_softmax = NnlutSoftmax(lut_details["cut_points"], lut_details["slopes"], lut_details["biases"])
            layer.self_attn.forward = types.MethodType(llama_attn_forward, layer.self_attn)
    device_map = infer_auto_device_map(model, no_split_module_classes=no_split_module_classes, max_memory=model_cfg.get("max_memory", None))
    dispatch_model(model, device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model"])
    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer),
        # tasks=["arc_easy", "arc_challenge", "boolq", "hellaswag", "lambada_openai", "openbookqa", "piqa", "social_iqa", "winogrande"],
        tasks=["openbookqa"],
        num_fewshot=None,
    )
    print(make_table(results))

if __name__ == "__main__":
    args = parse_args()
    main(args)