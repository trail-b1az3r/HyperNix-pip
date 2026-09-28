from hypernix.arch import infer_arch, map_tensor_name


def test_map_llama_style():
    assert map_tensor_name("model.layers.3.self_attn.q_proj.weight") == "blk.3.attn_q.weight"
    assert map_tensor_name("model.layers.0.mlp.down_proj.weight") == "blk.0.ffn_down.weight"
    assert map_tensor_name("model.embed_tokens.weight") == "token_embd.weight"
    assert map_tensor_name("lm_head.weight") == "output.weight"
    assert map_tensor_name("model.norm.weight") == "output_norm.weight"


def test_map_gpt_neox_style():
    assert map_tensor_name("transformer.h.5.attn.c_attn.weight") == "blk.5.attn_qkv.weight"
    assert map_tensor_name("transformer.h.5.attn.c_proj.weight") == "blk.5.attn_output.weight"
    assert map_tensor_name("transformer.h.5.mlp.c_fc.weight") == "blk.5.ffn_up.weight"


def test_infer_arch_any_size():
    import torch

    layers = 7
    hidden = 384
    ffn = 1024
    vocab = 12345
    sd = {
        "model.embed_tokens.weight": torch.zeros(vocab, hidden),
        "lm_head.weight": torch.zeros(vocab, hidden),
        "model.norm.weight": torch.zeros(hidden),
    }
    for i in range(layers):
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.zeros(hidden)
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(ffn, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(ffn, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(hidden, ffn)

    info = infer_arch(sd)
    assert info.n_layers == layers
    assert info.n_embd == hidden
    assert info.n_ff == ffn
    assert info.vocab_size == vocab
    assert info.n_head >= 1
