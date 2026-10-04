import torch
from utils import get_args, compare
from Llama import TransformerBlockLlama
from Llama3_1 import TransformerBlockLlama3
from GPT import TransformerBlockGPT

if __name__ == "__main__":

    args = get_args()
    if args.filename:
        dic_model = torch.load(args.filename)
    else:
        head_dim = 128
        dim =  head_dim * args.n_heads 
        ffn_dim = args.ffn_dim
        TP_param = 8 if args.GPT3_175B_TP_8 else 1
        n_heads = args.n_heads // TP_param
        n_kv_heads = args.n_kv_heads if args.Llama_GQA else n_heads
        num_experts = 128 
        num_experts_per_token = 8 
        hidden_dim = 2048 if args.Qwen3_30B_A3B else 4096 if args.Qwen3_235B_A22B else 0
        moe_hidden_dim = ffn_dim // num_experts_per_token 
        dic_model = {
            "TP_param": torch.tensor(TP_param),
            "dim": torch.tensor(dim),
            "n_heads": torch.tensor(n_heads),
            "x": torch.zeros((1, 1, dim)),
            "SANorm": torch.zeros(dim),
            "FFNNorm": torch.zeros(dim),
            "sa": torch.zeros((1, 1, dim)),
            "h": torch.zeros((1, 1, dim)),
            "out": torch.zeros((1, 1, dim)),
            "wq": torch.zeros((dim // TP_param, dim)),
            "wk": torch.zeros((head_dim * n_kv_heads), dim),
            "wv": torch.zeros((head_dim * n_kv_heads), dim),
            "xq": torch.zeros((1, 1, dim)),
            "xk": torch.zeros((1, 1, head_dim * n_heads)),
            "xv": torch.zeros((1, 1, head_dim * n_heads)),
            "start_pos": torch.tensor(args.seqlen - 1),
            "cache_k": torch.zeros((1, args.seqlen, n_kv_heads, head_dim)),
            "cache_v": torch.zeros((1, args.seqlen, n_kv_heads, head_dim)),
            "scores": torch.zeros((1, n_heads, 1, args.seqlen)),
            "output": torch.zeros((1, 1, dim)),
            "wo": torch.zeros((dim // TP_param, dim)),
            "w1": torch.zeros((ffn_dim // TP_param, dim)),
            "w3": torch.zeros((ffn_dim // TP_param, dim)),
            "w2": torch.zeros((dim // TP_param, ffn_dim)),
            "ffn": torch.zeros((1, 1, dim)),
            "num_experts": torch.tensor(num_experts),  # for Qwen MoE
            "hidden_dim": torch.tensor(hidden_dim),    # for Qwen MoE
            "num_experts_per_token": torch.tensor(num_experts_per_token),  # for Qwen MoE
            "moe_hidden_dim": torch.tensor(moe_hidden_dim)  # for Qwen MoE
        }
    if args.Llama_GQA:
        dic_model["n_kv_heads"] = torch.tensor(n_kv_heads)
    
    TB = TransformerBlockLlama3(dic_model, args) if args.Llama_GQA or args.Llama or args.filename else TransformerBlockGPT(dic_model, args)
    # print("Variable\t Dimension\t\t\t Rows required\n")
    TB.memory_mapping()

    # Original Version
    # if args.only_trace:
    #     if args.embedding:
    #         TB.trace_only_embedding()
    #     elif args.only_FC:
    #         TB.trace_only_FC()
    #     else:
    #         TB.trace_only()

    #     TB.finish()
    #     TB.file.close()


    operator_map = {
        "trace_rms": TB.trace_rms,
        "trace_qkv_proj": TB.trace_qkv_proj,
        "trace_rope": TB.trace_rope,
        "trace_attn_score": TB.trace_attn_score,
        "trace_attn_mask": TB.trace_attn_mask,
        "trace_attn_softmax": TB.trace_attn_softmax,
        "trace_attn_o": TB.trace_attn_o,
        "trace_wo_proj": TB.trace_wo_proj,
        "trace_router": TB.trace_router,
        "trace_w1_proj": TB.trace_w1_proj,
        "trace_w3_proj": TB.trace_w3_proj,
        "trace_ffn_af": TB.trace_ffn_af,
        "trace_w2_proj": TB.trace_w2_proj,
    }


    # # Single Operator Version
    if args.only_trace:
        if args.embedding:
            pass
        elif args.only_FC:
            pass
        else:
            if args.operator != "all":
                operator_map[args.operator]()
            else:
                TB.trace_only()

        TB.finish()
        TB.file.close()

    elif args.pim_memory_mapping:
        dic_model = torch.load(args.filename)
        TB.memory_mapping_verification()
        print("\n============ {} Functional Verification ============".format(args.filename.split("/")[-1].split(".")[0]))
        sa_aim = TB.self_attention_aim()
        # sa_aim = TB.self_attention()
        out_aim = TB.FFN_aim(sa_aim)
        compare(out_aim[0][0], TB.out[0][0], "AiM out")
        TB.finish()
        TB.file.close()
    else:
        dic_model = torch.load(args.filename)
        print("dic_model.keys():", list(dic_model.keys()))
        sa = TB.self_attention()
        out = TB.FFN(sa)
        compare(out, TB.out, "out")
        TB.file.close()
