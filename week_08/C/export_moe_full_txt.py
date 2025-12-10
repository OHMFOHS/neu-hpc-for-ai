import json
import numpy as np

# ====== 读取已有的 MLP & Router JSON ======
mlp = json.load(open("deepseek_v3_mlp_testcases.json"))
router = json.load(open("deepseek_v3_router_testcases.json"))

H = mlp["config"]["hidden_size"]        # 2
I = mlp["config"]["intermediate_size"]  # 4

# 设置 MoE 配置
E = 4   # 用前 4 个 experts
K = 2   # top-2
HAS_SHARED = 1

# ====== Router 权重: [E, H] ======
W_router_full = np.array(router["weights"]["weight"], dtype=np.float32)  # [16, H]
W_router = W_router_full[:E]  # [E, H]
np.savetxt("moe_router_weights.txt", W_router)

# ====== Expert & Shared MLP 权重: 直接用同一组 ======
W_gate = np.array(mlp["weights"]["gate_proj.weight"], dtype=np.float32)   # [I, H]
W_up   = np.array(mlp["weights"]["up_proj.weight"],   dtype=np.float32)   # [I, H]
W_down = np.array(mlp["weights"]["down_proj.weight"], dtype=np.float32)   # [H, I]

for e in range(E):
    np.savetxt(f"moe_expert{e}_w_gate.txt", W_gate)
    np.savetxt(f"moe_expert{e}_w_up.txt",   W_up)
    np.savetxt(f"moe_expert{e}_w_down.txt", W_down)

# shared expert 也用同一组
np.savetxt("moe_shared_w_gate.txt", W_gate)
np.savetxt("moe_shared_w_up.txt",   W_up)
np.savetxt("moe_shared_w_down.txt", W_down)

# ====== 选择一个 test 输入，生成 MoE reference 输出 ======
# 用 MLP testcase 的第二个，形状 (2,3,2) -> T=6 tokens
test = mlp["tests"][1]
X = np.array(test["input"], dtype=np.float32).reshape(-1, H)   # [T, H]
T = X.shape[0]

def silu(x):
    return x / (1.0 + np.exp(-x))

def mlp_forward(x):
    # x: (H,)
    gate = W_gate @ x      # (I,)
    up   = W_up @ x        # (I,)
    inter = silu(gate) * up
    y = W_down @ inter     # (H,)
    return y

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)

# Router logits & probs
logits = X @ W_router.T           # [T, E]
probs = softmax(logits, axis=-1)  # [T, E]

# top-k 选择
idx = np.argsort(probs, axis=-1)[:, -K:]   # [T, K], 最后 K 个是 top-k（升序）
w = np.take_along_axis(probs, idx, axis=-1)  # [T, K]

# 归一化 top-k 概率，使其和为 1
w_sum = np.sum(w, axis=-1, keepdims=True)   # [T,1]
w = w / w_sum

Y = np.zeros((T, H), dtype=np.float32)

for t in range(T):
    xt = X[t]
    y_t = np.zeros((H,), dtype=np.float32)

    # routed experts
    for kk in range(K):
        e_idx = int(idx[t, kk])
        wt = float(w[t, kk])
        y_exp = mlp_forward(xt)   # 所有 expert 权重相同
        y_t += wt * y_exp

    # shared expert
    y_shared = mlp_forward(xt)
    y_t += y_shared

    Y[t] = y_t

# ====== 导出 txt ======
# 配置：H I E K HAS_SHARED T
with open("moe_config.txt", "w") as f:
    f.write(f"{H} {I} {E} {K} {HAS_SHARED} {T}\n")

np.savetxt("moe_input.txt", X)   # [T,H]
np.savetxt("moe_ref.txt",   Y)   # [T,H]

print("Exported MoE txt files:")
print("  moe_config.txt")
print("  moe_router_weights.txt")
print("  moe_expert*_w_*.txt")
print("  moe_shared_w_*.txt")
print("  moe_input.txt")
print("  moe_ref.txt")
