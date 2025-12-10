#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// ========== 简单 struct 定义 ==========

typedef struct {
    int hidden_size;       // H
    int intermediate_size; // I
    float *w_gate;         // [I, H]
    float *w_up;           // [I, H]
    float *w_down;         // [H, I]
} DeepseekV3MLP;

typedef struct {
    int hidden_size;  // H
    int num_experts;  // E
    float *w;         // [E, H]
} DeepseekV3TopkRouter;

typedef struct {
    int hidden_size;        // H
    int intermediate_size;  // I
    int num_experts;        // E
    int top_k;              // K
    int has_shared;         // 0 or 1
    int num_tokens;         // T

    DeepseekV3TopkRouter router;
    DeepseekV3MLP *experts; // size E
    DeepseekV3MLP shared;   // only valid if has_shared == 1
} DeepseekV3MoE;

// ========== 工具函数 ==========

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

void load_floats(const char *path, float *buf, int n) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        if (fscanf(f, "%f", &buf[i]) != 1) {
            fprintf(stderr, "Failed to read float from %s at index %d\n", path, i);
            fclose(f);
            exit(1);
        }
    }
    fclose(f);
}

// ========== MLP 前向 ==========

void mlp_forward(const DeepseekV3MLP *mlp,
                 const float *x,  // [H]
                 float *y)        // [H]
{
    int H = mlp->hidden_size;
    int I = mlp->intermediate_size;

    float *gate = (float *)malloc(sizeof(float) * I);
    float *up   = (float *)malloc(sizeof(float) * I);
    float *inter= (float *)malloc(sizeof(float) * I);

    // gate = W_gate * x
    for (int i = 0; i < I; ++i) {
        float s = 0.0f;
        const float *w_row = mlp->w_gate + i * H;
        for (int h = 0; h < H; ++h) {
            s += w_row[h] * x[h];
        }
        gate[i] = s;
    }

    // up = W_up * x
    for (int i = 0; i < I; ++i) {
        float s = 0.0f;
        const float *w_row = mlp->w_up + i * H;
        for (int h = 0; h < H; ++h) {
            s += w_row[h] * x[h];
        }
        up[i] = s;
    }

    // inter = silu(gate) * up
    for (int i = 0; i < I; ++i) {
        inter[i] = silu(gate[i]) * up[i];
    }

    // y = W_down * inter    W_down: [H, I]
    for (int h = 0; h < H; ++h) {
        float s = 0.0f;
        const float *w_row = mlp->w_down + h * I;
        for (int i = 0; i < I; ++i) {
            s += w_row[i] * inter[i];
        }
        y[h] = s;
    }

    free(gate);
    free(up);
    free(inter);
}

// ========== Router 前向 ==========

// logits: [T, E] = X [T,H] @ W^T [H,E] => [T,E]
void router_forward(const DeepseekV3TopkRouter *router,
                    const float *x,   // [T,H]
                    int T,
                    float *logits)    // [T,E]
{
    int H = router->hidden_size;
    int E = router->num_experts;

    for (int t = 0; t < T; ++t) {
        const float *xt = x + t * H;
        float *lt = logits + t * E;
        for (int e = 0; e < E; ++e) {
            const float *w_row = router->w + e * H;
            float s = 0.0f;
            for (int h = 0; h < H; ++h) {
                s += w_row[h] * xt[h];
            }
            lt[e] = s;
        }
    }
}

// 对每个 token 做 softmax，输入 logits[T,E]，输出 probs[T,E]
void softmax_per_token(const float *logits, float *probs, int T, int E)
{
    for (int t = 0; t < T; ++t) {
        const float *lt = logits + t * E;
        float *pt = probs + t * E;

        float m = -FLT_MAX;
        for (int e = 0; e < E; ++e) {
            if (lt[e] > m) m = lt[e];
        }

        float sum = 0.0f;
        for (int e = 0; e < E; ++e) {
            float v = expf(lt[e] - m);
            pt[e] = v;
            sum += v;
        }

        float inv = 1.0f / sum;
        for (int e = 0; e < E; ++e) {
            pt[e] *= inv;
        }
    }
}

// 每个 token top-k，输入 probs[T,E]
// 输出 topk_idx[T,K], topk_w[T,K]，并把 topk_w 归一化到和为 1
void topk_per_token(const float *probs,
                    int T, int E, int K,
                    int *topk_idx,    // [T,K]
                    float *topk_w)    // [T,K]
{
    for (int t = 0; t < T; ++t) {
        const float *pt = probs + t * E;
        int   *idx_t = topk_idx + t * K;
        float *w_t   = topk_w   + t * K;

        // 暴力 K 次 argmax
        float *tmp = (float *)malloc(sizeof(float) * E);
        for (int e = 0; e < E; ++e) tmp[e] = pt[e];

        for (int k = 0; k < K; ++k) {
            int best_e = 0;
            float best_v = -FLT_MAX;
            for (int e = 0; e < E; ++e) {
                if (tmp[e] > best_v) {
                    best_v = tmp[e];
                    best_e = e;
                }
            }
            idx_t[k] = best_e;
            w_t[k]   = best_v;
            tmp[best_e] = -FLT_MAX;
        }
        free(tmp);

        // 归一化 top-k 权重
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += w_t[k];
        float inv = 1.0f / sum;
        for (int k = 0; k < K; ++k) w_t[k] *= inv;
    }
}

// ========== MoE 前向 ==========

void moe_forward(const DeepseekV3MoE *moe,
                 const float *x,   // [T,H]
                 float *y)         // [T,H]
{
    int H = moe->hidden_size;
    int E = moe->num_experts;
    int K = moe->top_k;
    int T = moe->num_tokens;

    float *logits = (float *)malloc(sizeof(float) * T * E);
    float *probs  = (float *)malloc(sizeof(float) * T * E);
    int   *topk_idx = (int *)malloc(sizeof(int) * T * K);
    float *topk_w   = (float *)malloc(sizeof(float) * T * K);
    float *tmp_out  = (float *)malloc(sizeof(float) * H);
    float *shared_out = (float *)malloc(sizeof(float) * H);

    // 1) router logits
    router_forward(&moe->router, x, T, logits);

    // 2) softmax
    softmax_per_token(logits, probs, T, E);

    // 3) top-k
    topk_per_token(probs, T, E, K, topk_idx, topk_w);

    // 4) 计算每个 token 的输出
    for (int t = 0; t < T; ++t) {
        const float *xt = x + t * H;
        float *yt = y + t * H;

        // init y_t = 0
        for (int h = 0; h < H; ++h) yt[h] = 0.0f;

        // routed experts
        for (int k = 0; k < K; ++k) {
            int e_idx = topk_idx[t*K + k];
            float w = topk_w[t*K + k];
            const DeepseekV3MLP *exp = &moe->experts[e_idx];

            mlp_forward(exp, xt, tmp_out);
            for (int h = 0; h < H; ++h) {
                yt[h] += w * tmp_out[h];
            }
        }

        // shared expert
        if (moe->has_shared) {
            mlp_forward(&moe->shared, xt, shared_out);
            for (int h = 0; h < H; ++h) {
                yt[h] += shared_out[h];
            }
        }
    }

    free(logits);
    free(probs);
    free(topk_idx);
    free(topk_w);
    free(tmp_out);
    free(shared_out);
}

// ========== main: 从 txt 读入，跑 MoE，输出 c_out.txt ==========

int main(void)
{
    printf("Running full DeepseekV3MoE C implementation\n");

    // ---- 读取配置 ----
    int H, I, E, K, HAS_SHARED, T;
    {
        FILE *f = fopen("moe_config.txt", "r");
        if (!f) {
            fprintf(stderr, "Failed to open moe_config.txt\n");
            return 1;
        }
        if (fscanf(f, "%d %d %d %d %d %d", &H, &I, &E, &K, &HAS_SHARED, &T) != 6) {
            fprintf(stderr, "Failed to read moe_config.txt\n");
            fclose(f);
            return 1;
        }
        fclose(f);
    }

    printf("Config: H=%d I=%d E=%d K=%d HAS_SHARED=%d T=%d\n", H, I, E, K, HAS_SHARED, T);

    DeepseekV3MoE moe;
    moe.hidden_size = H;
    moe.intermediate_size = I;
    moe.num_experts = E;
    moe.top_k = K;
    moe.has_shared = HAS_SHARED;
    moe.num_tokens = T;

    // ---- 初始化 router ----
    moe.router.hidden_size = H;
    moe.router.num_experts = E;
    moe.router.w = (float *)malloc(sizeof(float) * E * H);
    load_floats("moe_router_weights.txt", moe.router.w, E * H);

    // ---- 初始化 experts ----
    moe.experts = (DeepseekV3MLP *)malloc(sizeof(DeepseekV3MLP) * E);

    for (int e = 0; e < E; ++e) {
        DeepseekV3MLP *m = &moe.experts[e];
        m->hidden_size = H;
        m->intermediate_size = I;
        m->w_gate = (float *)malloc(sizeof(float) * I * H);
        m->w_up   = (float *)malloc(sizeof(float) * I * H);
        m->w_down = (float *)malloc(sizeof(float) * H * I);

        char path[256];
        sprintf(path, "moe_expert%d_w_gate.txt", e);
        load_floats(path, m->w_gate, I * H);

        sprintf(path, "moe_expert%d_w_up.txt", e);
        load_floats(path, m->w_up, I * H);

        sprintf(path, "moe_expert%d_w_down.txt", e);
        load_floats(path, m->w_down, H * I);
    }

    // ---- shared expert ----
    if (HAS_SHARED) {
        moe.shared.hidden_size = H;
        moe.shared.intermediate_size = I;
        moe.shared.w_gate = (float *)malloc(sizeof(float) * I * H);
        moe.shared.w_up   = (float *)malloc(sizeof(float) * I * H);
        moe.shared.w_down = (float *)malloc(sizeof(float) * H * I);

        load_floats("moe_shared_w_gate.txt", moe.shared.w_gate, I*H);
        load_floats("moe_shared_w_up.txt",   moe.shared.w_up,   I*H);
        load_floats("moe_shared_w_down.txt", moe.shared.w_down, H*I);
    }

    // ---- 输入 tokens ----
    float *x = (float *)malloc(sizeof(float) * T * H);
    load_floats("moe_input.txt", x, T * H);

    // ---- 输出缓冲 ----
    float *y = (float *)malloc(sizeof(float) * T * H);

    // ---- MoE 前向 ----
    moe_forward(&moe, x, y);

    // ---- 写出结果 ----
    FILE *fout = fopen("c_out.txt", "w");
    if (!fout) {
        fprintf(stderr, "Failed to open c_out.txt for write\n");
        return 1;
    }
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            fprintf(fout, "%.9f", y[t*H + h]);
            if (h + 1 < H) fprintf(fout, " ");
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    printf("C MoE output written to c_out.txt\n");

    // ---- 简单打印前几个值 ----
    printf("First token output: ");
    for (int h = 0; h < H; ++h) {
        printf("%.7f ", y[h]);
    }
    printf("\n");

    // 内存释放（可选）
    free(moe.router.w);
    for (int e = 0; e < E; ++e) {
        free(moe.experts[e].w_gate);
        free(moe.experts[e].w_up);
        free(moe.experts[e].w_down);
    }
    free(moe.experts);
    if (HAS_SHARED) {
        free(moe.shared.w_gate);
        free(moe.shared.w_up);
        free(moe.shared.w_down);
    }
    free(x);
    free(y);

    return 0;
}
