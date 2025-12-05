# Generate deterministic test vectors for DeepseekV3MLP
# Produces JSON with config, weights, random inputs, and outputs.

import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

SEED_WEIGHTS = 1234
SEED_INPUTS = 4321

# Minimal ACT2FN
ACT2FN = {
    "silu": torch.nn.functional.silu,
    "relu": torch.relu,
}

# Simple config
@dataclass
class Config:
    hidden_size: int = 7168 # d_model
    intermediate_size: int = 2048 # d_expert
    hidden_act: str = "silu"   # options: silu, gelu, relu


class DeepseekV3MLP(nn.Module):
    def __init__(self, config: Config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# converts tensor to list
def tolist(t, decimals=7):
    a = t.detach().cpu().numpy().astype(np.float32)
    return np.around(a, decimals=decimals).tolist()


def main():
    torch.manual_seed(SEED_WEIGHTS)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    config = Config(hidden_size=2, intermediate_size=4, hidden_act="silu")
    mlp = DeepseekV3MLP(config).eval()

    weights = {
        "gate_proj.weight": tolist(mlp.gate_proj.weight, decimals=7),  # [intermediate, hidden]
        "up_proj.weight":   tolist(mlp.up_proj.weight, decimals=7),    # [intermediate, hidden]
        "down_proj.weight": tolist(mlp.down_proj.weight, decimals=7),  # [hidden, intermediate]
    }

    # Define test shapes: (batch, seq, hidden)
    test_shapes = [
        (1, 1, config.hidden_size),
        (2, 3, config.hidden_size),
        (4, 2, config.hidden_size),
    ]

    # Generate inputs and outputs
    tests = []
    rng = torch.Generator().manual_seed(SEED_INPUTS)
    for (b, s, h) in test_shapes:
        x = torch.randn((b, s, h), generator=rng, dtype=torch.float32)
        y = mlp(x)
        tests.append({
            "input_shape": [b, s, h],
            "output_shape": list(y.shape),
            "input": tolist(x, decimals=7),
            "output": tolist(y, decimals=7),
        })

    # Package
    payload = {
        "config": {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "hidden_act": config.hidden_act,
            "dtype": "float32",
        },
        "weights": weights,
        "tests": tests,
    }

    # Write and print
    out_path = "deepseek_v3_mlp_testcases.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)

    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()