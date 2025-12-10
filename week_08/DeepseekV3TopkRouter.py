
import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

SEED_WEIGHTS = 1234
SEED_INPUTS = 4321

# Simple config
@dataclass
class Config:
    hidden_size: int = 7168
    n_routed_experts: int = 256

class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


def tolist(t, decimals=7):
    a = t.detach().cpu().numpy().astype(np.float32)
    return np.around(a, decimals=decimals).tolist()

def main():
    torch.manual_seed(SEED_WEIGHTS)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    config = Config(
        hidden_size = 2,
        n_routed_experts = 16,
    )
    router = DeepseekV3TopkRouter(config).eval()
    # init the library tensor without editing its class
    with torch.no_grad():
        #  Kaiming uniform (matches common Linear defaults)
        nn.init.kaiming_uniform_(router.weight, a=math.sqrt(5))

    weights = {
        "weight": tolist(router.weight, decimals=7),  # [intermediate, hidden]
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
        router_logits = router(x)
        tests.append({
            "input_shape": [b, s, h],
            "output_shape": list(router_logits.shape),
            "input": tolist(x, decimals=7),
            "output": tolist(router_logits, decimals=7),
        })

    # Package
    payload = {
        "config": {
            "hidden_size": config.hidden_size,
            "n_routed_experts": config.n_routed_experts,
        },
        "weights": weights,
        "tests": tests,
    }

    # Write and print
    out_path = "deepseek_v3_router_testcases.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False,  indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()