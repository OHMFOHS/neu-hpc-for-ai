import numpy as np

c = np.loadtxt("c_out.txt")
ref = np.loadtxt("moe_ref.txt")

print("C output shape:", c.shape)
print("Ref shape:", ref.shape)

c_flat = c.reshape(-1)
r_flat = ref.reshape(-1)

print("C output (first 4):", c_flat[:4])
print("Ref      (first 4):", r_flat[:4])
print("Max abs error:", np.max(np.abs(c_flat - r_flat)))
