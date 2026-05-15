"""Minimal example: add Pulse to any HuggingFace DPO/SFT training."""

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl import DPOTrainer, DPOConfig
from covec_pulse import PulseCallback

# ---- Your existing training setup ----
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
# trainer = DPOTrainer(model=model, args=DPOConfig(...), ...)

# ---- Add Pulse: 1 line ----
# trainer.add_callback(PulseCallback(log_every=50))
# trainer.train()

# ---- That's it. Check pulse_outputs/pulse_probe.jsonl ----

# ---- Or use the low-level API in any training loop ----
import torch

from covec_pulse import probe_optimizer

model = torch.nn.Linear(64, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for name, p in model.named_parameters():
    p._param_name = f"model.layers.0.self_attn.q_proj.{name}"

for step in range(1, 101):
    loss = model(torch.randn(8, 64)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        snapshot = probe_optimizer(optimizer, step=step)
        if snapshot:
            print(f"step {step:3d}  HM/AM={snapshot.hm_am_ratio:.6e}  AM={snapshot.am:.4e}")
