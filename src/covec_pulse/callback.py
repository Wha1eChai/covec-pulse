"""HuggingFace Trainer callback — 1-line integration for training health monitoring."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from covec_pulse.core import probe_optimizer


class PulseCallback(TrainerCallback):
    """Drop-in callback for DPOTrainer / SFTTrainer / Trainer.

    Usage::

        from covec_pulse import PulseCallback
        trainer = DPOTrainer(model=model, args=args, ...)
        trainer.add_callback(PulseCallback())
        trainer.train()

        # With Scope server:
        trainer.add_callback(PulseCallback(endpoint="https://api.covec.dev/v1/probe"))
    """

    def __init__(
        self,
        log_every: int = 50,
        output_dir: str = "pulse_outputs",
        track_per_layer: bool = False,
        endpoint: str | None = None,
        scope_every: int | None = None,
    ):
        self.log_every = log_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "pulse_probe.jsonl"
        self._names_tagged = False

        self._transport = None
        self._scope_every = scope_every or (log_every * 2)
        if endpoint:
            from covec_pulse.transport import ScopeTransport
            self._transport = ScopeTransport(
                endpoint=endpoint,
                verdict_path=self.output_dir / "scope_verdict.jsonl",
            )

    def _tag_param_names(self, model, optimizer) -> None:
        if model is None or self._names_tagged:
            return
        id_to_name = {id(p): n for n, p in model.named_parameters()}
        for group in optimizer.param_groups:
            for param in group["params"]:
                if id(param) in id_to_name:
                    param._param_name = id_to_name[id(param)]
        self._names_tagged = True

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 0 or state.global_step % self.log_every != 0:
            return

        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        model = kwargs.get("model")
        self._tag_param_names(model, optimizer)

        snapshot = probe_optimizer(optimizer, step=state.global_step)
        if snapshot is None:
            return

        current_loss = None
        if state.log_history:
            current_loss = state.log_history[-1].get("loss")

        record = snapshot.to_dict()
        if current_loss is not None:
            record["train_loss"] = current_loss

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[Pulse] step={state.global_step:5d} "
            f"HM/AM={snapshot.hm_am_ratio:.4e} "
            f"AM={snapshot.am:.4e} HM={snapshot.hm:.4e}"
        )

        if self._transport and state.global_step % self._scope_every == 0:
            self._transport.send(record)
