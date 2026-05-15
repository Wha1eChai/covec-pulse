"""CLI entry point: pulse diagnose <checkpoint_dir>"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "diagnose":
        print("Usage: pulse diagnose <checkpoint_dir> [--json]")
        print("  Analyze a HuggingFace Trainer checkpoint for training health.")
        sys.exit(1)

    checkpoint_dir = Path(sys.argv[2])
    as_json = "--json" in sys.argv

    if not checkpoint_dir.is_dir():
        print(f"Error: {checkpoint_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    from covec_pulse.checkpoint import diagnose_checkpoint

    try:
        snapshot = diagnose_checkpoint(checkpoint_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if snapshot is None:
        print("No data found in checkpoint.")
        sys.exit(1)

    if as_json:
        print(json.dumps(snapshot.to_dict(), indent=2))
    else:
        print("Covec Pulse — Checkpoint Diagnosis")
        print(f"{'=' * 40}")
        print(f"Params analyzed:  {snapshot.n_params:,}")
        print(f"Tensors matched:  {snapshot.n_tensors}")
        print(f"AM:               {snapshot.am:.4e}")
        print(f"HM:               {snapshot.hm:.4e}")
        print(f"HM/AM ratio:      {snapshot.hm_am_ratio:.4e}")
        print()
        if snapshot.hm_am_ratio < 1e-10:
            print("DIAGNOSIS: CRITICAL")
            print("  HM/AM ratio near zero — most parameters are silent.")
            print("  This model is likely suffering from severe capability loss.")
            print("  Recommendation: reduce learning rate or stop training.")
        elif snapshot.hm_am_ratio < 1e-5:
            print("DIAGNOSIS: WARNING")
            print("  HM/AM ratio is low — significant parameter anisotropy.")
            print("  Monitor closely for capability degradation.")
        else:
            print("DIAGNOSIS: OK")
            print("  HM/AM ratio is in a reasonable range.")
            print("  Check downstream eval to confirm training effectiveness.")

        if snapshot.per_layer:
            print()
            print(f"Per-layer breakdown ({len(snapshot.per_layer)} layers):")
            for lid, ls in snapshot.per_layer.items():
                print(f"  {lid:>10s}  HM/AM={ls.hm_am:.4e}  AM={ls.am:.4e}")


if __name__ == "__main__":
    main()
