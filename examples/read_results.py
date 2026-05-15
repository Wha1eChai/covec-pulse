"""Read and summarize probe output after training."""

from covec_pulse.io import read_probe_jsonl, summary

records = read_probe_jsonl("pulse_outputs/pulse_probe.jsonl")
stats = summary(records)

print(f"Steps: {stats['step_range'][0]} - {stats['step_range'][1]}")
print(f"Records: {stats['n_records']}")
print(f"Params tracked: {stats['n_params']:,}")
print(f"HM/AM initial: {stats['hm_am_ratio_initial']:.4e}")
print(f"HM/AM final:   {stats['hm_am_ratio_final']:.4e}")
print(f"Trend: {stats['hm_am_trend']}")

if stats["hm_am_trend"] == "declining":
    print("\nWARNING: HM/AM dropped >50% — model may be losing pretrained capabilities.")
    print("Consider reducing learning rate or stopping training.")
