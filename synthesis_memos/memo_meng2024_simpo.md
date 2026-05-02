# Synthesis Memo — SimPO: Simple Preference Optimization with a Reference-Free Reward

**Paper:** Meng, Y., Xia, M., and Chen, D. "SimPO: Simple Preference Optimization with a Reference-Free Reward." *NeurIPS 2024*.
**Author:** Mistire Daniel | **Date:** 2026-05-01

---

## Core Claim

SimPO improves on DPO by removing the reference model and normalizing reward by sequence length. The reference-free formulation drops VRAM by ~50% versus DPO and removes the main hyperparameter sensitivity that makes DPO fragile at small scale. The length normalization prevents a known DPO failure mode: rewarding shorter responses regardless of quality because length-biased log-probabilities dominate the loss.

The margin term (γ) adds a target gap between chosen and rejected reward. Without it, training is satisfied as soon as chosen > rejected by any margin, which can push the model toward minimal-effort discrimination rather than strong policy alignment.

## Why SimPO Over DPO and ORPO for Tenacious-Bench

**DPO** requires a frozen reference model at training time. On a free Colab T4 (15 GB), loading Qwen 2.5 0.5B twice plus the LoRA adapter and optimizer states exceeds available VRAM. More critically, DPO's reference-model term is calibrated to the KL divergence between the trained policy and the *initial* policy — which makes sense for a generation model but is less well-motivated for a judge/critic that needs to distinguish compliant from non-compliant outreach regardless of what the base model prefers.

**ORPO** is also reference-free but does not normalize by sequence length. Tenacious outreach emails range from 40 to 120 words body. Under ORPO, a 40-word compliant email and a 120-word compliant email are penalized differently for the same policy, because longer sequences accumulate lower average log-probability. This would bias the judge toward short outputs regardless of rubric compliance — a systematic distortion on exactly the constraint (body ≤ 120 words) we care most about enforcing.

**SimPO** is reference-free (fits T4) and length-normalized (handles variable-length outreach correctly). The margin term (γ=0.5 in our run) was set conservatively given the small training set (102 pairs); a larger margin would be appropriate with 500+ pairs.

## Specific Disagreement

The paper's ablation (Table 4) shows SimPO outperforming DPO on AlpacaEval 2 and Arena-Hard across Llama and Mistral backbones. However, all ablated backbones are 7B–70B. The paper makes no claim about 0.5B scale, and the results do not transfer directly. At 0.5B, the binding constraint is generation capacity, not preference signal quality — the backbone struggles to follow multi-constraint instructions regardless of how well the preference signal is shaped. SimPO's advantage over DPO narrows (and may disappear) when the backbone is below the threshold where it can reliably represent the target behavior.

This matches our Delta A = 0.000 finding: the SimPO preference signal was correctly formatted and non-degenerate (training loss 0.6741, converging), but the backbone could not express the policy being trained. The paper's evaluation design does not expose this failure mode because it never tests sub-1B backbones.

**Implication for Tenacious-Bench v0.2:** Repeat the SimPO run on Qwen 2.5 7B. The preference signal is correctly constructed — the bottleneck is purely backbone scale. A 7B run with the same 102 pairs would directly test SimPO's claim at a scale where generation capacity is not the binding constraint.

## Design Choice Applied

γ=0.5, β=2.0 were used in the Tenacious-Bench training run. The paper's recommended γ=0.5 for small datasets is directly cited. The β=2.0 follows the paper's Table 3 best-performing configuration for 7B models — we retained it for 0.5B as a conservative starting point, not because it was specifically calibrated for sub-1B scale.
