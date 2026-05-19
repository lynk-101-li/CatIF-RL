# GRPO algorithm walkthrough

Detailed companion to manuscript §2.5 (CatIF-RL Optimization by KL-Regularized
GRPO), SI Table S3 (hyperparameters), and Algorithm S2 (full pseudocode).

## Two-level loop structure

CatIF-RL refines the supervised CatIF policy through three outer rounds of
sample -> score -> train, with an inner KL-regularized GRPO update at each
round:

```
for round k = 1, 2, 3:
    Σ_k = sample G=5 candidates per (enzyme, substrate) from the previous policy
    score Σ_k with DLKcat / UniKP / CataPro, normalize via frozen 10/90 quantile range
    D_k = {(enzyme, substrate, sigma, S_ensemble)}        # offline scored CSV
    policy_k = GRPO_train(previous_policy, ref=CatIF, D_k, inner_epochs=2)
```

The reference policy `pi_ref` is the original CatIF supervised checkpoint
(Sep24, epoch 228) frozen across all three rounds. The KL term anchors the
optimized policy to that fold-compatible distribution while letting the
sampling mass shift toward higher predicted catalytic activity.

## Inner loop highlights (SI Algorithm S2)

- **Group-relative advantages**: rewards within a group of G candidates per
  backbone are normalized by the within-group mean and standard deviation.
  This decouples optimization from global reward calibration.
- **Length-normalized log-prob**: the policy-gradient term uses the
  token-mean log-prob `mean_logp = sum_t log pi(a_t) / L`, avoiding length-
  induced gradient bias.
- **Clipped MSE KL proxy**: KL is estimated by the squared difference of the
  per-token mean log-probs to the reference, clipped to `c = 5.0`. The
  associated coefficient beta is updated multiplicatively (x1.5 above
  target, x0.7 below) and bounded in `[5e-4, 0.5]`.
- **Mutation-fraction penalty**: `lambda_mut * max(0, mu - mu_0)` activates
  only when the candidate's mutation fraction exceeds the free-mutation
  threshold `mu_0 = 0.30`.
- **Group filtering**: groups with fewer than three distinct reward values
  are dropped; within-group sequence deduplication is enabled.

## Hyperparameter sets (SI Table S3)

All three rounds share an identical optimizer / loss configuration. The only
per-round differences are the training-run length flag (50 for Round 1, 25
for Rounds 2 and 3) and the policy used as the warm-start checkpoint.

Operational inner-epoch budget `E_k = 2` is uniform: the epoch-02 checkpoint
of every round is selected for the next round's sampling and for the final
benchmark sweep.

Hyperparameters live in `catif_rl/config/grpo_round{1,2,3}.yaml`.

## Implementation

`catif_rl.training.grpo` is the canonical implementation. Its CLI matches
the script invocations under `scripts/05a_rl_round1.sh`, `5b`, and `5c`.
