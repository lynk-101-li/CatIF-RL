# Discrete subgraph inpainting (Algorithm S1)

Detailed companion to manuscript §2.6 (Motif-Preserving Partial Design) and
SI Algorithm S1.

## Setting

A binary mask `m` of length `N` selects which residues remain fixed to the
native sequence (`m_i = 1`) and which are designable (`m_i = 0`). The
benchmark case study uses the four catalytic / proton-transfer residues of
*Papaver bracteatum* salutaridine reductase (Asn152, Ser180, Tyr236,
Lys240); see `case_study/EC1.1.1.248_SalR/motif_mask.json`.

## Reverse step (per DDIM jump t -> s)

For each reverse step:

1. Sample a denoised state `x_s_var` from the model's marginalised reverse
   posterior `p_theta(x_s | x_t, G)`.
2. Forward-corrupt the native sequence to step `s` to obtain `x_s_fix`.
3. Fuse the two states under the mask: `x_s = m * x_s_fix + (1 - m) * x_s_var`.

## Iterative fusion (U=5)

CatIF uses a BLOSUM substitution kernel, so the closed-form jump-back
transition matrix `Q_{t|s}` of standard RePaint (Lugmayr et al., 2022) is
not available. In its place, the case-study runs use the BLOSUM iterative-
fusion variant: the reverse-step -> GT-injection step is repeated U = 5
times at each outer step against the unchanged mask. This progressively
refines compatibility between the redesigned region and the surrounding
native context.

## Final step (s = 0)

At the final reverse step, no jump-back is performed (U_t = 1). The
designable positions emit one-hot residues by argmax (when `diverse =
False`) or by categorical sample (when `diverse = True`); fixed positions
remain at their native identity.

## Implementation

`catif_rl.sampling.inpaint` implements the algorithm verbatim. CLI:

```bash
python -m catif_rl.sampling.inpaint \
  --pdb case_study/EC1.1.1.248_SalR/native.pdb \
  --mask 151,179,235,239 \
  --ckpt checkpoints/catif_rl_R3_epoch02.pt \
  --u 5
```

Mask indices are 0-based.
