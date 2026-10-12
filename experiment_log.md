# Experiment log: mask-free fusion control

MICE re-implementation on FLUX.2-klein-4B (2026-10-10 to 2026-10-12)

## Evaluation axes

Different methods fail on different axes, so every result is scored on all five.

| Axis | Meaning |
|---|---|
| **Separation** | No object fusion between neighbouring instances |
| **Disentanglement** | No semantic or attribute leakage between instances (SDL) |
| **Harmonization** | Composition, lighting and DoF fit the scene; no pasted-in look |
| **Fidelity** | Each instance becomes the requested target |
| **Quality** | No artifacts, colour patches or "fever dream" look |

## Reference points

| Method | Separation | Disentanglement | Harmonization | Notes |
|---|---|---|---|---|
| **MICE** (masked latent self-attention + text binding) | ✓✓ | ✓ | ✗ (worst in dense scenes) | The method to beat |
| **`--free_latent`** (MICE text binding, latent attention open) | ✗ | ~ | ✓ | The baseline all steering experiments start from |
| **Colour-point method** | ✓✓ (no merging at all) | ✗ (semantic leakage everywhere) | ✓ | Wins if you only score fusion + harmonization; its latents end up very different per instance |

## Experiments (all on `--free_latent`)

### 1. Latent drift

Files: `flux2/instance_drift.py`, `infer_flux2_mice_drift.py`

Push each instance's x_t away from its neighbours between steps.

| Variant | Result |
|---|---|
| Uniform push, no similarity gate | Instances become flat colour patches: a uniform latent shift reads as colour |
| Similarity gate | "Helps" mostly by shrinking the push |
| Per-token noise / direction jitter (up to 150°) | Only the shared component matters: jitter 150° at strength 1 ≈ no jitter at strength 0.2 |
| Initial-noise drift | Breaks the image. Region means of pure noise are ≈ 0, so any offset is out of distribution |
| Push + release | The release pulls the object back to its original shape and doesn't fix the colour |
| **Best** | Strength ≈ 0.2 gives ~40% of MICE's anti-fusion, but the image is foggy and colourful |

**Takeaway:** every invented-drift knob reduces to "strength". The model absorbs per-token randomness.

### 2. Flow-level neighbour-orthogonal guidance

Files: `flux2/neighbor_ortho.py`, `infer_flux2_mice_ortho.py`

Edit the predicted clean image x̂₁: remove the neighbour-like part (α) and amplify the rest (β).

| Variant | Result |
|---|---|
| α | Slightly better separation than drift; removes colour (as APG predicts) |
| Raising α | Separation plateaus |
| β | CFG-like, high-contrast "vintage" look |
| Excluding steps one at a time | Colour loss comes mostly from steps 2–3 |

### 3. In-model block steering

Files: `flux2/block_ortho.py`, `infer_flux2_mice_block_ortho.py`, `summarize_block_ortho.py`

The same α/β split, applied to each transformer block's write (h_out − h_in) on the instance tokens. MICE's text binding stays on.

| Variant | Result |
|---|---|
| α = 0.1, all steps | Separation slightly better than flow level; no obvious artifacts |
| α = 0.3 | The colour shift returns |
| α in general | Perturbs the image less, but separation is limited and **fidelity suffers**: the edit drifts away from the target |
| β in general | Separates more, but distorts, harmonizes worse and leaves colour patches; the patch-like edit look remains |
| Later steps only | Colour, noise and nonsense content appear, so colour and detail form late |
| Step 0 only | Separation saturates at α ≈ 0.3; strong β still shifts colour |
| Steps 0,1 | Higher ceiling; α = 0.9 separates more but gives a "fever dream" look |
| Steps 0,1,2 | No better than steps 0,1, so outlines are decided early |
| `power` (σ decay) | Almost no effect on klein at 4 steps, because σ ≈ 1 until the last step |
| **α by block** (steps 0,1, α = 0.9) | All blocks ≈ excluding singles 10–19 ≈ excluding singles 5–19. Excluding all singles gives much worse separation. Fidelity improves slightly as more late singles are excluded |
| **β by block** (α = 0) | Steering only the double blocks keeps most of the separation and removes most of the fidelity and colour-patch damage |
| **Local pair weights** (`--ortho_local_tau`, α = 0.9, β = 0.5) | Each token is steered only from the neighbour it is near. Visibly better image quality than global weights at the same α/β; separation unchanged (no gain, no loss) |
| **Overall** | Much better than flow level, but **not close to MICE's separation**. Harmonization against MICE is mixed, case by case. With MICE's text binding on, disentanglement is better than the colour-point method's |

## Established facts

- **Schedule (klein, 4 steps, 1024²):** σ = 1.000 → 0.967 → 0.908 → 0.767 → 0, so each step moves the latent by Δσ = 0.033, 0.059, 0.141, 0.767. The last step does 77% of the trajectory.
- **When:** outlines and separation are decided in steps 0–1. Colour and detail form late.
- **Where:** the blocks that matter for separation are the doubles and singles 0–4. Late singles 5–19 add no separation. This matches the attention observation that late singles attend within the instance.
- **Damage axes:** α damages fidelity, β damages quality, masking damages harmonization.
- **Tokens:** one token is a 2×2 block of the 32-channel VAE latent, so 128 channels covering 16×16 pixels. Masks are mapped to the token grid by nearest interpolation, so boundary tokens hold both instances. An earlier finding: a mask margin improved quality, because the ring tokens contain the instance.

## Open questions

1. Why does the colour-point method never fuse, yet leak semantics everywhere? Could one cause explain both?
2. Separation and disentanglement look like they come from different mechanisms: colour-point gives separation, while cross-attention (text) masking gives disentanglement. Is that a real split?
3. Where does harmonization form, in steps and in blocks? Fusion is mapped; harmony isn't.
4. Untested baseline: MICE masking only at steps 0,1 (`--masking_steps 0,1`).

## Parked

- **PDE attention-energy transport** (`transport_energy_compare.py`): costly, and the role of the three modalities (text, generated image, source image) is unresolved.
- **Anti-fusion guidance:** proposed, not yet discussed.
