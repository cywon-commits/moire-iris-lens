## Variational model of the near zone (2026-11-03, added)

Ansatz (twist, φm=0):
- Inner disk r < r₁ is rigidly over-twisted by b: u = i(θ+b)z − iC/z̄.
- A commensurate ring sits at r₁.
- Outside: u = iθz − iC_eff/z̄.

Continuity at the ring gives r₁ equal to both Einstein radii, r₁² = |C|/(θ+b) = |C_eff|/θ. Hence

    m_eff = C/κ_in,  κ_in = i(θ+b),  ρ = C_eff/C = θ/(θ+b).

**Kinematic test.** The measured inner over-rotation, as the r-weighted mean of δω over 3.5–9.5, predicts ρ:

| case | inner b/θ | θ/(θ+b) | ρ from node fit |
|---|---|---|---|
| twist, V0 = 0.0025 | 0.84 | 0.54 | 0.57 |
| twist, V0 = 0.005 | 1.21 | 0.45 | 0.43 |
| twist, V0 = 0.01 | 1.60 | 0.38 | 0.31 |
| twist, V0 = 0.02 | 2.00 | 0.33 | 0.22 |
| twist, K/μ = 2 | 1.32 | 0.43 | 0.33 |
| mismatch, using b = δdiv/2 | 0.72–1.35 | 0.58–0.43 | 0.80–0.40 |

The relation holds semi-quantitatively; the box ±72 measurements scatter by about 0.1–0.3.

**Energy.** With elastic cost A = μθ (twist; A = (μ+K)θ in mismatch) and wall tension Γ, where the wall length density is approximated by σ₁ + σ₂ of ∇u:

    E/(2π|C|) = A(1−ρ)²/ρ + Γ(ln ρ + 1 − ρ)

Minimizing gives ρ = A/(Γ − A), with a threshold: ρ = 1 if Γ ≤ 2A.

- ρ does not depend on |C|, consistent with P1.
- Calibrating Γ from the background energy f/(2θ) gives the right trend but too weak screening. The dense near zone is dominated by shear, so the energy density e(G) of a sheared or inverted moiré is needed. This is requested as S0 in `tasks/TASK_screening_scan.md`.
- Expected origin of the twist K trend: mixed-character walls in the inverted zone carry a tensile part, so a stiffer K means larger gain and more screening.
