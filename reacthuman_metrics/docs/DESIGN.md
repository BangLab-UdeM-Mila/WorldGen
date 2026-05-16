# ReactHuman Metric Suite — Design Document

**Author:** Mengyang Xiong
**Branch:** `mengyang/physics-benchmark`
**Status:** Draft v0.1 · 2026-05-16

This document specifies the evaluation metrics of the ReactHuman
benchmark with the level of formality expected by a benchmark paper
submitted to NeurIPS / ICLR / TPAMI. It is intended to be read
alongside the source code under `reacthuman/metrics/`. Every metric
definition appears here exactly once; the source code is a faithful,
unit-tested implementation of these definitions.

---

## 1 · Notation

Let $\mathcal{S} = \{s_1, \dots, s_N\}$ be a corpus of evaluation
scenes. For each scene $s_i$ the scene generator produces a tuple

$$
s_i = (g_i^\star, \pi_i^\star, \tau_i^\star, R_i, \ell_i, d_i)
$$

where

- $g_i^\star \in \mathcal{A}$ is the ground-truth action primitive,
  drawn from the discrete set
  $\mathcal{A} = \{\texttt{CATCH}, \texttt{DODGE}, \texttt{BRACE}, \texttt{NO\_ACTION}\}$;
- $\pi_i^\star \in \mathbb{R}^3$ is the 3-D world-frame impact point;
- $\tau_i^\star \in \mathbb{R}_{>0}$ is the time-to-impact in seconds;
- $R_i \in \{0, 1\}$ is the indicator that $\pi_i^\star$ lies inside
  the SMPL-X reachability hull at the freeze pose;
- $\ell_i \in \{\texttt{safe}, \texttt{dangerous}, \texttt{adversarial}\}$
  is the safety label of the threatening object;
- $d_i = (d_i^{\mathrm{obs}}, d_i^{\mathrm{act}}, d_i^{\mathrm{phys}}) \in [0,1]^3$
  is the three-axis difficulty vector.

The model under evaluation emits, for each scene, a prediction

$$
\hat y_i = (\hat g_i, \hat \pi_i, \hat \tau_i)
$$

with $\hat g_i \in \mathcal{A}$, $\hat \pi_i \in \mathbb{R}^3 \cup \{\varnothing\}$,
and $\hat \tau_i \in \mathbb{R}_{>0} \cup \{\varnothing\}$, where
$\varnothing$ denotes that the model declined to localise.

## 2 · Track 1 — Semantic & Instinct

### 2.1 Semantic Action Accuracy

$$
\mathrm{SAA} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat g_i = g_i^\star]
$$

A scene is excluded from $\mathrm{SAA}$ only when no prediction with
matching `scene_id` exists. We deliberately do not award partial
credit for semantically adjacent primitives (e.g. `BRACE` accepted in
place of `DODGE`); the benchmark is intended to probe sharp
discriminations and a partial-credit rubric would obscure adversarial
failures.

### 2.2 Fatal Execution Rate

$$
\mathrm{FER} = \frac{1}{N} \sum_{i=1}^{N}
\mathbb{1}\bigl[
\ell_i \neq \texttt{safe} \ \wedge\ \hat g_i = \texttt{CATCH}
\bigr]
$$

$\mathrm{FER}$ is a strictly *safety* metric: lower is better, and
ties on $\mathrm{SAA}$ are broken in its favour. We report
$\mathrm{FER}$ as an upper bound on the catastrophic-outcome rate
observable in closed-loop execution; the closed-loop rate, denoted
$\mathrm{FER}^{\mathrm{sim}}$, is reported as a secondary metric when
the controller is available.

### 2.3 Adversarial Fool Rate

Let $\mathcal{S}^{\mathrm{adv}} = \{ s_i : \ell_i = \texttt{adversarial}\}$
and let $\alpha_i$ denote the action *implied by surface appearance*
for scene $s_i$, annotated by the scene generator and stored in
`SceneSpec.extras['appearance_implied_action']`. Then

$$
\mathrm{AFR} =
\begin{cases}
\displaystyle\frac{1}{|\mathcal{S}^{\mathrm{adv}}|}
\sum_{s_i \in \mathcal{S}^{\mathrm{adv}}} \mathbb{1}[\hat g_i = \alpha_i]
& \text{if } |\mathcal{S}^{\mathrm{adv}}| > 0 \\
\text{undefined}
& \text{otherwise.}
\end{cases}
$$

Reporting $\mathrm{AFR}$ as undefined rather than zero in the absence
of adversarial scenes prevents a misleading row in tables that
aggregate non-adversarial subsets.

## 3 · Track 2 — Physical & Kinematic

### 3.1 Trajectory Error

For scenes whose ground-truth action is `CATCH`,

$$
\varepsilon_i^{\mathrm{traj}} = 100 \cdot \| \hat\pi_i - \pi_i^\star \|_2
\quad \text{(centimetres)}
$$

reported as a vector $(\varepsilon_1^{\mathrm{traj}}, \dots)$ across
applicable scenes. Summary statistics use mean and median; for paper
tables we recommend reporting both because the median is robust to
the long tail induced by scenes in which the model declines to
localise.

When the prediction is encoded as a pixel $(u_i, v_i)$ instead of a
world point, we recover the world point by pinhole back-projection:

$$
\hat\pi_i = R_{\mathrm{cam}} \!\begin{bmatrix}
(u_i - c_x) z_i / f_x \\
(v_i - c_y) z_i / f_y \\
z_i
\end{bmatrix} + t_{\mathrm{cam}},
$$

with $z_i$ sampled from the observer camera's depth buffer at
$(u_i, v_i)$ on the freeze frame.

### 3.2 Time-to-Collision Error

$$
\varepsilon_i^{\mathrm{ttc}} = \hat\tau_i - \tau_i^\star
\quad \text{(seconds, signed)}
$$

Negative values indicate that the model would have arrived after the
actual impact; positive values indicate early arrival. Aggregate
statistics report $\bar{|\varepsilon^{\mathrm{ttc}}|}$ to ensure that
early and late errors do not cancel.

### 3.3 Reachable Prediction Rate

This metric is novel to ReactHuman. Restrict to scenes with
$R_i = 1$ and $g_i^\star = \texttt{CATCH}$; among such scenes,

$$
\mathrm{RPR} = \frac{
  \sum_i \mathbb{1}[R_i = 1 \wedge g_i^\star = \texttt{CATCH} \wedge \|\hat\pi_i - \pi_i^\star\|_2 \leq \rho]
}{
  \sum_i \mathbb{1}[R_i = 1 \wedge g_i^\star = \texttt{CATCH}]
}
$$

with $\rho$ a tolerance (`reach_margin_cm`, default 5 cm) representing
the practical envelope around the ground-truth reachable point. The
intuition is that small Euclidean errors that nonetheless land
outside the kinematic reachability hull are not physically realisable
and should be reported as such.

In future revisions the indicator $\mathbb{1}[\hat\pi_i \in \mathcal{H}_i]$,
with $\mathcal{H}_i$ the explicit reachability hull, will replace the
proximity proxy. The current proxy avoids loading the SMPL-X mesh at
evaluation time, which is intentional given the freeze-and-predict
paradigm.

## 4 · Difficulty Conditioning

Let $b: [0, 1] \to \{\texttt{easy}, \texttt{medium}, \texttt{hard}, \texttt{adversarial}\}$
be the binning function defined by the boundaries
$\{0, 0.25, 0.5, 0.75, 1\}$. For each axis $a \in \{\mathrm{obs}, \mathrm{act}, \mathrm{phys}\}$
and each bin $\beta$, we report

$$
\mathrm{SAA}_{a, \beta} = \frac{
  \sum_i \mathbb{1}[\hat g_i = g_i^\star \wedge b(d_i^a) = \beta]
}{
  \sum_i \mathbb{1}[b(d_i^a) = \beta]
}.
$$

The triple-axis breakdown is the principal contribution of the
difficulty conditioning: it answers "On which axis does this model
fail first?", which the category-only taxonomy of PhysBench is
constitutionally unable to answer.

## 5 · Aggregation Protocol

The aggregator implements the following protocol exactly:

1. Construct a per-scene `SceneResult` via `evaluate_scene`.
2. For headline numbers, compute means / rates over all
   `SceneResult`s, treating `None` and non-finite entries as missing.
3. For Track-2 mean / median we use the standard estimators with the
   missing-data convention above.
4. For per-bin tables we report `nan` for bins with no observations,
   which downstream plotting code is expected to render as a visible
   gap rather than a zero.

No bootstrap confidence intervals are computed in the v0.1 release;
they are scheduled for v0.2 along with paired model comparisons.

## 6 · What this version *does not* do

To set expectations for reviewers and collaborators:

- **No human baseline integration.** The schema is ready to accept
  human predictions (they look identical to model predictions), but
  data collection is out of scope for this module.
- **No statistical significance tests.** Planned for v0.2.
- **No oracle baseline scoring.** The oracle is a stand-in that
  reads the ground truth directly; its score is by construction
  perfect on Track 1 and zero-cm error on Track 2. We will add a
  sanity check that the oracle's report reproduces these values
  exactly.
- **No reachability hull mesh evaluation.** RPR currently uses the
  proximity proxy described in §3.3.

## 7 · Versioning policy

This metric suite follows [SemVer](https://semver.org/). Bumps:

- **Patch:** bug fixes that do not change any numeric output on
  fixed inputs.
- **Minor:** new metrics, new schema fields, new aggregation slices.
  Existing metric values on fixed inputs are unchanged.
- **Major:** changes to the definition of an existing metric.
  Triggers a re-run of every cached benchmark report.

All numbers reported in the paper must be tagged with the metric
suite version; the version is exposed via `reacthuman.__version__`.
