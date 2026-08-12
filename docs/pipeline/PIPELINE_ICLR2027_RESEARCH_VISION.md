# CauVid: ICLR 2027 Research Pipeline Review

**Proposed high-level figure:** [PIPELINE_ICLR2027_RESEARCH_VISION.pdf](./PIPELINE_ICLR2027_RESEARCH_VISION.pdf)

## Recommended paper thesis

**Auditable neuro-symbolic causal video understanding from noisy real-world
trajectories.** The paper should center on converting uncertain object-centric
video measurements into validated symbolic event representations, then learning
and testing causal hypotheses through interventions and counterfactuals.

## Strongest implemented modules

| Module | Research value |
|---|---|
| Real-world object trajectories | Detection, tracking, 3D positions, ego-relative motion, and per-signal provenance provide an object-centric basis. |
| Ego-state candidate consensus | Multiple threshold hypotheses, semantic penalties, confidence-weighted evidence, and constrained dynamic programming expose uncertainty rather than hiding it. |
| Symbolic cohort abstraction | Track signals become interpretable cues and rule-conditioned cohorts. |
| Multi-hypothesis trajectory repair | Several hypotheses are evaluated while numerical correction remains deterministic. |
| Hard symbolic validation | Physical validity, observation retention, class consistency, anomaly checks, and validation severity constrain selection. |
| Audit and provenance | Visualizations, dashboards, caches, versioned statistics, and complete selection reasons support reproducibility and falsification. |

The strongest single paper module is the closed loop:

```text
uncertain trajectory → symbolic cohort → multiple repair hypotheses
→ deterministic signal recomputation → hard symbolic validation
→ calibrated selection or unresolved output
```

## Missing modules needed for a stronger causal claim

The accompanying figure shows these modules in gray because they are proposed,
not currently implemented end to end.

| Proposed module | Why it is needed |
|---|---|
| Uncertainty-aware object-centric encoder | Jointly model measurement uncertainty and identity persistence instead of treating perception only as fixed preprocessing. |
| Probabilistic dynamic scene graph | Represent objects, events, relations, confidence, and temporal scope as probabilistic grounded atoms. |
| Temporal structural causal model | Separate ego action, object state, context, and observation noise; explicitly model latent confounding and selection effects. |
| Intervention and counterfactual engine | Support `do(·)` tests, controlled trajectory perturbations, and counterfactual event rollouts rather than relying on association. |
| Differentiable temporal rule learner | Learn weighted executable rules while preserving symbolic constraints and traceable proofs. |
| Causal event and risk heads | Produce useful scene-level predictions with explanations, uncertainty, and abstention. |
| Causal evaluation and falsification suite | Measure intervention consistency, counterfactual validity, OOD generalization, calibration, proof faithfulness, and robustness to tracking/depth corruption. |

## Recommended experimental structure

1. Evaluate perception and repair under controlled tracking, depth, and
   ego-motion corruptions.
2. Compare neural-only, symbolic-only, LLM-only, and full neuro-symbolic models.
3. Test cross-dataset and weather/location/camera shifts.
4. Evaluate causal hypotheses on held-out interventions or carefully defined
   natural experiments.
5. Report task accuracy together with calibration, abstention, rule fidelity,
   counterfactual consistency, and computational cost.
6. Include ablations for cohort rules, repair search, hard validation,
   uncertainty propagation, causal graph learning, and counterfactual training.

## Positioning

The architecture follows the object-centric premise that compositional video
reasoning benefits from persistent object representations, and the
neuro-symbolic premise that probabilistic ground atoms and executable rules make
reasoning traceable. Its causal contribution must be established through an
explicit structural model and intervention/counterfactual evaluation—not by
using “causal” as a synonym for temporal prediction.

Relevant primary references include
[NS-DR](https://openreview.net/forum?id=HkxYzANYDB),
[NS-FR](https://openreview.net/forum?id=UkgBSwjxwe),
[object-centric causal representation learning](https://openreview.net/forum?id=r9FsiXZxZt),
[iCITRIS](https://openreview.net/forum?id=itZ6ggvMnzS), and
[STAR](https://arxiv.org/abs/2405.09711).

## Scope warning

This redesign can make the project substantially more coherent and competitive,
but no architecture diagram can guarantee acceptance. The manuscript will need
a precise learning problem, a technically novel method, strong baselines,
statistically sound experiments, and evidence that the causal variables and
counterfactual claims are identifiable or appropriately limited.
