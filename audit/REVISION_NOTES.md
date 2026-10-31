# Revision notes and editor-comment resolution

## Scope and retained material

The title, authors, general introduction, historical framing, and most of Related Work are retained. Introductory contribution statements, claims about explicit political effects, results, Methods, and corresponding Discussion conclusions were changed wherever they were not supported. The mathematical compartment structure is retained; its terminology, rates, demographic interpretation, and relationship to the empirical reduction are clarified. The bibliography preserves the supplied library and citation keys wherever possible, including unused entries, for compatibility. The DOI/published version of the religion-diversification paper and the missing physics-informed-neural-network reference were repaired; clearly misapplied references were moved or removed from the particular claim rather than cited as false support.

This is a scientific reconciliation, not a cosmetic rescue of the old numerical results. No outcomes were selected for being more favorable. The executed output, not the earlier reported value, is authoritative.

## Comment-by-comment changes

| Concern | Resolution |
|---|---|
| Text and Table 1 use different scenario parameters | Complete executable rate tables, initial counts, demographic settings, mutation matrices, horizons and event schedules now accompany the revised figures. |
| Scenario B has no mutation although absent groups supposedly emerge through mutation | Scenario B is now explicitly an externally introduced, externally forced transient experiment. Its five introductions and five adverse-rate changes are disclosed. No-introduction and no-rate-change ablations establish what causes the displayed waves. |
| Scenario C is not the symmetric coexistence experiment described | Replaced with equal behavioral rates, unequal initial prevalence and symmetric mutation 0.02/year. The coexistence equilibrium and all six constrained Jacobian eigenvalues were calculated. Weak mutation is shown separately to demonstrate that symmetry alone is insufficient. |
| Inconsistent time units and initial populations | All rates and horizons now use years; N0=10,000 in all stylized configurations. A/B run 216 years and C runs 1000 for mathematical relaxation. These are not calibrated historical times. |
| New Zealand observations described as rise-and-fall | Corrected to monotonically increasing no-religion reporting. Counts and denominators are tabulated explicitly. No fitted breakpoint and no observed-state reset. |
| Historical parameters incompatible with trajectories | Refit all curves, export complete parameter lists, use exact continuous propagation, and regenerate the figures from saved curves. Recruitment/loss estimates at nonnegative bounds are marked and not interpreted causally. |
| Comparison sample size, mutation inclusion and error ranges contradict Figure 5 | Exactly 30 configurations (10 per scenario), 20 replicates each, with A/C mutation retained and B forcing retained. Figure and summary are generated from the same saved case-level metrics. |
| Historical fitting does not validate proposed multi-religion model | Explicitly acknowledged in abstract, introduction, model-reduction derivation, Methods, Results and Discussion. These are reduced descriptive illustrations. Full-model empirical calibration is not claimed or fabricated. |
| Social/political dependence unspecified | Positive log-linear covariate link is specified and implemented as an optional input; no empirical covariate estimation or causal political effect is claimed. |
| Other solver inconsistency | ODE restarts now propagate the evolved endpoint at every scheduled introduction/rate change. Stochastic updates use competing hazards to avoid simultaneous incompatible events. Both share the same transition generator. |
| Phase-transition overclaim | Replaced categorical thresholds with a continuous deterministic finite-horizon map and saved both 216- and 432-year outputs. Substantial horizon dependence is explicitly reported. |
| Predictive overclaim and overfitting | Added fixed temporal holdouts, anchored linear/exponential baselines, a complexity diagnostic, and exclusion of the fixed initial point from fit metrics. Turkey and NZ linear trends outperform the selected diffusion curves on their holdouts; this is reported. |

## Additional historical-data correction

Turkey's upstream average-publisher column uses peak counts in 2017 and 2018. The revised series uses the peak column throughout, with a clearly defined peak-publisher/population observable. It is not treated as total affiliation. Primary-source checking coverage is documented rather than overstated.

For New Zealand the main observable is no-religion count divided by the all-resident census population. This differs from published valid-response/processed percentages in some years. Both series are retained separately, not mixed. The primary complement includes missing or unclassified affiliation, so it is not interpreted as an exclusively religious compartment. Census comparability remains a real data limitation. Random-rounding differences of three persons are documented.

## Numerical findings to retain

The actual executed results are in `code/results/historical_metrics.csv` and `simulation_summary.json`. Principal figures: Sweden rate-shift R2 about 0.996 (rather than the old inconsistent value); Turkey logistic R2 about 0.918; NZ primary-series R2 about 0.968. These are conditional in-sample fit statistics, excluding initialization, not mechanism-validation statistics. Holdout tables use training-only fits, not full-data curves. Scenario B peaks coincide with the imposed changes by design. Scenario C's protected-role equilibrium fraction is large and is explicitly not treated as realistic clergy prevalence.

## Verification performed

- Executed all main experiments, ablations, the 441-cell map at two horizons, 30 comparison configurations and step-size checks.
- Ran 17 regression tests, including literal-equation agreement, mass balance, non-negativity, absent-group invariance, event continuity, seed conservation, stochastic repeatability, and continuous reduced-model propagation.
- Generated numerical LaTeX assets from saved outputs.
- Checked all citation keys resolve, compiled PDFLaTeX/BibTeX, and visually inspected rendered manuscript/figure pages.
- Included configurations, data provenance, per-replicate seeds, environment versions, numerical outputs and source hashes.

## Limits that must not be edited away

The revised paper is not a full empirical calibration of the multi-religion system, does not identify social/political causal effects, does not establish a bifurcation from the heatmap, and does not show general predictive superiority. The forcing schedule in B, finite-population/finite-step approximation, toy role composition in C, and sparse heterogeneous historical measurements are stated limitations. The case studies are illustrative, not preregistered prospective validations.

Authors should review scientific framing, their institutional/author declarations, and the journal's submission requirements before submitting. The supplied bibliography was not independently verified entry-by-entry beyond the targeted corrections and data-source checks described above.
