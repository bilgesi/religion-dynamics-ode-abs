# Upstream code lineage

Public project: https://github.com/bilgesi/religion-dynamics-ode-abs

Access: 22 October 2026. Public web/raw source files were inspected directly. A complete Git clone and immutable commit identifier were not obtained; this package does not invent a commit hash. Reproducibility of the revised work rests on the delivered source/configuration/data snapshots and SHA-256 manifest, not a moving upstream branch.

Inspected upstream components included:

- `src/scenarios.py`: actual A/B/C rate settings.
- `src/model/params.py`: parameter representation and disabled context functions.
- `src/ode/ode_system.py`: original S/B/M/P equations and ordering.
- `src/ode/solve_ode.py` and `src/ode/solve_ode_with_seeding.py`: integration and event handling.
- `src/abs/sim_abs.py`: individual-state stochastic implementation.
- `experiments/scenarios/run_scenarios.py`: scheduled introductions.
- Historical-fitting implementation: logistic-capacity variant and the New Zealand observation-anchored breakpoint fit.
- Repository raw Sweden, Turkey and New Zealand series and MIT license.

The revised `code/src/model.py` retains the published compartment ordering and mathematical transition mechanisms but replaces the implementation with a common transition-hazard matrix. The ODE and stochastic process share these hazards. An independent literal-equation test verifies the revised RHS against the written equations. Count-based sampling is valid because the model assumes exchangeable individuals and homogeneous mixing; it is not a network model. The Monte Carlo algorithm is explicitly a finite-step approximation, not an exact continuous-time sampler.

Scenario A retains the actual upstream rate values. Scenario B retains the upstream imposed introduction/adverse-rate-change design, with all events reported and the event solver corrected. Scenario C is deliberately replaced by a documented symmetric coexistence configuration with a local stability check. Time is consistently annual; the old weekly labels are not retained as biological or social timing claims. Historical fitting is revised rather than copying incompatible upstream parameter exports.

The original MIT notice is retained. The new repository layout is intentionally simple (`code/`, `paper/`, `audit/`) and does not preserve obsolete entrypoints that could regenerate inconsistent old figures.
