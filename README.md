# Religious Spread Model (ODE + ABS)

A computational framework for modeling religious dynamics using both Ordinary Differential Equations (ODE) and Agent-Based Simulation (ABS) approaches.

## Abstract

Understanding the spread and decline of religious groups in populations is a fundamental challenge in social dynamics. In this study, we present a hybrid computational framework that integrates deterministic mean-field dynamics (ODE) with stochastic agent-based simulation (ABS) to model multi-strain religious dynamics. The framework enables rapid scenario exploration through ODE simulations and detailed behavior modeling through ABS, allowing researchers to study complex religious spread patterns, phase transitions, and historical data fitting.

## Table of Contents

1. [Code Usage](#code-usage)
2. [The Model](#the-model)
3. [Key Scenarios](#key-scenarios)
4. [Data Preparation](#data-preparation)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [Bug Reports](#bug-reports)
8. [Contact](#contact)

## Code Usage

### Run the experiments shown in the paper:

1. Clone the repository
2. Install the `requirements.txt` file
3. Run experiments from the project root directory:

```bash
# Run all scenario experiments (A, B, C)
python experiments/scenarios/run_scenarios.py

# Generate phase transition heatmap
python experiments/heatmaps/heatmap_abc_abs.py

# Historical data fitting
python experiments/historical_fit/fig_historic_fit.py
```

### Use in your project:

1. Clone the repository
2. Install the `requirements.txt` file
3. Import and use the model components:

```python
from src.scenarios import scenario_A, scenario_B, scenario_C
from src.ode.ode_solver import run_ode
from src.abs.sim_abs import run_abs

# Load a scenario
params, init = scenario_A()

# Run ODE simulation
ode_result = run_ode(params, S0, B0, M0, P0)

# Run ABS simulation
abs_result = run_abs(params, roles, rel_ids, seed=42)
```

### Demo:

Example scripts demonstrating various use cases are available in the `experiments/` directory:
- `experiments/scenarios/` - Scenario runners for A, B, C
- `experiments/historical_fit/` - Historical data fitting examples
- `experiments/heatmaps/` - Phase transition analysis

## The Model

The Religious Spread Model is constructed from two complementary approaches:

1. **ODE Model**: Deterministic mean-field dynamics for rapid scenario exploration and parameter sweeps. Implements a multi-strain compartmental model with compartments S (susceptible), B_r (believers), M_r (missionaries), and P_r (practitioners) for each religion r.

2. **ABS Model**: Stochastic agent-based simulation for detailed behavior modeling and validation. Each agent transitions between compartments based on probabilistic rates, enabling the study of stochastic effects and finite-population dynamics.

### Model Components

- **Conversion Dynamics**: Susceptible individuals (S) convert to religion r via rate λ_r(t) = β_r(t) × M_r / N
- **Role Transitions**: Believers (B_r) and Missionaries (M_r) transition via σ_r and κ_r rates
- **Practice Transitions**: B_r → P_r and M_r → P_r transitions via τB_r and τM_r
- **Disaffiliation**: B_r, M_r, P_r → S transitions via ρB_r, ρM_r, ρP_r
- **Mutation**: Missionaries can mutate between strains via ν matrix (schism dynamics)
- **Demography**: Birth (b) and death (μ) rates applied to all compartments

The framework allows easy insertion of domain knowledge through parameter configuration, scenario definitions, and context-dependent rate modulation.

## Key Scenarios

The framework includes three predefined scenarios representing different religious dynamics regimes:

### Scenario A: Dominant Religion Replacement
- **Description**: Large religion emerging from another large religion
- **Mechanism**: Mutation from parent religion (r=1) to new religion (r=2) on missionaries enables emergence
- **Use Case**: Modeling schisms and major religious splits

### Scenario B: Sequential Cult Invasions
- **Description**: Small cults come and go with piecewise time-varying parameters
- **Mechanism**: Multiple cult strains (r=2..6) seeded at scheduled times, each with pre-crash (high growth) and post-crash (rapid decay) phases
- **Use Case**: Modeling transient religious movements and cult dynamics

### Scenario C: Transient Minority Dynamics
- **Description**: Second religion rises from small seed, reaches peak, then dies out
- **Mechanism**: High initial conversion rate balanced by high disaffiliation rates
- **Use Case**: Modeling short-lived religious movements and failed conversions

## Data Preparation

The data files to be analyzed should be CSV files, with each column containing numerical values of each variable. Historical data fitting scripts expect time series data with columns for:
- Time points
- Population compartments (S, B, M, P) for each religion
- Optional: demographic data (birth rates, death rates)

Processed time series are stored in `data/processed/`, while raw data from different countries (New Zealand, Finland, Sweden, Jehovah's Witnesses from Turkey) are stored in `data/raw/`.

## Dependencies

1. numpy >= 1.24.0
2. scipy >= 1.10.0
3. pandas >= 2.0.0
4. matplotlib >= 3.7.0

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions to this project! Pull requests are very welcome. Please send us an email with your suggestions or requests, or open an issue on GitHub.

## Bug Reports

Report bugs and issues on the GitHub Issues page. We guarantee a reply as fast as we can! :)

## Contact

- **Bilge Taskin** - bilgetaskinn@outlook.com | [LinkedIn](https://www.linkedin.com/in/bilgesi/)
- **Teddy Lazebnik** - teddy.lazebnik@ju.se | [LinkedIn](https://www.linkedin.com/in/teddy-lazebnik/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
