# Pipeline Forecasting Model

Two-Layer Hybrid Forecasting Model combining Cohort-Based Vintage Analysis with Stage-Weighted Probability Modeling.

## Directory Structure

```
ROOT/
├── data/
│   └── fact_snapshots.csv
├── scripts/
│   ├── generate_forecast.py
│   └── generate_mock.py
├── exports/
│   ├── forecast_2026.csv
│   └── assumptions_log.json
└── validation/
    ├── backtest_results.csv
    └── backtest_summary.json
```

## Usage

Set parameters at top of `generate_forecast.py`, then run:

```python
# In Fabric notebook, override these before calling run_forecast()
GENERATE_MOCK = False
RUN_BACKTEST = True
SCENARIO = 'base'
BACKTEST_DATE = '2025-01-01'
ACTUALS_THROUGH = '2025-12-31'

from scripts.generate_forecast import run_forecast
results = run_forecast()
```

## Architecture

### Layer 1: Cohort-Based Vintage Analysis

Addresses the incomplete cohort problem and establishes expected timing of deal closure.

- Groups historical deals by creation vintage (month/week)
- Tracks cumulative win behavior over time
- Measures how probability of win increases as deals age
- Prevents recent cohorts from appearing artificially weak

**Output**: Lookup table of Cumulative Win Percentage by Week of Age, partitioned by market segment

### Layer 2: Stage-Weighted Probability Modeling

Provides the operational, deal-level forecast for active pipeline.

- Assigns probabilities to deals based on current stage
- Applies penalties for pipeline stagnation (staleness)
- Produces auditable forecast (Deal Value × Probability)

**Output**: Monthly expected revenue derived from open opportunities

### Forecast Calculation Flow

1. **Active Pipeline (Layer 2)**: Identify open deals as of 12/31/2025, apply stage probabilities with staleness penalties
2. **Future Pipeline (Layer 1)**: Generate synthetic deals for Jan-Dec 2026 based on historical volume, apply vintage curves for closure timing

## Key Assumptions

| Assumption | Implementation |
|------------|----------------|
| Skipper Logic | Deals appearing directly as Closed Lost without Qualified/Alignment/Solutioning receive 0.5 weight in volume calculations |
| Right Censoring | Open deals at end of 2025 are projected using vintage maturity curves, not assumed lost |
| Staleness Threshold | 90th percentile of stage duration; deals exceeding this receive 0.8 probability penalty |
| Closed Deal Persistence | Closed deals persist indefinitely; logic identifies first snapshot where closure occurs |

## Scenario Levers

| Lever | Implementation |
|-------|----------------|
| Win Rate Improvement | Multiply stage probabilities by uplift factor |
| Deal Volume Growth | Increase count of future synthetic deals |
| Revenue per Deal | Apply pricing uplift to future deal revenue |

## Output Metrics

Per spec, monthly forecast includes:

| Metric | Description |
|--------|-------------|
| open_pipeline_deal_count | Total open deals |
| open_pipeline_revenue | Total unweighted pipeline value |
| expected_won_deal_count | Probability-weighted deal count |
| expected_won_revenue | Probability-weighted revenue |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| GENERATE_MOCK | bool | False | Run mock data generator first |
| RUN_BACKTEST | bool | False | Run historical validation |
| SCENARIO | str | 'base' | 'base', 'growth', or 'conservative' |
| BACKTEST_DATE | str | '2025-01-01' | Forecast-from date for validation |
| ACTUALS_THROUGH | str | '2025-12-31' | Actuals end date for validation |

## Data Schema

Required columns in `data/fact_snapshots.csv`:

| Column | Type | Description |
|--------|------|-------------|
| deal_id | string | Unique identifier |
| date_created | date | Deal creation date |
| date_closed | date | First snapshot where deal closed (null if open) |
| date_snapshot | date | Weekly snapshot date |
| stage | string | Pipeline stage at snapshot |
| net_revenue | float | Deal value |
| market_segment | string | Large Market, Mid Market/SMB, Indirect |