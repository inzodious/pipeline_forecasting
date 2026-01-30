# Pipeline Forecasting Model (V10)

## Overview
This project implements a **Two-Layer Hybrid Forecasting Model** combining cohort-based vintage analysis (Layer 1) with stage-weighted probability modeling (Layer 2) to forecast closed won revenue for 2026. The model is designed to handle sparse data in high-value segments (Large Market) by blending historical and recent performance metrics to produce a stable, defensible forecast.

## Project Directory
```
pipeline_forecasting/
├── data/
│   ├── fact_snapshots.csv          # Primary data: weekly snapshots of all deals
│   └── 20251226_sample_summary.csv  # Sample/supplementary data
├── docs/
│   ├── forecasting_guidelines.md   # Methodology & data context
│   ├── forecast_model_validation.md # Validation notes (Verbal=Won, backtest)
│   └── Pipeline_Forecasting_Model_V6.pptx
├── scripts/
│   ├── generate_forecast_v10.py    # Main forecast generator (Production)
│   ├── validation/
│   │   └── data_exploration.py     # Assumptions export → assumptions_log
│   └── archive/
│       ├── generate_forecast_v9.py # Previous version
│       ├── generate_forecast_v8.py
│       ├── generate_forecast_v7.py
│       ├── generate_forecast_v6.py
│       └── generate_forecast_v4.py
├── exports/
│   ├── forecast_2026.csv           # Monthly FY26 forecast output
│   ├── assumptions.json            # Config snapshot for auditability
│   └── goal_seek_analysis.csv      # What-if levers for revenue targets
├── validation/
│   ├── backtest_results.csv        # Segment-level backtest vs actuals
│   └── backtest_monthly.csv        # Monthly backtest breakdown
├── assumptions_log/                # Historical assumptions by month/segment
│   ├── volume_by_month.csv
│   ├── win_rates_by_month.csv
│   └── deal_size_by_month.csv
├── README.md                        # This file
└── .gitignore
```

Methodology details and validation notes live in `docs/`.

## Dependencies
- **Python 3.8+**
- **pandas**: Data manipulation and aggregation
- **numpy**: Numerical operations (weighting, decay curves)

To run the forecast (from project root):
```bash
python scripts/generate_forecast_v10.py
```

## Historical Interpretation (What We Use)
The model interprets history as follows; see `docs/forecasting_guidelines.md` for full context.

| Metric | Interpretation |
|--------|----------------|
| **Win Rates** | **% of Won Revenue / Total Pipeline Revenue** (same denominator as volume; `volume_weight` applied so skippers are half-weighted). |
| **Volume Creation** | Configurable baseline (see below) with **skipper logic**: deals that first appear as Closed Lost get `volume_weight = 0.5`. |
| **Sales Cycle & Seasonality** | **Months-to-close** distribution by segment (when new deals close); **monthly volume seasonality** from all-time creation by month. |
| **Starting Open Pipeline** | Deals **not** closed as of the last snapshot on or before `ACTUALS_THROUGH` (e.g. 2025-12-26); stage probabilities use **stage before exit** (per guidelines). |

## Methodology & Architecture

### Layer 1: Future Pipeline (The "When")
Forecasts revenue from deals **not yet created** in 2026.

- **Volume Baseline** (configurable via `VOLUME_BASELINE`):
  - `T12` — Trailing 12-month average (conservative)
  - `ALL_TIME` — All-time monthly average
  - `BLENDED` — 50/50 blend of T12 and All-Time (default, balances recency and stability)
  - `CAPACITY` — 65% peak + 35% T12 (aggressive, for capacity-based planning)
  
- **30% Drop Anchor**: Regardless of chosen baseline, if T12 drops >30% vs all-time, the model automatically uses a 50/50 blend to prevent over-pessimism.

- **Win Rate**: Revenue-based (Won Revenue / Total Pipeline Revenue), blended 55% T12 + 45% All-Time.

- **Deal Size**: Blended 55% T12 + 45% All-Time, with reversion to all-time if T12 wins < 15.

- **Timing**: Uses segment-specific months-to-close distribution from historical wins.

### Layer 2: Active Pipeline (The "How Much")
Forecasts revenue from deals **currently open** as of year-end 2025.

- **Stage Probabilities**: Uses the **stage before exit** (Qualified, Alignment, Solutioning), not the closure stage (Closed Won/Lost). This ensures open deals in working stages get non-zero probabilities.
- **Credibility Weighting**: If a segment/stage combo has <50 observed exits, it blends the segment-specific win rate with a global stage average.
- **Time-to-Close (V10)**: Computes average days remaining to close per `(market_segment, stage)` from historical snapshots. Each open deal gets an `expected_close_date = snapshot_date + avg_days_remaining`. Revenue is placed in the month containing the expected close date (no decay over months). This fixes the v9 issue where harmonic decay pulled revenue forward from early-stage deals.
- **Staleness Decay (V10)**: Uses segment-specific **sigmoid decay** instead of binary 0.8× penalty:
  - Formula: `staleness_factor = 1 / (1 + exp(k × (age_weeks - threshold)))`
  - **Large Market**: k=0.1 (slow decay, long cycles)
  - **SMB**: k=0.5 (fast decay, stale deals likely dead)
  - **Mid Market/Indirect**: k=0.25 (moderate)
  - Threshold remains 95th percentile age per (segment, stage)

## Parameters

All configuration is centralized at the top of `scripts/generate_forecast_v10.py`:

### Volume Baseline
```python
VOLUME_BASELINE = 'BLENDED'  # Options: 'T12', 'ALL_TIME', 'BLENDED', 'CAPACITY'

# Optional segment-specific overrides
VOLUME_BASELINE_BY_SEGMENT = {
    # 'Large Market': 'CAPACITY',  # Example: use capacity for Large Market only
}
```

**Calibration Results (with 1.0 levers):**

| Baseline | FY26 Forecast | Backtest Variance | Use Case |
|----------|---------------|-------------------|----------|
| `BLENDED` | ~$14.5M | -22.3% vs actuals | **Default**: Conservative, defensible baseline |
| `CAPACITY` | ~$21.6M | +22.4% vs actuals | Growth scenarios, capacity-based planning |
| `T12` | ~$15.1M | -11.4% vs actuals | Very conservative, recent-only |
| `ALL_TIME` | ~$13.9M | -33% vs actuals | Historical average baseline |

**Recommendation**: Use `T12` as the default for conservative, audit-friendly forecasts. Use `CAPACITY` or adjust `SCENARIO_LEVERS` to reach higher targets (~$20-23M).

### Scenario Levers
```python
SCENARIO_LEVERS = {
    'Indirect':     {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Large Market': {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Mid Market':   {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'SMB':          {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Other':        {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}
}
```

### Model Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `STALENESS_K_BY_SEGMENT` | Large Market: 0.1<br>Mid Market: 0.25<br>Indirect: 0.25<br>SMB: 0.5<br>Other: 0.3 | Sigmoid decay rate per segment (higher k = faster decay past threshold) |
| `DEFAULT_DAYS_REMAINING` | 90 | Fallback days-to-close for (segment, stage) with no history |
| `SKIPPER_WEIGHT` | 0.5 | Weight for deals that first appear as Closed Lost |
| `CREDIBILITY_THRESHOLD` | 50 | Minimum sample size for pure segment-specific probabilities |
| `WIN_RATE_T12_WEIGHT` | 0.55 | Weight for T12 in win rate blend |
| `DEAL_SIZE_T12_WEIGHT` | 0.55 | Weight for T12 in deal size blend |
| `MIN_WINS_FOR_T12_SIZE` | 15 | Minimum T12 wins to use T12 avg_size |

## Levers

The forecast is driven by three primary levers applied at the **segment level** in `SCENARIO_LEVERS`:

1. **Volume Multiplier**: Adjusts the expected number of new deals created
2. **Win Rate Multiplier**: Adjusts the conversion rate from creation to closed won
3. **Deal Size Multiplier**: Adjusts the average net revenue per deal

## Metrics / Drivers

| Metric | Calculation | Notes |
|--------|-------------|-------|
| **Win Rate** | Won Revenue / Total Pipeline Revenue | Revenue-weighted; skipper logic applies (0.5× weight) |
| **Volume Creation** | Configurable baseline (T12, ALL_TIME, BLENDED, CAPACITY) | Skipper deals (first appear as Closed Lost) get 0.5× weight |
| **Deal Size** | Average net revenue per won deal | Blended 55% T12 + 45% All-Time; reverts to all-time if T12 wins < 15 |
| **Time-to-Close** | Average days from snapshot to close per (segment, stage) | T12-weighted; used in Layer 2 for expected close dates |
| **Staleness Factor** | `1 / (1 + exp(k × (age - threshold)))` | Sigmoid decay; k varies by segment (0.1–0.5) |

## Backtesting Overview

The backtest validates the model by "predicting" 2025 using data available as of Jan 1, 2025.

**Model-Only Approach:**
- **Active Pipeline**: Forecasts deals open at start of 2025 using stage-before-exit probabilities, time-to-close logic, and sigmoid staleness decay
- **Future Pipeline**: Forecasts deals created in 2025 using historical baseline with a **single reconciliation lever** (volume_multiplier) per segment
- **No Actuals Substitution**: The forecast is purely model-driven; actual values are only used for comparison, not inserted into the forecast

**Validation Checks:**
- Probability cap: Ensures no stage probability exceeds 1.0
- Revenue conservation: Verifies Layer 2 input weighted revenue matches output revenue
- Snapshot consistency: Detects >30% drops in deal count per snapshot (possible ETL issues)

**Output:**
- `validation/backtest_results.csv`: Segment-level model_forecast vs actual with variance %
- `validation/backtest_monthly.csv`: Monthly breakdown for trend analysis

**Interpretation:**
- The backtest reconciliation lever shows what multiplier would have been needed to match actuals
- This validates the model structure while documenting the gap between baseline and actual performance

## Goal Seek Overview

When a **revenue target per segment** is set in `GOAL_WON_REVENUE`, the script calculates required levers to hit that number:
- `required_combined_multiplier`: Total multiplier needed (target / current_forecast)
- `suggested_volume_mult`, `suggested_win_rate_mult`, `suggested_deal_size_mult`: Cube-root split for even distribution across levers

**Output:** `exports/goal_seek_analysis.csv` (CSV only, no console logging)

**Usage:** Review the required multipliers to assess target feasibility. Update `SCENARIO_LEVERS` to match suggested values if targets are achievable.

## Outputs

| File | Description |
|------|-------------|
| `exports/forecast_2026.csv` | Monthly FY26 forecast by segment |
| `exports/assumptions.json` | Configuration snapshot for auditability |
| `exports/goal_seek_analysis.csv` | Required levers to hit revenue targets |
| `validation/backtest_results.csv` | Segment-level backtest vs actuals |
| `validation/backtest_monthly.csv` | Monthly backtest breakdown |
| `assumptions_log/*.csv` | Historical volume, win rates, deal size by month/segment |

## Future Upgrade Opportunities

1. **Stock vs Flow Win Rates**: Separate win rates for deals created >6 months ago (stock) vs fresh deals (flow) to capture different conversion dynamics
2. **Gaussian Spread**: Add optional ±1 month Gaussian distribution around expected close dates to model variance
3. **PySpark Migration**: Refactor to PySpark for production scale; vectorize remaining iterrows() loops
4. **Audit Log**: Add per-deal audit trail (deal_id, action, reason, impact) for "why was this deal downgraded?" queries
5. **Seasonality Adjustments**: Add quarterly weighting (e.g., Q4 flush) for monthly precision
6. **Lead Source Attribution**: Differentiate Marketing-sourced vs. Sales-sourced leads
7. **Churn Modeling**: Add churn/contraction layer for "Net Revenue" forecasting
