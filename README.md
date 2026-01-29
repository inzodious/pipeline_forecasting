# Pipeline Forecasting Model V6

## Overview
This project implements a **Two-Layer Hybrid Forecasting Model** combining cohort-based vintage analysis (Layer 1) with stage-weighted probability modeling (Layer 2) to forecast closed won revenue for 2026. The model is designed to handle sparse data in high-value segments (Large Market) by blending historical and recent performance metrics to produce a stable, defensible forecast.

## Project Directory
```
pipeline_forecasting/
├── data/
│   ├── fact_snapshots.csv          # Primary data: weekly snapshots of all deals
│   └── sample_data_pivoted.csv     # Excel pivot for validation
├── scripts/
│   ├── generate_forecast_v6.py     # Main forecast generator (Production)
│   ├── check_multipliers.py        # Diagnostic: Backtest multiplier validation
│   ├── check_timing.py             # Diagnostic: Timing distribution analysis
│   └── diagnose_backtest.py        # Diagnostic: Backtest variance deep dive
├── exports/
│   ├── forecast_2026.csv           # Monthly FY26 forecast output
│   ├── assumptions.json            # Scenario lever settings
│   └── goal_seek_analysis.csv      # What-if scenarios for targets
├── validation/
│   ├── backtest_results.csv        # Segment-level variance summary
│   └── backtest_monthly.csv        # Month-by-month backtest detail
├── assumptions_log/
│   ├── volume_by_month.csv         # Historical deal creation trends
│   ├── win_rates_by_month.csv      # Historical win rates by cohort
│   ├── deal_size_by_month.csv      # Historical deal size trends
│   └── win_rates_by_stage.csv      # Stage-level probabilities
├── README.md                        # This file
└── forecasting_guidelines.md        # Methodology details
```

## Dependencies
- **Python 3.8+**
- **pandas**: Data manipulation and aggregation
- **numpy**: Numerical operations (weighting, decay curves)

To run the forecast:
```bash
python scripts/generate_forecast_v6.py
```

## Methodology & Architecture

### Layer 1: Future Pipeline (The "When")
Forecasts revenue from deals **not yet created** in 2026.
- **Base Logic**: Uses T12M average monthly deal creation volume, win rates, and average deal size.
- **Stability Anchors**: 
  - If 2025 volume dropped >30% vs. all-time history, the model anchors to a 50/50 blend of T12M and All-Time volume.
  - Deal size uses a weighted average but reverts to all-time averages if the T12M sample size is small (<15 wins).
- **Timing**: Uses a blended distribution (50% All-Time / 50% Recent) to project when new deals will close.

### Layer 2: Active Pipeline (The "How Much")
Forecasts revenue from deals **currently open** as of year-end 2025.
- **Base Logic**: Assigns probability based on the deal's current stage.
- **Blended Probabilities**: Uses a "Credibility Weighting" mechanism. If a segment/stage combo has <50 observed exits, it blends the segment-specific win rate with a global stage average to prevent unrealistic zero-probabilities.
- **Decay**: Applies a "Staleness Penalty" (0.8x) to deals older than the 95th percentile duration.
- **Timing**: Large Market deals use a flat 12-month distribution to reflect long sales cycles; other segments use a front-loaded decay.

## Drivers
The forecast is driven by three primary levers:
1. **Volume Multiplier**: Adjusts the expected number of new deals created.
2. **Win Rate Multiplier**: Adjusts the conversion rate from creation to closed won.
3. **Deal Size Multiplier**: Adjusts the average net revenue per deal.

These drivers are applied at the **segment level** in `SCENARIO_LEVERS` within `scripts/generate_forecast_v6.py`.

## Inputs/Levers/Metrics

### Current Scenario: Growth Recovery (Scenario B)
Assumes a recovery in deal volume and conversion efficiency for 2026.

| Segment | Volume | Win Rate | Deal Size |
|---------|--------|----------|-----------|
| **Indirect** | 1.4x | 1.2x | 1.1x |
| **Large Market** | 1.8x | 1.4x | 1.2x |
| **Mid Market** | 1.4x | 1.2x | 1.0x |
| **SMB** | 1.4x | 1.1x | 1.0x |
| **Other** | 1.0x | 1.0x | 1.0x |

### Key Metrics
- **Skipper Weight (0.5)**: Penalty for deals that jump straight to "Closed Lost" without working stages.
- **Staleness Penalty (0.8)**: Probability reduction for deals exceeding 95th percentile age.
- **Credibility Threshold (50)**: Minimum sample size required to use pure segment-specific win rates.

## Forecast Summary
**Total Forecasted Revenue (FY26):** ~$19.63M

| Segment | Expected Revenue | Deal Count |
|---------|------------------|------------|
| **Large Market** | $9.18M | 31.0 |
| **Indirect** | $5.75M | 291.6 |
| **Mid Market** | $4.14M | 158.8 |
| **SMB** | $0.55M | 116.2 |
| **Other** | $0.01M | 3.1 |

**YoY Comparison:** The forecast is within **14%** of 2025 actuals ($22.8M), representing a realistic but conservative recovery target.

## Backtest Summary
The model was validated by "predicting" 2025 performance using data available as of Jan 1, 2025.

| Segment | Forecast (Backtest) | Actuals 2025 | Variance | Status |
|---------|---------------------|--------------|----------|--------|
| **Large Market** | $7.78M | $8.12M | -4.2% | ✅ Excellent |
| **Indirect** | $5.93M | $7.84M | -24.3% | ⚠️ Under |
| **Mid Market** | $4.48M | $6.45M | -30.5% | ⚠️ Under |
| **Total** | **$18.34M** | **$22.82M** | **-19.7%** | ✅ Acceptable |

**Note:** Backtest uses "Perfect Prediction Mode" which applies actual 2025 volume/win-rate multipliers to validate the core logic, rather than guessing levers.

## Further Enhancement Opportunities
1. **Seasonality Adjustments**: Currently, volume is distributed evenly. Adding quarterly weighting (e.g., Q4 flush) would improve monthly precision.
2. **Lead Source Attribution**: Differentiating between Marketing-sourced vs. Sales-sourced leads could refine the "Future Pipeline" layer.
3. **Stage Velocity**: Incorporating time-in-stage velocity metrics could provide a more dynamic "Staleness" penalty.
4. **Churn Modeling**: For recurring revenue components, adding a churn/contraction layer would move this from "New Business" to "Net Revenue" forecasting.

## Versioning/Status/Validation
- **Version**: 6.0 (January 2026)
- **Status**: **Production Ready**
- **Validation**:
  - Validated against 2025 actuals with <20% aggregate variance.
  - Logic updated to handle small sample sizes (Large Market) robustly.
  - "Goal Seek" analysis included to help leadership set targets.
