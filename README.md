# Hybrid Sales Pipeline Forecasting Model

Two-Layer forecasting system combining Cohort-Based Vintage Analysis (strategic baseline) with Stage-Weighted Probability Modeling (operational overlay).

## Quick Start

```bash
# Generate mock data for testing
python mock_data_generator.py

# Run full 2026 forecast
python forecast_pipeline.py

# Run with backtesting validation
python run_forecast.py --backtest
```

## Architecture

### Layer 1: Cohort-Based Vintage Analysis
- Groups deals by creation vintage (month/week)
- Tracks cumulative win behavior over time
- Handles incomplete cohort problem (recent deals appearing artificially weak)
- Output: Vintage maturity curves by segment

### Layer 2: Stage-Weighted Probability Modeling
- Assigns probabilities based on current pipeline stage
- Applies staleness penalties for stagnant deals
- Output: Deal-level expected revenue

## Key Assumptions

| Assumption | Implementation |
|------------|----------------|
| Skipper Logic | Deals appearing directly as Closed Lost without active stages receive 0.5 weight |
| Right Censoring | Open deals projected using vintage maturity curves, not assumed lost |
| Staleness | Deals exceeding 90th percentile stage duration receive 0.8 penalty multiplier |

## Scenario Levers

Modify `SCENARIO_LEVERS` in `forecast_pipeline.py`:

```python
SCENARIO_LEVERS = {
    'win_rate_uplift': 1.0,      # Multiply win rates
    'deal_volume_growth': 1.0,   # Multiply future deal counts
    'revenue_per_deal_uplift': 1.0,  # Multiply deal values
    'target_revenue': None,      # Target for back-solve analysis
}
```

## Output Files

| File | Description |
|------|-------------|
| `exports/forecast_2026.csv` | Monthly forecast by segment |
| `exports/assumptions_log.json` | All model parameters and derived metrics |
| `exports/backtest_results.csv` | Segment-level backtest comparison |
| `exports/backtest_summary.json` | Overall backtest metrics |

## Data Schema

Required columns in `data/fact_snapshots.csv`:

| Column | Type | Description |
|--------|------|-------------|
| deal_id | string | Unique deal identifier |
| date_created | date | Deal creation date |
| date_closed | date | Deal close date (null if open) |
| date_snapshot | date | Snapshot date |
| stage | string | Pipeline stage at snapshot |
| net_revenue | float | Deal value |
| market_segment | string | Market segment |

## Migration to PySpark

The Python script is designed for direct translation to PySpark. Key changes:
- Replace pandas DataFrames with Spark DataFrames
- Replace groupby operations with Spark window functions
- Use Spark SQL for aggregations