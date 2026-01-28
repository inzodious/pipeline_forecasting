# Analysis of generate_forecast_v4.py Issues

## Critical Problems Identified

### 1. **Over-Amplification: Double Application of Win Rate Multiplier**
In `forecast_future_pipeline()`, the win rate multiplier is applied AFTER the probability has already been applied:
```python
'expected_revenue': total_cohort_value * marginal_win_pct * SCENARIO_LEVERS['win_rate_multiplier']
'expected_count': vol * marginal_win_pct * SCENARIO_LEVERS['win_rate_multiplier']
```
This is incorrect. The multiplier should adjust the **probability** (marginal_win_pct), not be a direct multiplier on the result.

### 2. **Vintage Curve Logic: Incorrect Marginal Calculation**
The current logic calculates marginal wins for each forecast month by comparing:
- `age_weeks_now` (age of cohort in forecast month)
- `age_weeks_prev` (age 4 weeks earlier)

**Problem**: For a cohort created in Jan 2026:
- In Jan 2026 forecast: calculates marginal wins from week 0 to week 0
- In Feb 2026 forecast: calculates marginal wins from week 0 to week 4
- In Mar 2026 forecast: calculates marginal wins from week 4 to week 8
- etc.

This treats each forecast month independently, but all these results are **summed together**, causing revenue from the same deals to be counted multiple times across months.

### 3. **Active Pipeline Missing Time Progression**
Active pipeline only looks at deals open on ACTUALS_THROUGH date and assigns them all to that single month. It doesn't project when these deals will close over the forecast period.

### 4. **Forecast Output Extends to 2027**
The forecast shows data through 2027-04, even though FORECAST_END is 2026-12-31. This suggests the aggregation is including cohort projections beyond the forecast window.

### 5. **Backtest Variance Issues**
From validation/forecast_2025.csv:
- Large Market: -16.8% variance (underestimate)
- Mid Market: -42.8% variance (HUGE underestimate)
- Indirect: +18.7% variance (overestimate)

## Root Cause Analysis

The fundamental issue is that the **vintage maturity curve represents CUMULATIVE win probability**, but the code is trying to extract **marginal** probabilities for each forecast month and then summing them all together.

**Correct Logic Should Be:**
1. For each cohort (created in month M), calculate total expected value based on max maturity
2. Distribute that total value across forecast months based on when deals will close (marginal increments)
3. Each deal's value should only be counted ONCE across all forecast months

**Current Logic Does:**
1. For each cohort, for each forecast month, calculate marginal value
2. Sum all these together → **DOUBLE/TRIPLE COUNTING**

## Solution Architecture

### Layer 1 (Future Pipeline) - Correct Implementation:
1. For each creation month in 2026, estimate total cohort value (volume × avg size)
2. Calculate FINAL expected wins using max vintage maturity rate
3. Distribute those wins across forecast months based on aging curve shape
4. Each $1 of cohort value should appear in exactly ONE forecast month

### Layer 2 (Active Pipeline) - Correct Implementation:
1. Take open deals as of forecast date
2. For each deal, estimate closure probability AND timing
3. Assign to appropriate forecast month based on expected closure date
4. Use stage probability × staleness penalty × vintage timing

### Backtest Logic:
1. Use data only up to backtest date
2. Forecast forward
3. Compare against actuals that occurred after backtest date
4. Should have LOW variance if logic is sound
