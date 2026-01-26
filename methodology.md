# Pipeline Revenue Forecasting Model V3
## Implementation Year Based Forecasting

---

## The Fundamental Change

**V2 asked:** "How much revenue will close in 2025?"
**V3 asks:** "How much revenue will we win with 2025 implementation dates?"

This distinction matters because:
- A deal closing in December 2024 might have January 2025 implementation
- A deal closing in November 2025 might have January 2026 implementation
- Management reports typically focus on implementation year, not close date

---

## Model Architecture

### Component 1: Active Pipeline (Implementation Year Filtered)

```
Active Pipeline for 2025 = Deals where:
  - Stage ∈ {Qualified, Alignment, Solutioning, Verbal}
  - implementation_year = 2025
```

**Why this matters:** In V2, we included ALL active deals regardless of implementation year. This caused over-forecasting because deals with 2026 implementation dates were counted toward 2025.

### Component 2: Stage Conversion Rates

```
P(Win | Stage) = Historical win rate for deals that:
  - Passed through that stage
  - Have implementation_year < target_year (to avoid data leakage)
```

**Why filter by impl_year:** Deals with future implementation years may have different characteristics. We only train on "completed" cohorts.

### Component 3: Future Pipeline Projection

```
Future Revenue = Prior Year Impl Revenue × Conservatism × Growth
```

**Logic:** If we won $10M of 2024 implementation year deals, we might expect to win approximately $10M × 0.85 = $8.5M of 2025 implementation year deals from pipeline not yet created.

---

## Why V2 Failed (79% Variance)

### Problem 1: Wrong Pipeline Filter

V2 included **all active deals** regardless of implementation year:

| Segment | Active Deals | Active Value | Actual Won |
|---------|--------------|--------------|------------|
| Large Market | 38 | $17.7M | $2.9M |

Many of those 38 deals had **2026 implementation years** and shouldn't have been in the 2025 forecast.

### Problem 2: Wrong Training Data

V2 calculated conversion rates from all historical deals. But Large Market deals with 2024 implementation may have different win rates than Large Market deals with 2025 implementation.

### Problem 3: Insufficient Conservatism

V2 used 90% conservatism, but the actual YoY variance was larger.

---

## Running the Diagnostics

Before running the forecast model, run `diagnostics.py` to understand your data:

```bash
python scripts/diagnostics.py data/fact_snapshots.csv
```

This will show:
1. Implementation year distribution
2. Active pipeline by implementation year
3. Actual conversion rates by segment
4. Historical stage conversion rates
5. Recommended conservatism factor

---

## Key Diagnostic Questions

### Q1: What's in the active pipeline by implementation year?

```
Active Pipeline at 2025-01-01 by Implementation Year:
  2024: 50 deals, $3M    → These should NOT be in 2025 forecast
  2025: 100 deals, $8M   → These ARE the 2025 forecast
  2026: 30 deals, $5M    → These should NOT be in 2025 forecast
```

### Q2: What happened to the 2025 impl_year active pipeline?

```
2025 Impl Year Active Pipeline Outcomes:
  Won: 40 deals, $4M
  Lost: 45 deals, $3M
  Still Open: 15 deals, $1M  → These may slip to 2026 impl
```

### Q3: What's the actual conversion rate by segment?

```
Large Market (impl_year=2025):
  Active: 20 deals, $10M
  Won: 4 deals, $2M
  Deal Conversion: 20%
  Value Conversion: 20%
```

---

## Model Calibration

### Setting Conservatism Factor

1. Run diagnostics to get YoY impl year revenue:
   - 2024 impl year won: $X
   - 2025 impl year won: $Y
   - Ratio: Y/X

2. Use ratio as starting point for conservatism:
   - If ratio = 0.85, consider conservatism = 0.80-0.90
   - If ratio = 1.10, consider conservatism = 0.95-1.00

### Setting Conversion Rates

The model calculates rates automatically from historical data. But verify they make sense:

| Stage | Expected Range | If Outside Range |
|-------|----------------|------------------|
| Qualified | 15-25% | Check data quality |
| Alignment | 30-50% | Check stage definitions |
| Solutioning | 50-70% | Check stage definitions |
| Verbal | 70-90% | Check stage definitions |

---

## Validation Process

### Step 1: Run Diagnostics

```bash
python scripts/diagnostics.py
```

### Step 2: Review Pipeline by Implementation Year

Ensure the implementation year filter makes sense. If most deals have null implementation dates, the model won't work.

### Step 3: Run Backtest

```bash
python scripts/forecast_model_v3.py
```

### Step 4: Analyze Variance by Component

```
Forecast = Active Expected + Future Expected
Actual = From Active + From Future

Check:
- Active Expected vs From Active (pipeline accuracy)
- Future Expected vs From Future (baseline accuracy)
```

### Step 5: Segment Deep-Dive

If a segment has high variance:
1. Check sample size (thin segments have high variance)
2. Check if conversion rates match observed rates
3. Check if baseline year is representative

---

## Expected Accuracy

| Metric | Target | V2 Result | V3 Target |
|--------|--------|-----------|-----------|
| Aggregate Variance | ±10% | +79% | ±10% |
| Large Market | ±20% | +203% | ±20% |
| Mid Market | ±15% | +19% | ±15% |

**Note:** Large Market will always have higher variance due to:
- Fewer deals (small sample)
- Larger deal sizes (single deal can swing results)
- Longer sales cycles (more uncertainty)

---

## Dual Forecasting (Optional)

If management wants BOTH close date and implementation year forecasts:

```python
# Forecast 1: Implementation Year 2026
results_impl = run_backtest(df, config, scenario, '2026-01-01', target_impl_year=2026)

# Forecast 2: Close Date 2026 
# (requires modifying the model to filter by expected close date instead)
```

---

## Troubleshooting

### "Too few deals in active pipeline"

The implementation year filter may be too restrictive. Check:
- Are implementation dates populated?
- Are deals being assigned the correct implementation year?

### "Conversion rates are 0% or 100%"

Not enough training data. Consider:
- Using global rates instead of segment-specific
- Increasing the training window
- Combining similar segments

### "Still seeing high variance"

Check if "Still Open" deals are significant. If many 2025 impl year deals are still open, they may:
- Slip to 2026 implementation
- Eventually close won (meaning your forecast was right, just early)

---

## Summary

V3's key insight: **Filter everything by implementation year**.

1. Active pipeline → Only deals with target impl year
2. Conversion rates → Only train on prior impl year cohorts
3. Future baseline → Use prior impl year total as baseline

This aligns the forecast with how management measures results.