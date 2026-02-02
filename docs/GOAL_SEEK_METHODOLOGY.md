# Goal-Seek Pipeline Forecasting: Complete Technical Documentation

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Data Model](#2-data-model)
3. [Step 1: Deal Facts Construction](#3-step-1-deal-facts-construction)
4. [Step 2: Skipper Logic](#4-step-2-skipper-logic)
5. [Step 3: T12 Volume Calculation](#5-step-3-t12-volume-calculation)
6. [Step 4: Win Rate Calculation](#6-step-4-win-rate-calculation)
7. [Step 5: Stage Probability Calculation](#7-step-5-stage-probability-calculation)
8. [Step 6: Open Pipeline Valuation](#8-step-6-open-pipeline-valuation)
9. [Step 7: Timing Distribution](#9-step-7-timing-distribution)
10. [Step 8: Baseline Annual Revenue](#10-step-8-baseline-annual-revenue)
11. [The Goal-Seek Model](#11-the-goal-seek-model)
12. [Excel Output Reference](#12-excel-output-reference)
13. [Worked Example](#13-worked-example)
14. [Assumptions & Limitations](#14-assumptions--limitations)

---

## 1. Executive Summary

This script transforms weekly CRM snapshot data into a **Goal-Seek ready Excel model** for FY2026 revenue forecasting. The output allows executives to:

1. See baseline forecast derived purely from historical data
2. Adjust three levers (Volume, Win Rate, Deal Size) per segment
3. Use Excel's Goal Seek to solve for "what multiplier do I need to hit $X target?"

**The Core Formula:**

```
FY26 Forecast = Future Pipeline Revenue + Active Pipeline Revenue

Where:
  Future Pipeline Revenue = Avg Monthly Volume × 12 × Win Rate × Avg Deal Size
  Active Pipeline Revenue = Σ (Open Deal Revenue × Stage Probability)
```

---

## 2. Data Model

### 2.1 Input: Weekly Snapshots

The source data (`fact_snapshots.csv`) contains **weekly snapshots** of every deal in the CRM. Each row represents the state of one deal on one snapshot date.

| Column | Type | Description |
|--------|------|-------------|
| `deal_id` | int | Unique deal identifier |
| `date_snapshot` | date | When this snapshot was taken (weekly, typically Friday) |
| `date_created` | date | When the deal was first created in CRM |
| `date_closed` | date | When the deal closed (null if still open) |
| `market_segment` | string | Large Market, Mid Market, SMB, Indirect, Other |
| `stage` | string | Current pipeline stage at snapshot time |
| `net_revenue` | float | Deal value in dollars |

### 2.2 Stage Definitions

| Stage | Meaning |
|-------|---------|
| `Qualified` | Initial qualification complete |
| `Alignment` | Stakeholder alignment in progress |
| `Solutioning` | Solution design / proposal stage |
| `Verbal` | Verbal commitment received (counts as WON) |
| `Closed Won` | Contract signed (counts as WON) |
| `Closed Lost` | Deal lost |

**Critical Definition:** `Won = (stage == 'Closed Won') OR (stage == 'Verbal')`

This is applied consistently everywhere in the model.

---

## 3. Step 1: Deal Facts Construction

### 3.1 Purpose
Convert weekly snapshots into a **single row per deal** with its final outcome.

### 3.2 Logic

```python
# For each deal_id:
# 1. Get the FIRST snapshot (for creation date, initial revenue, segment)
# 2. Get the FIRST snapshot where is_closed=True (for final outcome)
# 3. Merge them together
```

### 3.3 Resulting Columns

| Column | Source | Description |
|--------|--------|-------------|
| `deal_id` | First snapshot | Unique identifier |
| `date_created` | First snapshot | Deal creation date |
| `market_segment` | First snapshot | Segment assignment |
| `net_revenue` | Final snapshot (if closed) or first | Deal value |
| `date_closed` | First closed snapshot | When it closed (null if open) |
| `stage` | First closed snapshot | Final stage (or 'Open' if not closed) |
| `won` | Derived | True if stage in ['Closed Won', 'Verbal'] |
| `created_month` | Derived | Period('2025-03') for grouping |

### 3.4 Why "First Closed Snapshot"?

Deals can appear as closed in multiple subsequent snapshots. We use the **first** closed snapshot to capture the true closure date, not when it was last reported.

---

## 4. Step 2: Skipper Logic

### 4.1 The Problem

Some deals are entered into the CRM **after they've already been lost**. Their first snapshot shows `stage = 'Closed Lost'`. These deals never had a chance to be worked - they "skipped" the pipeline entirely.

### 4.2 The Solution

Deals whose **first-ever stage** is `Closed Lost` receive a **0.5 weight** in volume calculations.

```python
first_stage = df.groupby('deal_id')['stage'].first()
skippers = first_stage[first_stage == 'Closed Lost'].index
deals.loc[deals['deal_id'].isin(skippers), 'volume_weight'] = 0.5
# All other deals get volume_weight = 1.0
```

### 4.3 Why 0.5?

This is a business judgment call:
- **0.0** would ignore them entirely (but they did exist)
- **1.0** would treat them as real pipeline (but they were never worked)
- **0.5** is a middle ground that acknowledges their existence while discounting their impact

### 4.4 Where Skipper Weight is Applied

| Metric | Applied? | Effect |
|--------|----------|--------|
| Volume counts | ✅ Yes | Skippers count as 0.5 deals |
| Win rate denominator | ✅ Yes | Skippers contribute 0.5 to total pipeline |
| Win rate numerator | ✅ Yes | If a skipper somehow won, it's 0.5 |
| Deal size | ❌ No | Uses actual revenue values |
| Open pipeline | ❌ No | Skippers are already closed |

---

## 5. Step 3: T12 Volume Calculation

### 5.1 Purpose
Calculate the **average number of deals created per month** over the trailing 12 months.

### 5.2 Definition

```
T12 = Trailing 12 Months = Most recent 12 months of data
Cutoff = max(date_created) in the dataset
T12 Start = Cutoff - 12 months
```

### 5.3 Calculation

```python
# Step 1: Filter to T12 deals
t12_deals = deals[deals['date_created'] >= t12_start]

# Step 2: Group by segment and created_month, sum volume_weight
monthly = t12_deals.groupby(['market_segment', 'created_month']).agg(
    raw_count = count of deal_id,           # Actual deal count
    adj_volume = sum of volume_weight,      # Skipper-adjusted count
    total_revenue = sum of net_revenue      # Total $ created
)

# Step 3: Calculate monthly average per segment
avg_monthly_volume = total_adj_volume / months_in_t12
```

### 5.4 Output Columns (T12_Volume Sheet)

| Column | Description |
|--------|-------------|
| `month` | The creation month (e.g., "2025-06") |
| `market_segment` | Segment name |
| `raw_count` | Actual number of deals created |
| `adj_volume` | Skipper-adjusted volume (raw - 0.5 per skipper) |
| `total_revenue` | Sum of net_revenue for deals created this month |

### 5.5 Why T12?

- **Recent enough** to reflect current business conditions
- **Long enough** to smooth out monthly volatility
- **Standard practice** in financial forecasting

---

## 6. Step 4: Win Rate Calculation

### 6.1 The Formula

**Win Rate (Revenue-Based):**

```
Win Rate = Won Revenue / Total Pipeline Revenue

Where:
  Won Revenue = Σ (net_revenue × volume_weight) for deals where won=True
  Total Pipeline Revenue = Σ (net_revenue × volume_weight) for all deals
```

### 6.2 Why Revenue-Based?

Consider two scenarios:

| Scenario | Deals | Won | Count-Based WR | Revenue-Based WR |
|----------|-------|-----|----------------|------------------|
| A | 10 × $10K, 1 × $500K | The $500K wins | 9% (1/11) | 83% ($500K/$600K) |
| B | 10 × $10K, 1 × $500K | 5 × $10K win | 45% (5/11) | 8% ($50K/$600K) |

Revenue-based win rate better predicts **dollars**, which is what we're forecasting.

### 6.3 T12 vs All-Time Blending

We calculate win rates for both periods and blend them:

```python
blended_win_rate = (t12_win_rate × 0.55) + (all_time_win_rate × 0.45)
```

**Why blend?**
- **T12** captures recent performance trends
- **All-Time** provides stability and larger sample size
- **55/45 split** slightly favors recency while maintaining stability

### 6.4 Output Columns (Win_Rates Sheet)

| Column | Description |
|--------|-------------|
| `t12_total_pipeline_revenue` | Total revenue × volume_weight for T12 deals |
| `t12_won_revenue` | Won revenue × volume_weight for T12 deals |
| `t12_win_rate_revenue` | t12_won_revenue / t12_total_pipeline_revenue |
| `t12_total_deals` | Sum of volume_weight for T12 |
| `t12_won_deals` | Sum of volume_weight for T12 wins |
| `t12_win_rate_count` | t12_won_deals / t12_total_deals |
| `t12_avg_won_deal_size` | Average net_revenue for T12 won deals |
| `t12_sample_size` | Count of T12 deals (unweighted) |
| `all_*` | Same metrics for all-time period |
| `blended_win_rate` | 0.55 × t12 + 0.45 × all_time |
| `blended_deal_size` | 0.55 × t12_avg + 0.45 × all_time_avg |

---

## 7. Step 5: Stage Probability Calculation

### 7.1 Purpose

Calculate **P(Win | Deal was in Stage X before closing)** for use in valuing open pipeline.

### 7.2 Critical Distinction: Stage BEFORE Exit

We use the **stage the deal was in immediately before it closed**, NOT the final stage.

**Why?**
- A deal that closes as "Closed Won" tells us nothing about stage probability
- We need to know: "Of deals that were in Solutioning before they closed, what % won?"

### 7.3 The Algorithm

```python
# Step 1: Find all deals that closed
first_closed = df[df['is_closed']].groupby('deal_id').first()

# Step 2: For each closed deal, find the snapshot BEFORE the close snapshot
prev_snap = df[df['date_snapshot'] < close_snap].groupby('deal_id').last()
stage_before_exit = prev_snap['stage']

# Step 3: Calculate probability per (segment, stage)
for each (segment, stage) combination:
    wins = count of deals where is_closed_won=True (weighted)
    total = count of all exits from this stage (weighted)
    probability = wins / total
```

### 7.4 T12 Weighting

Recent exits are weighted 3× more than older exits:

```python
weight = 3.0 if close_date >= t12_start else 1.0

weighted_wins = Σ (is_closed_won × weight)
weighted_total = Σ (weight)
probability = weighted_wins / weighted_total
```

### 7.5 Credibility Weighting

For segment/stage combinations with few observations, we blend toward the global average:

```python
CREDIBILITY_THRESHOLD = 50

credibility = min(weighted_total / CREDIBILITY_THRESHOLD, 1.0)
blended_prob = (raw_probability × credibility) + (global_prob × (1 - credibility))

# Floor: Never go below 95% of global average
final_probability = max(blended_prob, global_prob × 0.95)
```

**Example:**
- Large Market / Solutioning has only 20 weighted exits → credibility = 0.4
- Raw probability = 60%, Global probability = 50%
- Blended = (60% × 0.4) + (50% × 0.6) = 24% + 30% = 54%

### 7.6 Output Columns (Stage_Probabilities Sheet)

| Column | Description |
|--------|-------------|
| `market_segment` | Segment name |
| `stage` | The stage deals were in before exiting |
| `weighted_wins` | Sum of (is_won × weight) |
| `weighted_total` | Sum of weight for all exits |
| `unweighted_exits` | Raw count of deals that exited from this stage |
| `raw_probability` | weighted_wins / weighted_total |
| `global_prob` | Stage-level probability across all segments |
| `credibility` | min(weighted_total/50, 1.0) |
| `blended_probability` | Credibility-weighted blend |
| `final_probability` | max(blended, 0.95 × global) |

---

## 8. Step 6: Open Pipeline Valuation

### 8.1 Purpose

Calculate the **expected revenue** from deals currently open as of the forecast start date.

### 8.2 Identifying Open Deals

```python
# Find the most recent snapshot on or before ACTUALS_THROUGH date
cutoff = '2025-12-26'
last_snapshot = max(date_snapshot where date_snapshot <= cutoff)

# Open deals = deals that are NOT closed on that snapshot
open_deals = df[(df['date_snapshot'] == last_snapshot) & (df['is_closed'] == False)]
```

### 8.3 Assigning Probabilities

Each open deal gets the `final_probability` from its current (segment, stage):

```python
open_deals = open_deals.merge(
    stage_probs[['market_segment', 'stage', 'final_probability']],
    on=['market_segment', 'stage']
)
open_deals['probability'] = final_probability  # 0 if no match found
```

### 8.4 Expected Revenue Calculation

```
expected_revenue = net_revenue × probability
```

### 8.5 Output Columns (Open_Pipeline Sheet)

| Column | Description |
|--------|-------------|
| `deal_id` | Unique identifier |
| `market_segment` | Segment |
| `stage` | Current stage |
| `net_revenue` | Deal value ($) |
| `age_days` | Days since creation |
| `age_weeks` | Weeks since creation |
| `probability` | P(Win) based on segment/stage |
| `expected_revenue` | net_revenue × probability |
| `date_created` | Deal creation date |

### 8.6 Aggregated Metrics

```python
open_pipeline_value = Σ net_revenue           # Total unweighted value
expected_pipeline_value = Σ expected_revenue  # Probability-weighted value
```

---

## 9. Step 7: Timing Distribution

### 9.1 Purpose

Understand **how long deals take to close** by segment. Used for context, not directly in the goal-seek model.

### 9.2 Calculation

```python
# Only won deals with a close date
won_deals = deals[(deals['won'] == True) & (deals['date_closed'].notna())]

# Calculate months to close (capped at 11)
months_to_close = min(11, round((date_closed - date_created).days / 30))

# Group and calculate percentages
for each (segment, months_to_close):
    count = number of deals
    pct = count / total_wins_for_segment
```

### 9.3 Output Columns (Timing_Distribution Sheet)

| Column | Description |
|--------|-------------|
| `market_segment` | Segment |
| `months_to_close` | 0, 1, 2, ... 11 |
| `count` | Number of deals that closed in this many months |
| `total` | Total won deals for segment |
| `pct` | count / total |

---

## 10. Step 8: Baseline Annual Revenue

### 10.1 The Formula

```
Baseline Annual Revenue = Avg Monthly Volume × 12 × Blended Win Rate × Blended Deal Size
```

This is the **Future Pipeline** component - revenue expected from deals not yet created.

### 10.2 Component Sources

| Component | Source | Description |
|-----------|--------|-------------|
| Avg Monthly Volume | T12_Volume | Skipper-adjusted monthly average |
| Blended Win Rate | Win_Rates | 55% T12 + 45% All-Time (revenue-based) |
| Blended Deal Size | Win_Rates | 55% T12 + 45% All-Time average won deal size |

### 10.3 Example Calculation

**Large Market:**
- Avg Monthly Volume = 7.75 deals/month
- Blended Win Rate = 11.25%
- Blended Deal Size = $185,361

```
Baseline = 7.75 × 12 × 0.1125 × $185,361 = $1,939,550
```

---

## 11. The Goal-Seek Model

### 11.1 Total Forecast Formula

```
Total Forecast = Future Pipeline Revenue + Active Pipeline Revenue

Where:
  Future Pipeline = Adj Volume × 12 × Adj Win Rate × Adj Deal Size
  Active Pipeline = Expected Pipeline Value (fixed from open deals)
```

### 11.2 The Three Levers

| Lever | Formula | Default |
|-------|---------|---------|
| Volume Multiplier | Adj Volume = Base Volume × Volume Mult | 1.0 |
| Win Rate Multiplier | Adj Win Rate = min(Base WR × WR Mult, 1.0) | 1.0 |
| Deal Size Multiplier | Adj Deal Size = Base Size × Size Mult | 1.0 |

**Note:** Win Rate is capped at 100% (1.0) to prevent impossible forecasts.

### 11.3 The Goal Seek Process

1. **Set Target:** Enter your revenue goal in the "Revenue Target" cell
2. **Gap Calculation:** Gap = Target - Total Forecast
3. **Goal Seek:** Ask Excel to set Gap = 0 by changing one lever

**Excel Steps:**
1. Data → What-If Analysis → Goal Seek
2. Set cell: [Gap to Target cell]
3. To value: 0
4. By changing cell: [Volume Mult, Win Rate Mult, or Deal Size Mult]

### 11.4 Which Lever to Adjust?

| Lever | Business Meaning | Realistic Range |
|-------|------------------|-----------------|
| Volume | More salespeople, better lead gen, market expansion | 0.8× to 1.5× |
| Win Rate | Better sales execution, product-market fit | 0.9× to 1.3× |
| Deal Size | Price increases, upselling, larger customers | 0.9× to 1.2× |

**Warning:** If Goal Seek returns a multiplier > 2.0×, the target is likely unrealistic.

---

## 12. Excel Output Reference

### 12.1 Sheet: Summary

One row per segment with key metrics for quick reference.

| Column | Formula/Source |
|--------|----------------|
| market_segment | From data |
| avg_monthly_volume | T12 adjusted volume / months in T12 |
| blended_win_rate | 0.55 × T12 + 0.45 × All-Time |
| blended_deal_size | 0.55 × T12 + 0.45 × All-Time |
| t12_sample_size | Count of T12 deals |
| open_deals | Count of open deals |
| open_pipeline_value | Σ net_revenue for open deals |
| expected_pipeline_value | Σ (net_revenue × probability) |
| baseline_annual_revenue | vol × 12 × wr × size |

### 12.2 Sheet: Goal_Seek

**Section 1: Baseline Metrics (Row 4-9)**
- Locked values from historical data
- Reference these in formulas, never edit

**Section 2: Scenario Levers (Row 12-17)**
- Yellow cells = user inputs
- Blue text = editable values
- Gap to Target = formula

**Section 3: Forecast Output (Row 20-26)**
- All formulas referencing Baseline × Levers
- Total row at bottom

### 12.3 Formula Reference (Goal_Seek Sheet)

Assuming Large Market is in row 6 (baseline) and row 15 (levers):

| Cell | Formula | Description |
|------|---------|-------------|
| B24 | `=B6*B15` | Adj Volume = Base Vol × Vol Mult |
| C24 | `=MIN(C6*C15,1)` | Adj WR = min(Base WR × WR Mult, 100%) |
| D24 | `=D6*D15` | Adj Size = Base Size × Size Mult |
| E24 | `=B24*12*C24*D24` | Future Pipeline Revenue |
| F24 | `=F6` | Active Pipeline (fixed) |
| G24 | `=E24+F24` | Total Forecast |
| F15 | `=E15-G24` | Gap = Target - Forecast |

---

## 13. Worked Example

### 13.1 Scenario: Large Market Target of $5,000,000

**Current State:**
- Baseline Annual Revenue: $1,939,550
- Expected Pipeline: $1,244,979
- Total Baseline Forecast: $3,184,529
- Target: $5,000,000
- Gap: $1,815,471

**What multiplier do we need?**

Future Pipeline must increase from $1,939,550 to $3,755,021 ($5M - $1.245M active)

Required multiplier on Future Pipeline = $3,755,021 / $1,939,550 = **1.94×**

**Option A: Volume Only**
- Need Volume Multiplier = 1.94
- Meaning: Nearly double the deal creation rate
- Feasibility: Requires significant headcount or market expansion

**Option B: Even Split (Cube Root)**
- Each lever = 1.94^(1/3) = 1.25
- Volume: 1.25× (25% more deals)
- Win Rate: 1.25× (25% better conversion)  
- Deal Size: 1.25× (25% larger deals)
- Feasibility: More balanced but still aggressive

### 13.2 Reality Check

| Multiplier | Assessment |
|------------|------------|
| < 1.2× | Achievable with execution improvement |
| 1.2× - 1.5× | Requires investment or market changes |
| 1.5× - 2.0× | Aggressive; needs structural change |
| > 2.0× | Likely unrealistic; revisit target |

---

## 14. Assumptions & Limitations

### 14.1 Key Assumptions

| Assumption | Implication |
|------------|-------------|
| Verbal = Won | Verbal commitments are treated as closed won |
| Skipper Weight = 0.5 | Direct-to-lost deals half-count in volume |
| T12 = Recent Trend | Trailing 12 months represents current performance |
| 55/45 Blend | Slight preference for recent data over all-time |
| Credibility Threshold = 50 | Segments with <50 exits blend toward global |
| Stage Before Exit | Probability based on last working stage, not closure stage |

### 14.2 What This Model Does NOT Include

| Factor | Status |
|--------|--------|
| Seasonality | Not modeled (assumes even monthly distribution) |
| Sales cycle timing | Captured in Timing_Distribution but not in forecast |
| Staleness penalties | Not applied in this simplified model |
| Segment-specific growth | Assumes same multipliers apply to future |
| Market conditions | No external factor adjustments |
| Churn/contraction | Only models new business |

### 14.3 When to Update

- **Monthly:** Refresh with new snapshot data
- **Quarterly:** Review and adjust baseline assumptions
- **Annually:** Full recalibration of blend weights and thresholds

---

## Appendix A: Quick Reference Formulas

```
Volume Weight:
  = 0.5 if first_stage == 'Closed Lost'
  = 1.0 otherwise

Win Rate (Revenue):
  = Σ(won_revenue × volume_weight) / Σ(all_revenue × volume_weight)

Blended Win Rate:
  = 0.55 × T12_win_rate + 0.45 × All_time_win_rate

Stage Probability:
  raw_prob = weighted_wins / weighted_total
  credibility = min(weighted_total / 50, 1.0)
  blended = raw_prob × credibility + global_prob × (1 - credibility)
  final = max(blended, 0.95 × global_prob)

Expected Pipeline:
  = Σ(net_revenue × final_probability) for all open deals

Baseline Annual Revenue:
  = avg_monthly_volume × 12 × blended_win_rate × blended_deal_size

Total Forecast:
  = (Adj_Volume × 12 × Adj_WR × Adj_Size) + Expected_Pipeline
```

---

## Appendix B: File Manifest

| File | Purpose |
|------|---------|
| `export_goal_seek_baseline.py` | Main export script |
| `goal_seek_baseline.xlsx` | Output workbook |
| `generate_sample_data.py` | Test data generator |
| `recalc.py` | Excel formula recalculation |
| `fact_snapshots.csv` | Source data (not included) |

---

## Appendix C: Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Win rate = 0 for a segment | No wins in T12 or all-time | Check data; may need to exclude segment |
| Stage probability = 0 | No exits from that stage | Defaults to global probability × 0.95 |
| Expected pipeline = 0 | No open deals or all in Closed stages | Verify ACTUALS_THROUGH date |
| Goal Seek returns >3× | Target is unrealistic | Lower target or reconsider strategy |
| Baseline differs from Python forecast | Different blend weights or logic | Verify both use same formulas |

---

## Appendix D: Actual Output Walkthrough (Test Data)

This section shows actual output from the script with test data to validate the math.

### Summary by Segment

| Segment | Avg Vol | Win Rate | Deal Size | Baseline Rev |
|---------|---------|----------|-----------|--------------|
| Indirect | 31.0 | 15.7% | $44,188 | $2,582,997 |
| Large Market | 7.8 | 11.3% | $185,361 | $1,939,550 |
| Mid Market | 24.8 | 16.7% | $63,822 | $3,168,910 |
| Other | 5.3 | 24.2% | $23,735 | $367,858 |
| SMB | 38.7 | 21.9% | $34,756 | $3,528,065 |

### Open Pipeline

- **Total Open Deals:** 150
- **Unweighted Value:** $10,092,790
- **Expected Value:** $2,152,282
- **Implied Win Rate:** 21.3%

### Total Baseline Forecast

| Component | Value |
|-----------|-------|
| Future Pipeline (new deals) | $11,587,379 |
| Active Pipeline (open deals) | $2,152,282 |
| **TOTAL FY26 BASELINE** | **$13,739,662** |

### Validation Check

**Formula:** `Baseline = AvgVol × 12 × WinRate × DealSize`

**Large Market:**
```
7.75 × 12 × 0.1125 × $185,361 = $1,939,339 ✓
```

**SMB:**
```
38.7 × 12 × 0.2186 × $34,756 = $3,528,354 ✓
```

Small differences from table values are due to rounding in display vs. full precision in calculations.

---

*Document Version: 1.0*  
*Last Updated: 2026-02-02*  
*Compatible With: export_goal_seek_baseline.py v1.0*
