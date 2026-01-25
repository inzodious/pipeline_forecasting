# 📊 Technical Specification: Hybrid Sales Pipeline Forecasting Model

## 1. Executive Summary

This document defines the requirements, assumptions, and logic for implementing a **Two-Layer Hybrid Forecasting Model** for a deal-based service business. The model combines:

- **Cohort-Based Vintage Analysis** (strategic baseline)
- **Stage-Weighted Probability Modeling** (operational overlay)

This hybrid approach is designed to satisfy enterprise requirements for:
- Auditability
- Defensibility
- Stability over time
- Excel-level verifiability

**Primary Objective**  
Project **monthly 2026 deal volume and revenue by market segment** using historical **weekly snapshot data (1/1/2024 – 12/31/2025)**.

---

## 2. Data Context & Core Assumptions

### 2.1 Source Data Structure

The model will be implemented in **Microsoft Fabric using PySpark**, operating on weekly deal snapshots.

**Historical Training Window**  
- 1/1/2024 – 12/31/2025

**Key Fields**
- deal_id
- date_created
- date_closed
- date_snapshot
- stage
- net_revenue
- market_segment

**Data Reality**  
The dataset reflects operational CRM imperfections, including:
- Deals skipping pipeline stages
- Retroactive or delayed data entry
- Persistent closed deals across snapshots

---

### 2.2 Critical Modeling Assumptions

To produce stable and realistic estimates of deal volume and win probability, the following assumptions apply:

#### Skipper Logic (Direct-to-Lost Deals)
Deals that first appear directly as **Closed Lost** without ever passing through **Qualified, Alignment, or Solutioning** are partially credited.

- These deals receive a **0.5 weight** in deal volume calculations
- This assumption compensates for known data entry gaps

#### Open Deal Valuation (Right Censoring)
- Deals that remain open at the end of 2025 are **right-censored**, not assumed lost
- Their expected value is projected using historical cohort maturity patterns

#### Closed Deal Persistence
- Closed deals persist indefinitely in snapshots
- Forecast logic must identify the **first snapshot where closure occurs** to establish the true `date_closed`

---

## 3. Methodology: “The Vintage-Weighted Hybrid”

### Layer 1: Cohort-Based Vintage Analysis (The “When”)

**Purpose**  
Addresses the *incomplete cohort problem* and establishes expected timing of deal closure.

**Function**
- Group historical deals by **creation month or week (vintage)**
- Track cumulative win behavior over time
- Measure how probability of win increases as deals age

**Why This Layer Exists**
- Prevents recent cohorts (e.g., Q4 2025) from appearing artificially weak
- Establishes realistic revenue timing assumptions

**Primary Output**
- Lookup table of **Cumulative Win Percentage by Week of Age**, partitioned by market segment

---

### Layer 2: Stage-Weighted Probability (The “How Much”)

**Purpose**  
Provides the operational, deal-level forecast for active pipeline.

**Function**
- Assign probabilities to deals based on current stage
- Apply penalties for pipeline stagnation

**Why This Layer Exists**
- Produces an auditable forecast (Deal Value × Probability)
- Aligns with Finance and CFO expectations

**Primary Output**
- Monthly expected revenue derived directly from open opportunities

---

## 4. Implementation Inputs & Logic (PySpark)

### 4.1 Derived Inputs

The following inputs must be calculated prior to generating forecasts:

| Metric | Logic Description |
|------|-------------------|
| Vintage Maturity Curve | Cumulative % of total qualified revenue that is closed-won by week of age (t=0 → t=max), segmented by market_segment |
| Stage Probability | Historical ratio of Closed Won deals to total deals exiting each stage |
| Staleness Threshold | 90th percentile DSO per stage; deals exceeding this age receive a probability penalty |

---

### 4.2 Handling “Skippers” (Direct-to-Lost)

In the **Deal Volume** calculation step:

```
IF stage == 'Closed Lost'
AND previous_stage IS NULL
THEN volume_weight = 0.5
ELSE volume_weight = 1.0
```

---

### 4.3 Forecast Calculation Flow

#### Active Pipeline Forecast (Layer 2)

1. Identify all open deals as of **12/31/2025**
2. Apply stage-based win probabilities
3. If deal age exceeds stage-specific staleness threshold:
   - Apply penalty multiplier (e.g., 0.8)

#### Future Pipeline Forecast (Layer 1)

1. Generate synthetic “dummy deals” for **Jan–Dec 2026** based on historical deal volume trends
2. Apply vintage maturity curves to project closure timing
3. Convert projected wins into monthly expected revenue

---

## 5. Management Scenario Levers

The model must support scenario testing through a **parameter table**, without modifying source data.

| Lever | Implementation Logic |
|------|----------------------|
| Win Rate Improvement | Multiply final maturity curve asymptote by uplift factor; adjust stage probabilities proportionally |
| Deal Volume Growth | Increase count of future dummy deals |
| Revenue per Deal | Apply pricing uplift to future dummy deal revenue |
| Target Back-Solve | Compute revenue gap vs target; divide by average win rate to derive required incremental pipeline |

---

## 6. Output Requirements

### 6.1 Final Forecast Table (2026)

Forecast outputs must be aggregated **monthly (EOM)** and by **market segment**.

**Metrics Required**

- **Total Open Pipeline**
  - Deal Count
  - Revenue (unweighted)

- **Expected Won Pipeline**
  - Deal Count (probability-weighted)
  - Revenue (probability-weighted)

Expected wins must include:
- Active pipeline (Layer 2)
- Future pipeline cohorts (Layer 1)

---

### 6.2 Validation Outputs (Back-Testing)

To validate stability, the model must support historical back-testing (e.g., forecasting 2025 using data available as of January 2025):

- Predicted revenue vs. actual closed-won revenue
- Variance percentage

The objective is **directional correctness with explainable variance**, not exact precision.

---

## 7. Governance & Verifiability

- **Platform**: Microsoft Fabric (PySpark notebooks)
- **Excel Auditability**:
  - Layer 2 calculations must be reproducible in Excel
  - Deal Value × Probability calculations must reconcile exactly
- **Configuration Management**:
  - All assumptions (e.g., staleness thresholds, penalty factors) must reside in configuration tables
  - No hard-coded business logic in Python

---

## Closing Note

This hybrid methodology intentionally prioritizes:
- Trust over cleverness
- Stability over overfitting
- Explainability over black-box accuracy

The result is a forecasting system suitable for executive planning, scenario modeling, and long-term operational use.