## 📊 Sales Pipeline Forecasting for an EAP Company (Refined Prompt)

### Objective
Design a **professional-grade, enterprise-credible sales pipeline forecasting methodology** for a **client / deal-based service business**, using historical deal snapshot data.

The forecast will:
- Project **2026 monthly performance**
- Support **scenario-based management levers**
- Be implemented in **Microsoft Fabric (PySpark)**
- Remain **fully explainable, auditable, and Excel-verifiable**

The goal is **not maximum theoretical accuracy**, but a model that is:
- Directionally correct
- Stable over time
- Defensible to executives
- Capable of back-testing against known historical outcomes

---

## 1. Source Data Description

The source data consists of **weekly deal snapshots** spanning **1/1/2024 – 12/31/2025**. Each snapshot shares the same schema:

```
deal_id
deal_name
deal_owner
market_segment
net_revenue
date_created
date_closed
date_implementation
date_snapshot
stage
```

### Snapshot Behavior
- Snapshots are captured **weekly**
- Deals may change:
  - Stage
  - Revenue
  - Implementation date
- **Closed deals persist indefinitely** in the dataset with their final status
- Deal creation dates range from **1/1/2020 – 12/31/2025**

The dataset represents an **operational CRM reality**, including known data entry inconsistencies.

---

## 2. Pipeline & Stage Behavior

### Stages (Typical Flow)
1. Qualified  
2. Alignment  
3. Solutioning  
4. Closed Won / Closed Lost

### Observed Behaviors
- Approximately **68% of deals first appear in “Qualified”**
- A non-trivial number of deals **skip the pipeline entirely** and first appear as:
  - Closed Won
  - Closed Lost

These behaviors are believed to reflect **process and data entry gaps**, not true absence of pipeline activity.

📌 **Modeling Assumption**  
Deals that first appear directly as *Closed Lost* without ever appearing in Qualified should be **counted at 50% weight** when calculating deal volume.

This assumption is intended to balance realism with data imperfections and should remain adjustable.

---

## 3. Market Segments

Forecasting must be performed separately for each **market segment**, recognizing distinct behavior patterns:

- **Large Market (LM)** – highest revenue per deal
- **Mid Market (MM)**
- **Small Market (SMB)**
- **Indirect**
  - Mixed deal sizes
  - Consistently **2nd highest deal volume in terms of revenue** in aggregate

Win rates, revenue distributions, and pipeline velocity vary:
- Month-to-month
- By market segment

---

## 4. Key Date Logic

- `date_closed` is a **derived column** that:
  - Identifies the exact snapshot date when a deal first reached a closed status
  - Is populated consistently across all snapshot rows for that deal

- **Days Sales Outstanding (DSO)** is defined as:
  
  `date_closed - date_created`

DSO differs:
- By creation cohort
- By market segment

DSO should be treated as a **distribution**, not a single point estimate.

---

## 5. Required Derived Metrics (Forecast Inputs)

### 1️⃣ Deal Volume (Deal Entry)

- Monthly count of deals created
- Includes deals that first appear in **any stage**
- Deals that skip directly to *Closed Lost* are counted at **0.5 weight**

This metric represents **pipeline inflow**, not sales performance.

---

### 2️⃣ Win Rate / Revenue Conversion

Initial framing (subject to refinement):

> By **deal creation month**:  
> `Won Revenue ÷ Total Qualified Revenue`

Example:
- 10 deals created in January
- 2 Closed Won, 8 Closed Lost
- Deal win rate = **20%**
- Revenue win rate =  
  `Revenue from Closed Won deals ÷ Total created revenue`

⚠️ **Known Complication**
- Recent creation cohorts do not yet have sufficient closed outcomes
- Snapshot-based or naïve monthly win rates can be misleading

The forecasting method should explicitly address:
- Incomplete cohorts
- Time lag between creation and close
- Segment-specific variation

Guidance is requested on **best-practice enterprise approaches** for deriving and applying win probabilities.

---

### 3️⃣ Pipeline Velocity (DSO)

- Represents time-to-close behavior
- Used to:
  - Shift deal inflow into expected close periods
  - Determine revenue timing

Velocity assumptions should avoid false precision and favor **stable historical patterns**.

---

### 4️⃣ Starting Pipeline (Carryover)

- Deals created in **Q4 2025** will largely remain open entering 2026
- These deals must be:
  - Included in Q1 2026 pipeline according to DSO patterns,
  - Closed probabilistically over subsequent months

The forecast should model the pipeline as a **continuous system**, not a calendar reset.

---

## 6. Scenario & Management Levers

The forecasting framework must support controlled scenario adjustments, including:

1. **Win Rate Improvement**  
   - e.g., +5% relative or absolute uplift

2. **Deal Volume Growth**  
   - e.g., +5% increase in pipeline inflow

3. **Revenue per Deal Increase**  
   - e.g., +5% pricing uplift by market segment

4. **Top-Line Revenue Target**
  - OPTIONAL: The model should be capable of forecasting with all other inputs and assumptions with this value set to 0.
   - e.g., Management mandates **$30M annual revenue**
   - Model must back-solve:
     - Required pipeline volume
     - Segment allocation

Levers should adjust assumptions **once**, avoiding stacked optimism.

---

## 7. Forecast Output Requirements

For forward-looking forecasts (e.g., 2026), produce **monthly (EOM)** outputs by market segment:

### Deal Counts
- Total open pipeline deal count
- Total expected won deal count (probability-weighted)

### Revenue
- Total open pipeline revenue (unweighted)
- Total expected won revenue (probability-weighted)

## 7a. Back-Testing & Validation Outputs (Historical Periods Only)

When forecasting historical periods for validation purposes, also compute:

### Deal Counts
- Total actual closed-won deal count

### Revenue
- Total actual closed-won revenue


Deal-level forecasts are **not required** for final outputs.

---

## 8. Technical & Governance Constraints

- Implemented in **Microsoft Fabric**
- Core logic housed in a **PySpark notebook**
- Intermediate tables exposed for inspection
- Excel recreation of logic must be feasible
- Assumptions must be:
  - Explicit
  - Versioned
  - Documented

---

## 9. Requested Deliverable

Provide **at least 5 professional, enterprise-accepted forecasting methods** that:

- Are commonly used by large organizations
- Apply to **service-based, deal-driven revenue models**
- Prioritize:
  - Explainability
  - Stability
  - Governance

Each method should include:
- Conceptual overview
- How it uses this dataset
- How it handles incomplete cohorts
- Back-testing methodology
- Strengths and limitations
- Why an enterprise would select it
- Any known examples of organizations that utilize this method w/ references

All methods must support:
- Historical back-testing
- Forecast vs actual comparison
- Acceptable, explainable variance

The resulting methodology should be suitable for **automation, executive communication, and long-term operational use**.

