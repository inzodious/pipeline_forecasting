# Pipeline Forecasting Engine: Technical Review & Optimization Plan

---

## 1. The Critical Flaw: Layer 2 Deal Timing

**Current Score Impact:** High Negative Impact on Logic & Precision

The most significant logical gap in the current script is how **Active Pipeline (Layer 2)** is distributed over time.

### Current Behavior

* The script uses a **harmonic decay** (`1 / (i + 1)` or `1 / (i + 1)^2`) for all open deals, immediately starting in **Month 1**.

### The Problem

* This assumes every open deal, regardless of stage, begins generating probability-weighted revenue next month.
* In reality, a **Large Market** deal in **Discovery** might have a **6-month cycle time**.
* By smearing revenue starting in Month 1, revenue is pulled forward incorrectly, resulting in:

  * Near-term **over-forecasting**
  * Long-term **under-forecasting**

### The Fix: Stage-Specific *Time-to-Close* Injection

Instead of arbitrary decay, the model must calculate a specific `expected_close_date` for **every deal** in the pipeline.

#### Recommended Logic Update

##### 1. Calculate Average Cycle Time Remaining

For every `market_segment + stage` combination, calculate the historical **average days remaining to Closed Won**.

**Examples:**

* `Large Market / Proposal` → **45 days**
* `Large Market / Discovery` → **120 days**

##### 2. Apply to Individual Deals

```text
Predicted Close Date = Snapshot Date + Avg Days Remaining (Stage, Segment)
```

##### 3. Distribute Revenue

* Place probability-weighted revenue in the **month containing the predicted close date**.
* Optionally apply a **Gaussian (bell curve)** around the predicted close (±1 month) to model variance.
* The **center of gravity** for revenue must shift based on stage — not start at `t = 0`.

---

## 2. Deep Dive: Staleness & the “Conservative Trap”

**Current Score Impact:** Medium Negative Impact on Intelligence

Your backtest showed a **-30% variance (under-forecast)** in Large Market. This strongly suggests the **staleness logic is too aggressive** for complex deals.

### Current Logic

* `stale_after_weeks` is set to the **95th percentile** of deal age for a given stage.
* If `age > threshold`, probability is multiplied by **0.8**.

### The Flaw

* Large Market deals are **idiosyncratic**.
* A deal can sit in **Legal** for months and still close.
* A binary penalty (`0.8x`) is a **blunt instrument** that systematically underestimates complex deals.

### The Fix: Segment-Specific Sigmoid Decay

Do not use cliffs — use **curves**.

Large Market deals should decay **slowly**.
SMB deals should decay **quickly**.

#### Proposed Math

Create a `staleness_factor` that scales continuously based on how far past the threshold a deal is:

```text
Staleness Factor = 1 / (1 + e^(k · (Current Age − Threshold)))
```

#### Parameterization

* **SMB**

  * High `k` (≈ 0.5)
  * If stale, it is likely dead

* **Large Market**

  * Low `k` (≈ 0.1)
  * Deals degrade gently over time, reflecting long cycles

---

## 3. Production Readiness Grading

To reach **Production Ready**, grading must shift from subjective qualities to system-level guarantees:

* Observability
* Robustness
* Precision

---

### Category 1: Predictive Precision

**Target:** 9 / 10
**Current:** 7 / 10

#### Issues

* Strong high-level logic
* Poor **temporal precision**

#### Required Actions

* Implement **Time-to-Close logic** (Section 1)
* Refine **Win Rate** modeling

##### Win Rate Improvement

* Current: Blends **Global** and **Trailing-12 (T12)**
* Improvement:

  * Separate **Stock** vs **Flow**

    * *Stock*: Deals created > 6 months ago (lower win rates)
    * *Flow*: Fresh deals (higher win rates)

##### Math Validation

* Ensure:

```text
Σ(probabilities per deal_id) ≤ 1.0
```

*(Low risk today due to single-state snapshots, but critical if multi-path logic is added.)*

---

### Category 2: Observability & Logging

**Target:** 10 / 10
**Current:** 2 / 10

`print()` statements are not acceptable in production forecasting systems.

#### Required Changes

##### Replace `print()` with Structured Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Loaded {len(df)} snapshots.")
logger.warning(
    f"Segment 'Other' has low volume ({vol}). Defaulting to global priors."
)
```

##### Add Audit Logs (“The Why”)

Create a dedicated audit log (table or file) that records **decisions**, not just data.

**Example Record:**

* Deal ID: `123`
* Action: `Penalty Applied`
* Reason: `Age 45 weeks > Threshold 40 weeks`
* Impact: `Probability reduced from 0.40 → 0.32`

**Value:** Analysts can answer *“Why was this deal downgraded?”* without reading code.

---

### Category 3: Robustness & Data Validation

**Target:** 8 / 10
**Current:** 5 / 10

The current script assumes clean CSVs. In PySpark and production systems, this assumption will fail.

#### Required Validation Checks

* **Zombie Check**

  * `date_closed ≥ date_created`

* **Negative Revenue Check**

  * `net_revenue ≥ 0`

* **Stage Integrity Check**

  * If `is_closed_won == True`, stage must be `Closed Won` or `Verbal`

* **Null Trap (Fail Loudly)**

```python
required_cols = ['deal_id', 'market_segment', 'stage', 'date_snapshot']

missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")
```

---

### Category 4: Scenario Reliability

**Target:** 9 / 10
**Current:** 8 / 10

The **Goal Seek** logic is mathematically sound but operationally dangerous.

#### Critique

* A **4.8× multiplier** for Large Market is not actionable.
* This output approaches **hallucination-level guidance** for decision-makers.

#### Required Improvement: Sanity Caps

* Add explicit bounds to Goal Seek output:

```text
If required_multiplier > 1.5:
    Output: "Target unattainable via optimization. Structural change required."
```

This prevents the business from chasing impossible targets.

---

## 4. Summary of Validations to Add

Insert these checks into `run_backtest()` or the main execution flow:

| Metric               | Validation Check                       | Purpose                                          |
| -------------------- | -------------------------------------- | ------------------------------------------------ |
| Probability Cap      | `assert df['prob'].max() <= 1.0`       | Prevent >100% win likelihood                     |
| Revenue Conservation | `sum(forecast) ≈ sum(pipeline × prob)` | Ensure Layer 2 doesn’t create or destroy revenue |
| Stage Progression    | `t1_stage != t2_stage`                 | Detect invalid T12 velocity assumptions          |
| Snapshot Consistency | `count(deal_id) per snapshot`          | Detect ETL failures masquerading as sales drops  |

---

## 5. Next Step: Refactoring for PySpark

Since this is moving to **PySpark**, the architecture must shift from **iterative** to **vectorized** logic.

### Current (Will Not Scale)

```python
for _, deal in open_deals.iterrows():
    ...
```

### Future (Production-Safe)

* Compute the following as **DataFrame columns**:

  * `time_to_close`
  * `staleness_penalty`
  * `expected_revenue`
* Use:

  * `.withColumn()`
  * Window functions
  * Columnar aggregations

**Note:** The *Time-to-Close* logic is actually **simpler and safer** in vectorized PySpark than in the current loop-based implementation.
