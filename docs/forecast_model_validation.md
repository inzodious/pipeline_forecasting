# Forecast Model Validation (generate_forecast_v8.py)

## Verbal Treated as Won — Verified

The 2026 forecast **does include Verbal as Won** everywhere. Verification:

| Location | Logic |
|----------|--------|
| **load_snapshots()** | `is_closed_won = stage in ['Closed Won', 'Verbal']` |
| **build_deal_facts()** | `won = stage in ['Closed Won', 'Verbal']` (deal-level outcome) |
| **build_stage_probabilities()** | Uses `is_closed_won` from snapshots (Verbal counts as win) |
| **forecast_future_pipeline()** | Win rate and timing use `deals['won']` and `won_deals = deals[deals['won'] & ...]` |
| **forecast_active_pipeline()** | Open deals = `~is_closed`; Verbal/Closed Won are closed, so excluded from "open" (correct) |
| **Backtest actuals** | `all_act_won` uses `actual_full_deals['won']` (Verbal + Closed Won) |
| **FY26 actual_2025 (YoY)** | `deals['won']` with date_closed in 2025 (Verbal + Closed Won) |

No loose ends: every place that defines or uses "won" / "closed won" consistently includes Verbal.

---

## Why FY26 Base (~$15M) vs 2025 Actual (>$20M)

- **2025 actual** (Verbal + Closed Won, date_closed in 2025): **~$23.7M** (725 deals).
- **FY26 base forecast** (1.0x levers): **~$15.4M** (active + future pipeline only).

The forecast is **active pipeline + future pipeline only**. It does **not** add "pre-won" revenue (deals already Verbal/Closed at period start that close in the period). The backtest uses the same rule: Model = active + future only; Actual = total closed won in the year. So:

- Backtest: Model **$21.7M** vs Actual **$23.7M** — the ~$2M gap is pre-won (already won before 2025, closed in 2025).
- FY26 base **$15.4M** is the same *type* of number (active + future only).

The drop from backtest model **$21.7M** to FY26 base **$15.4M** is because:

1. **Backtest** uses "perfect prediction" levers (volume multiplier per segment set to reconcile 2025-created won revenue), so it intentionally matches 2025.
2. **FY26 base** uses 1.0x levers; future pipeline is T12 volume × win rate × deal size with no reconciliation uplift.

So with base levers, a lower number than 2025 actual is expected. For a "similar pattern" to 2025 (~$20M+), use scenario levers (e.g. Growth Recovery in README ~$19M) or Goal Seek targets.

---

## Backtest

No changes made. Backtest logic is unchanged and continues to compare model (active + future) to actual 2025 closed won (Verbal + Closed Won).
