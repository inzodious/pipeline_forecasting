import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================

DATA_PATH = './data/fact_snapshots.csv'

EXPORT_DIR = './exports'
VALIDATION_DIR = './validation'
ASSUMPTIONS_DIR = './assumptions_log'

FORECAST_START = '2026-01-01'
FORECAST_END = '2026-12-31'
ACTUALS_THROUGH = '2025-12-26'

RUN_BACKTEST = False

SCENARIO_LEVERS = {
    'volume_multiplier': 1.0,
    'win_rate_multiplier': 1.0,
    'deal_size_multiplier': 1.0
}

RUN_GOAL_SEEK = True
GOAL_SEEK_MODE = 'volume'   # 'volume' or 'win_rate'
GOAL_WON_REVENUE = {
    'Large Market': 14_000_000,
    'Mid Market/SMB': 7_000_000,
    'Indirect': 3_000_000
}

STALENESS_PENALTY = 0.8
SKIPPER_WEIGHT = 0.5

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(ASSUMPTIONS_DIR, exist_ok=True)

# ======================================================
# LOAD & NORMALIZE
# ======================================================

def load_snapshots():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=['date_created', 'date_closed', 'date_snapshot']
    )

    df = df.sort_values(['deal_id', 'date_snapshot'])

    df['is_closed_won'] = df['stage'] == 'Closed Won'
    df['is_closed_lost'] = df['stage'] == 'Closed Lost'
    df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']

    return df

# ======================================================
# DEAL-LEVEL FACT TABLE (CRITICAL)
# ======================================================

def build_deal_facts(df):
    first = (
        df.groupby('deal_id')
          .first()
          .reset_index()
    )

    closed = (
        df[df['is_closed']]
        .groupby('deal_id')
        .first()
        .reset_index()[['deal_id', 'date_closed', 'stage']]
    )

    deals = first.merge(closed, on='deal_id', how='left')
    deals['created_month'] = deals['date_created'].dt.to_period('M')
    deals['won'] = deals['stage'] == 'Closed Won'

    return deals

# ======================================================
# SKIPPER LOGIC (VOLUME ONLY)
# ======================================================

def apply_skipper_weights(df):
    first_stage = df.groupby('deal_id')['stage'].first()
    skippers = first_stage[first_stage == 'Closed Lost'].index

    df['volume_weight'] = 1.0
    df.loc[df['deal_id'].isin(skippers), 'volume_weight'] = SKIPPER_WEIGHT
    return df

# ======================================================
# ASSUMPTIONS EXPORTS
# ======================================================

def export_assumptions(deals):
    volume = (
        deals.groupby(['market_segment', 'created_month'])
             .agg(deals_created=('deal_id', 'count'))
             .reset_index()
    )

    win = (
        deals.groupby(['market_segment', 'created_month'])
             .agg(
                 deals=('deal_id', 'count'),
                 wins=('won', 'sum')
             )
             .reset_index()
    )
    win['win_rate'] = win['wins'] / win['deals']

    size = (
        deals.groupby(['market_segment', 'created_month'])
             .agg(avg_deal_size=('net_revenue', 'mean'))
             .reset_index()
    )

    volume.to_csv(f'{ASSUMPTIONS_DIR}/volume_by_month.csv', index=False)
    win.to_csv(f'{ASSUMPTIONS_DIR}/win_rates_by_month.csv', index=False)
    size.to_csv(f'{ASSUMPTIONS_DIR}/deal_size_by_month.csv', index=False)

# ======================================================
# VINTAGE MATURITY CURVES (DEAL-LEVEL)
# ======================================================

def build_vintage_curves(deals):
    won = deals[deals['won'] & deals['date_closed'].notna()].copy()
    won['weeks_to_close'] = (
        (won['date_closed'] - won['date_created']).dt.days // 7
    ).clip(lower=0)

    curve = (
        won.groupby(['market_segment', 'weeks_to_close'])
           .agg(won_revenue=('net_revenue', 'sum'))
           .groupby(level=0)
           .cumsum()
           .reset_index()
    )

    totals = (
        won.groupby('market_segment')['net_revenue']
           .sum()
           .rename('total')
    )

    curve = curve.merge(totals, on='market_segment')
    curve['cum_win_pct'] = curve['won_revenue'] / curve['total']

    return curve[['market_segment', 'weeks_to_close', 'cum_win_pct']]

# ======================================================
# STAGE PROBABILITIES + STALENESS
# ======================================================

def build_stage_probabilities(df):
    exits = df[df['is_closed']]
    probs = (
        exits.groupby(['market_segment', 'stage'])
             .agg(
                 wins=('is_closed_won', 'sum'),
                 total=('deal_id', 'count')
             )
             .reset_index()
    )
    probs['prob'] = probs['wins'] / probs['total']
    return probs[['market_segment', 'stage', 'prob']]

def build_staleness_thresholds(df):
    df['age_weeks'] = (
        (df['date_snapshot'] - df['date_created']).dt.days // 7
    ).clip(lower=0)

    return (
        df.groupby(['market_segment', 'stage'])['age_weeks']
          .quantile(0.9)
          .reset_index(name='stale_after_weeks')
    )

# ======================================================
# ACTIVE PIPELINE FORECAST (LAYER 2)
# ======================================================

def forecast_active_pipeline(df, stage_probs, staleness):
    snap = df[df['date_snapshot'] == pd.to_datetime(ACTUALS_THROUGH)]
    open_deals = snap[~snap['is_closed']].copy()

    open_deals['age_weeks'] = (
        (open_deals['date_snapshot'] - open_deals['date_created']).dt.days // 7
    )

    open_deals = open_deals.merge(stage_probs, on=['market_segment', 'stage'], how='left')
    open_deals = open_deals.merge(staleness, on=['market_segment', 'stage'], how='left')

    open_deals['prob'] = open_deals['prob'].fillna(0)
    open_deals.loc[
        open_deals['age_weeks'] > open_deals['stale_after_weeks'],
        'prob'
    ] *= STALENESS_PENALTY

    open_deals['expected_revenue'] = open_deals['net_revenue'] * open_deals['prob']
    open_deals['expected_count'] = open_deals['prob']

    open_deals['month'] = pd.to_datetime(ACTUALS_THROUGH).to_period('M').to_timestamp()
    return open_deals[['month', 'market_segment', 'expected_revenue', 'expected_count']]

# ======================================================
# FUTURE PIPELINE (AGING COHORTS)
# ======================================================

def forecast_future_pipeline(deals, curves):
    base = (
        deals.groupby(['market_segment', 'created_month'])
             .agg(
                 volume=('deal_id', 'count'),
                 avg_size=('net_revenue', 'mean')
             )
             .reset_index()
    )

    months = pd.period_range(FORECAST_START, FORECAST_END, freq='M')
    rows = []

    for _, r in base.iterrows():
        for m in months:
            age_weeks = max((m.to_timestamp() - r['created_month'].to_timestamp()).days // 7, 0)
            rows.append({
                'month': m.to_timestamp(),
                'market_segment': r['market_segment'],
                'age_weeks': age_weeks,
                'volume': r['volume'] * SCENARIO_LEVERS['volume_multiplier'],
                'avg_size': r['avg_size'] * SCENARIO_LEVERS['deal_size_multiplier']
            })

    future = pd.DataFrame(rows)

    future = future.merge(
        curves,
        left_on=['market_segment', 'age_weeks'],
        right_on=['market_segment', 'weeks_to_close'],
        how='left'
    )

    future['cum_win_pct'] = (
        future.groupby('market_segment')['cum_win_pct']
              .ffill()
              .fillna(0)
    )

    future['cum_win_pct'] *= SCENARIO_LEVERS['win_rate_multiplier']

    future['expected_revenue'] = future['volume'] * future['avg_size'] * future['cum_win_pct']
    future['expected_count'] = future['volume'] * future['cum_win_pct']

    return future[['month', 'market_segment', 'expected_revenue', 'expected_count']]

# ======================================================
# GOAL SEEK
# ======================================================

def run_goal_seek(forecast):
    summary = (
        forecast.groupby('market_segment')
                .agg(won_revenue=('expected_revenue', 'sum'))
                .reset_index()
    )

    results = []

    for _, r in summary.iterrows():
        target = GOAL_WON_REVENUE.get(r['market_segment'])
        if not target or r['won_revenue'] == 0:
            continue

        multiplier = target / r['won_revenue']

        results.append({
            'market_segment': r['market_segment'],
            'baseline_won_revenue': round(r['won_revenue'], 2),
            'target_won_revenue': target,
            'required_multiplier': round(multiplier, 3),
            'mode': GOAL_SEEK_MODE
        })

    pd.DataFrame(results).to_csv(
        f'{EXPORT_DIR}/goal_seek_results.csv',
        index=False
    )

# ======================================================
# MAIN
# ======================================================

def run_forecast():
    snapshots = load_snapshots()
    snapshots = apply_skipper_weights(snapshots)

    deals = build_deal_facts(snapshots)
    export_assumptions(deals)

    curves = build_vintage_curves(deals)
    stage_probs = build_stage_probabilities(snapshots)
    staleness = build_staleness_thresholds(snapshots)

    active = forecast_active_pipeline(snapshots, stage_probs, staleness)
    future = forecast_future_pipeline(deals, curves)

    forecast = (
        pd.concat([active, future])
          .groupby(['month', 'market_segment'])
          .sum()
          .reset_index()
    )

    forecast.to_csv(f'{EXPORT_DIR}/forecast_2026.csv', index=False)

    with open(f'{EXPORT_DIR}/assumptions.json', 'w') as f:
        json.dump(SCENARIO_LEVERS, f, indent=4)

    if RUN_GOAL_SEEK:
        run_goal_seek(forecast)

    return forecast

if __name__ == "__main__":
    run_forecast()
