import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================

DATA_PATH = '../data/fact_snapshots.csv'
EXPORT_DIR = '../exports'
VALIDATION_DIR = '../validation'

FORECAST_START = '2026-01-01'
FORECAST_END = '2026-12-31'
ACTUALS_THROUGH = '2025-12-31'

RUN_BACKTEST = True
BACKTEST_ASOF = '2025-01-01'

SCENARIO_LEVERS = {
    'volume_multiplier': 1.00,
    'win_rate_multiplier': 1.00,
    'deal_size_multiplier': 1.00
}

STALENESS_PENALTY = 0.8
SKIPPER_WEIGHT = 0.5

RUN_GOAL_SEEK = True

GOAL_SEEK_MODE = 'volume'   # 'volume' or 'win_rate'

GOAL_WON_REVENUE = {
    'Large Market': 14_000_000,
    'Mid Market/SMB': 6_500_000,
    'Indirect': 2_000_000
}


os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# ======================================================
# LOAD & NORMALIZE
# ======================================================

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=[
        'date_created', 'date_closed', 'date_snapshot'
    ])

    df = df.sort_values(['deal_id', 'date_snapshot'])
    df['created_month'] = df['date_created'].dt.to_period('M')
    df['age_weeks'] = ((df['date_snapshot'] - df['date_created']).dt.days // 7).clip(lower=0)

    df['is_closed_won'] = df['stage'] == 'Closed Won'
    df['is_closed_lost'] = df['stage'] == 'Closed Lost'
    df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']

    return df

# ======================================================
# SKIPPER LOGIC
# ======================================================

def apply_skipper_weights(df):
    first_stage = df.groupby('deal_id')['stage'].first()
    skippers = first_stage[first_stage == 'Closed Lost'].index

    df['volume_weight'] = 1.0
    df.loc[df['deal_id'].isin(skippers), 'volume_weight'] = SKIPPER_WEIGHT
    return df

# ======================================================
# LAYER 1 — VINTAGE MATURITY CURVES
# ======================================================

def build_vintage_curves(df):
    closed = df[df['is_closed_won']].copy()
    closed['week_of_win'] = closed['age_weeks']

    curves = (
        closed
        .groupby(['market_segment', 'week_of_win'])
        .agg(won_revenue=('net_revenue', 'sum'))
        .groupby(level=0)
        .cumsum()
        .reset_index()
    )

    totals = (
        closed
        .groupby('market_segment')['net_revenue']
        .sum()
        .rename('total_revenue')
    )

    curves = curves.merge(totals, on='market_segment')
    curves['cum_win_pct'] = curves['won_revenue'] / curves['total_revenue']
    return curves[['market_segment', 'week_of_win', 'cum_win_pct']]

# ======================================================
# LAYER 2 — STAGE PROBABILITIES & STALENESS
# ======================================================

def build_stage_probabilities(df):
    exits = df[df['is_closed']]
    stage_probs = (
        exits
        .groupby(['market_segment', 'stage'])
        .agg(
            won=('is_closed_won', 'sum'),
            total=('deal_id', 'count')
        )
        .reset_index()
    )
    stage_probs['prob'] = stage_probs['won'] / stage_probs['total']
    return stage_probs[['market_segment', 'stage', 'prob']]

def build_staleness_thresholds(df):
    df['stage_age'] = df.groupby(['deal_id', 'stage'])['age_weeks'].transform('max')
    return (
        df.groupby(['market_segment', 'stage'])['stage_age']
        .quantile(0.9)
        .rename('stale_after_weeks')
        .reset_index()
    )

# ======================================================
# ACTIVE PIPELINE FORECAST (LAYER 2)
# ======================================================

def forecast_active_pipeline(df, stage_probs, staleness):
    snapshot = df[df['date_snapshot'] == pd.to_datetime(ACTUALS_THROUGH)]
    open_deals = snapshot[~snapshot['is_closed']].copy()

    open_deals = open_deals.merge(
        stage_probs,
        on=['market_segment', 'stage'],
        how='left'
    )

    open_deals = open_deals.merge(
        staleness,
        on=['market_segment', 'stage'],
        how='left'
    )

    open_deals['prob'] = open_deals['prob'].fillna(0)
    open_deals['is_stale'] = open_deals['age_weeks'] > open_deals['stale_after_weeks']
    open_deals.loc[open_deals['is_stale'], 'prob'] *= STALENESS_PENALTY

    open_deals['expected_revenue'] = open_deals['net_revenue'] * open_deals['prob']
    open_deals['expected_count'] = open_deals['prob']

    return open_deals

# ======================================================
# FUTURE PIPELINE FORECAST (LAYER 1)
# ======================================================

def forecast_future_pipeline(df, vintage_curves):
    hist = df[df['date_created'] <= ACTUALS_THROUGH]
    base = (
        hist
        .groupby(['market_segment', 'created_month'])
        .agg(
            volume=('deal_id', 'nunique'),
            avg_size=('net_revenue', 'mean')
        )
        .reset_index()
    )

    monthly = (
        base
        .groupby('market_segment')
        .agg(
            volume=('volume', 'mean'),
            avg_size=('avg_size', 'mean')
        )
        .reset_index()
    )

    monthly['volume'] *= SCENARIO_LEVERS['volume_multiplier']
    monthly['avg_size'] *= SCENARIO_LEVERS['deal_size_multiplier']

    months = pd.period_range(FORECAST_START, FORECAST_END, freq='M')
    rows = []

    for _, r in monthly.iterrows():
        for m in months:
            rows.append({
                'forecast_month': m.to_timestamp(),
                'market_segment': r['market_segment'],
                'volume': r['volume'],
                'avg_size': r['avg_size']
            })

    future = pd.DataFrame(rows)
    future = future.merge(vintage_curves, on='market_segment', how='left')
    future['expected_revenue'] = (
        future['volume'] *
        future['avg_size'] *
        future['cum_win_pct'] *
        SCENARIO_LEVERS['win_rate_multiplier']
    )

    future['expected_count'] = future['volume'] * future['cum_win_pct']
    return future

# ======================================================
# AGGREGATION
# ======================================================

def aggregate(active, future):
    active['month'] = pd.to_datetime(ACTUALS_THROUGH).to_period('M').to_timestamp()
    future['month'] = future['forecast_month']

    all_rows = pd.concat([
        active[['month', 'market_segment', 'expected_revenue', 'expected_count']],
        future[['month', 'market_segment', 'expected_revenue', 'expected_count']]
    ])

    return (
        all_rows
        .groupby(['month', 'market_segment'])
        .sum()
        .reset_index()
    )

# ======================================================
# BACKTEST
# ======================================================

def run_backtest(df):
    df_bt = df[df['date_snapshot'] <= BACKTEST_ASOF]
    vintage = build_vintage_curves(df_bt)
    stage_probs = build_stage_probabilities(df_bt)
    staleness = build_staleness_thresholds(df_bt)

    active = forecast_active_pipeline(df_bt, stage_probs, staleness)
    future = forecast_future_pipeline(df_bt, vintage)
    forecast = aggregate(active, future)

    actuals = (
        df[df['is_closed_won']]
        .groupby(df['date_closed'].dt.to_period('M'))
        ['net_revenue']
        .sum()
        .reset_index()
    )

    forecast.to_csv(f'{VALIDATION_DIR}/backtest_results.csv', index=False)

# ======================================================
# GOAL SEEK
# ======================================================

def run_goal_seek(baseline_forecast):
    results = []

    summary = (
        baseline_forecast
        .groupby('market_segment')
        .agg(
            baseline_won_revenue=('expected_revenue', 'sum'),
            baseline_won_count=('expected_count', 'sum')
        )
        .reset_index()
    )

    for _, row in summary.iterrows():
        segment = row['market_segment']
        baseline_rev = row['baseline_won_revenue']

        target_rev = GOAL_WON_REVENUE.get(segment)
        if not target_rev or baseline_rev == 0:
            continue

        multiplier = target_rev / baseline_rev

        if GOAL_SEEK_MODE == 'volume':
            volume_multiplier = multiplier
            win_rate_multiplier = 1.0
        elif GOAL_SEEK_MODE == 'win_rate':
            volume_multiplier = 1.0
            win_rate_multiplier = min(multiplier, 1.0)
        else:
            raise ValueError("Invalid GOAL_SEEK_MODE")

        results.append({
            'market_segment': segment,
            'baseline_won_revenue': round(baseline_rev, 2),
            'target_won_revenue': round(target_rev, 2),
            'required_multiplier': round(multiplier, 3),
            'volume_multiplier': round(volume_multiplier, 3),
            'win_rate_multiplier': round(win_rate_multiplier, 3),
            'goal_seek_mode': GOAL_SEEK_MODE
        })

    return pd.DataFrame(results)


# ======================================================
# MAIN
# ======================================================

def run_forecast():
    df = load_data()
    df = apply_skipper_weights(df)

    vintage = build_vintage_curves(df)
    stage_probs = build_stage_probabilities(df)
    staleness = build_staleness_thresholds(df)

    active = forecast_active_pipeline(df, stage_probs, staleness)
    future = forecast_future_pipeline(df, vintage)

    baseline = aggregate(active, future)
    baseline.to_csv(f'{EXPORT_DIR}/forecast_2026_baseline.csv', index=False)

    if RUN_GOAL_SEEK:
        goal_seek = run_goal_seek(baseline)
        goal_seek.to_csv(f'{EXPORT_DIR}/goal_seek_results.csv', index=False)

    with open(f'{EXPORT_DIR}/assumptions_log.json', 'w') as f:
        json.dump({
            'scenario_levers': SCENARIO_LEVERS,
            'goal_seek_mode': GOAL_SEEK_MODE,
            'goal_won_revenue': GOAL_WON_REVENUE
        }, f, indent=4)

    return baseline


if __name__ == "__main__":
    run_forecast()
