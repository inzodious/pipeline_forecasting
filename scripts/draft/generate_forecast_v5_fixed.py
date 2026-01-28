import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================

DATA_PATH = './data/fact_snapshots.csv'
TEST_MODE = False  # Set to True to use test data

EXPORT_DIR = './exports'
VALIDATION_DIR = './validation'
ASSUMPTIONS_DIR = './assumptions_log'

FORECAST_START = '2026-01-01'
FORECAST_END = '2026-12-31'
ACTUALS_THROUGH = '2025-12-26'

RUN_BACKTEST = True
BACKTEST_DATE = '2025-01-01'  # Forecast from this date
BACKTEST_THROUGH = '2025-12-31'  # Compare against actuals through this date

SCENARIO_LEVERS = {
    'volume_multiplier': 1.0,
    'win_rate_multiplier': 1.0,
    'deal_size_multiplier': 1.0
}

# Goal Seek / What-If Analysis
RUN_GOAL_SEEK = True
GOAL_SEEK_MODE = 'volume'  # 'volume' or 'win_rate' - which lever to adjust
GOAL_WON_REVENUE = {
    'Large Market': 500_000,
    'Mid Market/SMB': 1_200_000,
    'Indirect': 400_000
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
    path = './data/test_snapshots.csv' if TEST_MODE else DATA_PATH
    df = pd.read_csv(
        path,
        parse_dates=['date_created', 'date_closed', 'date_snapshot']
    )

    df = df.sort_values(['deal_id', 'date_snapshot'])

    df['is_closed_won'] = df['stage'] == 'Closed Won'
    df['is_closed_lost'] = df['stage'] == 'Closed Lost'
    df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']

    return df

# ======================================================
# DEAL-LEVEL FACT TABLE
# ======================================================

def build_deal_facts(df, cutoff_date=None):
    """
    Build deal-level facts, optionally limiting to data before cutoff_date
    """
    if cutoff_date:
        df = df[df['date_snapshot'] <= pd.to_datetime(cutoff_date)].copy()
    
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

    first = first.drop(columns=['stage', 'date_closed'], errors='ignore')

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

def export_assumptions(deals, stage_probs, snapshots):
    """Export historical assumptions for volume, win rates, deal size, and stage probabilities"""
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
    
    # Export stage-level win rates - get the UNADJUSTED probabilities
    # Calculate from scratch without multipliers to show pure historical rates
    exits = snapshots[snapshots['is_closed']]
    historical_stage_probs = (
        exits.groupby(['market_segment', 'stage'])
             .agg(
                 wins=('is_closed_won', 'sum'),
                 total=('deal_id', 'count')
             )
             .reset_index()
    )
    historical_stage_probs['win_rate'] = historical_stage_probs['wins'] / historical_stage_probs['total']
    historical_stage_probs = historical_stage_probs[['market_segment', 'stage', 'wins', 'total', 'win_rate']]
    historical_stage_probs.to_csv(f'{ASSUMPTIONS_DIR}/win_rates_by_stage.csv', index=False)

# ======================================================
# VINTAGE MATURITY CURVES (CUMULATIVE)
# ======================================================

def build_vintage_curves(deals):
    """
    CORRECTED: Build cumulative win % by age in weeks
    This represents: of all deals CREATED, what % have closed WON by week N
    The curve asymptotes to the actual win rate (not 100%)
    """
    rows = []
    
    for segment in deals['market_segment'].unique():
        segment_deals = deals[deals['market_segment'] == segment].copy()
        total_deals = len(segment_deals)
        
        if total_deals == 0:
            continue
        
        # Get won deals and calculate their age at closure
        won_deals = segment_deals[segment_deals['won'] & segment_deals['date_closed'].notna()].copy()
        
        if len(won_deals) == 0:
            # No wins - create a flat 0% curve
            for week in range(0, 53):
                rows.append({
                    'market_segment': segment,
                    'weeks_to_close': week,
                    'cum_win_pct': 0.0
                })
            continue
        
        won_deals['weeks_to_close'] = (
            (won_deals['date_closed'] - won_deals['date_created']).dt.days // 7
        ).clip(lower=0)
        
        max_weeks = int(won_deals['weeks_to_close'].max())
        
        # For each week, calculate cumulative wins as % of ALL deals created
        for week in range(0, max_weeks + 1):
            wins_by_week = len(won_deals[won_deals['weeks_to_close'] <= week])
            cum_win_pct = wins_by_week / total_deals  # Denominator is ALL deals, not just wins
            
            rows.append({
                'market_segment': segment,
                'weeks_to_close': week,
                'cum_win_pct': cum_win_pct
            })
    
    return pd.DataFrame(rows)

def get_curve_rate(curves, segment, age_weeks):
    """Helper to get cumulative win rate for a segment at a given age"""
    segment_curve = curves[curves['market_segment'] == segment]
    rate = segment_curve.loc[segment_curve['weeks_to_close'] <= age_weeks, 'cum_win_pct'].max()
    
    if pd.isna(rate):
        # If age is beyond our data, use max
        rate = segment_curve['cum_win_pct'].max()
        if pd.isna(rate):
            rate = 0
    
    return rate

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
    
    # Apply win rate multiplier to probabilities
    probs['prob'] = probs['prob'] * SCENARIO_LEVERS['win_rate_multiplier']
    probs['prob'] = probs['prob'].clip(upper=1.0)  # Cap at 100%
    
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
# LAYER 2: ACTIVE PIPELINE FORECAST (WITH TIMING)
# ======================================================

def forecast_active_pipeline(df, stage_probs, staleness, curves, forecast_months, cutoff_date=None):
    """
    FIXED: Projects when active deals will close across the forecast period
    """
    if cutoff_date is None:
        cutoff_date = ACTUALS_THROUGH
    
    # Get the last snapshot on or before the cutoff date
    available_snapshots = df[df['date_snapshot'] <= pd.to_datetime(cutoff_date)]
    if len(available_snapshots) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    last_snapshot_date = available_snapshots['date_snapshot'].max()
    snap = df[df['date_snapshot'] == last_snapshot_date]
    open_deals = snap[~snap['is_closed']].copy()
    
    if len(open_deals) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])

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

    # For each open deal, distribute expected value across forecast months
    # based on vintage curve (when it's likely to close)
    rows = []
    
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    for _, deal in open_deals.iterrows():
        current_age = deal['age_weeks']
        segment = deal['market_segment']
        base_prob = deal['prob']
        
        # Current cumulative win probability at current age
        current_cum_prob = get_curve_rate(curves, segment, current_age)
        
        for month in forecast_months:
            # Age at this forecast month
            weeks_from_now = ((month - cutoff_dt).days // 7)
            future_age = current_age + weeks_from_now
            
            if weeks_from_now <= 0:
                continue
            
            # Calculate previous month's age (4 weeks earlier)
            prev_age = future_age - 4
            if prev_age < current_age:
                prev_age = current_age
            
            # Marginal probability of closing in this specific month
            future_cum_prob = get_curve_rate(curves, segment, future_age)
            prev_cum_prob = get_curve_rate(curves, segment, prev_age)
            
            marginal_prob = future_cum_prob - prev_cum_prob
            
            if marginal_prob > 0:
                # Combine stage probability with timing probability
                combined_prob = base_prob * marginal_prob
                
                rows.append({
                    'month': month,
                    'market_segment': segment,
                    'expected_revenue': deal['net_revenue'] * combined_prob,
                    'expected_count': combined_prob
                })
    
    if len(rows) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    result = pd.DataFrame(rows)
    return result.groupby(['month', 'market_segment']).sum().reset_index()

# ======================================================
# LAYER 1: FUTURE PIPELINE (FIXED - NO DOUBLE COUNTING)
# ======================================================

def forecast_future_pipeline(deals, curves, forecast_start, forecast_end):
    """
    FIXED: Each cohort's value is distributed across months, not summed multiple times
    """
    # Calculate baseline assumptions from history
    baseline_assumptions = (
        deals.groupby('market_segment')
             .agg(
                 avg_monthly_vol=('deal_id', lambda x: x.count() / deals['created_month'].nunique()),
                 avg_size=('net_revenue', 'mean'),
                 final_win_rate=('won', 'mean')  # Historical win rate
             )
             .reset_index()
    )

    forecast_months = pd.period_range(forecast_start, forecast_end, freq='M')
    creation_months = pd.period_range(forecast_start, forecast_end, freq='M')
    
    rows = []

    for m_created in creation_months:
        for _, r in baseline_assumptions.iterrows():
            segment = r['market_segment']
            
            # Apply scenario levers
            vol = r['avg_monthly_vol'] * SCENARIO_LEVERS['volume_multiplier']
            size = r['avg_size'] * SCENARIO_LEVERS['deal_size_multiplier']
            
            # Calculate max maturity rate from curves
            segment_curve = curves[curves['market_segment'] == segment]
            max_maturity_rate = segment_curve['cum_win_pct'].max()
            if pd.isna(max_maturity_rate):
                max_maturity_rate = r['final_win_rate']
            
            # Apply win rate multiplier to the probability
            adjusted_win_rate = max_maturity_rate * SCENARIO_LEVERS['win_rate_multiplier']
            adjusted_win_rate = min(adjusted_win_rate, 1.0)  # Cap at 100%
            
            # Total expected value from this cohort (across ALL future months)
            total_cohort_expected_value = vol * size * adjusted_win_rate
            total_cohort_expected_count = vol * adjusted_win_rate
            
            # Now distribute this total across forecast months based on timing curve
            prev_cum_rate = 0
            
            for m_forecast in forecast_months:
                if m_forecast < m_created:
                    continue
                
                # Age of cohort in this forecast month
                age_weeks = (m_forecast.to_timestamp() - m_created.to_timestamp()).days // 7
                
                # Cumulative % of final wins that have occurred by this age
                cum_rate = get_curve_rate(curves, segment, age_weeks)
                
                # Marginal % of wins occurring in this specific month
                marginal_rate = cum_rate - prev_cum_rate
                
                if marginal_rate > 0:
                    # Allocate portion of total expected value to this month
                    rows.append({
                        'month': m_forecast.to_timestamp(),
                        'market_segment': segment,
                        'expected_revenue': total_cohort_expected_value * marginal_rate,
                        'expected_count': total_cohort_expected_count * marginal_rate
                    })
                
                prev_cum_rate = cum_rate

    return pd.DataFrame(rows)

# ======================================================
# BACKTEST
# ======================================================

def run_backtest(snapshots):
    """
    Forecast from BACKTEST_DATE through BACKTEST_THROUGH using only data available at BACKTEST_DATE
    Compare against actuals
    """
    print(f"\n{'='*60}")
    print(f"RUNNING BACKTEST: Forecast from {BACKTEST_DATE} through {BACKTEST_THROUGH}")
    print(f"{'='*60}\n")
    
    # Use only data available as of backtest date
    backtest_cutoff = pd.to_datetime(BACKTEST_DATE)
    historical_data = snapshots[snapshots['date_snapshot'] < backtest_cutoff].copy()
    
    # Build assumptions from historical data only
    deals = build_deal_facts(historical_data)
    curves = build_vintage_curves(deals)
    stage_probs = build_stage_probabilities(historical_data)
    staleness = build_staleness_thresholds(historical_data)
    
    # Forecast period
    forecast_months = pd.period_range(BACKTEST_DATE, BACKTEST_THROUGH, freq='M')
    
    # Active pipeline as of backtest date
    active = forecast_active_pipeline(
        historical_data, stage_probs, staleness, curves, 
        [m.to_timestamp() for m in forecast_months]
    )
    
    # Future pipeline for backtest period
    future = forecast_future_pipeline(deals, curves, BACKTEST_DATE, BACKTEST_THROUGH)
    
    # Combine forecasts
    if len(active) > 0 and len(future) > 0:
        forecast = pd.concat([active, future]).groupby(['month', 'market_segment']).sum().reset_index()
    elif len(active) > 0:
        forecast = active
    elif len(future) > 0:
        forecast = future
    else:
        forecast = pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    # Calculate actuals for backtest period
    backtest_end = pd.to_datetime(BACKTEST_THROUGH)
    actual_deals = build_deal_facts(snapshots[snapshots['date_snapshot'] <= backtest_end])
    
    actuals = actual_deals[
        (actual_deals['won']) & 
        (actual_deals['date_closed'] >= backtest_cutoff) &
        (actual_deals['date_closed'] <= backtest_end)
    ].copy()
    
    actuals['close_month'] = actuals['date_closed'].dt.to_period('M').dt.to_timestamp()
    
    actual_summary = (
        actuals.groupby(['market_segment', 'close_month'])
        .agg(
            actual_revenue=('net_revenue', 'sum'),
            actual_count=('deal_id', 'count')
        )
        .reset_index()
    )
    
    # Total by segment
    forecast_total = forecast.groupby('market_segment').agg(
        forecasted_revenue=('expected_revenue', 'sum'),
        forecasted_count=('expected_count', 'sum')
    ).reset_index()
    
    actual_total = actual_summary.groupby('market_segment').agg(
        actual_revenue=('actual_revenue', 'sum'),
        actual_count=('actual_count', 'sum')
    ).reset_index()
    
    comparison = forecast_total.merge(actual_total, on='market_segment', how='outer').fillna(0)
    comparison['variance_pct'] = ((comparison['forecasted_revenue'] - comparison['actual_revenue']) / 
                                   comparison['actual_revenue'].replace(0, 1)) * 100
    comparison['variance_abs'] = comparison['forecasted_revenue'] - comparison['actual_revenue']
    
    print("\nBACKTEST RESULTS BY SEGMENT:")
    print("="*80)
    print(comparison.to_string(index=False))
    print("\n")
    
    # Save results
    comparison.to_csv(f'{VALIDATION_DIR}/backtest_results.csv', index=False)
    
    # Monthly comparison
    forecast['month'] = pd.to_datetime(forecast['month'])
    actual_summary['close_month'] = pd.to_datetime(actual_summary['close_month'])
    
    monthly = forecast.merge(
        actual_summary,
        left_on=['month', 'market_segment'],
        right_on=['close_month', 'market_segment'],
        how='outer'
    ).fillna(0)
    
    monthly.to_csv(f'{VALIDATION_DIR}/backtest_monthly.csv', index=False)
    
    return comparison

# ======================================================
# GOAL SEEK / WHAT-IF ANALYSIS
# ======================================================

def run_goal_seek(forecast):
    """
    Calculate required multipliers (volume or win_rate) to hit revenue targets
    """
    print(f"\n{'='*60}")
    print(f"GOAL SEEK ANALYSIS - Mode: {GOAL_SEEK_MODE.upper()}")
    print(f"{'='*60}\n")
    
    # Calculate current baseline forecast by segment
    summary = (
        forecast.groupby('market_segment')
                .agg(baseline_revenue=('expected_won_revenue', 'sum'))
                .reset_index()
    )
    
    results = []
    
    for _, r in summary.iterrows():
        segment = r['market_segment']
        baseline = r['baseline_revenue']
        target = GOAL_WON_REVENUE.get(segment)
        
        if not target:
            continue
        
        if baseline == 0:
            multiplier = None
            gap = target
        else:
            multiplier = target / baseline
            gap = target - baseline
        
        results.append({
            'market_segment': segment,
            'baseline_revenue': round(baseline, 2),
            'target_revenue': target,
            'revenue_gap': round(gap, 2),
            'required_multiplier': round(multiplier, 3) if multiplier else None,
            'lever': GOAL_SEEK_MODE
        })
    
    goal_df = pd.DataFrame(results)
    goal_df.to_csv(f'{EXPORT_DIR}/goal_seek_analysis.csv', index=False)
    
    print("GOAL SEEK RESULTS:")
    print("="*80)
    print(goal_df.to_string(index=False))
    print(f"\nTo hit targets, adjust {GOAL_SEEK_MODE}_multiplier in SCENARIO_LEVERS")
    print(f"Results saved to: {EXPORT_DIR}/goal_seek_analysis.csv\n")
    
    return goal_df

# ======================================================
# MAIN
# ======================================================

def run_forecast():
    snapshots = load_snapshots()
    snapshots = apply_skipper_weights(snapshots)

    deals = build_deal_facts(snapshots)
    
    curves = build_vintage_curves(deals)
    stage_probs = build_stage_probabilities(snapshots)
    staleness = build_staleness_thresholds(snapshots)
    
    # Export assumptions (including stage win rates)
    export_assumptions(deals, stage_probs, snapshots)

    # Generate forecast months
    forecast_months = pd.period_range(FORECAST_START, FORECAST_END, freq='M')
    
    # Layer 2: Active pipeline
    active = forecast_active_pipeline(
        snapshots, stage_probs, staleness, curves,
        [m.to_timestamp() for m in forecast_months]
    )
    
    # Layer 1: Future pipeline
    future = forecast_future_pipeline(deals, curves, FORECAST_START, FORECAST_END)

    # Combine
    if len(active) > 0 and len(future) > 0:
        forecast = pd.concat([active, future]).groupby(['month', 'market_segment']).sum().reset_index()
    elif len(active) > 0:
        forecast = active
    elif len(future) > 0:
        forecast = future
    else:
        print("WARNING: No forecast generated")
        return None

    # Format output
    forecast['month'] = pd.to_datetime(forecast['month']).dt.to_period('M')
    forecast = forecast.rename(columns={
        'month': 'forecast_month',
        'expected_revenue': 'expected_won_revenue',
        'expected_count': 'expected_won_deal_count'
    })
    
    forecast.to_csv(f'{EXPORT_DIR}/forecast_2026.csv', index=False)

    with open(f'{EXPORT_DIR}/assumptions.json', 'w') as f:
        json.dump(SCENARIO_LEVERS, f, indent=4)

    print(f"\nForecast generated: {len(forecast)} rows")
    print("\nSummary by Segment:")
    summary = forecast.groupby('market_segment').agg(
        total_revenue=('expected_won_revenue', 'sum'),
        total_deals=('expected_won_deal_count', 'sum')
    )
    print(summary)

    # Run backtest if enabled
    if RUN_BACKTEST:
        run_backtest(snapshots)
    
    # Run goal seek if enabled
    if RUN_GOAL_SEEK:
        run_goal_seek(forecast)

    return forecast

if __name__ == "__main__":
    result = run_forecast()
