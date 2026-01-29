import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)

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

RUN_BACKTEST = True
BACKTEST_DATE = '2025-01-01'
BACKTEST_THROUGH = '2025-12-31'

SCENARIO_LEVERS = {
    'volume_multiplier': 1.0,
    'win_rate_multiplier': 1.0,
    'deal_size_multiplier': 1.0
}

RUN_GOAL_SEEK = True
GOAL_WON_REVENUE = {
    'Large Market': 14_750_000,
    'Mid Market': 7_800_000
}

STALENESS_PENALTY = 0.8
SKIPPER_WEIGHT = 0.5

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(ASSUMPTIONS_DIR, exist_ok=True)

# ======================================================
# DATA LOADING
# ======================================================

def load_snapshots():
    df = pd.read_csv(DATA_PATH, parse_dates=['date_created', 'date_closed', 'date_snapshot'])
    df = df.sort_values(['deal_id', 'date_snapshot'])
    df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])
    df['is_closed_lost'] = df['stage'] == 'Closed Lost'
    df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']
    return df

# ======================================================
# DEAL FACTS
# ======================================================

def build_deal_facts(df, cutoff_date=None):
    if cutoff_date:
        df = df[df['date_snapshot'] <= pd.to_datetime(cutoff_date)].copy()
    
    first = df.groupby('deal_id').first().reset_index()
    closed = df[df['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed', 'stage']]
    first = first.drop(columns=['stage', 'date_closed'], errors='ignore')
    deals = first.merge(closed, on='deal_id', how='left')
    deals['created_month'] = deals['date_created'].dt.to_period('M')
    deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])
    
    first_stage = df.groupby('deal_id')['stage'].first()
    skippers = first_stage[first_stage == 'Closed Lost'].index
    deals['volume_weight'] = 1.0
    deals.loc[deals['deal_id'].isin(skippers), 'volume_weight'] = SKIPPER_WEIGHT
    
    return deals

# ======================================================
# PROBABILITIES & STALENESS
# ======================================================

def build_stage_probabilities(df, weight_recent=True):
    exits = df[df['is_closed']].copy()
    
    if weight_recent:
        cutoff_date = exits['date_snapshot'].max()
        trailing_12_start = cutoff_date - pd.DateOffset(months=12)
        exits['weight'] = np.where(exits['date_snapshot'] >= trailing_12_start, 3.0, 1.0)
    else:
        exits['weight'] = 1.0
    
    probs = exits.groupby(['market_segment', 'stage']).apply(
        lambda x: pd.Series({
            'wins': (x['is_closed_won'] * x['weight']).sum(),
            'total': x['weight'].sum()
        })
    ).reset_index()
    
    probs['prob'] = (probs['wins'] / probs['total']).clip(upper=1.0)
    return probs[['market_segment', 'stage', 'prob']]

def build_staleness_thresholds(df):
    df = df.copy()
    df['age_weeks'] = ((df['date_snapshot'] - df['date_created']).dt.days // 7).clip(lower=0)
    return df.groupby(['market_segment', 'stage'])['age_weeks'].quantile(0.9).reset_index(name='stale_after_weeks')

# ======================================================
# LAYER 2: ACTIVE PIPELINE FORECAST
# ======================================================

def forecast_active_pipeline(df, stage_probs, staleness, forecast_months, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = ACTUALS_THROUGH
    
    available_snapshots = df[df['date_snapshot'] <= pd.to_datetime(cutoff_date)]
    if len(available_snapshots) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    last_snapshot_date = available_snapshots['date_snapshot'].max()
    open_deals = df[(df['date_snapshot'] == last_snapshot_date) & (~df['is_closed'])].copy()
    
    if len(open_deals) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])

    open_deals['age_weeks'] = ((open_deals['date_snapshot'] - open_deals['date_created']).dt.days // 7)
    open_deals = open_deals.merge(stage_probs, on=['market_segment', 'stage'], how='left')
    open_deals = open_deals.merge(staleness, on=['market_segment', 'stage'], how='left')
    open_deals['prob'] = open_deals['prob'].fillna(0)
    
    stale_mask = open_deals['age_weeks'] > open_deals['stale_after_weeks']
    open_deals.loc[stale_mask, 'prob'] *= STALENESS_PENALTY
    
    # Distribute across forecast period with simple decay
    rows = []
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    for _, deal in open_deals.iterrows():
        total_prob = deal['prob']
        if total_prob == 0:
            continue
            
        # Distribute probability across months with front-loading
        num_months = len(forecast_months)
        weights = np.array([1/(i+1) for i in range(num_months)])
        weights = weights / weights.sum()
        
        for month, weight in zip(forecast_months, weights):
            month_prob = total_prob * weight
            rows.append({
                'month': month,
                'market_segment': deal['market_segment'],
                'expected_revenue': deal['net_revenue'] * month_prob,
                'expected_count': month_prob
            })
    
    if len(rows) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index()

# ======================================================
# LAYER 1: FUTURE PIPELINE
# ======================================================

def forecast_future_pipeline(deals, forecast_start, forecast_end, weight_recent=True):
    deals_copy = deals.copy()
    total_months = deals_copy['created_month'].nunique()
    
    if weight_recent:
        cutoff_date = deals_copy['date_created'].max()
        trailing_12_start = cutoff_date - pd.DateOffset(months=12)
        deals_copy['weight'] = np.where(deals_copy['date_created'] >= trailing_12_start, 3.0, 1.0)
    else:
        deals_copy['weight'] = 1.0
    
    # Calculate baseline metrics per segment
    baseline = deals_copy.groupby('market_segment').apply(
        lambda x: pd.Series({
            'deal_count': x['volume_weight'].sum(),
            'weighted_revenue': (x['net_revenue'] * x['weight']).sum(),
            'weighted_wins': (x['won'].astype(float) * x['weight']).sum(),
            'total_weight': x['weight'].sum()
        })
    ).reset_index()
    
    baseline['avg_monthly_vol'] = baseline['deal_count'] / total_months
    baseline['avg_size'] = baseline['weighted_revenue'] / baseline['total_weight']
    baseline['win_rate'] = baseline['weighted_wins'] / baseline['total_weight']
    
    # Calculate timing distribution from won deals  
    won_deals = deals[deals['won'] & deals['date_closed'].notna()].copy()
    won_deals['months_to_close'] = ((won_deals['date_closed'] - won_deals['date_created']).dt.days / 30).clip(lower=0, upper=11).round().astype(int)
    
    timing_dist = won_deals.groupby(['market_segment', 'months_to_close']).size().reset_index(name='count')
    timing_totals = timing_dist.groupby('market_segment')['count'].sum().reset_index(name='total')
    timing_dist = timing_dist.merge(timing_totals, on='market_segment')
    timing_dist['pct'] = timing_dist['count'] / timing_dist['total']
    
    forecast_months = pd.period_range(forecast_start, forecast_end, freq='M')
    creation_months = pd.period_range(forecast_start, forecast_end, freq='M')
    
    rows = []
    
    for m_created in creation_months:
        for _, r in baseline.iterrows():
            segment = r['market_segment']
            vol = r['avg_monthly_vol'] * SCENARIO_LEVERS['volume_multiplier']
            size = r['avg_size'] * SCENARIO_LEVERS['deal_size_multiplier']
            win_rate = r['win_rate'] * SCENARIO_LEVERS['win_rate_multiplier']
            win_rate = min(win_rate, 1.0)
            
            segment_timing = timing_dist[timing_dist['market_segment'] == segment]
            if len(segment_timing) == 0:
                # Default: distribute evenly over 6 months
                segment_timing = pd.DataFrame({
                    'months_to_close': range(6),
                    'pct': [1/6] * 6
                })
            
            total_expected_revenue = vol * size * win_rate
            total_expected_count = vol * win_rate
            
            for _, timing_row in segment_timing.iterrows():
                months_offset = int(timing_row['months_to_close'])
                close_month = m_created + months_offset
                
                if close_month < pd.Period(forecast_start, 'M') or close_month > pd.Period(forecast_end, 'M'):
                    continue
                
                pct = timing_row['pct']
                
                rows.append({
                    'month': close_month.to_timestamp(),
                    'market_segment': segment,
                    'expected_revenue': total_expected_revenue * pct,
                    'expected_count': total_expected_count * pct
                })
    
    if len(rows) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index()

# ======================================================
# BACKTEST
# ======================================================

def run_backtest(snapshots):
    print(f"\n{'='*60}")
    print(f"BACKTEST")
    print(f"{'='*60}")
    
    backtest_cutoff = pd.to_datetime(BACKTEST_DATE)
    backtest_end = pd.to_datetime(BACKTEST_THROUGH)
    
    # Build all deals including 2025
    actual_full_data = snapshots[snapshots['date_snapshot'] <= backtest_end].copy()
    actual_deals = build_deal_facts(actual_full_data)
    
    # Get 2025 created deals for future pipeline metrics
    deals_created_2025 = actual_deals[
        (actual_deals['date_created'] >= backtest_cutoff) &
        (actual_deals['date_created'] <= backtest_end)
    ].copy()
    
    # Build 2025 probabilities from actual 2025 exits
    stage_probs_2025 = build_stage_probabilities(
        snapshots[(snapshots['date_snapshot'] >= backtest_cutoff) & (snapshots['date_snapshot'] <= backtest_end)],
        weight_recent=False
    )
    staleness = build_staleness_thresholds(actual_full_data)
    
    # Historical data for active pipeline (deals open at start of 2025)
    historical_data = snapshots[snapshots['date_snapshot'] < backtest_cutoff].copy()
    
    forecast_months = pd.period_range(BACKTEST_DATE, BACKTEST_THROUGH, freq='M')
    
    # Active pipeline forecast
    active = forecast_active_pipeline(
        historical_data, stage_probs_2025, staleness,
        [m.to_timestamp() for m in forecast_months],
        cutoff_date=backtest_cutoff - pd.Timedelta(days=1)
    )
    
    # Future pipeline forecast using actual 2025 creation metrics
    future = forecast_future_pipeline(deals_created_2025, BACKTEST_DATE, BACKTEST_THROUGH, weight_recent=False)
    
    # Combine
    all_parts = [active, future]
    all_parts = [p for p in all_parts if len(p) > 0]
    
    if len(all_parts) > 0:
        forecast = pd.concat(all_parts).groupby(['month', 'market_segment']).sum().reset_index()
    else:
        forecast = pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    # Actuals - Use FINAL status from last available snapshot in the period
    last_snapshot_date = snapshots[snapshots['date_snapshot'] <= backtest_end]['date_snapshot'].max()
    last_snapshot = snapshots[snapshots['date_snapshot'] == last_snapshot_date]
    won_in_final = last_snapshot[last_snapshot['stage'].isin(['Closed Won', 'Verbal'])].copy()
    won_in_final = won_in_final[
        (won_in_final['date_closed'] >= backtest_cutoff) &
        (won_in_final['date_closed'] <= backtest_end) &
        (won_in_final['date_closed'].notna())
    ]
    
    actuals = won_in_final[['deal_id', 'market_segment', 'net_revenue', 'date_closed']].copy()
    
    actuals['close_month'] = actuals['date_closed'].dt.to_period('M').dt.to_timestamp()
    
    actual_summary = actuals.groupby(['market_segment', 'close_month']).agg(
        actual_revenue=('net_revenue', 'sum'),
        actual_count=('deal_id', 'count')
    ).reset_index()
    
    # Compare totals
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
    
    print("\nBACKTEST RESULTS:")
    print(comparison[['market_segment', 'forecasted_revenue', 'actual_revenue', 'variance_pct']].to_string(index=False))
    print()
    
    comparison.to_csv(f'{VALIDATION_DIR}/backtest_results.csv', index=False)
    
    # Monthly
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
# GOAL SEEK
# ======================================================

def run_goal_seek_analysis(deals, forecast_start, forecast_end):
    print(f"\n{'='*60}")
    print(f"GOAL SEEK ANALYSIS")
    print(f"{'='*60}")
    
    baseline = deals.groupby('market_segment').apply(
        lambda x: pd.Series({
            'historical_monthly_volume': x['volume_weight'].sum() / x['created_month'].nunique(),
            'historical_deal_size': x['net_revenue'].mean(),
            'historical_win_rate': x['won'].mean()
        })
    ).reset_index()
    
    # Calculate monthly needed for 12 month period
    results = []
    
    for segment, target in GOAL_WON_REVENUE.items():
        seg_data = baseline[baseline['market_segment'] == segment]
        if len(seg_data) == 0:
            continue
        
        hist_vol = seg_data['historical_monthly_volume'].iloc[0]
        hist_size = seg_data['historical_deal_size'].iloc[0]
        hist_wr = seg_data['historical_win_rate'].iloc[0]
        
        # Monthly target
        monthly_target = target / 12
        
        # Option 1: Adjust volume only
        required_vol = monthly_target / (hist_size * hist_wr) if (hist_size * hist_wr) > 0 else 0
        
        # Option 2: Adjust win rate only
        required_wr = monthly_target / (hist_vol * hist_size) if (hist_vol * hist_size) > 0 else 0
        required_wr = min(required_wr, 1.0)
        
        results.append({
            'market_segment': segment,
            'fy26_target': target,
            'monthly_target': round(monthly_target, 2),
            'historical_monthly_volume': round(hist_vol, 1),
            'required_monthly_volume': round(required_vol, 1),
            'volume_change_pct': round(((required_vol / hist_vol) - 1) * 100, 1) if hist_vol > 0 else 0,
            'historical_win_rate': round(hist_wr * 100, 1),
            'required_win_rate': round(required_wr * 100, 1),
            'win_rate_change_pct': round(((required_wr / hist_wr) - 1) * 100, 1) if hist_wr > 0 else 0
        })
    
    goal_df = pd.DataFrame(results)
    goal_df.to_csv(f'{EXPORT_DIR}/goal_seek_analysis.csv', index=False)
    print(goal_df.to_string(index=False))
    print()
    return goal_df

# ======================================================
# EXPORT ASSUMPTIONS
# ======================================================

def calculate_stage_win_rates(snapshots):
    """Calculate win rates for deals that REACHED each stage (progression-based) by month and market segment"""
    
    # Get all deals with their creation month and final status
    deals_final = snapshots.groupby('deal_id').agg({
        'date_created': 'first',
        'net_revenue': 'first',
        'market_segment': 'first',
        'stage': 'last',
        'is_closed_won': 'last'
    }).reset_index()
    
    deals_final['created_month'] = deals_final['date_created'].dt.to_period('M')
    
    # Define stage hierarchy
    open_stages = ['Qualified', 'Alignment', 'Solutioning']
    
    # Filter snapshots to only open stages
    open_snapshots = snapshots[snapshots['stage'].isin(open_stages)].copy()
    
    if len(open_snapshots) == 0:
        return pd.DataFrame(columns=['month', 'market_segment', 'stage', 'total_volume', 'total_won_volume', 'total_revenue', 'total_won_revenue'])
    
    # Get all unique combinations of deal_id and stage they reached (vectorized)
    stage_reached = open_snapshots[['deal_id', 'stage']].drop_duplicates()
    stage_reached = stage_reached.rename(columns={'stage': 'stage_reached'})
    
    # Merge with deal facts
    analysis = deals_final.merge(stage_reached, on='deal_id', how='inner')
    
    # Group by month, segment, and stage reached
    results = analysis.groupby(['created_month', 'market_segment', 'stage_reached']).agg(
        total_volume=('deal_id', 'count'),
        total_won_volume=('is_closed_won', 'sum'),
        total_revenue=('net_revenue', 'sum'),
        total_won_revenue=('net_revenue', lambda x: x[analysis.loc[x.index, 'is_closed_won']].sum())
    ).reset_index()
    
    results = results.rename(columns={'stage_reached': 'stage', 'created_month': 'month'})
    
    # Define stage order for sorting
    stage_order = {'Qualified': 1, 'Alignment': 2, 'Solutioning': 3}
    results['stage_order'] = results['stage'].map(stage_order)
    results = results.sort_values(['month', 'market_segment', 'stage_order'])
    results = results.drop(columns=['stage_order'])
    
    return results

def export_assumptions(deals, snapshots):
    # Volume by month and segment - raw counts only
    volume = deals.groupby(['created_month', 'market_segment']).agg(
        total_volume=('deal_id', 'count')
    ).reset_index()
    volume = volume.rename(columns={'created_month': 'month'})
    
    # Win rates by month and segment - raw data only (no calculated percentages)
    win = deals.groupby(['created_month', 'market_segment']).agg(
        total_volume=('deal_id', 'count'),
        total_won_volume=('won', 'sum'),
        total_revenue=('net_revenue', 'sum')
    ).reset_index()
    
    # Calculate total won revenue
    won_deals_revenue = deals[deals['won']].groupby(['created_month', 'market_segment'])['net_revenue'].sum().reset_index(name='total_won_revenue')
    win = win.merge(won_deals_revenue, on=['created_month', 'market_segment'], how='left')
    win['total_won_revenue'] = win['total_won_revenue'].fillna(0)
    win = win.rename(columns={'created_month': 'month'})
    
    # Deal size by month and segment - raw data
    size = deals.groupby(['created_month', 'market_segment']).agg(
        total_volume=('deal_id', 'count'),
        total_revenue=('net_revenue', 'sum')
    ).reset_index()
    size = size.rename(columns={'created_month': 'month'})
    
    # Stage-based win rates for open stages - raw data only
    stage_win_rates = calculate_stage_win_rates(snapshots)
    
    volume.to_csv(f'{ASSUMPTIONS_DIR}/volume_by_month.csv', index=False)
    win.to_csv(f'{ASSUMPTIONS_DIR}/win_rates_by_month.csv', index=False)
    size.to_csv(f'{ASSUMPTIONS_DIR}/deal_size_by_month.csv', index=False)
    stage_win_rates.to_csv(f'{ASSUMPTIONS_DIR}/win_rates_by_stage.csv', index=False)

# ======================================================
# MAIN
# ======================================================

def run_forecast():
    print(f"\n{'='*60}")
    print(f"FORECAST GENERATOR V5")
    print(f"{'='*60}")
    
    snapshots = load_snapshots()
    deals = build_deal_facts(snapshots)
    
    stage_probs = build_stage_probabilities(snapshots, weight_recent=True)
    staleness = build_staleness_thresholds(snapshots)
    
    export_assumptions(deals, snapshots)
    
    if RUN_BACKTEST:
        run_backtest(snapshots)
    
    print(f"\n{'='*60}")
    print(f"FY26 FORECAST")
    print(f"{'='*60}")
    
    forecast_months = pd.period_range(FORECAST_START, FORECAST_END, freq='M')
    
    active = forecast_active_pipeline(
        snapshots, stage_probs, staleness,
        [m.to_timestamp() for m in forecast_months]
    )
    
    future = forecast_future_pipeline(deals, FORECAST_START, FORECAST_END, weight_recent=True)
    
    all_parts = [active, future]
    all_parts = [p for p in all_parts if len(p) > 0]
    
    if len(all_parts) > 0:
        forecast = pd.concat(all_parts).groupby(['month', 'market_segment']).sum().reset_index()
    else:
        print("WARNING: No forecast generated")
        return None

    forecast['month'] = pd.to_datetime(forecast['month']).dt.to_period('M')
    forecast = forecast.rename(columns={
        'month': 'forecast_month',
        'expected_revenue': 'expected_won_revenue',
        'expected_count': 'expected_won_deal_count'
    })
    
    forecast.to_csv(f'{EXPORT_DIR}/forecast_2026.csv', index=False)

    with open(f'{EXPORT_DIR}/assumptions.json', 'w') as f:
        json.dump(SCENARIO_LEVERS, f, indent=4)

    summary = forecast.groupby('market_segment').agg(
        total_revenue=('expected_won_revenue', 'sum'),
        total_deals=('expected_won_deal_count', 'sum')
    )
    
    print("\nFORECAST SUMMARY:")
    print(summary)
    print()
    
    if RUN_GOAL_SEEK:
        run_goal_seek_analysis(deals, FORECAST_START, FORECAST_END)

    return forecast

if __name__ == "__main__":
    result = run_forecast()
