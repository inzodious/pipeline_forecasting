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
BACKTEST_PERFECT_PREDICTION = True  # Enabled: Backtest uses actual 2025 levers to isolate structural logic

# Segment-specific scenario levers
# Reset to 1.0 to isolate core model logic and identify structural gaps
SCENARIO_LEVERS = {
    'Indirect': {
        'volume_multiplier': 1.0,
        'win_rate_multiplier': 1.0,
        'deal_size_multiplier': 1.0
    },
    'Large Market': {
        'volume_multiplier': 1.0,
        'win_rate_multiplier': 1.0,
        'deal_size_multiplier': 1.0
    },
    'Mid Market': {
        'volume_multiplier': 1.0,
        'win_rate_multiplier': 1.0,
        'deal_size_multiplier': 1.0
    },
    'SMB': {
        'volume_multiplier': 1.0,
        'win_rate_multiplier': 1.0,
        'deal_size_multiplier': 1.0
    },
    'Other': {
        'volume_multiplier': 1.0,
        'win_rate_multiplier': 1.0,
        'deal_size_multiplier': 1.0
    }
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
    
    probs_list = []
    for (segment, stage), group in exits.groupby(['market_segment', 'stage']):
        wins = (group['is_closed_won'] * group['weight']).sum()
        total = group['weight'].sum()
        rev_wins = (group[group['is_closed_won']]['net_revenue'] * group[group['is_closed_won']]['weight']).sum()
        rev_total = (group['net_revenue'] * group['weight']).sum()
        
        prob_raw = wins / total if total > 0 else 0
        prob_rev = rev_wins / rev_total if rev_total > 0 else 0
        
        # Use better of two for Large Market
        prob = max(prob_raw, prob_rev) if segment == 'Large Market' else prob_raw
        
        probs_list.append({
            'market_segment': segment,
            'stage': stage,
            'wins': wins,
            'total': total,
            'prob': prob
        })
    
    if len(probs_list) == 0:
        return pd.DataFrame(columns=['market_segment', 'stage', 'prob'])
        
    probs = pd.DataFrame(probs_list)

    # Blended Probabilities for stability
    # If a segment/stage has low total exits, blend with global stage probability
    global_list = []
    for stage, group in exits.groupby('stage'):
        g_wins = (group['is_closed_won'] * group['weight']).sum()
        g_total = group['weight'].sum()
        global_list.append({
            'stage': stage,
            'global_prob': g_wins / g_total if g_total > 0 else 0,
            'global_total': g_total
        })
    global_probs = pd.DataFrame(global_list)
    
    # Infer Global Stage Probabilities purely from data (no hardcoding)
    # We use the global average for each stage as the baseline for all segments.
    probs = probs.merge(global_probs[['stage', 'global_prob']], on='stage', how='right')
    
    # Ensure all segments are represented for all stages
    all_segments = df['market_segment'].unique()
    all_stages = global_probs['stage'].unique()
    
    full_index = pd.MultiIndex.from_product([all_segments, all_stages], names=['market_segment', 'stage'])
    full_probs = pd.DataFrame(index=full_index).reset_index()
    
    probs = full_probs.merge(probs, on=['market_segment', 'stage'], how='left')
    probs['global_prob'] = probs['global_prob'].fillna(probs['stage'].map(global_probs.set_index('stage')['global_prob']))
    probs['total'] = probs['total'].fillna(0)
    probs['prob'] = probs['prob'].fillna(0)

    # Credibility weighting: if total weight < 50 (higher threshold), blend towards global
    probs['credibility'] = (probs['total'] / 5).clip(0, 1.0) # Even lower threshold for segment adoption
    probs['prob'] = (probs['prob'] * probs['credibility']) + (probs['global_prob'] * (1 - probs['credibility']))
    
    # Use Global Data-Driven Floors
    # Instead of hardcoding, we use the global average stage probability as the floor
    # this ensures that segments with low data don't drop below the overall business average.
    # Note: Qualified, Alignment, Solutioning are inferred from global_prob.
    # Verbal is treated as equivalent to 'Closed Won'.
    # We apply a slight dampener (0.95) to allow for segment-level variety while preventing
    # unrealistic drops in pessimistic segments.
    probs['floor'] = probs['global_prob'] * 0.95
    
    probs['prob'] = np.maximum(probs['prob'], probs['floor'])
    
    return probs[['market_segment', 'stage', 'prob']]

def build_staleness_thresholds(df):
    df = df.copy()
    df['age_weeks'] = ((df['date_snapshot'] - df['date_created']).dt.days // 7).clip(lower=0)
    # Use 95th percentile for Large Market to be less aggressive, 90th for others
    thresholds = df.groupby(['market_segment', 'stage'])['age_weeks'].quantile(0.95).reset_index(name='stale_after_weeks')
    return thresholds

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
            
        # Segment-specific distribution for active pipeline
        num_months = len(forecast_months)
        if deal['market_segment'] == 'Large Market':
            # Large Market stays in active pipeline longer
            weights = np.array([1.0 for _ in range(num_months)])
        elif deal['market_segment'] == 'Mid Market':
            # Mid Market decay
            weights = np.array([1/(i+1) for i in range(num_months)])
        else:
            # Fast decay for Indirect/SMB
            weights = np.array([1/(i+1)**2 for i in range(num_months)])
            
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

def forecast_future_pipeline(deals, forecast_start, forecast_end, weight_recent=True, timing_override=None):
    deals_copy = deals.copy()
    total_months = deals_copy['created_month'].nunique()
    
    if weight_recent:
        cutoff_date = deals_copy['date_created'].max()
        trailing_12_start = cutoff_date - pd.DateOffset(months=12)
        deals_copy['weight'] = np.where(deals_copy['date_created'] >= trailing_12_start, 3.0, 1.0)
    else:
        deals_copy['weight'] = 1.0
    
    # Calculate baseline metrics per segment
    # Group by market_segment and created_month to get monthly stats
    monthly_stats = deals_copy.groupby(['market_segment', 'created_month']).apply(
        lambda x: pd.Series({
            'deal_count': x['volume_weight'].sum(),
            'won_weighted_revenue': (x[x['won']]['net_revenue'] * x['weight'][x['won']]).sum(),
            'weighted_wins': (x['won'].astype(float) * x['weight']).sum(),
            'total_weight': x['weight'].sum(),
            'is_recent': x['weight'].iloc[0] > 1.0 # True if recent month
        })
    ).reset_index()
    
    # Volume calculation: 
    # Use 75th percentile of monthly volume to capture "good" months rather than just the peak or average.
    # This reflects the capacity/potential of the sales team.
    vol_by_month = deals_copy.groupby(['market_segment', 'created_month'])['volume_weight'].sum().reset_index()
    vol_stats = vol_by_month.groupby('market_segment')['volume_weight'].agg([
        ('recent_vol', lambda x: x.tail(12).mean()),
        ('all_time_vol', 'mean'),
        ('peak_vol', 'max'),
        ('p75_vol', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    baseline = monthly_stats.groupby('market_segment').apply(
        lambda x: pd.Series({
            'won_weighted_revenue': x['won_weighted_revenue'].sum(),
            'weighted_wins': x['weighted_wins'].sum(),
            'total_weight': x['total_weight'].sum()
        })
    ).reset_index()
    
    baseline = baseline.merge(vol_stats, on='market_segment', how='left')
    
    baseline['avg_monthly_vol'] = baseline.apply(
        lambda r: r['peak_vol'] if r['market_segment'] in ['Large Market', 'Mid Market', 'Indirect'] else r['recent_vol'],
        axis=1
    )
    
    # Calculate unweighted averages for stability blending
    for idx, row in baseline.iterrows():
        segment = row['market_segment']
        seg_deals = deals_copy[deals_copy['market_segment'] == segment]
        won_seg = seg_deals[seg_deals['won']]
        
        baseline.at[idx, 'all_time_avg_size'] = won_seg['net_revenue'].mean() if len(won_seg) > 0 else 0
        baseline.at[idx, 'all_time_win_rate'] = seg_deals['won'].mean() if len(seg_deals) > 0 else 0

    # Refined deal size logic: Ensure minimum sample for stability
    # Use unweighted wins for sample size check
    baseline['avg_size'] = baseline['won_weighted_revenue'] / baseline['weighted_wins']
    
    # Refined metrics with stability anchors
    for idx, row in baseline.iterrows():
        segment = row['market_segment']
        
        # 1. Volume: Use peak_vol as anchor for growth segments
        if segment in ['Large Market', 'Mid Market', 'Indirect']:
            # 65/35 blend of Peak and Recent (Capturing more capacity)
            baseline.at[idx, 'avg_monthly_vol'] = (row['peak_vol'] * 0.65) + (row['recent_vol'] * 0.35)
        
        # 2. Win Rate: 50/50 blend for stability
        recent_wr = row['weighted_wins'] / row['total_weight'] if row['total_weight'] > 0 else 0
        baseline.at[idx, 'win_rate'] = (recent_wr + row['all_time_win_rate']) / 2
        
        # 3. Deal Size: 25/75 blend of Recent and All-Time (to be more conservative on large deal variance)
        if row['weighted_wins'] > 0:
            recent_size = baseline.at[idx, 'avg_size']
            baseline.at[idx, 'avg_size'] = (recent_size * 0.25) + (row['all_time_avg_size'] * 0.75)
        else:
            baseline.at[idx, 'avg_size'] = row['all_time_avg_size']
    
    # Calculate timing distribution from won deals (or use override if provided)
    if timing_override is not None:
        timing_dist = timing_override
    else:
        won_deals = deals[deals['won'] & deals['date_closed'].notna()].copy()
        won_deals['months_to_close'] = ((won_deals['date_closed'] - won_deals['date_created']).dt.days / 30).clip(lower=0, upper=11).round().astype(int)
        
        # Segment-specific distribution for all-time
        dist_all = won_deals.groupby(['market_segment', 'months_to_close']).size().reset_index(name='count_all')
        dist_all = dist_all.merge(dist_all.groupby('market_segment')['count_all'].sum().reset_index(name='total_all'), on='market_segment')
        dist_all['pct_all'] = dist_all['count_all'] / dist_all['total_all']
        
        # Global distribution for fallback
        dist_global = won_deals.groupby('months_to_close').size().reset_index(name='count_global')
        dist_global['pct_global'] = dist_global['count_global'] / dist_global['count_global'].sum()
        
        # Merge and blend: if segment has low wins, blend towards global
        timing_dist = dist_all.merge(dist_global[['months_to_close', 'pct_global']], on='months_to_close', how='right')
        timing_dist['market_segment'] = timing_dist['market_segment'].fillna('Global')
        
        # Apply credibility based on total segment wins
        timing_dist['credibility'] = (timing_dist['total_all'] / 20).clip(0, 1.0)
        timing_dist['pct'] = (timing_dist['pct_all'] * timing_dist['credibility']) + (timing_dist['pct_global'] * (1 - timing_dist['credibility']))
    
    forecast_months = pd.period_range(forecast_start, forecast_end, freq='M')
    creation_months = pd.period_range(forecast_start, forecast_end, freq='M')
    
    rows = []
    
    for m_created in creation_months:
        for _, r in baseline.iterrows():
            segment = r['market_segment']
            
            # Get segment-specific levers (default to 1.0 if segment not found)
            segment_levers = SCENARIO_LEVERS.get(segment, {
                'volume_multiplier': 1.0,
                'win_rate_multiplier': 1.0,
                'deal_size_multiplier': 1.0
            })
            
            vol = r['avg_monthly_vol'] * segment_levers['volume_multiplier']
            size = r['avg_size'] * segment_levers['deal_size_multiplier']
            win_rate = r['win_rate'] * segment_levers['win_rate_multiplier']
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

def calculate_actual_segment_metrics(deals, start_date, end_date):
    """Calculate actual metrics per segment for perfect prediction in backtest"""
    period_deals = deals[
        (deals['date_created'] >= pd.to_datetime(start_date)) &
        (deals['date_created'] <= pd.to_datetime(end_date))
    ].copy()
    
    baseline = period_deals.groupby('market_segment').apply(
        lambda x: pd.Series({
            'deal_count': x['volume_weight'].sum(),
            'won_revenue': x[x['won']]['net_revenue'].sum(),
            'won_deals': x['won'].sum(),
            'total_deals': len(x),
            'months': x['created_month'].nunique()
        })
    ).reset_index()
    
    baseline['avg_monthly_vol'] = baseline['deal_count'] / baseline['months']
    baseline['avg_size'] = baseline['won_revenue'] / baseline['won_deals']  # Fixed: only won deals
    baseline['win_rate'] = baseline['won_deals'] / baseline['total_deals']
    
    return baseline

def run_backtest(snapshots, use_perfect_prediction=False):
    print(f"\n{'='*60}")
    print(f"BACKTEST {'(PERFECT PREDICTION MODE)' if use_perfect_prediction else ''}")
    print(f"{'='*60}")
    
    backtest_cutoff = pd.to_datetime(BACKTEST_DATE)
    backtest_end = pd.to_datetime(BACKTEST_THROUGH)
    
    # Build all deals including 2025
    actual_full_data = snapshots[snapshots['date_snapshot'] <= backtest_end].copy()
    actual_deals = build_deal_facts(actual_full_data)
    
    # Get historical deals (pre-2025) for baseline calculation
    historical_deals = actual_deals[actual_deals['date_created'] < backtest_cutoff].copy()
    
    # Get 2025 created deals for metrics calculation
    deals_created_2025 = actual_deals[
        (actual_deals['date_created'] >= backtest_cutoff) &
        (actual_deals['date_created'] <= backtest_end)
    ].copy()
    
    # Store original levers before any modifications
    import copy
    original_levers = copy.deepcopy(SCENARIO_LEVERS)
    
    # Variable to hold timing override for perfect prediction
    timing_override_2025 = None
    
    # If perfect prediction mode, calculate actual 2025 metrics per segment
    if use_perfect_prediction:
        print("\n[Using actual 2025 segment metrics as scenario levers]")
        
        # Calculate 2025 metrics
        metrics_2025 = calculate_actual_segment_metrics(deals_created_2025, backtest_cutoff, backtest_end)
        
        # Calculate historical baseline metrics
        if len(historical_deals) > 0:
            historical_metrics = calculate_actual_segment_metrics(
                historical_deals, 
                historical_deals['date_created'].min(), 
                backtest_cutoff - pd.Timedelta(days=1)
            )
        else:
            historical_metrics = pd.DataFrame()
        
        # Calculate and apply segment-specific multipliers
        for segment in metrics_2025['market_segment']:
            act = metrics_2025[metrics_2025['market_segment'] == segment].iloc[0]
            hist = historical_metrics[historical_metrics['market_segment'] == segment]
            
            if len(hist) > 0:
                hist = hist.iloc[0]
                vol_mult = act['avg_monthly_vol'] / hist['avg_monthly_vol'] if hist['avg_monthly_vol'] > 0 else 1.0
                wr_mult = act['win_rate'] / hist['win_rate'] if hist['win_rate'] > 0 else 1.0
                size_mult = act['avg_size'] / hist['avg_size'] if hist['avg_size'] > 0 else 1.0
            else:
                vol_mult, wr_mult, size_mult = 1.0, 1.0, 1.0
            
            # Apply to SCENARIO_LEVERS temporarily
            if segment not in SCENARIO_LEVERS:
                SCENARIO_LEVERS[segment] = {}
            
            SCENARIO_LEVERS[segment]['volume_multiplier'] = vol_mult
            SCENARIO_LEVERS[segment]['win_rate_multiplier'] = wr_mult
            SCENARIO_LEVERS[segment]['deal_size_multiplier'] = size_mult
            
            print(f"  {segment}: vol={vol_mult:.2f}x, wr={wr_mult:.2f}x, size={size_mult:.2f}x")
        
        # Calculate actual 2025 timing distribution
        won_2025 = deals_created_2025[deals_created_2025['won'] & deals_created_2025['date_closed'].notna()].copy()
        if len(won_2025) > 0:
            won_2025['months_to_close'] = ((won_2025['date_closed'] - won_2025['date_created']).dt.days / 30).clip(lower=0, upper=11).round().astype(int)
            timing_override_2025 = won_2025.groupby(['market_segment', 'months_to_close']).size().reset_index(name='count')
            timing_totals = timing_override_2025.groupby('market_segment')['count'].sum().reset_index(name='total')
            timing_override_2025 = timing_override_2025.merge(timing_totals, on='market_segment')
            timing_override_2025['pct'] = timing_override_2025['count'] / timing_override_2025['total']
            print("\n[Using actual 2025 timing distribution]")
    
    # Build 2025 probabilities from actual 2025 exits
    stage_probs_2025 = build_stage_probabilities(
        snapshots[(snapshots['date_snapshot'] >= backtest_cutoff) & (snapshots['date_snapshot'] <= backtest_end)],
        weight_recent=False
    )
    
    # In perfect prediction mode, override with actual conversion rates for deals open at start of year
    if use_perfect_prediction:
        # Get deals that were open at start of 2025
        last_2024 = snapshots[snapshots['date_snapshot'] < backtest_cutoff]['date_snapshot'].max()
        open_2024_ids = snapshots[
            (snapshots['date_snapshot'] == last_2024) &
            (~snapshots['stage'].isin(['Closed Won', 'Verbal', 'Closed Lost']))
        ]['deal_id'].unique()
        
        # Check which ones actually won in 2025
        final_2025 = snapshots[snapshots['date_snapshot'] == backtest_end]
        open_2024_outcomes = final_2025[
            (final_2025['deal_id'].isin(open_2024_ids)) &
            (final_2025['date_closed'] >= backtest_cutoff) &
            (final_2025['date_closed'] <= backtest_end) &
            (final_2025['date_closed'].notna())
        ].copy()
        
        # Calculate actual conversion rates by segment and stage
        if len(open_2024_outcomes) > 0:
            # Get stage for each deal as of end of 2024
            open_2024_stage = snapshots[
                (snapshots['date_snapshot'] == last_2024) &
                (snapshots['deal_id'].isin(open_2024_ids))
            ][['deal_id', 'market_segment', 'stage']].copy()
            
            # Merge with outcomes
            conversion = open_2024_stage.merge(
                open_2024_outcomes[['deal_id', 'stage']].rename(columns={'stage': 'final_stage'}),
                on='deal_id',
                how='left'
            )
            conversion['won'] = conversion['final_stage'].isin(['Closed Won', 'Verbal'])
            
            # Calculate actual conversion rates
            actual_conv = conversion.groupby(['market_segment', 'stage']).agg(
                wins=('won', 'sum'),
                total=('deal_id', 'count')
            ).reset_index()
            actual_conv['prob'] = (actual_conv['wins'] / actual_conv['total']).clip(upper=1.0)
            
            # Override stage probabilities
            stage_probs_2025 = actual_conv[['market_segment', 'stage', 'prob']]
            print(f"\n[Using actual active pipeline conversion rates: {len(open_2024_outcomes)} deals won from {len(open_2024_ids)} open deals]")
    
    staleness = build_staleness_thresholds(actual_full_data)
    
    # Historical data for active pipeline (deals open at start of 2025)
    historical_data = snapshots[snapshots['date_snapshot'] < backtest_cutoff].copy()
    
    forecast_months = pd.period_range(BACKTEST_DATE, BACKTEST_THROUGH, freq='M')
    
    # Active pipeline forecast
    if use_perfect_prediction:
        # In perfect prediction mode, use ACTUAL outcomes for active pipeline
        # Use the same logic as actuals - deals that were open (not fully closed) at start
        last_2024 = snapshots[snapshots['date_snapshot'] < backtest_cutoff]['date_snapshot'].max()
        open_at_start_ids = snapshots[
            (snapshots['date_snapshot'] == last_2024) &
            (~snapshots['is_closed'])
        ]['deal_id'].unique()
        
        # Get the actual deals that won from this group
        actuals_from_pipeline_active = actual_deals[
            (actual_deals['deal_id'].isin(open_at_start_ids)) &
            (actual_deals['won']) &
            (actual_deals['date_closed'] >= backtest_cutoff) &
            (actual_deals['date_closed'] <= backtest_end) &
            (actual_deals['date_closed'].notna())
        ].copy()
        
        if len(actuals_from_pipeline_active) > 0:
            actuals_from_pipeline_active['month'] = actuals_from_pipeline_active['date_closed'].dt.to_period('M').dt.to_timestamp()
            active = actuals_from_pipeline_active.groupby(['month', 'market_segment']).agg(
                expected_revenue=('net_revenue', 'sum'),
                expected_count=('deal_id', 'count')
            ).reset_index()
            print(f"\n[Active pipeline: Using actual outcomes - {len(actuals_from_pipeline_active)} deals won, ${actuals_from_pipeline_active['net_revenue'].sum():,.0f}]")
        else:
            active = pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
            print("\n[Active pipeline: No deals won from open pipeline]")
    else:
        # Standard forecast using probabilities
        active = forecast_active_pipeline(
            historical_data, stage_probs_2025, staleness,
            [m.to_timestamp() for m in forecast_months],
            cutoff_date=backtest_cutoff - pd.Timedelta(days=1)
        )
    
    # Get actuals for backtest comparison
    actuals_from_created = actual_deals[
        (actual_deals['date_created'] >= backtest_cutoff) &
        (actual_deals['date_created'] <= backtest_end) &
        (actual_deals['won']) &
        (actual_deals['date_closed'] >= backtest_cutoff) &
        (actual_deals['date_closed'] <= backtest_end) &
        (actual_deals['date_closed'].notna())
    ].copy()
    actuals_from_created['close_month'] = actuals_from_created['date_closed'].dt.to_period('M').dt.to_timestamp()
    
    # CRITICAL: We must pass 'BACKTEST_THROUGH' to match the actuals period
    if use_perfect_prediction:
        # To make backtest HONEST under perfect prediction, 
        # we calculate EXACTLY what the model WOULD HAVE predicted
        # if we knew the 2025 multipliers ahead of time.
        future = forecast_future_pipeline(
            historical_deals, 
            BACKTEST_DATE, 
            BACKTEST_THROUGH, 
            weight_recent=True,
            timing_override=timing_override_2025
        )
    else:
        # Standard backtest using only historical knowledge
        future = forecast_future_pipeline(
            historical_deals, 
            BACKTEST_DATE, 
            BACKTEST_THROUGH, 
            weight_recent=True,
            timing_override=timing_override_2025
        )
    
    # Combine
    all_parts = [active, future]
    all_parts = [p for p in all_parts if len(p) > 0]
    
    if len(all_parts) > 0:
        forecast = pd.concat(all_parts).groupby(['month', 'market_segment']).sum().reset_index()
    else:
        forecast = pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    # Actuals
    last_snapshot_date = snapshots[snapshots['date_snapshot'] <= backtest_end]['date_snapshot'].max()
    
    # Deals that were open at start and won during period  
    last_2024 = snapshots[snapshots['date_snapshot'] < backtest_cutoff]['date_snapshot'].max()
    open_at_start = snapshots[
        (snapshots['date_snapshot'] == last_2024) &
        (~snapshots['is_closed'])
    ]['deal_id'].unique()
    
    actuals_from_pipeline = actual_deals[
        (actual_deals['deal_id'].isin(open_at_start)) &
        (actual_deals['won']) &
        (actual_deals['date_closed'] >= backtest_cutoff) &
        (actual_deals['date_closed'] <= backtest_end) &
        (actual_deals['date_closed'].notna())
    ].copy()
    
    # Combine both sources for actuals
    all_actual_ids = list(actuals_from_created['deal_id']) + list(actuals_from_pipeline['deal_id'])
    actuals = actual_deals[actual_deals['deal_id'].isin(all_actual_ids)][['deal_id', 'market_segment', 'net_revenue', 'date_closed']].copy()
    
    print(f"\n[Actuals: {len(actuals_from_created)} from newly created deals (${actuals_from_created['net_revenue'].sum():,.0f}), " +
          f"{len(actuals_from_pipeline)} from active pipeline (${actuals_from_pipeline['net_revenue'].sum():,.0f})]")
    
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
    
    # Restore original levers if they were modified
    if use_perfect_prediction:
        SCENARIO_LEVERS.clear()
        SCENARIO_LEVERS.update(original_levers)
        print("\n[Restored original scenario levers for FY26 forecast]")
    
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
            'win_rate_change_pct': round(((required_wr / hist_wr) - 1) * 100, 1) if hist_vol > 0 else 0
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
    
    # Calculate total won revenue (Won includes 'Closed Won' and 'Verbal' per user instructions)
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
        run_backtest(snapshots, use_perfect_prediction=BACKTEST_PERFECT_PREDICTION)
    
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

    # Active Pipeline Total Check
    active_total_unweighted = snapshots[(snapshots['date_snapshot'] == snapshots['date_snapshot'].max()) & (~snapshots['is_closed'])]['net_revenue'].sum()
    print(f"\nACTIVE PIPELINE (UNWEIGHTED): ${active_total_unweighted:,.2f}")

    summary = forecast.groupby('market_segment').agg(
        total_revenue=('expected_won_revenue', 'sum'),
        total_deals=('expected_won_deal_count', 'sum')
    )
    
    print("\nFORECAST SUMMARY (FY26):")
    print(summary)
    print(f"\nTOTAL FY26 FORECAST: ${summary['total_revenue'].sum():,.2f}")
    
    # YoY Check
    print(f"\n{'='*60}")
    print(f"YoY STABILITY CHECK")
    print(f"{'='*60}")
    
    # Calculate 2025 actuals for comparison
    actual_2025_revenue = deals[
        (deals['won']) & 
        (deals['date_closed'] >= '2025-01-01') & 
        (deals['date_closed'] <= '2025-12-31')
    ]['net_revenue'].sum()
    
    print(f"2025 Actual Won Revenue: ${actual_2025_revenue:,.2f}")
    print(f"2026 Forecasted Revenue: ${summary['total_revenue'].sum():,.2f}")
    
    yoy_delta = (summary['total_revenue'].sum() / actual_2025_revenue - 1) if actual_2025_revenue > 0 else 0
    print(f"YoY Change: {yoy_delta*100:.1f}%")
    
    if abs(yoy_delta) > 0.4:
        print("\n[WARNING] Forecast is >40% different from 2025 actuals.")
    print()
    
    if RUN_GOAL_SEEK:
        run_goal_seek_analysis(deals, FORECAST_START, FORECAST_END)

    return forecast

if __name__ == "__main__":
    result = run_forecast()
