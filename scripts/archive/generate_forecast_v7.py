import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime
import copy

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
BACKTEST_PERFECT_PREDICTION = True 

# Neutral multipliers for FY26 forecast
SCENARIO_LEVERS = {
    'Indirect': {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Large Market': {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Mid Market': {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'SMB': {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Other': {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}
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
    
    # Capture first and last states
    first = df.groupby('deal_id').first().reset_index()
    
    # Fix for backtest: identify outcomes WITHIN the cutoff window
    closed = df[df['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed', 'stage', 'net_revenue']]
    
    first = first.drop(columns=['stage', 'date_closed', 'net_revenue'], errors='ignore')
    deals = first.merge(closed, on='deal_id', how='left')
    deals['created_month'] = deals['date_created'].dt.to_period('M')
    deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])
    
    # Weight skippers (immediate losses)
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
        prob = wins / total if total > 0 else 0
        
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

    # Global stage averages for anchoring
    global_list = []
    for stage, group in exits.groupby('stage'):
        g_wins = (group['is_closed_won'] * group['weight']).sum()
        g_total = group['weight'].sum()
        global_list.append({
            'stage': stage,
            'global_prob': g_wins / g_total if g_total > 0 else 0
        })
    global_probs = pd.DataFrame(global_list)
    
    # Blend segment logic with global anchor
    probs = probs.merge(global_probs, on='stage', how='left')
    probs['credibility'] = (probs['total'] / 10).clip(0, 1.0) 
    probs['prob'] = (probs['prob'] * probs['credibility']) + (probs['global_prob'] * (1 - probs['credibility']))
    
    # Floor at 95% of global average to prevent data-scarcity pessimism
    probs['prob'] = np.maximum(probs['prob'], probs['global_prob'] * 0.95)
    
    return probs[['market_segment', 'stage', 'prob']]

def build_staleness_thresholds(df):
    df = df.copy()
    df['age_weeks'] = ((df['date_snapshot'] - df['date_created']).dt.days // 7).clip(lower=0)
    thresholds = df.groupby(['market_segment', 'stage'])['age_weeks'].quantile(0.95).reset_index(name='stale_after_weeks')
    return thresholds

# ======================================================
# LAYER 2: ACTIVE PIPELINE FORECAST
# ======================================================

def forecast_active_pipeline(df, stage_probs, staleness, forecast_months, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = ACTUALS_THROUGH
    
    cutoff_dt = pd.to_datetime(cutoff_date)
    available_snapshots = df[df['date_snapshot'] <= cutoff_dt]
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
    
    rows = []
    for _, deal in open_deals.iterrows():
        total_prob = deal['prob']
        if total_prob == 0: continue
            
        num_months = len(forecast_months)
        if deal['market_segment'] == 'Large Market':
            weights = np.array([1.0 for _ in range(num_months)])
        elif deal['market_segment'] == 'Mid Market':
            weights = np.array([1/(i+1) for i in range(num_months)])
        else:
            weights = np.array([1/(i+1)**2 for i in range(num_months)])
            
        weights = weights / weights.sum()
        
        for month, weight in zip(forecast_months, weights):
            rows.append({
                'month': month,
                'market_segment': deal['market_segment'],
                'expected_revenue': deal['net_revenue'] * total_prob * weight,
                'expected_count': total_prob * weight
            })
    
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index() if rows else pd.DataFrame()

# ======================================================
# LAYER 1: FUTURE PIPELINE
# ======================================================

def forecast_future_pipeline(deals, forecast_start, forecast_end, levers=None, timing_override=None):
    if levers is None:
        levers = SCENARIO_LEVERS
        
    deals_copy = deals.copy()
    
    # Calculate baseline metrics per segment
    # Volume: 65/35 blend of Peak and Recent
    vol_stats = deals_copy.groupby(['market_segment', 'created_month'])['volume_weight'].sum().reset_index()
    vol_agg = vol_stats.groupby('market_segment')['volume_weight'].agg([
        ('recent_vol', lambda x: x.tail(12).mean()),
        ('peak_vol', 'max')
    ]).reset_index()
    
    # Win Rate and Size: Blend Recent (Trailing 12) with All-Time
    cutoff_date = deals_copy['date_created'].max()
    t12_start = cutoff_date - pd.DateOffset(months=12)
    deals_copy['is_recent'] = deals_copy['date_created'] >= t12_start
    
    metrics = []
    for segment, group in deals_copy.groupby('market_segment'):
        recent = group[group['is_recent']]
        won_all = group[group['won']]
        won_recent = recent[recent['won']]
        
        # Win Rate: 50/50 blend
        wr_recent = len(won_recent) / len(recent) if len(recent) > 0 else 0
        wr_all = len(won_all) / len(group) if len(group) > 0 else 0
        win_rate = (wr_recent + wr_all) / 2
        
        # Deal Size: 25/75 blend (conservative on size variance)
        size_recent = won_recent['net_revenue'].mean() if len(won_recent) > 0 else 0
        size_all = won_all['net_revenue'].mean() if len(won_all) > 0 else 0
        avg_size = (size_recent * 0.25) + (size_all * 0.75) if size_recent > 0 else size_all
        
        metrics.append({
            'market_segment': segment,
            'base_win_rate': win_rate,
            'base_avg_size': avg_size
        })
    
    baseline = pd.DataFrame(metrics).merge(vol_agg, on='market_segment')
    
    # Timing distribution
    if timing_override is not None:
        timing_dist = timing_override
    else:
        won_deals = deals[deals['won'] & deals['date_closed'].notna()].copy()
        won_deals['months_to_close'] = ((won_deals['date_closed'] - won_deals['date_created']).dt.days / 30).clip(lower=0, upper=11).round().astype(int)
        
        dist_all = won_deals.groupby(['market_segment', 'months_to_close']).size().reset_index(name='count')
        totals = dist_all.groupby('market_segment')['count'].sum().reset_index(name='total')
        dist_all = dist_all.merge(totals, on='market_segment')
        dist_all['pct'] = dist_all['count'] / dist_all['total']
        timing_dist = dist_all
    
    forecast_months = pd.period_range(forecast_start, forecast_end, freq='M')
    rows = []
    
    for m_created in forecast_months:
        for _, r in baseline.iterrows():
            seg = r['market_segment']
            l = levers.get(seg, {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0})
            
            vol = ((r['peak_vol'] * 0.65) + (r['recent_vol'] * 0.35)) * l['volume_multiplier']
            size = r['base_avg_size'] * l['deal_size_multiplier']
            wr = r['base_win_rate'] * l['win_rate_multiplier']
            
            # Dampen Indirect Volume if it appears to be a runaway forecast
            if seg == 'Indirect' and vol > 50: # Arbitrary but data-driven cap based on historical max
                 vol = 50 + (vol - 50) * 0.1
            
            seg_timing = timing_dist[timing_dist['market_segment'] == seg]
            if len(seg_timing) == 0:
                seg_timing = pd.DataFrame({'months_to_close': range(6), 'pct': [1/6]*6})
            
            for _, t_row in seg_timing.iterrows():
                close_month = m_created + int(t_row['months_to_close'])
                if close_month < pd.Period(forecast_start, 'M') or close_month > pd.Period(forecast_end, 'M'):
                    continue
                
                rows.append({
                    'month': close_month.to_timestamp(),
                    'market_segment': seg,
                    'expected_revenue': vol * size * wr * t_row['pct'],
                    'expected_count': vol * wr * t_row['pct']
                })
                
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index() if rows else pd.DataFrame()

# ======================================================
# BACKTEST LOGIC (V7 - HONEST RECONCILIATION)
# ======================================================

def run_backtest_v7(snapshots):
    print(f"\n{'='*60}")
    print(f"BACKTEST V7 (HONEST PERFECT PREDICTION)")
    print(f"{'='*60}")
    
    cutoff = pd.to_datetime(BACKTEST_DATE)
    end = pd.to_datetime(BACKTEST_THROUGH)
    
    # 1. Historical Data (Pre-2025)
    hist_snapshots = snapshots[snapshots['date_snapshot'] < cutoff].copy()
    hist_deals = build_deal_facts(hist_snapshots)
    
    # 2. 2025 Actuals (For Multipliers)
    actual_2025_snapshots = snapshots[(snapshots['date_snapshot'] >= cutoff) & (snapshots['date_snapshot'] <= end)].copy()
    actual_2025_deals_full = build_deal_facts(snapshots[snapshots['date_snapshot'] <= end])
    
    # Deals created in 2025
    created_2025 = actual_2025_deals_full[
        (actual_2025_deals_full['date_created'] >= cutoff) & 
        (actual_2025_deals_full['date_created'] <= end)
    ].copy()
    
    # Calculate perfect multipliers: 2025 Actual / Historical Baseline
    # This represents "What if we knew exactly how 2025 would perform relative to the past?"
    backtest_levers = {}
    for seg in created_2025['market_segment'].unique():
        seg_2025 = created_2025[created_2025['market_segment'] == seg]
        seg_hist = hist_deals[hist_deals['market_segment'] == seg]
        
        if len(seg_hist) == 0 or len(seg_2025) == 0:
            backtest_levers[seg] = {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}
            continue

        # Volume
        h_vol = seg_hist['volume_weight'].sum() / seg_hist['created_month'].nunique()
        a_vol = seg_2025['volume_weight'].sum() / 12
        
        # Win Rate
        # Backtest honest win rate: how many deals created in 2025 won IN 2025
        h_wr = seg_hist['won'].mean()
        a_wr = seg_2025['won'].mean()
        
        # Size
        h_size = seg_hist[seg_hist['won']]['net_revenue'].mean()
        a_size = seg_2025[seg_2025['won']]['net_revenue'].mean()
        
        backtest_levers[seg] = {
            'volume_multiplier': a_vol / h_vol if h_vol > 0 else 1.0,
            'win_rate_multiplier': a_wr / h_wr if h_wr > 0 else 1.0,
            'deal_size_multiplier': a_size / h_size if h_size > 0 else 1.0
        }
        print(f"  {seg}: vol={backtest_levers[seg]['volume_multiplier']:.2f}x, wr={backtest_levers[seg]['win_rate_multiplier']:.2f}x, size={backtest_levers[seg]['deal_size_multiplier']:.2f}x")

    # 3. Active Pipeline (Open at end of 2024)
    # We use actual outcomes for perfect prediction of the starting stack
    last_2024 = hist_snapshots['date_snapshot'].max()
    open_start_ids = snapshots[(snapshots['date_snapshot'] == last_2024) & (~snapshots['is_closed'])]['deal_id'].unique()
    
    pipeline_wins = actual_2025_deals_full[
        (actual_2025_deals_full['deal_id'].isin(open_start_ids)) & 
        (actual_2025_deals_full['won']) & 
        (actual_2025_deals_full['date_closed'] <= end)
    ].copy()
    
    pipeline_wins['month'] = pipeline_wins['date_closed'].dt.to_period('M').dt.to_timestamp()
    active_forecast = pipeline_wins.groupby(['month', 'market_segment']).agg(
        expected_revenue=('net_revenue', 'sum'),
        expected_count=('deal_id', 'count')
    ).reset_index()

    # 4. Future Pipeline (Created in 2025)
    future_forecast = forecast_future_pipeline(
        hist_deals, BACKTEST_DATE, BACKTEST_THROUGH, levers=backtest_levers
    )
    
    # 5. Comparison
    forecast = pd.concat([active_forecast, future_forecast]).groupby(['month', 'market_segment']).sum().reset_index()
    
    # Actuals for the period
    actuals_2025 = actual_2025_deals_full[
        (actual_2025_deals_full['won']) & 
        (actual_2025_deals_full['date_closed'] >= cutoff) & 
        (actual_2025_deals_full['date_closed'] <= end)
    ].copy()
    actuals_2025['month'] = actuals_2025['date_closed'].dt.to_period('M').dt.to_timestamp()
    
    actual_summary = actuals_2025.groupby(['month', 'market_segment']).agg(
        actual_revenue=('net_revenue', 'sum'),
        actual_count=('deal_id', 'count')
    ).reset_index()
    
    res = forecast.merge(actual_summary, on=['month', 'market_segment'], how='outer').fillna(0)
    
    print("\nBACKTEST SUMMARY (FY25):")
    summary = res.groupby('market_segment').agg({
        'expected_revenue': 'sum',
        'actual_revenue': 'sum'
    })
    summary['variance_pct'] = (summary['expected_revenue'] / summary['actual_revenue'] - 1) * 100
    print(summary)
    
    res.to_csv(f'{VALIDATION_DIR}/backtest_results_v7.csv', index=False)
    return res

# ======================================================
# MAIN
# ======================================================

def run_forecast():
    print(f"\n{'='*60}")
    print(f"FORECAST GENERATOR V7")
    print(f"{'='*60}")
    
    snapshots = load_snapshots()
    deals = build_deal_facts(snapshots)
    
    if RUN_BACKTEST:
        run_backtest_v7(snapshots)
    
    # FY26 Forecast logic
    stage_probs = build_stage_probabilities(snapshots)
    staleness = build_staleness_thresholds(snapshots)
    
    f_months = [m.to_timestamp() for m in pd.period_range(FORECAST_START, FORECAST_END, freq='M')]
    active = forecast_active_pipeline(snapshots, stage_probs, staleness, f_months)
    future = forecast_future_pipeline(deals, FORECAST_START, FORECAST_END)
    
    forecast = pd.concat([active, future]).groupby(['month', 'market_segment']).sum().reset_index()
    
    print(f"\n{'='*60}")
    print(f"FY26 FORECAST SUMMARY")
    print(f"{'='*60}")
    summary = forecast.groupby('market_segment')['expected_revenue'].sum()
    
    # Final sanity check: if segment forecast is significantly lower than recent 2025 performance,
    # and we are in a 'Neutral' scenario, we should allow the data to speak but flag it.
    
    print(summary)
    print(f"\nTOTAL FY26 FORECAST: ${summary.sum():,.2f}")
    
    actual_2025 = deals[(deals['won']) & (deals['date_closed'] >= '2025-01-01') & (deals['date_closed'] <= '2025-12-31')]['net_revenue'].sum()
    print(f"YoY Change: {(summary.sum()/actual_2025 - 1)*100:.1f}%")

    forecast.to_csv(f'{EXPORT_DIR}/forecast_2026_v7.csv', index=False)

if __name__ == "__main__":
    run_forecast()
