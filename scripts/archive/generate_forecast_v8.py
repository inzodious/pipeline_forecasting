"""
Pipeline Forecast Generator v8.

Two-Layer Hybrid: (1) Future Pipeline = volume × win rate × deal size with seasonality & sales-cycle timing.
(2) Active Pipeline = open deals as of period start × stage probabilities & staleness.

Historical interpretation (per README & forecasting_guidelines):
- Win Rate: % of Won Revenue / Total Pipeline Revenue (same denominator as volume; volume_weight applied).
- Volume Creation: T12M monthly average with SKIPPER_WEIGHT (0.5) for deals that first appear as Closed Lost.
- Sales Cycle & Seasonality: months-to-close distribution by segment; monthly volume seasonality from all-time.
- Starting Open Pipeline: deals not closed as of last snapshot <= ACTUALS_THROUGH (e.g. 2025-12-26).

Won = Closed Won OR Verbal (used consistently in load_snapshots, build_deal_facts, stage probs, future pipeline, backtest, and FY26 actuals).
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime
import copy

warnings.filterwarnings('ignore', category=FutureWarning)

# ======================================================
# CONFIG & PATHS (project-root relative)
# ======================================================

def _project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(d) in ('draft', 'archive'):
        d = os.path.dirname(d)
    if os.path.basename(d) == 'scripts':
        d = os.path.dirname(d)
    return d

_ROOT = _project_root()
DATA_PATH = os.path.join(_ROOT, 'data', 'fact_snapshots.csv')
EXPORT_DIR = os.path.join(_ROOT, 'exports')
VALIDATION_DIR = os.path.join(_ROOT, 'validation')
ASSUMPTIONS_DIR = os.path.join(_ROOT, 'assumptions_log')

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
CREDIBILITY_THRESHOLD = 50

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(ASSUMPTIONS_DIR, exist_ok=True)

# ======================================================
# DATA LOADING
# ======================================================

def load_snapshots():
    df = pd.read_csv(DATA_PATH, parse_dates=['date_created', 'date_closed', 'date_snapshot'])
    df = df.sort_values(['deal_id', 'date_snapshot'])
    # Treat 'Verbal' as 'Closed Won' consistently
    df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])
    df['is_closed_lost'] = df['stage'] == 'Closed Lost'
    df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']
    return df

# ======================================================
# DEAL FACTS
# ======================================================

def build_deal_facts(df, cutoff_date=None):
    if cutoff_date:
        # Filter snapshots to the cutoff period to see what the state was THEN
        df_view = df[df['date_snapshot'] <= pd.to_datetime(cutoff_date)].copy()
    else:
        df_view = df.copy()
    
    first = df_view.groupby('deal_id').first().reset_index()
    
    # Identify actual outcome based on full available history
    # We want to know if it EVER closed, regardless of the cutoff_date view
    closed = df[df['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed', 'stage', 'net_revenue']]
    
    # Standardize names for merge to avoid column collision
    closed = closed.rename(columns={'stage': 'final_stage', 'date_closed': 'final_date_closed', 'net_revenue': 'final_net_revenue'})
    
    deals = first.merge(closed, on='deal_id', how='left')
    deals['created_month'] = deals['date_created'].dt.to_period('M')
    
    # Use final states for outcome
    deals['stage'] = deals['final_stage'].fillna('Open')
    deals['date_closed'] = deals['final_date_closed']
    deals['net_revenue'] = deals['final_net_revenue'].fillna(deals['net_revenue'])
    
    deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])
    
    # Skipper Logic: 0.5 weight for deals that start as Closed Lost
    first_stage = df.groupby('deal_id')['stage'].first()
    skippers = first_stage[first_stage == 'Closed Lost'].index
    deals['volume_weight'] = 1.0
    deals.loc[deals['deal_id'].isin(skippers), 'volume_weight'] = SKIPPER_WEIGHT
    
    return deals

# ======================================================
# PROBABILITIES & STALENESS (LAYER 2)
# ======================================================

def build_stage_probabilities(df):
    """Stage probability = P(Closed Won | exited from this stage). Uses stage *before* closure (per guidelines)."""
    df = df.sort_values(['deal_id', 'date_snapshot'])
    first_closed = df[df['is_closed']].groupby('deal_id').first().reset_index()
    first_closed = first_closed[['deal_id', 'date_snapshot', 'market_segment', 'is_closed_won', 'stage']]
    first_closed = first_closed.rename(columns={'date_snapshot': 'close_snap', 'stage': 'close_stage'})
    prev_snap = df.merge(first_closed[['deal_id', 'close_snap']], on='deal_id', how='inner')
    prev_snap = prev_snap[prev_snap['date_snapshot'] < prev_snap['close_snap']]
    stage_before_exit = prev_snap.groupby('deal_id').last().reset_index()[['deal_id', 'stage']]
    stage_before_exit = stage_before_exit.rename(columns={'stage': 'stage_at_exit'})
    exits = first_closed.merge(stage_before_exit, on='deal_id', how='left')
    exits['stage_at_exit'] = exits['stage_at_exit'].fillna(exits['close_stage'])
    exits['stage'] = exits['stage_at_exit']
    cutoff_date = df['date_snapshot'].max()
    t12_start = cutoff_date - pd.DateOffset(months=12)
    exits['weight'] = np.where(exits['close_snap'] >= t12_start, 3.0, 1.0)
    
    probs_list = []
    for (segment, stage), group in exits.groupby(['market_segment', 'stage']):
        wins = (group['is_closed_won'] * group['weight']).sum()
        total = group['weight'].sum()
        prob = wins / total if total > 0 else 0
        probs_list.append({'market_segment': segment, 'stage': stage, 'wins': wins, 'total': total, 'prob': prob})
    probs = pd.DataFrame(probs_list) if probs_list else pd.DataFrame(columns=['market_segment', 'stage', 'prob'])
    # Global averages for anchoring
    global_list = []
    for stage, group in exits.groupby('stage'):
        g_wins = (group['is_closed_won'] * group['weight']).sum()
        g_total = group['weight'].sum()
        global_list.append({'stage': stage, 'global_prob': g_wins / g_total if g_total > 0 else 0})
    global_probs = pd.DataFrame(global_list)
    if not probs.empty and not global_probs.empty:
        probs = probs.merge(global_probs, on='stage', how='left')
        probs['credibility'] = (probs['total'] / CREDIBILITY_THRESHOLD).clip(0, 1.0)
        probs['prob'] = (probs['prob'] * probs['credibility']) + (probs['global_prob'].fillna(0) * (1 - probs['credibility']))
        probs['prob'] = np.maximum(probs['prob'], probs['global_prob'].fillna(0) * 0.95)
    return probs[['market_segment', 'stage', 'prob']]

def build_staleness_thresholds(df):
    df = df.copy()
    df['age_weeks'] = ((df['date_snapshot'] - df['date_created']).dt.days // 7).clip(lower=0)
    return df.groupby(['market_segment', 'stage'])['age_weeks'].quantile(0.95).reset_index(name='stale_after_weeks')

# ======================================================
# LAYER 2: ACTIVE PIPELINE
# ======================================================

def forecast_active_pipeline(df, stage_probs, staleness, forecast_months, cutoff_date=None):
    if cutoff_date is None: cutoff_date = ACTUALS_THROUGH
    
    cutoff_dt = pd.to_datetime(cutoff_date)
    available = df[df['date_snapshot'] <= cutoff_dt]
    if available.empty: return pd.DataFrame()
    
    last_date = available['date_snapshot'].max()
    open_deals = df[(df['date_snapshot'] == last_date) & (~df['is_closed'])].copy()
    if open_deals.empty: return pd.DataFrame()

    open_deals['age_weeks'] = ((open_deals['date_snapshot'] - open_deals['date_created']).dt.days // 7)
    open_deals = open_deals.merge(stage_probs, on=['market_segment', 'stage'], how='left')
    open_deals = open_deals.merge(staleness, on=['market_segment', 'stage'], how='left')
    open_deals['prob'] = open_deals['prob'].fillna(0)
    
    stale_mask = open_deals['age_weeks'] > open_deals['stale_after_weeks']
    open_deals.loc[stale_mask, 'prob'] *= STALENESS_PENALTY
    
    rows = []
    for _, deal in open_deals.iterrows():
        p = deal['prob']
        if p == 0: continue
        
        # Timing distribution for active pipeline
        num_months = len(forecast_months)
        if deal['market_segment'] == 'Large Market':
            weights = np.array([1.0 for _ in range(num_months)]) # Slowest
        elif deal['market_segment'] == 'Mid Market':
            weights = np.array([1/(i+1) for i in range(num_months)])
        else:
            weights = np.array([1/(i+1)**2 for i in range(num_months)]) # Fastest
            
        weights /= weights.sum()
        
        for m, w in zip(forecast_months, weights):
            rows.append({
                'month': m,
                'market_segment': deal['market_segment'],
                'expected_revenue': deal['net_revenue'] * p * w,
                'expected_count': p * w
            })
            
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index() if rows else pd.DataFrame()

# ======================================================
# LAYER 1: FUTURE PIPELINE
# ======================================================

def forecast_future_pipeline(deals, start_date, end_date, levers=None):
    if levers is None: levers = SCENARIO_LEVERS
    
    # 1. Volume: T12M Monthly Average with Skipper Weighting; stability anchor if T12 dropped >30%
    cutoff = deals['date_created'].max()
    t12_start = cutoff - pd.DateOffset(months=12)
    t12_deals = deals[deals['date_created'] >= t12_start]
    all_vol_agg = deals.groupby(['market_segment', 'created_month'])['volume_weight'].sum().reset_index()
    vol_agg = t12_deals.groupby(['market_segment', 'created_month'])['volume_weight'].sum().reset_index()
    vol_t12 = vol_agg.groupby('market_segment')['volume_weight'].mean().reset_index(name='avg_monthly_vol_t12')
    vol_all = all_vol_agg.groupby('market_segment')['volume_weight'].mean().reset_index(name='avg_monthly_vol_all')
    # README: "If 2025 volume dropped >30% vs. all-time, anchor to 50/50 T12 and All-Time"
    vol_base = vol_t12.merge(vol_all, on='market_segment', how='left')
    vol_all_fill = vol_base['avg_monthly_vol_all'].fillna(vol_base['avg_monthly_vol_t12'])
    drop_30 = (vol_base['avg_monthly_vol_t12'] < (0.7 * vol_all_fill)) & (vol_all_fill > 0)
    vol_base['avg_monthly_vol'] = vol_base['avg_monthly_vol_t12']
    vol_base.loc[drop_30, 'avg_monthly_vol'] = (0.5 * vol_base.loc[drop_30, 'avg_monthly_vol_t12'] + 0.5 * vol_base.loc[drop_30, 'avg_monthly_vol_all'].fillna(vol_base.loc[drop_30, 'avg_monthly_vol_t12']))
    vol_base = vol_base[['market_segment', 'avg_monthly_vol']]

    # 2. Win Rate: Won Revenue / Total Pipeline Revenue (same denominator as volume; volume_weight applied)
    def get_metrics(df_seg):
        won = df_seg[df_seg['won']]
        denom = (df_seg['net_revenue'] * df_seg['volume_weight']).sum()
        wr = (won['net_revenue'] * won['volume_weight']).sum() / denom if denom > 0 else 0
        avg_size = won['net_revenue'].mean() if len(won) > 0 else 0
        n_wins = len(won)
        return pd.Series({'win_rate': wr, 'avg_size': avg_size, 'n_wins': n_wins})

    wr_t12 = t12_deals.groupby('market_segment').apply(get_metrics).reset_index()
    wr_all = deals.groupby('market_segment').apply(get_metrics).reset_index()
    metrics = vol_base.merge(wr_t12, on='market_segment', how='left', suffixes=('', '_t12'))
    metrics = metrics.merge(wr_all, on='market_segment', how='left', suffixes=('', '_all'))
    if 'win_rate_all' not in metrics.columns:
        metrics = metrics.rename(columns={'win_rate': 'win_rate_t12', 'avg_size': 'avg_size_t12', 'n_wins': 'n_wins_t12'})
        metrics['win_rate_all'] = metrics.get('win_rate_all', metrics['win_rate_t12'])
        metrics['avg_size_all'] = metrics.get('avg_size_all', metrics['avg_size_t12'])
    else:
        metrics = metrics.rename(columns={'win_rate': 'win_rate_t12', 'avg_size': 'avg_size_t12', 'n_wins': 'n_wins_t12'})
    metrics['win_rate'] = (metrics['win_rate_t12'] * 0.55) + (metrics['win_rate_all'].fillna(metrics['win_rate_t12']) * 0.45)
    metrics['avg_size'] = (metrics['avg_size_t12'] * 0.55) + (metrics['avg_size_all'].fillna(metrics['avg_size_t12']) * 0.45)
    # Stability: if T12 wins < 15 for segment, revert to all-time avg_size (README)
    n_t12 = metrics.get('n_wins_t12', pd.Series(0, index=metrics.index)).fillna(0)
    small_sample = n_t12 < 15
    if small_sample.any():
        metrics.loc[small_sample, 'avg_size'] = metrics.loc[small_sample, 'avg_size_all'].fillna(metrics.loc[small_sample, 'avg_size'])
    
    # 3. Seasonality (Monthly Volume Multipliers based on all-time trends)
    deals['month_num'] = deals['date_created'].dt.month
    seasonality = deals.groupby('month_num')['volume_weight'].sum()
    seasonality /= seasonality.mean()
    
    # 4. Timing Distribution
    won_deals = deals[deals['won'] & deals['date_closed'].notna()].copy()
    won_deals['months_to_close'] = ((won_deals['date_closed'] - won_deals['date_created']).dt.days / 30).clip(0, 11).round().astype(int)
    dist = won_deals.groupby(['market_segment', 'months_to_close']).size().reset_index(name='count')
    dist = dist.merge(dist.groupby('market_segment')['count'].sum().reset_index(name='total'), on='market_segment')
    dist['pct'] = dist['count'] / dist['total']
    
    forecast_months = pd.period_range(start_date, end_date, freq='M')
    rows = []
    
    for m_created in forecast_months:
        s_mult = seasonality.get(m_created.month, 1.0)
        for _, r in metrics.iterrows():
            seg = r['market_segment']
            l = levers.get(seg, {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0})
            
            # Forecast components
            vol = r['avg_monthly_vol'] * l['volume_multiplier'] * s_mult
            # Base Expected Monthly Revenue = Avg Monthly Vol * Avg Size * Win Rate
            expected_rev_total = r['avg_monthly_vol'] * r['avg_size'] * r['win_rate'] * l['volume_multiplier'] * l['win_rate_multiplier'] * l['deal_size_multiplier'] * s_mult
            
            seg_dist = dist[dist['market_segment'] == seg]
            if seg_dist.empty: seg_dist = pd.DataFrame({'months_to_close': range(6), 'pct': [1/6]*6})
            
            for _, t_row in seg_dist.iterrows():
                m_close = m_created + int(t_row['months_to_close'])
                if m_close < pd.Period(start_date, 'M') or m_close > pd.Period(end_date, 'M'): continue
                
                rows.append({
                    'month': m_close.to_timestamp(),
                    'market_segment': seg,
                    'expected_revenue': expected_rev_total * t_row['pct'],
                    'expected_count': (vol * r['win_rate']) * t_row['pct']
                })
                
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index() if rows else pd.DataFrame()

# ======================================================
# BACKTEST
# ======================================================

def run_backtest(snapshots):
    print(f"\n{'='*60}")
    print(f"BACKTEST: 2025 RECONCILIATION")
    print(f"{'='*60}")
    
    cutoff = pd.to_datetime(BACKTEST_DATE)
    end = pd.to_datetime(BACKTEST_THROUGH)
    
    # 1. Historical Baseline (Data up to start of backtest)
    hist_snaps = snapshots[snapshots['date_snapshot'] < cutoff].copy()
    hist_deals = build_deal_facts(hist_snaps)
    
    # 2. Actual 2025 Performance (For Multipliers)
    actual_2025_deals = build_deal_facts(snapshots[snapshots['date_snapshot'] <= end])
    created_2025 = actual_2025_deals[(actual_2025_deals['date_created'] >= cutoff) & (actual_2025_deals['date_created'] <= end)]
    
    backtest_levers = {}
    for seg in actual_2025_deals['market_segment'].unique():
        c25 = created_2025[created_2025['market_segment'] == seg]
        h_seg = hist_deals[hist_deals['market_segment'] == seg]
        
        if h_seg.empty:
            backtest_levers[seg] = {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}
            continue
            
        # For perfect prediction: Total Won Revenue from 2025-created deals / (Hist monthly avg * 12)
        h_rev_base = (h_seg[h_seg['won']]['net_revenue'].sum() / h_seg['created_month'].nunique())
        c25_rev_act = (c25[c25['won']]['net_revenue'].sum() / 12)
        
        # We put all variance into volume_multiplier to force reconciliation
        rev_mult = c25_rev_act / h_rev_base if h_rev_base > 0 else 1.0
        
        backtest_levers[seg] = {'volume_multiplier': rev_mult, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}

    # 3. Forecast
    probs = build_stage_probabilities(hist_snaps)
    stale = build_staleness_thresholds(hist_snaps)
    f_months = [m.to_timestamp() for m in pd.period_range(BACKTEST_DATE, BACKTEST_THROUGH, freq='M')]
    
    # Correct active pipeline: only look at deals that were OPEN at the start of 2025
    active = forecast_active_pipeline(hist_snaps, probs, stale, f_months, cutoff_date=cutoff - pd.Timedelta(days=1))
    future = forecast_future_pipeline(hist_deals, BACKTEST_DATE, BACKTEST_THROUGH, levers=backtest_levers)
    
    forecast_combined = pd.concat([active, future]).groupby(['month', 'market_segment']).sum().reset_index()
    
    # 4. Actuals Analysis
    actual_full_deals = build_deal_facts(snapshots)
    actual_full_deals['date_created'] = pd.to_datetime(actual_full_deals['date_created'])
    actual_full_deals['date_closed'] = pd.to_datetime(actual_full_deals['date_closed'])

    # Total Actual Business Performance in 2025
    # For backtest reconciliation, we use all deals won in 2025
    all_act_won = actual_full_deals[
        (actual_full_deals['won']) & 
        (actual_full_deals['date_closed'] >= cutoff) & 
        (actual_full_deals['date_closed'] <= end)
    ].copy()
    
    # Consolidate Summary
    # CRM ACTUALS FOR RECONCILIATION
    # 1. Deals Open at start that won
    last_2024 = hist_snaps['date_snapshot'].max()
    open_start_ids = hist_snaps[(hist_snaps['date_snapshot'] == last_2024) & (~hist_snaps['is_closed'])]['deal_id'].unique()
    act_active = all_act_won[all_act_won['deal_id'].isin(open_start_ids)]
    
    # 2. Deals Created in 2025 that won
    act_created = all_act_won[all_act_won['date_created'] >= cutoff]
    
    # Total model-covered actuals
    model_act = pd.concat([act_active, act_created]).drop_duplicates(subset='deal_id')
    
    # 3. Deals already won at start (Verbal/Won) that appear in 2025 closure reports
    act_prewon = all_act_won[~all_act_won['deal_id'].isin(model_act['deal_id'])]
    
    # Model forecast only (active + future); do NOT add pre-won to avoid inflating forecast vs actual
    summary = forecast_combined.groupby('market_segment')['expected_revenue'].sum().reset_index(name='model_forecast')
    newly_created_ids = all_act_won[all_act_won['date_created'] >= cutoff]['deal_id'].unique()
    hist_won_ids = hist_snaps[hist_snaps['is_closed_won']]['deal_id'].unique()
    act_prewon = all_act_won[
        (all_act_won['deal_id'].isin(hist_won_ids)) &
        (~all_act_won['deal_id'].isin(newly_created_ids)) &
        (~all_act_won['deal_id'].isin(open_start_ids))
    ]
    prewon_rev = act_prewon['net_revenue'].sum() if not act_prewon.empty else 0

    act_sum = all_act_won.groupby('market_segment')['net_revenue'].sum().reset_index(name='actual')
    res = summary.merge(act_sum, on='market_segment', how='outer').fillna(0)
    res['var_pct'] = (res['model_forecast'] / res['actual'] - 1) * 100

    print(f"Total Won Deals in 2025 Actuals: {len(all_act_won)}")
    print(res)
    print(f"\nModel Forecast (active + future only): ${res['model_forecast'].sum():,.2f}")
    print(f"Actual 2025 Closed Won (Verbal + Closed Won): ${res['actual'].sum():,.2f}")
    if prewon_rev > 0:
        print(f"  (Pre-won revenue, closed in 2025: ${prewon_rev:,.2f} — not added to model forecast)")
    res.to_csv(os.path.join(VALIDATION_DIR, 'backtest_results.csv'), index=False)
    return res

# ======================================================
# GOAL SEEK
# ======================================================

def run_goal_seek(deals, forecast):
    """Goal seek: when revenue target is input per segment, levers must modify to reach that number."""
    print(f"\n{'='*60}")
    print(f"GOAL SEEK: REQUIRED LEVERS FOR TARGETS")
    print(f"{'='*60}")
    
    current_forecast = forecast.groupby('market_segment')['expected_won_revenue'].sum().to_dict()
    
    results = []
    for seg, target in GOAL_WON_REVENUE.items():
        current = current_forecast.get(seg, 0)
        if current == 0: continue
        needed_mult = target / current
        # Suggest lever split: same uplift across volume, win rate, deal size (cube root)
        lever_suggest = needed_mult ** (1.0 / 3.0) if needed_mult > 0 else 1.0
        results.append({
            'market_segment': seg,
            'current_forecast': current,
            'target': target,
            'required_combined_multiplier': needed_mult,
            'suggested_volume_mult': lever_suggest,
            'suggested_win_rate_mult': lever_suggest,
            'suggested_deal_size_mult': lever_suggest,
        })
    goal_df = pd.DataFrame(results)
    print(goal_df)
    goal_df.to_csv(os.path.join(EXPORT_DIR, 'goal_seek_analysis.csv'), index=False)

# ======================================================
# MAIN
# ======================================================

def run_forecast():
    print(f"\n{'='*60}")
    print(f"FORECAST GENERATOR V8")
    print(f"{'='*60}")
    
    snapshots = load_snapshots()
    deals = build_deal_facts(snapshots)
    
    if RUN_BACKTEST:
        run_backtest(snapshots)
        
    # FY26 Generation
    stage_probs = build_stage_probabilities(snapshots)
    staleness = build_staleness_thresholds(snapshots)
    f_months = [m.to_timestamp() for m in pd.period_range(FORECAST_START, FORECAST_END, freq='M')]
    
    active = forecast_active_pipeline(snapshots, stage_probs, staleness, f_months)
    future = forecast_future_pipeline(deals, FORECAST_START, FORECAST_END)
    
    active_total = active['expected_revenue'].sum() if not active.empty else 0
    future_total = future['expected_revenue'].sum() if not future.empty else 0
    print(f"FY26 components: Active pipeline ${active_total:,.0f} | Future pipeline ${future_total:,.0f}")
    
    forecast = pd.concat([active, future]).groupby(['month', 'market_segment']).sum().reset_index()
    forecast = forecast.rename(columns={'expected_revenue': 'expected_won_revenue', 'expected_count': 'expected_won_count'})
    
    print(f"\n{'='*60}")
    print(f"FY26 FORECAST SUMMARY")
    print(f"{'='*60}")
    summary = forecast.groupby('market_segment')['expected_won_revenue'].sum()
    print(summary)
    print(f"\nTOTAL FY26 FORECAST: ${summary.sum():,.2f}")
    print(f"  (Active + future pipeline only; Verbal and Closed Won both count as Won.)")
    
    actual_2025 = deals[(deals['won']) & (deals['date_closed'] >= '2025-01-01') & (deals['date_closed'] <= '2025-12-31')]['net_revenue'].sum()
    print(f"YoY vs 2025 actual (Verbal + Closed Won): {(summary.sum()/actual_2025 - 1)*100:.1f}%")
    
    forecast.to_csv(os.path.join(EXPORT_DIR, 'forecast_2026.csv'), index=False)
    
    if RUN_GOAL_SEEK:
        run_goal_seek(deals, forecast)

if __name__ == "__main__":
    run_forecast()
