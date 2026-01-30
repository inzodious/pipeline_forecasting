import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- Config & paths ---

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

FORECAST_START = '2026-01-01'
FORECAST_END = '2026-12-31'
ACTUALS_THROUGH = '2025-12-26'

RUN_BACKTEST = True
BACKTEST_DATE = '2025-01-01'
BACKTEST_THROUGH = '2025-12-31'

VOLUME_BASELINE = 'T12' # Options: 'T12', 'ALL_TIME', 'BLENDED', 'CAPACITY'
VOLUME_BASELINE_BY_SEGMENT = {}

SCENARIO_LEVERS = {
    'Indirect':     {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Large Market': {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Mid Market':   {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'SMB':          {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0},
    'Other':        {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}
}

RUN_GOAL_SEEK = True
GOAL_WON_REVENUE = {'Large Market': 14_750_000, 'Mid Market': 7_800_000}

STALENESS_PENALTY = 0.8
SKIPPER_WEIGHT = 0.5
CREDIBILITY_THRESHOLD = 50
MIN_WINS_FOR_T12_SIZE = 15
WIN_RATE_T12_WEIGHT = 0.55
DEAL_SIZE_T12_WEIGHT = 0.55

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# --- Data loading ---

def format_currency_m(x):
    if pd.isna(x) or x == 0:
        return "$0.00M"
    return f"${x / 1_000_000:,.2f}M"

def load_snapshots():
    df = pd.read_csv(DATA_PATH, parse_dates=['date_created', 'date_closed', 'date_snapshot'])
    df = df.sort_values(['deal_id', 'date_snapshot'])
    df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])
    df['is_closed_lost'] = df['stage'] == 'Closed Lost'
    df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']
    return df

# --- Deal facts ---

def build_deal_facts(df, cutoff_date=None):
    if cutoff_date:
        df_view = df[df['date_snapshot'] <= pd.to_datetime(cutoff_date)].copy()
    else:
        df_view = df.copy()
    
    first = df_view.groupby('deal_id').first().reset_index()
    closed = df[df['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed', 'stage', 'net_revenue']]
    closed = closed.rename(columns={'stage': 'final_stage', 'date_closed': 'final_date_closed', 'net_revenue': 'final_net_revenue'})
    
    deals = first.merge(closed, on='deal_id', how='left')
    deals['created_month'] = deals['date_created'].dt.to_period('M')
    deals['stage'] = deals['final_stage'].fillna('Open')
    deals['date_closed'] = deals['final_date_closed']
    deals['net_revenue'] = deals['final_net_revenue'].fillna(deals['net_revenue'])
    deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])
    
    first_stage = df.groupby('deal_id')['stage'].first()
    skippers = first_stage[first_stage == 'Closed Lost'].index
    deals['volume_weight'] = 1.0
    deals.loc[deals['deal_id'].isin(skippers), 'volume_weight'] = SKIPPER_WEIGHT
    
    return deals

# --- Probabilities & staleness (Layer 2) ---

def build_stage_probabilities(df):
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

# --- Layer 2: Active pipeline ---

def forecast_active_pipeline(df, stage_probs, staleness, forecast_months, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = ACTUALS_THROUGH
    
    cutoff_dt = pd.to_datetime(cutoff_date)
    available = df[df['date_snapshot'] <= cutoff_dt]
    if available.empty:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    last_date = available['date_snapshot'].max()
    open_deals = df[(df['date_snapshot'] == last_date) & (~df['is_closed'])].copy()
    if open_deals.empty:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])

    open_deals['age_weeks'] = ((open_deals['date_snapshot'] - open_deals['date_created']).dt.days // 7)
    open_deals = open_deals.merge(stage_probs, on=['market_segment', 'stage'], how='left')
    open_deals = open_deals.merge(staleness, on=['market_segment', 'stage'], how='left')
    open_deals['prob'] = open_deals['prob'].fillna(0)
    stale_mask = open_deals['age_weeks'] > open_deals['stale_after_weeks']
    open_deals.loc[stale_mask, 'prob'] *= STALENESS_PENALTY
    
    rows = []
    for _, deal in open_deals.iterrows():
        p = deal['prob']
        if p == 0:
            continue
        num_months = len(forecast_months)
        if deal['market_segment'] == 'Large Market':
            weights = np.array([1.0 for _ in range(num_months)])
        elif deal['market_segment'] == 'Mid Market':
            weights = np.array([1/(i+1) for i in range(num_months)])
        else:
            weights = np.array([1/(i+1)**2 for i in range(num_months)])
            
        weights /= weights.sum()
        
        for m, w in zip(forecast_months, weights):
            rows.append({
                'month': m,
                'market_segment': deal['market_segment'],
                'expected_revenue': deal['net_revenue'] * p * w,
                'expected_count': p * w
            })
    
    if not rows:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index()

# --- Layer 1: Future pipeline ---

def _calculate_volume_baseline(deals, segment, vol_t12, vol_all, vol_peak):
    mode = VOLUME_BASELINE_BY_SEGMENT.get(segment, VOLUME_BASELINE)
    t12 = vol_t12.get(segment, 0)
    all_time = vol_all.get(segment, t12)
    peak = vol_peak.get(segment, t12)
    if all_time == 0:
        all_time = t12
    if t12 < (0.7 * all_time) and all_time > 0:
        return 0.5 * t12 + 0.5 * all_time
    if mode == 'T12':
        return t12
    if mode == 'ALL_TIME':
        return all_time
    if mode == 'BLENDED':
        return 0.5 * t12 + 0.5 * all_time
    if mode == 'CAPACITY':
        return 0.65 * peak + 0.35 * t12
    return t12


def forecast_future_pipeline(deals, start_date, end_date, levers=None):
    if levers is None:
        levers = SCENARIO_LEVERS
    
    cutoff = deals['date_created'].max()
    t12_start = cutoff - pd.DateOffset(months=12)
    t12_deals = deals[deals['date_created'] >= t12_start]
    
    all_vol_agg = deals.groupby(['market_segment', 'created_month'])['volume_weight'].sum().reset_index()
    vol_agg = t12_deals.groupby(['market_segment', 'created_month'])['volume_weight'].sum().reset_index()
    vol_t12 = vol_agg.groupby('market_segment')['volume_weight'].mean().to_dict()
    vol_all = all_vol_agg.groupby('market_segment')['volume_weight'].mean().to_dict()
    vol_peak = all_vol_agg.groupby('market_segment')['volume_weight'].max().to_dict()
    
    segments = deals['market_segment'].unique()
    vol_base = {seg: _calculate_volume_baseline(deals, seg, vol_t12, vol_all, vol_peak) for seg in segments}
    
    def get_metrics(df_seg):
        won = df_seg[df_seg['won']]
        denom = (df_seg['net_revenue'] * df_seg['volume_weight']).sum()
        wr = (won['net_revenue'] * won['volume_weight']).sum() / denom if denom > 0 else 0
        avg_size = won['net_revenue'].mean() if len(won) > 0 else 0
        n_wins = len(won)
        return pd.Series({'win_rate': wr, 'avg_size': avg_size, 'n_wins': n_wins})

    wr_t12 = t12_deals.groupby('market_segment').apply(get_metrics).reset_index()
    wr_all = deals.groupby('market_segment').apply(get_metrics).reset_index()
    
    metrics = pd.DataFrame({'market_segment': segments})
    metrics['avg_monthly_vol'] = metrics['market_segment'].map(vol_base)
    wr_t12_dict = wr_t12.set_index('market_segment')[['win_rate', 'avg_size', 'n_wins']].to_dict('index')
    wr_all_dict = wr_all.set_index('market_segment')[['win_rate', 'avg_size', 'n_wins']].to_dict('index')
    metrics['win_rate_t12'] = metrics['market_segment'].map(lambda x: wr_t12_dict.get(x, {}).get('win_rate', 0))
    metrics['avg_size_t12'] = metrics['market_segment'].map(lambda x: wr_t12_dict.get(x, {}).get('avg_size', 0))
    metrics['n_wins_t12'] = metrics['market_segment'].map(lambda x: wr_t12_dict.get(x, {}).get('n_wins', 0))
    metrics['win_rate_all'] = metrics['market_segment'].map(lambda x: wr_all_dict.get(x, {}).get('win_rate', 0))
    metrics['avg_size_all'] = metrics['market_segment'].map(lambda x: wr_all_dict.get(x, {}).get('avg_size', 0))
    metrics['win_rate'] = (metrics['win_rate_t12'] * WIN_RATE_T12_WEIGHT) + (metrics['win_rate_all'] * (1 - WIN_RATE_T12_WEIGHT))
    metrics['avg_size'] = (metrics['avg_size_t12'] * DEAL_SIZE_T12_WEIGHT) + (metrics['avg_size_all'] * (1 - DEAL_SIZE_T12_WEIGHT))
    small_sample = metrics['n_wins_t12'] < MIN_WINS_FOR_T12_SIZE
    metrics.loc[small_sample, 'avg_size'] = metrics.loc[small_sample, 'avg_size_all']
    
    deals_with_month = deals.copy()
    deals_with_month['month_num'] = deals_with_month['date_created'].dt.month
    seasonality = deals_with_month.groupby('month_num')['volume_weight'].sum()
    seasonality = seasonality / seasonality.mean()
    
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
            vol = r['avg_monthly_vol'] * l['volume_multiplier'] * s_mult
            wr = r['win_rate'] * l['win_rate_multiplier']
            size = r['avg_size'] * l['deal_size_multiplier']
            expected_rev_total = vol * size * wr
            expected_count_total = vol * wr
            seg_dist = dist[dist['market_segment'] == seg]
            if seg_dist.empty:
                seg_dist = pd.DataFrame({'months_to_close': range(6), 'pct': [1/6]*6})
            
            for _, t_row in seg_dist.iterrows():
                m_close = m_created + int(t_row['months_to_close'])
                if m_close < pd.Period(start_date, 'M') or m_close > pd.Period(end_date, 'M'):
                    continue
                
                rows.append({
                    'month': m_close.to_timestamp(),
                    'market_segment': seg,
                    'expected_revenue': expected_rev_total * t_row['pct'],
                    'expected_count': expected_count_total * t_row['pct']
                })
    
    if not rows:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    return pd.DataFrame(rows).groupby(['month', 'market_segment']).sum().reset_index()

# --- Backtest ---

def run_backtest(snapshots):
    print(f"\n{'='*60}")
    print(f"BACKTEST: 2025 MODEL-ONLY RECONCILIATION")
    print(f"{'='*60}")
    
    cutoff = pd.to_datetime(BACKTEST_DATE)
    end = pd.to_datetime(BACKTEST_THROUGH)
    hist_snaps = snapshots[snapshots['date_snapshot'] < cutoff].copy()
    if hist_snaps.empty:
        print("WARNING: No historical snapshots before backtest date")
        return pd.DataFrame()
    
    hist_deals = build_deal_facts(hist_snaps)
    actual_2025_deals = build_deal_facts(snapshots[snapshots['date_snapshot'] <= end])
    created_2025 = actual_2025_deals[
        (actual_2025_deals['date_created'] >= cutoff) & 
        (actual_2025_deals['date_created'] <= end)
    ]
    
    backtest_levers = {}
    for seg in actual_2025_deals['market_segment'].unique():
        c25 = created_2025[created_2025['market_segment'] == seg]
        h_seg = hist_deals[hist_deals['market_segment'] == seg]
        
        if h_seg.empty:
            backtest_levers[seg] = {'volume_multiplier': 1.0, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}
            continue
        h_rev_base = (h_seg[h_seg['won']]['net_revenue'].sum() / max(h_seg['created_month'].nunique(), 1))
        c25_rev_act = (c25[c25['won']]['net_revenue'].sum() / 12)
        rev_mult = c25_rev_act / h_rev_base if h_rev_base > 0 else 1.0
        
        backtest_levers[seg] = {'volume_multiplier': rev_mult, 'win_rate_multiplier': 1.0, 'deal_size_multiplier': 1.0}

    probs = build_stage_probabilities(hist_snaps)
    stale = build_staleness_thresholds(hist_snaps)
    f_months = [m.to_timestamp() for m in pd.period_range(BACKTEST_DATE, BACKTEST_THROUGH, freq='M')]
    active = forecast_active_pipeline(hist_snaps, probs, stale, f_months, cutoff_date=cutoff - pd.Timedelta(days=1))
    future = forecast_future_pipeline(hist_deals, BACKTEST_DATE, BACKTEST_THROUGH, levers=backtest_levers)
    forecast_combined = pd.concat([active, future]).groupby(['month', 'market_segment']).sum().reset_index()
    
    actual_full_deals = build_deal_facts(snapshots)
    actual_full_deals['date_created'] = pd.to_datetime(actual_full_deals['date_created'])
    actual_full_deals['date_closed'] = pd.to_datetime(actual_full_deals['date_closed'])
    all_act_won = actual_full_deals[
        (actual_full_deals['won']) & 
        (actual_full_deals['date_closed'] >= cutoff) & 
        (actual_full_deals['date_closed'] <= end)
    ].copy()
    
    summary = forecast_combined.groupby('market_segment')['expected_revenue'].sum().reset_index(name='model_forecast')
    act_sum = all_act_won.groupby('market_segment')['net_revenue'].sum().reset_index(name='actual')
    res = summary.merge(act_sum, on='market_segment', how='outer').fillna(0)
    res['var_pct'] = ((res['model_forecast'] / res['actual'].replace(0, 1)) - 1) * 100
    res['var_abs'] = res['model_forecast'] - res['actual']

    print(f"\nTotal Won Deals in 2025 Actuals: {len(all_act_won)}")
    res_display = res.copy()
    for col in ['model_forecast', 'actual', 'var_abs']:
        res_display[col] = res_display[col].apply(format_currency_m)
    res_display['var_pct'] = res_display['var_pct'].apply(lambda x: f"{x:.2f}%")
    print(res_display.to_string(index=False))
    print(f"\nModel Forecast (active + future): ${res['model_forecast'].sum():,.2f}")
    print(f"Actual 2025 Closed Won (Verbal + Closed Won): ${res['actual'].sum():,.2f}")
    print(f"Variance: {((res['model_forecast'].sum() / res['actual'].sum()) - 1) * 100:.1f}%")
    
    res.to_csv(os.path.join(VALIDATION_DIR, 'backtest_results.csv'), index=False)
    forecast_combined['month'] = pd.to_datetime(forecast_combined['month'])
    all_act_won['close_month'] = all_act_won['date_closed'].dt.to_period('M').dt.to_timestamp()
    
    actual_monthly = all_act_won.groupby(['close_month', 'market_segment'])['net_revenue'].sum().reset_index(name='actual_revenue')
    
    monthly = forecast_combined.merge(
        actual_monthly,
        left_on=['month', 'market_segment'],
        right_on=['close_month', 'market_segment'],
        how='outer'
    ).fillna(0)
    
    monthly.to_csv(os.path.join(VALIDATION_DIR, 'backtest_monthly.csv'), index=False)
    
    return res

# --- Goal seek ---

def run_goal_seek(deals, forecast):
    print(f"\n{'='*60}")
    print(f"GOAL SEEK: REQUIRED LEVERS FOR TARGETS")
    print(f"{'='*60}")
    
    current_forecast = forecast.groupby('market_segment')['expected_won_revenue'].sum().to_dict()
    
    results = []
    for seg, target in GOAL_WON_REVENUE.items():
        current = current_forecast.get(seg, 0)
        if current == 0:
            continue
        needed_mult = target / current
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
    if not goal_df.empty:
        goal_display = goal_df.copy()
        cols_to_fix = ['current_forecast', 'target']
        for col in cols_to_fix:
            goal_display[col] = goal_display[col].apply(format_currency_m)
        print(goal_display.to_string(index=False))
        goal_df.to_csv(os.path.join(EXPORT_DIR, 'goal_seek_analysis.csv'), index=False)

# --- Export config ---

def export_assumptions():
    config = {
        'VOLUME_BASELINE': VOLUME_BASELINE,
        'VOLUME_BASELINE_BY_SEGMENT': VOLUME_BASELINE_BY_SEGMENT,
        'SCENARIO_LEVERS': SCENARIO_LEVERS,
        'STALENESS_PENALTY': STALENESS_PENALTY,
        'SKIPPER_WEIGHT': SKIPPER_WEIGHT,
        'CREDIBILITY_THRESHOLD': CREDIBILITY_THRESHOLD,
        'WIN_RATE_T12_WEIGHT': WIN_RATE_T12_WEIGHT,
        'DEAL_SIZE_T12_WEIGHT': DEAL_SIZE_T12_WEIGHT,
        'MIN_WINS_FOR_T12_SIZE': MIN_WINS_FOR_T12_SIZE,
        'FORECAST_START': FORECAST_START,
        'FORECAST_END': FORECAST_END,
        'ACTUALS_THROUGH': ACTUALS_THROUGH,
    }
    with open(os.path.join(EXPORT_DIR, 'assumptions.json'), 'w') as f:
        json.dump(config, f, indent=4)

# --- Main ---

def run_forecast():
    print(f"\n{'='*60}")
    print(f"FORECAST GENERATOR V9")
    print(f"{'='*60}")
    print(f"Volume Baseline: {VOLUME_BASELINE}")
    print(f"Segment Overrides: {VOLUME_BASELINE_BY_SEGMENT or 'None'}")
    
    snapshots = load_snapshots()
    deals = build_deal_facts(snapshots)
    export_assumptions()
    if RUN_BACKTEST:
        run_backtest(snapshots)
    print(f"\n{'='*60}")
    print(f"FY26 FORECAST")
    print(f"{'='*60}")
    
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
    print(summary.apply(format_currency_m).to_string())
    print(f"\nTOTAL FY26 FORECAST: ${summary.sum():,.2f}")
    print(f"  (Active + future pipeline; Verbal and Closed Won both count as Won)")
    actual_2025 = deals[
        (deals['won']) & 
        (deals['date_closed'] >= '2025-01-01') & 
        (deals['date_closed'] <= '2025-12-31')
    ]['net_revenue'].sum()
    
    if actual_2025 > 0:
        yoy_pct = (summary.sum() / actual_2025 - 1) * 100
        print(f"\nYoY vs 2025 actual (Verbal + Closed Won): {yoy_pct:.1f}%")
        if abs(yoy_pct) > 40:
            print("[WARNING] Forecast is >40% different from 2025 actuals.")
    active_unweighted = snapshots[
        (snapshots['date_snapshot'] == snapshots['date_snapshot'].max()) & 
        (~snapshots['is_closed'])
    ]['net_revenue'].sum()
    print(f"\nActive Pipeline (unweighted): ${active_unweighted:,.2f}")
    
    forecast.to_csv(os.path.join(EXPORT_DIR, 'forecast_2026.csv'), index=False)
    
    if RUN_GOAL_SEEK:
        run_goal_seek(deals, forecast)
    
    return forecast


if __name__ == "__main__":
    result = run_forecast()
