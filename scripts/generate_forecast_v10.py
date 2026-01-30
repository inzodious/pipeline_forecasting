import pandas as pd
import numpy as np
import json
import os
import logging
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- Logging setup ---

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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

VOLUME_BASELINE = 'ALL_TIME'  # Options: 'T12', 'ALL_TIME', 'BLENDED', 'CAPACITY'
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

STALENESS_K_BY_SEGMENT = {
    'Large Market': 0.1,   
    'Mid Market':   0.25,  
    'Indirect':     0.25,  
    'SMB':          0.5,   
    'Other':        0.3    
}
DEFAULT_STALENESS_K = 0.25  

SKIPPER_WEIGHT = 0.5
CREDIBILITY_THRESHOLD = 50
MIN_WINS_FOR_T12_SIZE = 15
WIN_RATE_T12_WEIGHT = 0.55
DEAL_SIZE_T12_WEIGHT = 0.55

DEFAULT_DAYS_REMAINING = 90

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

REQUIRED_COLUMNS = ['deal_id', 'market_segment', 'stage', 'date_snapshot', 'date_created', 'net_revenue']

def validate_snapshots(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    critical_nulls = ['deal_id', 'market_segment', 'stage', 'date_snapshot']
    for col in critical_nulls:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(f"Column '{col}' has {null_count} null values")
    
    if 'date_closed' in df.columns:
        closed_rows = df[df['date_closed'].notna()].copy()
        if not closed_rows.empty:
            zombie_mask = closed_rows['date_closed'] < closed_rows['date_created']
            zombie_count = zombie_mask.sum()
            if zombie_count > 0:
                zombie_deals = closed_rows.loc[zombie_mask, 'deal_id'].unique()[:5]
                logger.warning(f"Found {zombie_count} zombie rows (date_closed < date_created). "
                             f"Sample deal_ids: {list(zombie_deals)}")
    
    if 'net_revenue' in df.columns:
        neg_rev_count = (df['net_revenue'] < 0).sum()
        if neg_rev_count > 0:
            logger.warning(f"Found {neg_rev_count} rows with negative net_revenue")
    
    if 'is_closed_won' in df.columns:
        won_wrong_stage = df[df['is_closed_won'] & ~df['stage'].isin(['Closed Won', 'Verbal'])]
        if len(won_wrong_stage) > 0:
            logger.warning(f"Found {len(won_wrong_stage)} rows where is_closed_won=True but stage not in (Closed Won, Verbal)")
    
    if 'date_snapshot' in df.columns:
        snap_counts = df.groupby('date_snapshot')['deal_id'].nunique()
        if len(snap_counts) > 1:
            pct_change = snap_counts.pct_change().dropna()
            large_drops = pct_change[pct_change < -0.3]
            if len(large_drops) > 0:
                logger.warning(f"Found {len(large_drops)} snapshots with >30% drop in deal count (possible ETL issue)")
    
    return True

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
    
    validate_snapshots(df)
    
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
    """Stage probability = P(Closed Won | exited from this stage)."""
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
    
    if not probs.empty and probs['prob'].max() > 1.0:
        logger.warning(f"Stage probabilities exceed 1.0 (max: {probs['prob'].max():.3f})")
    
    return probs[['market_segment', 'stage', 'prob']]


def build_staleness_thresholds(df):
    """Calculate 95th percentile age (in weeks) per (market_segment, stage)."""
    df = df.copy()
    df['age_weeks'] = ((df['date_snapshot'] - df['date_created']).dt.days // 7).clip(lower=0)
    return df.groupby(['market_segment', 'stage'])['age_weeks'].quantile(0.95).reset_index(name='stale_after_weeks')


def build_time_to_close_table(df):
    df = df.sort_values(['deal_id', 'date_snapshot'])
    
    closed_deals = df[df['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed']]
    
    df_with_close = df.merge(closed_deals, on='deal_id', how='inner', suffixes=('', '_final'))
    
    open_stages = ~df_with_close['stage'].isin(['Closed Won', 'Verbal', 'Closed Lost'])
    pre_close = df_with_close[
        (df_with_close['date_snapshot'] < df_with_close['date_closed']) & 
        open_stages
    ].copy()
    
    if pre_close.empty:
        logger.warning("No pre-close snapshots found for time-to-close calculation")
        return pd.DataFrame(columns=['market_segment', 'stage', 'avg_days_remaining'])
    
    pre_close['days_remaining'] = (pre_close['date_closed'] - pre_close['date_snapshot']).dt.days
    
    cutoff_date = df['date_snapshot'].max()
    t12_start = cutoff_date - pd.DateOffset(months=12)
    pre_close['weight'] = np.where(pre_close['date_closed'] >= t12_start, 3.0, 1.0)
    
    ttc_list = []
    for (segment, stage), group in pre_close.groupby(['market_segment', 'stage']):
        weighted_sum = (group['days_remaining'] * group['weight']).sum()
        weight_total = group['weight'].sum()
        avg_days = weighted_sum / weight_total if weight_total > 0 else DEFAULT_DAYS_REMAINING
        ttc_list.append({
            'market_segment': segment, 
            'stage': stage, 
            'avg_days_remaining': avg_days,
            'sample_size': len(group)
        })
    
    ttc = pd.DataFrame(ttc_list)
    return ttc[['market_segment', 'stage', 'avg_days_remaining']]


def compute_staleness_factor(age_weeks, threshold, segment):
    k = STALENESS_K_BY_SEGMENT.get(segment, DEFAULT_STALENESS_K)
    
    if isinstance(age_weeks, (pd.Series, np.ndarray)):
        diff = age_weeks - threshold
        return 1.0 / (1.0 + np.exp(k * diff))
    else:
        diff = age_weeks - threshold
        return 1.0 / (1.0 + np.exp(k * diff))

def forecast_active_pipeline(df, stage_probs, staleness, time_to_close, forecast_start, forecast_end, cutoff_date=None):
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

    open_deals = open_deals.merge(stage_probs, on=['market_segment', 'stage'], how='left')
    open_deals['prob'] = open_deals['prob'].fillna(0)
    
    open_deals = open_deals.merge(staleness, on=['market_segment', 'stage'], how='left')
    open_deals['age_weeks'] = ((open_deals['date_snapshot'] - open_deals['date_created']).dt.days // 7)
    
    open_deals['stale_after_weeks'] = open_deals['stale_after_weeks'].fillna(52)  # Default 1 year
    open_deals['staleness_factor'] = open_deals.apply(
        lambda row: compute_staleness_factor(
            row['age_weeks'], 
            row['stale_after_weeks'], 
            row['market_segment']
        ), 
        axis=1
    )
    open_deals['prob'] = open_deals['prob'] * open_deals['staleness_factor']
    
    open_deals = open_deals.merge(time_to_close, on=['market_segment', 'stage'], how='left')
    open_deals['avg_days_remaining'] = open_deals['avg_days_remaining'].fillna(DEFAULT_DAYS_REMAINING)
    open_deals['expected_close_date'] = open_deals['date_snapshot'] + pd.to_timedelta(open_deals['avg_days_remaining'], unit='D')
    open_deals['expected_close_month'] = open_deals['expected_close_date'].dt.to_period('M').dt.to_timestamp()
    
    forecast_start_dt = pd.to_datetime(forecast_start)
    forecast_end_dt = pd.to_datetime(forecast_end)
    
    open_deals['expected_close_month'] = open_deals['expected_close_month'].clip(
        lower=forecast_start_dt,
        upper=pd.to_datetime(forecast_end).replace(day=1)
    )
    
    open_deals = open_deals[open_deals['prob'] > 0].copy()
    
    if open_deals.empty:
        return pd.DataFrame(columns=['month', 'market_segment', 'expected_revenue', 'expected_count'])
    
    open_deals['expected_revenue'] = open_deals['net_revenue'] * open_deals['prob']
    open_deals['expected_count'] = open_deals['prob']
    
    result = open_deals.groupby(['expected_close_month', 'market_segment']).agg({
        'expected_revenue': 'sum',
        'expected_count': 'sum'
    }).reset_index()
    result = result.rename(columns={'expected_close_month': 'month'})
    
    input_weighted_rev = (open_deals['net_revenue'] * open_deals['prob']).sum()
    output_rev = result['expected_revenue'].sum()
    if abs(input_weighted_rev - output_rev) > 1e-6:
        logger.warning(f"Revenue conservation check: input={input_weighted_rev:,.0f}, output={output_rev:,.0f}")
    
    return result[['month', 'market_segment', 'expected_revenue', 'expected_count']]


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
    cutoff = pd.to_datetime(BACKTEST_DATE)
    end = pd.to_datetime(BACKTEST_THROUGH)
    hist_snaps = snapshots[snapshots['date_snapshot'] < cutoff].copy()
    if hist_snaps.empty:
        logger.warning("No historical snapshots before backtest date")
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
    ttc = build_time_to_close_table(hist_snaps)
    
    active = forecast_active_pipeline(
        hist_snaps, probs, stale, ttc, 
        BACKTEST_DATE, BACKTEST_THROUGH,
        cutoff_date=cutoff - pd.Timedelta(days=1)
    )
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

    backtest_forecast = res['model_forecast'].sum()
    backtest_actual = res['actual'].sum()
    backtest_var = ((backtest_forecast / backtest_actual) - 1) * 100 if backtest_actual > 0 else 0
    logger.info(f"Backtest 2025: Forecast {format_currency_m(backtest_forecast)} vs Actual {format_currency_m(backtest_actual)} ({backtest_var:+.1f}%)")
    
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
    """Run goal seek analysis and export to CSV (no logging)."""
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
        goal_df.to_csv(os.path.join(EXPORT_DIR, 'goal_seek_analysis.csv'), index=False)


# --- Export config ---

def export_assumptions():
    config = {
        'VOLUME_BASELINE': VOLUME_BASELINE,
        'VOLUME_BASELINE_BY_SEGMENT': VOLUME_BASELINE_BY_SEGMENT,
        'SCENARIO_LEVERS': SCENARIO_LEVERS,
        'STALENESS_K_BY_SEGMENT': STALENESS_K_BY_SEGMENT,
        'DEFAULT_STALENESS_K': DEFAULT_STALENESS_K,
        'DEFAULT_DAYS_REMAINING': DEFAULT_DAYS_REMAINING,
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
    setup_logging()
    
    snapshots = load_snapshots()
    deals = build_deal_facts(snapshots)
    export_assumptions()
    
    if RUN_BACKTEST:
        run_backtest(snapshots)
    
    stage_probs = build_stage_probabilities(snapshots)
    staleness = build_staleness_thresholds(snapshots)
    time_to_close = build_time_to_close_table(snapshots)
    
    active = forecast_active_pipeline(
        snapshots, stage_probs, staleness, time_to_close,
        FORECAST_START, FORECAST_END
    )
    future = forecast_future_pipeline(deals, FORECAST_START, FORECAST_END)
    
    forecast = pd.concat([active, future]).groupby(['month', 'market_segment']).sum().reset_index()
    forecast = forecast.rename(columns={'expected_revenue': 'expected_won_revenue', 'expected_count': 'expected_won_count'})
    
    summary = forecast.groupby('market_segment')['expected_won_revenue'].sum()
    forecast_total = summary.sum()
    
    actual_2025 = deals[
        (deals['won']) & 
        (deals['date_closed'] >= '2025-01-01') & 
        (deals['date_closed'] <= '2025-12-31')
    ]['net_revenue'].sum()
    
    yoy_pct = ((forecast_total / actual_2025) - 1) * 100 if actual_2025 > 0 else 0
    logger.info(f"FY26 Forecast: {format_currency_m(forecast_total)} vs FY25 Actual {format_currency_m(actual_2025)} ({yoy_pct:+.1f}% YoY)")
    
    if abs(yoy_pct) > 40:
        logger.warning("Forecast is >40% different from 2025 actuals.")
    
    forecast.to_csv(os.path.join(EXPORT_DIR, 'forecast_2026.csv'), index=False)
    
    if RUN_GOAL_SEEK:
        run_goal_seek(deals, forecast)
    
    return forecast


if __name__ == "__main__":
    result = run_forecast()
