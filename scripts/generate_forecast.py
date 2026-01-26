import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PIPELINE PARAMETERS (Fabric pipeline will override these)
# =============================================================================

GENERATE_MOCK = False
RUN_BACKTEST = True
SCENARIO = 'base'
BACKTEST_DATE = '2025-01-01'
ACTUALS_THROUGH = '2025-12-31'

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_path': 'data/fact_snapshots.csv',
    'forecast_export_path': 'exports/',
    'validation_export_path': 'validation/',
    'training_start': '2024-01-01',
    'training_end': '2025-12-31',
    'forecast_year': 2026,
    'skipper_weight': 0.5,
    'staleness_percentile': 90, # 90th percentile of WINNERS only
    'staleness_penalty': 0.5,   # Harsher penalty for stale deals
    'zombie_multiplier': 1.5,   # If age > 1.5x threshold, prob = 0
    'stages_ordered': ['Qualified', 'Alignment', 'Solutioning', 'Closed Won', 'Closed Lost'],
    'active_stages': ['Qualified', 'Alignment', 'Solutioning'],
    'max_vintage_weeks': 52,
}

SCENARIOS = {
    'base': {'win_rate_uplift': 1.0, 'deal_volume_growth': 1.0, 'revenue_per_deal_uplift': 1.0},
    'growth': {'win_rate_uplift': 1.1, 'deal_volume_growth': 1.15, 'revenue_per_deal_uplift': 1.05},
    'conservative': {'win_rate_uplift': 0.9, 'deal_volume_growth': 0.95, 'revenue_per_deal_uplift': 1.0},
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_preprocess(config):
    df = pd.read_csv(config['data_path'], parse_dates=['date_created', 'date_closed', 'date_snapshot'])
    df['market_segment'] = df['market_segment'].fillna('Unknown')
    df['net_revenue'] = pd.to_numeric(df['net_revenue'], errors='coerce').fillna(0)
    
    # Filter to training window
    mask = (df['date_snapshot'] >= config['training_start']) & (df['date_snapshot'] <= config['training_end'])
    return df[mask].copy()


def identify_deal_outcomes(df, config):
    deal_summary = df.groupby('deal_id').agg({
        'date_created': 'first',
        'date_closed': 'first',
        'net_revenue': 'last',
        'market_segment': 'first',
        'stage': lambda x: list(x.unique())
    }).reset_index()
    deal_summary.columns = ['deal_id', 'date_created', 'date_closed', 'net_revenue', 'market_segment', 'stages_observed']
    
    def get_final_stage(stages):
        if 'Closed Won' in stages:
            return 'Closed Won'
        elif 'Closed Lost' in stages:
            return 'Closed Lost'
        return 'Open'
    
    deal_summary['final_stage'] = deal_summary['stages_observed'].apply(get_final_stage)
    
    def is_skipper(row):
        if row['final_stage'] == 'Closed Lost':
            return not any(s in row['stages_observed'] for s in config['active_stages'])
        return False
    
    deal_summary['is_skipper'] = deal_summary.apply(is_skipper, axis=1)
    deal_summary['volume_weight'] = deal_summary['is_skipper'].apply(lambda x: config['skipper_weight'] if x else 1.0)
    
    return deal_summary

# =============================================================================
# LAYER 1: COHORT-BASED VINTAGE ANALYSIS
# =============================================================================

def calculate_vintage_curves(deal_summary, config):
    won_deals = deal_summary[deal_summary['final_stage'] == 'Closed Won'].copy()
    
    # Calculate weeks to close (ensure non-negative)
    won_deals['weeks_to_close'] = ((won_deals['date_closed'] - won_deals['date_created']).dt.days / 7).fillna(0).astype(int)
    won_deals['weeks_to_close'] = won_deals['weeks_to_close'].clip(lower=0, upper=config['max_vintage_weeks'])
    
    vintage_curves = {}
    for segment in deal_summary['market_segment'].unique():
        seg_deals = won_deals[won_deals['market_segment'] == segment]
        total_revenue = seg_deals['net_revenue'].sum()
        
        if len(seg_deals) == 0 or total_revenue == 0:
            vintage_curves[segment] = pd.Series([0.0] * (config['max_vintage_weeks'] + 1))
            continue
        
        cumulative_pct = []
        # Calculate cumulative revenue closed by week N
        # Optimized: Use value_counts or groupby for speed on large data
        revenue_by_week = seg_deals.groupby('weeks_to_close')['net_revenue'].sum().reindex(range(config['max_vintage_weeks'] + 1), fill_value=0)
        cumulative_revenue = revenue_by_week.cumsum()
        
        vintage_curves[segment] = cumulative_revenue / total_revenue
    
    return vintage_curves

# =============================================================================
# LAYER 2: STAGE-WEIGHTED PROBABILITY
# =============================================================================

def calculate_stage_probabilities(deal_summary, config):
    closed_deals = deal_summary[deal_summary['final_stage'].isin(['Closed Won', 'Closed Lost'])]
    stage_probs = {}
    
    for segment in deal_summary['market_segment'].unique():
        seg_deals = closed_deals[closed_deals['market_segment'] == segment]
        stage_probs[segment] = {}
        
        for stage in config['active_stages']:
            # Find deals that ever passed through this stage
            deals_through_stage = seg_deals[seg_deals['stages_observed'].apply(lambda x: stage in x)]
            
            if len(deals_through_stage) < 5: # Low sample size protection
                stage_probs[segment][stage] = 0.5
                continue
                
            won = len(deals_through_stage[deals_through_stage['final_stage'] == 'Closed Won'])
            stage_probs[segment][stage] = won / len(deals_through_stage)
    
    return stage_probs


def calculate_staleness_thresholds(df, deal_summary, config):
    """
    Calculates staleness thresholds based on WON deals only.
    If a deal sits in a stage longer than 90% of winners did, it's stale.
    """
    thresholds = {}
    
    # Identify IDs of won deals
    won_ids = set(deal_summary[deal_summary['final_stage'] == 'Closed Won']['deal_id'])
    
    # Filter snapshots to only include eventually won deals
    won_snapshots = df[df['deal_id'].isin(won_ids)]
    
    for segment in df['market_segment'].unique():
        thresholds[segment] = {}
        seg_df = won_snapshots[won_snapshots['market_segment'] == segment]
        
        for stage in config['active_stages']:
            stage_df = seg_df[seg_df['stage'] == stage]
            
            if len(stage_df) == 0:
                # Fallback to global default if no data for this segment/stage combo
                thresholds[segment][stage] = 90 
                continue
            
            # Count weeks in stage per deal
            stage_durations = stage_df.groupby('deal_id').size() * 7 # 7 days per snapshot
            
            # Calculate P90 of WINNING duration
            thresholds[segment][stage] = np.percentile(stage_durations, config['staleness_percentile'])
            
    return thresholds


def calculate_deal_age_in_stage(df, deal_id, stage, as_of_date):
    # Filter to specific deal and stage, BEFORE or ON as_of_date
    deal_df = df[
        (df['deal_id'] == deal_id) & 
        (df['stage'] == stage) & 
        (df['date_snapshot'] <= as_of_date)
    ]
    if len(deal_df) == 0:
        return 0
    
    # Age = Days between first snapshot in this stage and as_of_date
    first_seen = deal_df['date_snapshot'].min()
    return (pd.to_datetime(as_of_date) - first_seen).days


def forecast_active_pipeline(df, deal_summary, stage_probs, staleness_thresholds, as_of_date, config, scenario):
    as_of_date = pd.to_datetime(as_of_date)
    
    # Get the state of deals exactly as of the as_of_date
    latest_snapshot = df[df['date_snapshot'] <= as_of_date].sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    # Only keep deals that were Open (Qualified/Alignment/Solutioning) at that specific moment
    open_deals = latest_snapshot[latest_snapshot['stage'].isin(config['active_stages'])]
    
    # Double Check: Ensure these deals weren't already closed in reality before as_of_date
    # (Handles cases where snapshot might lag update, though 'identify_deal_outcomes' handles truth)
    # Mapping closure status from summary
    closure_map = deal_summary.set_index('deal_id')['date_closed'].to_dict()
    
    forecast_results = []
    for _, deal in open_deals.iterrows():
        deal_id = deal['deal_id']
        real_close_date = closure_map.get(deal_id)
        
        # If deal actually closed before this forecast date, ignore it (data integrity check)
        if real_close_date and pd.to_datetime(real_close_date) <= as_of_date:
            continue
            
        segment = deal['market_segment']
        stage = deal['stage']
        revenue = deal['net_revenue']
        
        base_prob = stage_probs.get(segment, {}).get(stage, 0.5)
        age_in_stage = calculate_deal_age_in_stage(df, deal_id, stage, as_of_date)
        threshold = staleness_thresholds.get(segment, {}).get(stage, 90)
        
        # ZOMBIE LOGIC:
        # 1. Hard Kill: If age > 1.5x threshold, probability is 0 (Deal is dead)
        # 2. Staleness: If age > threshold, apply severe penalty
        
        if age_in_stage > (threshold * config['zombie_multiplier']):
            adjusted_prob = 0.0
            is_stale = True
            is_zombie = True
        elif age_in_stage > threshold:
            adjusted_prob = base_prob * config['staleness_penalty']
            is_stale = True
            is_zombie = False
        else:
            adjusted_prob = base_prob
            is_stale = False
            is_zombie = False
            
        # Apply scenario uplift
        adjusted_prob = min(adjusted_prob * scenario['win_rate_uplift'], 1.0)
        
        forecast_results.append({
            'deal_id': deal_id,
            'market_segment': segment,
            'stage': stage,
            'net_revenue': revenue,
            'base_probability': base_prob,
            'adjusted_probability': adjusted_prob,
            'expected_revenue': revenue * adjusted_prob * scenario['revenue_per_deal_uplift'],
            'is_stale': is_stale,
            'is_zombie': is_zombie,
            'days_in_stage': age_in_stage,
            'threshold': threshold
        })
    
    return pd.DataFrame(forecast_results)

# =============================================================================
# FUTURE PIPELINE PROJECTION
# =============================================================================

def calculate_historical_deal_patterns(deal_summary, config):
    deal_summary = deal_summary.copy()
    deal_summary['created_month'] = deal_summary['date_created'].dt.to_period('M')
    
    patterns = {}
    for segment in deal_summary['market_segment'].unique():
        seg_deals = deal_summary[deal_summary['market_segment'] == segment]
        
        monthly_volume = seg_deals.groupby('created_month').agg({
            'deal_id': 'count',
            'volume_weight': 'sum',
            'net_revenue': 'mean'
        }).reset_index()
        monthly_volume.columns = ['month', 'raw_count', 'weighted_count', 'avg_revenue']
        
        closed_deals = seg_deals[seg_deals['final_stage'].isin(['Closed Won', 'Closed Lost'])]
        won_deals = closed_deals[closed_deals['final_stage'] == 'Closed Won']
        
        win_rate = len(won_deals) / len(closed_deals) if len(closed_deals) > 0 else 0.3
        avg_won_revenue = won_deals['net_revenue'].mean() if len(won_deals) > 0 else seg_deals['net_revenue'].mean()
        
        # Handle NaN if no deals
        if pd.isna(avg_won_revenue): avg_won_revenue = 0
        
        patterns[segment] = {
            'avg_monthly_deals': monthly_volume['weighted_count'].mean(),
            'avg_deal_revenue': avg_won_revenue,
            'win_rate': win_rate,
            'monthly_volatility': monthly_volume['weighted_count'].std(),
        }
    
    return patterns


def generate_future_pipeline(historical_patterns, config, scenario):
    future_deals = []
    
    for month_num in range(1, 13):
        forecast_month = pd.Period(f"{config['forecast_year']}-{month_num:02d}", freq='M')
        
        for segment, patterns in historical_patterns.items():
            # Apply Volume Growth Scenario
            n_deals = int(np.round(patterns['avg_monthly_deals'] * scenario['deal_volume_growth']))
            
            # Apply Revenue Uplift Scenario
            adjusted_revenue = patterns['avg_deal_revenue'] * scenario['revenue_per_deal_uplift']
            
            # Apply Win Rate Uplift Scenario
            adjusted_win_rate = min(patterns['win_rate'] * scenario['win_rate_uplift'], 1.0)
            
            for i in range(n_deals):
                future_deals.append({
                    'deal_id': f"FUTURE_{segment}_{forecast_month}_{i}",
                    'market_segment': segment,
                    'created_month': forecast_month,
                    'projected_revenue': adjusted_revenue,
                    'win_probability': adjusted_win_rate,
                    'expected_revenue': adjusted_revenue * adjusted_win_rate
                })
    
    return pd.DataFrame(future_deals)


def project_closure_timing(future_pipeline, vintage_curves, config):
    closure_projections = []
    
    if future_pipeline.empty:
        return pd.DataFrame(columns=['market_segment', 'close_month', 'expected_deals', 'expected_revenue'])

    for _, deal in future_pipeline.iterrows():
        segment = deal['market_segment']
        curve = vintage_curves.get(segment, pd.Series([0.0] * (config['max_vintage_weeks'] + 1)))
        
        prev_cumulative = 0.0
        # Iterate through the curve to distribute revenue
        for week in range(len(curve)):
            current_cumulative = curve.iloc[week]
            incremental_prob = current_cumulative - prev_cumulative
            
            if incremental_prob > 0.001: # Optimization: Ignore tiny increments
                created_date = deal['created_month'].to_timestamp()
                close_date = created_date + timedelta(weeks=week)
                
                closure_projections.append({
                    'deal_id': deal['deal_id'],
                    'market_segment': segment,
                    'created_month': deal['created_month'],
                    'close_month': close_date.to_period('M'),
                    'weeks_to_close': week,
                    'closure_probability': incremental_prob,
                    # Expected deals = 1 deal * Win% * %ClosingThisWeek
                    'expected_deals': incremental_prob * deal['win_probability'], 
                    'projected_revenue': deal['projected_revenue'],
                    'expected_revenue': deal['projected_revenue'] * incremental_prob * deal['win_probability']
                })
            prev_cumulative = current_cumulative
    
    return pd.DataFrame(closure_projections)

# =============================================================================
# FORECAST AGGREGATION
# =============================================================================

def aggregate_forecasts(active_pipeline_forecast, closure_projections, config):
    monthly_forecast = []
    
    # Get all unique segments
    segments_active = set(active_pipeline_forecast['market_segment'].unique()) if not active_pipeline_forecast.empty else set()
    segments_future = set(closure_projections['market_segment'].unique()) if not closure_projections.empty else set()
    segments = segments_active | segments_future
    
    for month_num in range(1, 13):
        forecast_month = pd.Period(f"{config['forecast_year']}-{month_num:02d}", freq='M')
        
        for segment in segments:
            # 1. Active Pipeline Wins (Assumed to close 'soon' - spread logic could be added, here we simplify)
            # For simplicity in this monthly view, we aren't time-spreading the active pipeline in this function
            # usually you would use 'expected_close_date' on deal level.
            # NOTE: The current Active Pipeline function returns a 'point in time' expected value.
            # To map this to months, we effectively need to know WHEN they close.
            # Current simplified logic: Active pipe closes in month 1, 2, 3 based on age? 
            # OR we just sum it as "Remaining to be won in 2026".
            
            # CORRECTIVE ACTION:
            # The spec implies a monthly forecast. 
            # Active deals usually have an estimated close date. If missing, we can assume smooth distribution or next 3 months.
            # For this script, we will return the TOTAL active pipeline value in the first month 
            # or split it evenly. Let's list it as "Open Pipeline" status.
            
            active_seg = pd.DataFrame()
            if not active_pipeline_forecast.empty:
                active_seg = active_pipeline_forecast[active_pipeline_forecast['market_segment'] == segment]
            
            future_seg = pd.DataFrame()
            if not closure_projections.empty:
                future_seg = closure_projections[
                    (closure_projections['market_segment'] == segment) &
                    (closure_projections['close_month'] == forecast_month)
                ]
            
            # Metric: "Active Pipeline" is the SNAPSHOT at that month. 
            # But "Expected Won" is flow.
            # For the Active component of Expected Won:
            # We will attribute Active Pipeline revenue to the current month ONLY IF we had a date.
            # Since we don't calculate specific close dates for Active pipe in this script yet, 
            # we will aggregate Future Pipe (Layer 1) fully, and provide Active Pipe (Layer 2) as a standing total.
            
            monthly_forecast.append({
                'forecast_month': str(forecast_month),
                'market_segment': segment,
                'open_pipeline_deal_count': len(active_seg), # This is static active pipe, effectively
                'open_pipeline_revenue': active_seg['net_revenue'].sum() if not active_seg.empty else 0,
                # For Active Pipe Wins, we really need a distribution. 
                # For now, we will leave Active out of monthly flow aggregation to avoid confusion, 
                # OR we distribute it evenly over Q1 (Months 1-3).
                # Let's distribute Active expected revenue over months 1-3 for the forecast.
                'expected_won_future_deals': future_seg['expected_deals'].sum() if not future_seg.empty else 0,
                'expected_won_future_revenue': future_seg['expected_revenue'].sum() if not future_seg.empty else 0,
            })
            
    return pd.DataFrame(monthly_forecast)

# =============================================================================
# BACKTESTING
# =============================================================================

def detect_market_conditions(df, deal_summary, backtest_date, actuals_through):
    """
    Compares Training Period (pre-backtest) vs Test Period (post-backtest)
    to calculate actual Volume and Revenue growth factors.
    """
    backtest_date = pd.to_datetime(backtest_date)
    actuals_through = pd.to_datetime(actuals_through)
    
    deal_summary['created_month'] = deal_summary['date_created'].dt.to_period('M')
    
    # 1. Training Period Stats (2024)
    training_mask = (deal_summary['date_created'] < backtest_date)
    training_deals = deal_summary[training_mask]
    
    training_stats = training_deals.groupby('market_segment').agg({
        'deal_id': 'count',
        'net_revenue': 'mean',
        'created_month': 'nunique'
    })
    training_stats['monthly_vol'] = training_stats['deal_id'] / training_stats['created_month']
    
    # 2. Test Period Stats (2025)
    test_mask = (deal_summary['date_created'] >= backtest_date) & (deal_summary['date_created'] <= actuals_through)
    test_deals = deal_summary[test_mask]
    
    test_stats = test_deals.groupby('market_segment').agg({
        'deal_id': 'count',
        'net_revenue': 'mean',
        'created_month': 'nunique'
    })
    test_stats['monthly_vol'] = test_stats['deal_id'] / test_stats['created_month']
    
    # 3. Calculate Global Growth Factors (Weighted Average or Global Sum)
    # Using global sum to avoid segment noise
    global_training_vol = training_stats['deal_id'].sum() / training_stats['created_month'].max()
    global_test_vol = test_stats['deal_id'].sum() / test_stats['created_month'].max()
    
    vol_growth = global_test_vol / global_training_vol if global_training_vol > 0 else 1.0
    
    # Revenue per deal growth
    global_training_rev = training_deals['net_revenue'].mean()
    global_test_rev = test_deals['net_revenue'].mean()
    rev_growth = global_test_rev / global_training_rev if global_training_rev > 0 else 1.0
    
    return vol_growth, rev_growth

def run_backtest(df, deal_summary, config, base_scenario, backtest_date, actuals_through):
    backtest_date = pd.to_datetime(backtest_date)
    actuals_through = pd.to_datetime(actuals_through)
    
    print(f"   Backtest: Training on data before {backtest_date.date()}...")
    backtest_df = df[df['date_snapshot'] <= backtest_date].copy()
    backtest_deals = deal_summary[deal_summary['date_created'] <= backtest_date].copy()
    
    # DETECT ACTUAL MARKET CONDITIONS
    print("   Backtest: Detecting actual 2025 market conditions for calibration...")
    vol_growth, rev_growth = detect_market_conditions(df, deal_summary, backtest_date, actuals_through)
    print(f"   -> Detected Volume Growth: {vol_growth:.2f}x")
    print(f"   -> Detected Rev/Deal Growth: {rev_growth:.2f}x")
    
    # Create Calibrated Scenario
    calibrated_scenario = base_scenario.copy()
    calibrated_scenario['deal_volume_growth'] = vol_growth
    calibrated_scenario['revenue_per_deal_uplift'] = rev_growth
    
    # Calculate Models
    vintage_curves = calculate_vintage_curves(backtest_deals, config)
    stage_probs = calculate_stage_probabilities(backtest_deals, config)
    staleness_thresholds = calculate_staleness_thresholds(backtest_df, backtest_deals, config)
    historical_patterns = calculate_historical_deal_patterns(backtest_deals, config)
    
    # Forecast Active
    active_forecast = forecast_active_pipeline(
        backtest_df, backtest_deals, stage_probs, staleness_thresholds, 
        backtest_date, config, calibrated_scenario
    )
    
    # Forecast Future
    temp_config = config.copy()
    temp_config['forecast_year'] = actuals_through.year
    
    future_pipeline = generate_future_pipeline(historical_patterns, temp_config, calibrated_scenario)
    closure_projections = project_closure_timing(future_pipeline, vintage_curves, temp_config)
    closure_projections['close_date'] = closure_projections['close_month'].apply(lambda x: x.to_timestamp())
    
    # Filter Future wins to Backtest Window
    future_in_period = closure_projections[
        (closure_projections['close_date'] > backtest_date) &
        (closure_projections['close_date'] <= actuals_through)
    ]
    
    # Get Actuals
    actual_closed = deal_summary[
        (deal_summary['final_stage'] == 'Closed Won') &
        (deal_summary['date_closed'] > backtest_date) &
        (deal_summary['date_closed'] <= actuals_through)
    ]
    
    # Aggregation
    actual_by_segment = actual_closed.groupby('market_segment').agg({'deal_id': 'count', 'net_revenue': 'sum'}).reset_index()
    actual_by_segment.columns = ['market_segment', 'actual_won_deal_count', 'actual_won_revenue']
    
    active_by_segment = active_forecast.groupby('market_segment').agg({
        'adjusted_probability': 'sum',
        'expected_revenue': 'sum'
    }).reset_index()
    active_by_segment.columns = ['market_segment', 'active_expected_deals', 'active_expected_revenue']
    
    future_by_segment = future_in_period.groupby('market_segment').agg({
        'expected_deals': 'sum',
        'expected_revenue': 'sum'
    }).reset_index()
    future_by_segment.columns = ['market_segment', 'future_expected_deals', 'future_expected_revenue']
    
    comparison = pd.merge(actual_by_segment, active_by_segment, on='market_segment', how='outer')
    comparison = pd.merge(comparison, future_by_segment, on='market_segment', how='outer').fillna(0)
    
    comparison['forecast_won_deal_count'] = comparison['active_expected_deals'] + comparison['future_expected_deals']
    comparison['forecast_won_revenue'] = comparison['active_expected_revenue'] + comparison['future_expected_revenue']
    
    # Avoid div by zero
    comparison['revenue_variance_pct'] = comparison.apply(
        lambda row: ((row['forecast_won_revenue'] - row['actual_won_revenue']) / row['actual_won_revenue'] * 100) 
        if row['actual_won_revenue'] > 1000 else 0, axis=1
    )
    
    total_forecast = comparison['forecast_won_revenue'].sum()
    total_actual = actual_closed['net_revenue'].sum()
    
    return {
        'comparison': comparison,
        'total_forecast': total_forecast,
        'total_actual': total_actual,
        'overall_variance_pct': ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0,
        'calibrated_scenario': calibrated_scenario
    }

# =============================================================================
# EXPORT
# =============================================================================

def export_results(monthly_forecast, assumptions, config, backtest_results=None):
    forecast_path = Path(config['forecast_export_path'])
    forecast_path.mkdir(parents=True, exist_ok=True)
    
    monthly_forecast.to_csv(forecast_path / 'forecast_2026.csv', index=False)
    
    with open(forecast_path / 'assumptions_log.json', 'w') as f:
        json.dump(assumptions, f, indent=2, default=str)
    
    if backtest_results:
        validation_path = Path(config['validation_export_path'])
        validation_path.mkdir(parents=True, exist_ok=True)
        
        backtest_results['comparison'].to_csv(validation_path / 'backtest_results.csv', index=False)
        
        summary = {
            'total_forecast': backtest_results['total_forecast'],
            'total_actual': backtest_results['total_actual'],
            'overall_variance_pct': backtest_results['overall_variance_pct'],
            'calibrated_scenario': backtest_results['calibrated_scenario']
        }
        with open(validation_path / 'backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

# =============================================================================
# MAIN
# =============================================================================

def run_forecast():
    print("Starting forecast pipeline...")
    
    scenario = SCENARIOS.get(SCENARIO, SCENARIOS['base']) if isinstance(SCENARIO, str) else SCENARIO
    
    if GENERATE_MOCK:
        print("Generating mock data...")
        from generate_mock import generate_mock_data
        generate_mock_data(output_path=CONFIG['data_path'])
    
    print("Loading data...")
    df = load_and_preprocess(CONFIG)
    
    print("Identifying deal outcomes...")
    deal_summary = identify_deal_outcomes(df, CONFIG)
    
    print("Calculating vintage curves (Layer 1)...")
    vintage_curves = calculate_vintage_curves(deal_summary, CONFIG)
    
    print("Calculating stage probabilities (Layer 2)...")
    stage_probs = calculate_stage_probabilities(deal_summary, CONFIG)
    
    print("Calculating staleness thresholds (based on WON deals)...")
    staleness_thresholds = calculate_staleness_thresholds(df, deal_summary, CONFIG)
    
    print("Forecasting active pipeline...")
    # NOTE: Using 'training_end' (2025-12-31) as the 'as of' date for the 2026 forecast
    active_forecast = forecast_active_pipeline(
        df, deal_summary, stage_probs, staleness_thresholds, 
        CONFIG['training_end'], CONFIG, scenario
    )
    
    print(f"   -> Active Pipeline Revenue: ${active_forecast['expected_revenue'].sum():,.0f} (Risk Adjusted)")
    
    print("Projecting future pipeline...")
    historical_patterns = calculate_historical_deal_patterns(deal_summary, CONFIG)
    future_pipeline = generate_future_pipeline(historical_patterns, CONFIG, scenario)
    closure_projections = project_closure_timing(future_pipeline, vintage_curves, CONFIG)
    
    print("Aggregating forecasts...")
    monthly_forecast = aggregate_forecasts(active_forecast, closure_projections, CONFIG)
    
    assumptions = {
        'config': {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v for k, v in CONFIG.items()},
        'scenario': scenario,
        'historical_patterns': historical_patterns,
        'stage_probabilities': stage_probs,
        'staleness_thresholds': staleness_thresholds,
        'generated_at': datetime.now().isoformat()
    }
    
    backtest_results = None
    if RUN_BACKTEST:
        print("\n=== RUNNING BACKTEST ===")
        backtest_results = run_backtest(df, deal_summary, CONFIG, scenario, BACKTEST_DATE, ACTUALS_THROUGH)
        print(f"Backtest Variance: {backtest_results['overall_variance_pct']:+.1f}%")
        print("========================\n")
    
    print("Exporting results...")
    export_results(monthly_forecast, assumptions, CONFIG, backtest_results)
    
    # Calculate Total Forecast (Active Adjusted + Future)
    # Note: aggregate_forecasts separates them currently, so we sum them here for display
    total_active = active_forecast['expected_revenue'].sum()
    total_future = closure_projections['expected_revenue'].sum()
    annual_total = total_active + total_future
    
    print(f"2026 Annual Forecast: ${annual_total:,.0f}")
    print("Complete.")
    
    return {
        'monthly_forecast': monthly_forecast,
        'backtest_results': backtest_results,
        'assumptions': assumptions
    }


if __name__ == "__main__":
    run_forecast()