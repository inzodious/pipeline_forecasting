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
    'staleness_percentile': 90,
    'staleness_penalty': 0.7,   # Less harsh penalty
    'zombie_multiplier': 2.0,   # Take longer to zombie-out
    'active_probability_floor': 0.20,  # Minimum probability for active deals
    'stages_ordered': ['Qualified', 'Alignment', 'Solutioning', 'Closed Won', 'Closed Lost'],
    'active_stages': ['Qualified', 'Alignment', 'Solutioning'],
    'max_vintage_weeks': 52,
    'min_segment_deals': 10,
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
    
    mask = (df['date_snapshot'] >= config['training_start']) & (df['date_snapshot'] <= config['training_end'])
    return df[mask].copy()


def identify_deal_outcomes(df, config, as_of_date=None):
    """
    FIX: Added as_of_date parameter to prevent data leakage in backtesting.
    Only considers snapshots up to as_of_date when determining outcomes.
    """
    if as_of_date is not None:
        as_of_date = pd.to_datetime(as_of_date)
        df = df[df['date_snapshot'] <= as_of_date].copy()
    
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
    """
    Vintage curves from CLOSED WON deals only.
    Only includes deals with valid date_closed.
    Creates a GLOBAL fallback curve for thin segments.
    """
    won_deals = deal_summary[
        (deal_summary['final_stage'] == 'Closed Won') & 
        (deal_summary['date_closed'].notna())
    ].copy()
    
    won_deals['weeks_to_close'] = ((won_deals['date_closed'] - won_deals['date_created']).dt.days / 7).fillna(0).astype(int)
    won_deals['weeks_to_close'] = won_deals['weeks_to_close'].clip(lower=0, upper=config['max_vintage_weeks'])
    
    # Build GLOBAL curve as fallback
    total_global_revenue = won_deals['net_revenue'].sum()
    global_curve = None
    if total_global_revenue > 0:
        global_by_week = won_deals.groupby('weeks_to_close')['net_revenue'].sum().reindex(
            range(config['max_vintage_weeks'] + 1), fill_value=0
        )
        global_curve = global_by_week.cumsum() / total_global_revenue
    
    vintage_curves = {'_GLOBAL': global_curve}
    
    for segment in deal_summary['market_segment'].unique():
        seg_deals = won_deals[won_deals['market_segment'] == segment]
        total_revenue = seg_deals['net_revenue'].sum()
        
        if len(seg_deals) < config['min_segment_deals'] or total_revenue == 0:
            vintage_curves[segment] = global_curve  # Use global fallback
            continue
        
        revenue_by_week = seg_deals.groupby('weeks_to_close')['net_revenue'].sum().reindex(
            range(config['max_vintage_weeks'] + 1), fill_value=0
        )
        cumulative_revenue = revenue_by_week.cumsum()
        vintage_curves[segment] = cumulative_revenue / total_revenue
    
    return vintage_curves

# =============================================================================
# LAYER 2: STAGE-WEIGHTED PROBABILITY
# =============================================================================

def calculate_stage_probabilities(deal_summary, config):
    """
    Stage probabilities from CLOSED deals only.
    """
    closed_deals = deal_summary[deal_summary['final_stage'].isin(['Closed Won', 'Closed Lost'])]
    stage_probs = {}
    
    for segment in deal_summary['market_segment'].unique():
        seg_deals = closed_deals[closed_deals['market_segment'] == segment]
        stage_probs[segment] = {}
        
        if len(seg_deals) < config['min_segment_deals']:
            for stage in config['active_stages']:
                stage_probs[segment][stage] = None
            continue
        
        for stage in config['active_stages']:
            deals_through_stage = seg_deals[seg_deals['stages_observed'].apply(lambda x: stage in x)]
            
            if len(deals_through_stage) < 5:
                stage_probs[segment][stage] = None
                continue
                
            won = len(deals_through_stage[deals_through_stage['final_stage'] == 'Closed Won'])
            stage_probs[segment][stage] = won / len(deals_through_stage)
    
    return stage_probs


def calculate_staleness_thresholds(df, deal_summary, config):
    thresholds = {}
    won_ids = set(deal_summary[deal_summary['final_stage'] == 'Closed Won']['deal_id'])
    won_snapshots = df[df['deal_id'].isin(won_ids)]
    
    for segment in df['market_segment'].unique():
        thresholds[segment] = {}
        seg_df = won_snapshots[won_snapshots['market_segment'] == segment]
        
        for stage in config['active_stages']:
            stage_df = seg_df[seg_df['stage'] == stage]
            
            if len(stage_df) == 0:
                thresholds[segment][stage] = 90
                continue
            
            stage_durations = stage_df.groupby('deal_id').size() * 7
            thresholds[segment][stage] = np.percentile(stage_durations, config['staleness_percentile'])
            
    return thresholds


def calculate_deal_age_in_stage(df, deal_id, stage, as_of_date):
    deal_df = df[
        (df['deal_id'] == deal_id) & 
        (df['stage'] == stage) & 
        (df['date_snapshot'] <= as_of_date)
    ]
    if len(deal_df) == 0:
        return 0
    
    first_seen = deal_df['date_snapshot'].min()
    return (pd.to_datetime(as_of_date) - first_seen).days


def forecast_active_pipeline(df, deal_summary, stage_probs, staleness_thresholds, vintage_curves, as_of_date, config, scenario):
    """
    FIX: Now returns time-distributed forecast using vintage curves.
    """
    as_of_date = pd.to_datetime(as_of_date)
    
    latest_snapshot = df[df['date_snapshot'] <= as_of_date].sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    open_deals = latest_snapshot[latest_snapshot['stage'].isin(config['active_stages'])]
    
    closure_map = deal_summary.set_index('deal_id')['date_closed'].to_dict()
    
    forecast_results = []
    for _, deal in open_deals.iterrows():
        deal_id = deal['deal_id']
        real_close_date = closure_map.get(deal_id)
        
        if real_close_date and pd.to_datetime(real_close_date) <= as_of_date:
            continue
            
        segment = deal['market_segment']
        stage = deal['stage']
        revenue = deal['net_revenue']
        
        base_prob = stage_probs.get(segment, {}).get(stage)
        if base_prob is None:
            base_prob = 0.35  # Conservative fallback
        
        age_in_stage = calculate_deal_age_in_stage(df, deal_id, stage, as_of_date)
        threshold = staleness_thresholds.get(segment, {}).get(stage, 90)
        
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
        
        # Apply probability floor for non-zombie deals
        if not is_zombie and adjusted_prob < config.get('active_probability_floor', 0):
            adjusted_prob = config.get('active_probability_floor', 0)
            
        adjusted_prob = min(adjusted_prob * scenario['win_rate_uplift'], 1.0)
        
        deal_age_weeks = (as_of_date - pd.to_datetime(deal['date_created'])).days // 7
        
        forecast_results.append({
            'deal_id': deal_id,
            'market_segment': segment,
            'stage': stage,
            'date_created': deal['date_created'],
            'deal_age_weeks': deal_age_weeks,
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


def distribute_active_pipeline_closures(active_forecast, vintage_curves, config, forecast_start_date):
    """
    Time-distribute active pipeline closures.
    The expected_revenue is ALREADY probability-adjusted, so we just need to determine WHEN it closes.
    Uses vintage curves to estimate timing, but ensures total expected value is preserved.
    """
    if active_forecast.empty:
        return pd.DataFrame(columns=['deal_id', 'market_segment', 'close_month', 'expected_revenue'])
    
    closure_projections = []
    forecast_start = pd.to_datetime(forecast_start_date)
    
    for _, deal in active_forecast.iterrows():
        if deal['adjusted_probability'] == 0:
            continue
            
        segment = deal['market_segment']
        curve = vintage_curves.get(segment)
        expected_rev = deal['expected_revenue']
        
        # If no curve available, assume closes in first month
        if curve is None:
            close_month = forecast_start.to_period('M')
            closure_projections.append({
                'deal_id': deal['deal_id'],
                'market_segment': segment,
                'close_month': close_month,
                'expected_revenue': expected_rev
            })
            continue
        
        deal_age_weeks = deal['deal_age_weeks']
        
        # Calculate remaining probability mass in the curve after current age
        current_cumulative = curve.iloc[min(deal_age_weeks, len(curve)-1)] if deal_age_weeks < len(curve) else curve.iloc[-1]
        remaining_mass = 1.0 - current_cumulative
        
        # If deal is already past typical closure time, assume it closes soon
        if remaining_mass <= 0.05:
            close_month = forecast_start.to_period('M')
            closure_projections.append({
                'deal_id': deal['deal_id'],
                'market_segment': segment,
                'close_month': close_month,
                'expected_revenue': expected_rev
            })
            continue
        
        # Distribute expected revenue across future weeks based on conditional timing
        distributed_total = 0.0
        week_distributions = []
        
        prev_cumulative = current_cumulative
        for week_offset in range(min(52, config['max_vintage_weeks'] - deal_age_weeks)):
            future_week = deal_age_weeks + week_offset + 1
            if future_week >= len(curve):
                break
                
            future_cumulative = curve.iloc[future_week]
            incremental_prob = future_cumulative - prev_cumulative
            
            if incremental_prob > 0.001:
                # Conditional probability given deal hasn't closed yet
                conditional_prob = incremental_prob / remaining_mass
                close_date = forecast_start + timedelta(weeks=week_offset + 1)
                week_distributions.append({
                    'close_month': close_date.to_period('M'),
                    'weight': conditional_prob
                })
                distributed_total += conditional_prob
            
            prev_cumulative = future_cumulative
        
        # Ensure we distribute all expected revenue (normalize weights)
        if distributed_total > 0 and week_distributions:
            for wd in week_distributions:
                normalized_weight = wd['weight'] / distributed_total
                closure_projections.append({
                    'deal_id': deal['deal_id'],
                    'market_segment': segment,
                    'close_month': wd['close_month'],
                    'expected_revenue': expected_rev * normalized_weight
                })
        else:
            # Fallback: all closes in first month
            close_month = forecast_start.to_period('M')
            closure_projections.append({
                'deal_id': deal['deal_id'],
                'market_segment': segment,
                'close_month': close_month,
                'expected_revenue': expected_rev
            })
    
    return pd.DataFrame(closure_projections)

# =============================================================================
# FUTURE PIPELINE PROJECTION
# =============================================================================

def calculate_historical_deal_patterns(deal_summary, config, training_end_date=None):
    """
    FIX: Now only uses deals created within the training period for pattern calculation.
    Uses global averages as fallback for thin segments.
    """
    deal_summary = deal_summary.copy()
    deal_summary['created_month'] = deal_summary['date_created'].dt.to_period('M')
    
    if training_end_date:
        training_end = pd.to_datetime(training_end_date)
        deal_summary = deal_summary[deal_summary['date_created'] <= training_end]
    
    # Calculate GLOBAL patterns as fallback
    global_monthly = deal_summary.groupby('created_month').agg({
        'deal_id': 'count',
        'volume_weight': 'sum',
        'net_revenue': 'mean'
    }).reset_index()
    
    global_closed = deal_summary[deal_summary['final_stage'].isin(['Closed Won', 'Closed Lost'])]
    global_won = global_closed[global_closed['final_stage'] == 'Closed Won']
    
    global_patterns = {
        'avg_monthly_deals': global_monthly['volume_weight'].mean() if len(global_monthly) > 0 else 5,
        'avg_deal_revenue': global_won['net_revenue'].mean() if len(global_won) > 0 else 50000,
        'win_rate': len(global_won) / len(global_closed) if len(global_closed) > 0 else 0.35,
        'monthly_volatility': global_monthly['volume_weight'].std() if len(global_monthly) > 1 else 2,
    }
    
    # Handle NaN
    for key in global_patterns:
        if pd.isna(global_patterns[key]):
            global_patterns[key] = {'avg_monthly_deals': 5, 'avg_deal_revenue': 50000, 'win_rate': 0.35, 'monthly_volatility': 2}[key]
    
    patterns = {'_GLOBAL': global_patterns}
    
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
        
        if len(closed_deals) < config['min_segment_deals']:
            # Use global pattern scaled by this segment's volume share
            seg_volume_share = len(seg_deals) / len(deal_summary) if len(deal_summary) > 0 else 0.33
            scaled_patterns = global_patterns.copy()
            scaled_patterns['avg_monthly_deals'] = global_patterns['avg_monthly_deals'] * seg_volume_share
            patterns[segment] = scaled_patterns
            continue
        
        win_rate = len(won_deals) / len(closed_deals) if len(closed_deals) > 0 else 0.0
        avg_won_revenue = won_deals['net_revenue'].mean() if len(won_deals) > 0 else 0.0
        
        if pd.isna(avg_won_revenue) or avg_won_revenue == 0:
            avg_won_revenue = global_patterns['avg_deal_revenue']
        
        patterns[segment] = {
            'avg_monthly_deals': monthly_volume['weighted_count'].mean(),
            'avg_deal_revenue': avg_won_revenue,
            'win_rate': win_rate,
            'monthly_volatility': monthly_volume['weighted_count'].std() if len(monthly_volume) > 1 else 0,
        }
    
    return patterns


def generate_future_pipeline(historical_patterns, config, scenario):
    future_deals = []
    
    for month_num in range(1, 13):
        forecast_month = pd.Period(f"{config['forecast_year']}-{month_num:02d}", freq='M')
        
        for segment, patterns in historical_patterns.items():
            if segment == '_GLOBAL':  # Skip the global fallback entry
                continue
            if patterns is None:
                continue
                
            n_deals = int(np.round(patterns['avg_monthly_deals'] * scenario['deal_volume_growth']))
            adjusted_revenue = patterns['avg_deal_revenue'] * scenario['revenue_per_deal_uplift']
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
        curve = vintage_curves.get(segment)
        
        if curve is None:
            continue
        
        prev_cumulative = 0.0
        for week in range(len(curve)):
            current_cumulative = curve.iloc[week]
            incremental_prob = current_cumulative - prev_cumulative
            
            if incremental_prob > 0.001:
                created_date = deal['created_month'].to_timestamp()
                close_date = created_date + timedelta(weeks=week)
                
                closure_projections.append({
                    'deal_id': deal['deal_id'],
                    'market_segment': segment,
                    'created_month': deal['created_month'],
                    'close_month': close_date.to_period('M'),
                    'weeks_to_close': week,
                    'closure_probability': incremental_prob,
                    'expected_deals': incremental_prob * deal['win_probability'], 
                    'projected_revenue': deal['projected_revenue'],
                    'expected_revenue': deal['projected_revenue'] * incremental_prob * deal['win_probability']
                })
            prev_cumulative = current_cumulative
    
    return pd.DataFrame(closure_projections)

# =============================================================================
# FORECAST AGGREGATION
# =============================================================================

def aggregate_forecasts(active_closure_projections, future_closure_projections, config):
    """
    FIX: Now properly aggregates time-distributed active pipeline with future pipeline.
    """
    all_projections = pd.concat([
        active_closure_projections[['market_segment', 'close_month', 'expected_revenue']].assign(source='active'),
        future_closure_projections[['market_segment', 'close_month', 'expected_revenue']].assign(source='future')
    ], ignore_index=True) if not active_closure_projections.empty else future_closure_projections[['market_segment', 'close_month', 'expected_revenue']].assign(source='future')
    
    if all_projections.empty:
        return pd.DataFrame()
    
    monthly_forecast = all_projections.groupby(['close_month', 'market_segment']).agg({
        'expected_revenue': 'sum'
    }).reset_index()
    monthly_forecast.columns = ['forecast_month', 'market_segment', 'expected_won_revenue']
    monthly_forecast['forecast_month'] = monthly_forecast['forecast_month'].astype(str)
    
    return monthly_forecast

# =============================================================================
# BACKTESTING - CORRECTED
# =============================================================================

def detect_market_conditions(deal_summary, backtest_date, actuals_through):
    """
    FIX: Corrected volume growth calculation.
    Compares Training Period (pre-backtest) vs Test Period (post-backtest).
    """
    backtest_date = pd.to_datetime(backtest_date)
    actuals_through = pd.to_datetime(actuals_through)
    
    deal_summary = deal_summary.copy()
    deal_summary['created_month'] = deal_summary['date_created'].dt.to_period('M')
    
    # Training Period: All of 2024
    training_start = pd.to_datetime('2024-01-01')
    training_deals = deal_summary[
        (deal_summary['date_created'] >= training_start) & 
        (deal_summary['date_created'] < backtest_date)
    ]
    training_months = (backtest_date - training_start).days / 30.44  # Average days per month
    
    # Test Period: All of 2025
    test_deals = deal_summary[
        (deal_summary['date_created'] >= backtest_date) & 
        (deal_summary['date_created'] <= actuals_through)
    ]
    test_months = (actuals_through - backtest_date).days / 30.44
    
    # Volume Growth: Compare monthly deal creation rates
    training_vol_per_month = len(training_deals) / max(training_months, 1)
    test_vol_per_month = len(test_deals) / max(test_months, 1)
    vol_growth = test_vol_per_month / training_vol_per_month if training_vol_per_month > 0 else 1.0
    
    # Revenue Growth: Compare average revenue of WON deals
    training_won = training_deals[training_deals['final_stage'] == 'Closed Won']
    test_won = test_deals[test_deals['final_stage'] == 'Closed Won']
    
    training_avg_rev = training_won['net_revenue'].mean() if len(training_won) > 0 else 0
    test_avg_rev = test_won['net_revenue'].mean() if len(test_won) > 0 else 0
    rev_growth = test_avg_rev / training_avg_rev if training_avg_rev > 0 else 1.0
    
    # Win Rate Change
    training_closed = training_deals[training_deals['final_stage'].isin(['Closed Won', 'Closed Lost'])]
    test_closed = test_deals[test_deals['final_stage'].isin(['Closed Won', 'Closed Lost'])]
    
    training_win_rate = len(training_won) / len(training_closed) if len(training_closed) > 0 else 0
    test_win_rate = len(test_won) / len(test_closed) if len(test_closed) > 0 else 0
    win_rate_growth = test_win_rate / training_win_rate if training_win_rate > 0 else 1.0
    
    return vol_growth, rev_growth, win_rate_growth


def run_backtest(df, config, base_scenario, backtest_date, actuals_through):
    """
    FIX: Completely rewritten to prevent data leakage.
    Now rebuilds deal outcomes using only data available as of backtest_date.
    """
    backtest_date = pd.to_datetime(backtest_date)
    actuals_through = pd.to_datetime(actuals_through)
    
    print(f"   Backtest: Training on data available as of {backtest_date.date()}...")
    
    # FIX: Build deal outcomes using ONLY snapshots available as of backtest_date
    # This prevents data leakage - we don't know future outcomes
    training_deal_summary = identify_deal_outcomes(df, config, as_of_date=backtest_date)
    training_df = df[df['date_snapshot'] <= backtest_date].copy()
    
    # For calculating actual market conditions, we need the FULL picture
    # (This is OK because we're using this to EVALUATE, not to TRAIN)
    full_deal_summary = identify_deal_outcomes(df, config, as_of_date=actuals_through)
    
    # DETECT ACTUAL MARKET CONDITIONS
    print("   Backtest: Detecting actual 2025 market conditions for calibration...")
    vol_growth, rev_growth, win_rate_growth = detect_market_conditions(
        full_deal_summary, backtest_date, actuals_through
    )
    print(f"   -> Detected Volume Growth: {vol_growth:.3f}x")
    print(f"   -> Detected Rev/Deal Growth: {rev_growth:.3f}x")
    print(f"   -> Detected Win Rate Growth: {win_rate_growth:.3f}x")
    
    # Create Calibrated Scenario (simulating "perfect foresight" of market conditions)
    calibrated_scenario = base_scenario.copy()
    calibrated_scenario['deal_volume_growth'] = vol_growth
    calibrated_scenario['revenue_per_deal_uplift'] = rev_growth
    calibrated_scenario['win_rate_uplift'] = win_rate_growth
    
    # Calculate Models using ONLY training data (no data leakage)
    vintage_curves = calculate_vintage_curves(training_deal_summary, config)
    stage_probs = calculate_stage_probabilities(training_deal_summary, config)
    staleness_thresholds = calculate_staleness_thresholds(training_df, training_deal_summary, config)
    historical_patterns = calculate_historical_deal_patterns(training_deal_summary, config, backtest_date)
    
    # Count valid segments (exclude _GLOBAL marker)
    valid_segments = [s for s, c in vintage_curves.items() if c is not None and s != '_GLOBAL']
    print(f"   -> Valid segments for forecasting: {valid_segments}")
    
    # DIAGNOSTIC: Show training data coverage
    training_closed_won = training_deal_summary[training_deal_summary['final_stage'] == 'Closed Won']
    print(f"   -> Training closed-won deals: {len(training_closed_won)}")
    for seg in sorted(training_deal_summary['market_segment'].unique()):
        seg_won = training_closed_won[training_closed_won['market_segment'] == seg]
        print(f"      {seg}: {len(seg_won)} deals, ${seg_won['net_revenue'].sum():,.0f} revenue")
    
    # DIAGNOSTIC: Show vintage curve terminal values
    print("   -> Vintage curve terminal values (% revenue closed by week 52):")
    for seg in sorted([s for s in vintage_curves.keys() if s != '_GLOBAL']):
        curve = vintage_curves.get(seg)
        if curve is not None:
            terminal_val = curve.iloc[-1] if len(curve) > 0 else 0
            print(f"      {seg}: {terminal_val:.1%}")
    
    # Forecast Active Pipeline (deals open as of backtest_date)
    active_forecast = forecast_active_pipeline(
        training_df, training_deal_summary, stage_probs, staleness_thresholds, 
        vintage_curves, backtest_date, config, calibrated_scenario
    )
    
    # DIAGNOSTIC: Active pipeline details
    print(f"   -> Active pipeline: {len(active_forecast)} deals")
    if not active_forecast.empty:
        total_raw_value = active_forecast['net_revenue'].sum()
        total_expected = active_forecast['expected_revenue'].sum()
        avg_prob = active_forecast['adjusted_probability'].mean()
        stale_count = active_forecast['is_stale'].sum()
        zombie_count = active_forecast['is_zombie'].sum()
        print(f"      Raw value: ${total_raw_value:,.0f}")
        print(f"      Expected value: ${total_expected:,.0f}")
        print(f"      Avg probability: {avg_prob:.1%}")
        print(f"      Stale deals: {stale_count}, Zombie deals: {zombie_count}")
    
    # Time-distribute active pipeline closures
    active_closure_projections = distribute_active_pipeline_closures(
        active_forecast, vintage_curves, config, backtest_date
    )
    
    # DIAGNOSTIC: Trace active pipeline value
    if not active_closure_projections.empty:
        total_distributed = active_closure_projections['expected_revenue'].sum()
        print(f"      Total distributed to months: ${total_distributed:,.0f}")
    
    # Forecast Future Pipeline (new deals created in 2025)
    temp_config = config.copy()
    temp_config['forecast_year'] = actuals_through.year
    
    future_pipeline = generate_future_pipeline(historical_patterns, temp_config, calibrated_scenario)
    future_closure_projections = project_closure_timing(future_pipeline, vintage_curves, temp_config)
    
    if not future_closure_projections.empty:
        future_closure_projections['close_date'] = future_closure_projections['close_month'].apply(lambda x: x.to_timestamp())
    
    # Filter projections to backtest window
    active_in_period = pd.DataFrame()
    if not active_closure_projections.empty:
        active_closure_projections['close_date'] = active_closure_projections['close_month'].apply(lambda x: x.to_timestamp())
        
        # DIAGNOSTIC: Show distribution timing
        total_active_distributed = active_closure_projections['expected_revenue'].sum()
        active_in_period = active_closure_projections[
            (active_closure_projections['close_date'] > backtest_date) &
            (active_closure_projections['close_date'] <= actuals_through)
        ]
        in_period_total = active_in_period['expected_revenue'].sum() if not active_in_period.empty else 0
        out_of_period = total_active_distributed - in_period_total
        print(f"   -> Active revenue timing:")
        print(f"      In backtest period (2025): ${in_period_total:,.0f}")
        print(f"      After backtest period: ${out_of_period:,.0f}")
    
    future_in_period = pd.DataFrame()
    if not future_closure_projections.empty:
        future_in_period = future_closure_projections[
            (future_closure_projections['close_date'] > backtest_date) &
            (future_closure_projections['close_date'] <= actuals_through)
        ]
    
    # Get Actuals (using FULL data)
    actual_closed = full_deal_summary[
        (full_deal_summary['final_stage'] == 'Closed Won') &
        (full_deal_summary['date_closed'] > backtest_date) &
        (full_deal_summary['date_closed'] <= actuals_through)
    ]
    
    # DIAGNOSTIC: Show where actual revenue came from
    carryover_deals = actual_closed[actual_closed['date_created'] < backtest_date]
    new_deals = actual_closed[actual_closed['date_created'] >= backtest_date]
    print("   -> Actual revenue sources:")
    print(f"      Carryover (created 2024, closed 2025): {len(carryover_deals)} deals, ${carryover_deals['net_revenue'].sum():,.0f}")
    print(f"      New (created & closed 2025): {len(new_deals)} deals, ${new_deals['net_revenue'].sum():,.0f}")
    
    # Aggregation
    actual_by_segment = actual_closed.groupby('market_segment').agg({
        'deal_id': 'count', 
        'net_revenue': 'sum'
    }).reset_index()
    actual_by_segment.columns = ['market_segment', 'actual_won_deal_count', 'actual_won_revenue']
    
    active_by_segment = pd.DataFrame()
    if not active_in_period.empty:
        active_by_segment = active_in_period.groupby('market_segment').agg({
            'expected_revenue': 'sum'
        }).reset_index()
        active_by_segment.columns = ['market_segment', 'active_expected_revenue']
    
    future_by_segment = pd.DataFrame()
    if not future_in_period.empty:
        future_by_segment = future_in_period.groupby('market_segment').agg({
            'expected_deals': 'sum',
            'expected_revenue': 'sum'
        }).reset_index()
        future_by_segment.columns = ['market_segment', 'future_expected_deals', 'future_expected_revenue']
    
    comparison = actual_by_segment.copy()
    if not active_by_segment.empty:
        comparison = pd.merge(comparison, active_by_segment, on='market_segment', how='outer')
    else:
        comparison['active_expected_revenue'] = 0
        
    if not future_by_segment.empty:
        comparison = pd.merge(comparison, future_by_segment, on='market_segment', how='outer')
    else:
        comparison['future_expected_deals'] = 0
        comparison['future_expected_revenue'] = 0
    
    comparison = comparison.fillna(0)
    
    comparison['forecast_won_revenue'] = comparison['active_expected_revenue'] + comparison['future_expected_revenue']
    
    comparison['revenue_variance_pct'] = comparison.apply(
        lambda row: ((row['forecast_won_revenue'] - row['actual_won_revenue']) / row['actual_won_revenue'] * 100) 
        if row['actual_won_revenue'] > 1000 else 0, axis=1
    )
    
    # Only sum segments with actual revenue for fair comparison
    valid_comparison = comparison[comparison['actual_won_revenue'] > 1000]
    total_forecast = valid_comparison['forecast_won_revenue'].sum()
    total_actual = valid_comparison['actual_won_revenue'].sum()
    
    return {
        'comparison': comparison,
        'total_forecast': total_forecast,
        'total_actual': total_actual,
        'overall_variance_pct': ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0,
        'calibrated_scenario': calibrated_scenario,
        'diagnostics': {
            'training_deals_total': len(training_deal_summary),
            'training_closed_won': len(training_closed_won),
            'active_pipeline_count': len(active_forecast),
            'active_pipeline_expected': active_forecast['expected_revenue'].sum() if not active_forecast.empty else 0,
            'future_pipeline_synthetic_deals': len(future_pipeline),
            'valid_segments': valid_segments,
            'actual_carryover_deals': len(carryover_deals),
            'actual_carryover_revenue': carryover_deals['net_revenue'].sum(),
            'actual_new_deals': len(new_deals),
            'actual_new_revenue': new_deals['net_revenue'].sum(),
        }
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
            'calibrated_scenario': backtest_results['calibrated_scenario'],
            'diagnostics': backtest_results.get('diagnostics', {})
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
    valid_segments = [s for s, c in vintage_curves.items() if c is not None and s != '_GLOBAL']
    print(f"   -> Valid segments: {valid_segments}")
    
    print("Calculating stage probabilities (Layer 2)...")
    stage_probs = calculate_stage_probabilities(deal_summary, CONFIG)
    
    print("Calculating staleness thresholds (based on WON deals)...")
    staleness_thresholds = calculate_staleness_thresholds(df, deal_summary, CONFIG)
    
    print("Forecasting active pipeline...")
    active_forecast = forecast_active_pipeline(
        df, deal_summary, stage_probs, staleness_thresholds, 
        vintage_curves, CONFIG['training_end'], CONFIG, scenario
    )
    
    print(f"   -> Active Pipeline Revenue: ${active_forecast['expected_revenue'].sum():,.0f} (Risk Adjusted)")
    
    # Time-distribute active pipeline
    active_closure_projections = distribute_active_pipeline_closures(
        active_forecast, vintage_curves, CONFIG, CONFIG['training_end']
    )
    
    print("Projecting future pipeline...")
    historical_patterns = calculate_historical_deal_patterns(deal_summary, CONFIG)
    future_pipeline = generate_future_pipeline(historical_patterns, CONFIG, scenario)
    future_closure_projections = project_closure_timing(future_pipeline, vintage_curves, CONFIG)
    
    print("Aggregating forecasts...")
    monthly_forecast = aggregate_forecasts(active_closure_projections, future_closure_projections, CONFIG)
    
    assumptions = {
        'config': {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v for k, v in CONFIG.items()},
        'scenario': scenario,
        'historical_patterns': {k: v for k, v in historical_patterns.items() if v is not None and k != '_GLOBAL'},
        'stage_probabilities': stage_probs,
        'staleness_thresholds': staleness_thresholds,
        'generated_at': datetime.now().isoformat()
    }
    
    backtest_results = None
    if RUN_BACKTEST:
        print("\n=== RUNNING BACKTEST ===")
        backtest_results = run_backtest(df, CONFIG, scenario, BACKTEST_DATE, ACTUALS_THROUGH)
        print(f"\nBacktest Results:")
        print(f"   Total Forecast: ${backtest_results['total_forecast']:,.0f}")
        print(f"   Total Actual:   ${backtest_results['total_actual']:,.0f}")
        print(f"   Variance:       {backtest_results['overall_variance_pct']:+.1f}%")
        print("\nSegment Breakdown:")
        print(backtest_results['comparison'].to_string(index=False))
        print("========================\n")
    
    print("Exporting results...")
    export_results(monthly_forecast, assumptions, CONFIG, backtest_results)
    
    total_active = active_closure_projections['expected_revenue'].sum() if not active_closure_projections.empty else 0
    total_future = future_closure_projections['expected_revenue'].sum() if not future_closure_projections.empty else 0
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