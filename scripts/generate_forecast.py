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
RUN_BACKTEST = False
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
    'staleness_penalty': 0.8,
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
    
    training_mask = (df['date_snapshot'] >= config['training_start']) & (df['date_snapshot'] <= config['training_end'])
    return df[training_mask].copy()


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
    won_deals['weeks_to_close'] = ((won_deals['date_closed'] - won_deals['date_created']).dt.days / 7).astype(int)
    won_deals['weeks_to_close'] = won_deals['weeks_to_close'].clip(lower=0, upper=config['max_vintage_weeks'])
    
    vintage_curves = {}
    for segment in deal_summary['market_segment'].unique():
        seg_deals = won_deals[won_deals['market_segment'] == segment]
        total_revenue = seg_deals['net_revenue'].sum()
        
        if len(seg_deals) == 0 or total_revenue == 0:
            vintage_curves[segment] = pd.Series([1.0] * (config['max_vintage_weeks'] + 1))
            continue
        
        cumulative_pct = []
        for week in range(config['max_vintage_weeks'] + 1):
            revenue_by_week = seg_deals[seg_deals['weeks_to_close'] <= week]['net_revenue'].sum()
            cumulative_pct.append(revenue_by_week / total_revenue)
        vintage_curves[segment] = pd.Series(cumulative_pct)
    
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
            deals_through_stage = seg_deals[seg_deals['stages_observed'].apply(lambda x: stage in x)]
            if len(deals_through_stage) == 0:
                stage_probs[segment][stage] = 0.5
                continue
            won = len(deals_through_stage[deals_through_stage['final_stage'] == 'Closed Won'])
            stage_probs[segment][stage] = won / len(deals_through_stage)
    
    return stage_probs


def calculate_staleness_thresholds(df, config):
    thresholds = {}
    
    for segment in df['market_segment'].unique():
        thresholds[segment] = {}
        seg_df = df[df['market_segment'] == segment]
        
        for stage in config['active_stages']:
            stage_df = seg_df[seg_df['stage'] == stage].sort_values(['deal_id', 'date_snapshot'])
            
            if len(stage_df) == 0:
                thresholds[segment][stage] = 90
                continue
            
            stage_durations = [len(deal_group) * 7 for _, deal_group in stage_df.groupby('deal_id')]
            thresholds[segment][stage] = np.percentile(stage_durations, config['staleness_percentile']) if stage_durations else 90
    
    return thresholds


def calculate_deal_age_in_stage(df, deal_id, stage, as_of_date):
    deal_df = df[(df['deal_id'] == deal_id) & (df['stage'] == stage)]
    if len(deal_df) == 0:
        return 0
    return (pd.to_datetime(as_of_date) - deal_df['date_snapshot'].min()).days


def forecast_active_pipeline(df, deal_summary, stage_probs, staleness_thresholds, as_of_date, config, scenario):
    as_of_date = pd.to_datetime(as_of_date)
    latest_snapshot = df[df['date_snapshot'] <= as_of_date].groupby('deal_id').last().reset_index()
    open_deals = latest_snapshot[latest_snapshot['stage'].isin(config['active_stages'])]
    
    forecast_results = []
    for _, deal in open_deals.iterrows():
        segment = deal['market_segment']
        stage = deal['stage']
        revenue = deal['net_revenue']
        
        base_prob = stage_probs.get(segment, {}).get(stage, 0.5)
        age_in_stage = calculate_deal_age_in_stage(df, deal['deal_id'], stage, as_of_date)
        threshold = staleness_thresholds.get(segment, {}).get(stage, 90)
        
        adjusted_prob = base_prob * config['staleness_penalty'] if age_in_stage > threshold else base_prob
        adjusted_prob = min(adjusted_prob * scenario['win_rate_uplift'], 1.0)
        
        forecast_results.append({
            'deal_id': deal['deal_id'],
            'market_segment': segment,
            'stage': stage,
            'net_revenue': revenue,
            'base_probability': base_prob,
            'adjusted_probability': adjusted_prob,
            'expected_revenue': revenue * adjusted_prob * scenario['revenue_per_deal_uplift'],
            'is_stale': age_in_stage > threshold,
            'days_in_stage': age_in_stage
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
    
    for _, deal in future_pipeline.iterrows():
        segment = deal['market_segment']
        curve = vintage_curves.get(segment, pd.Series([1.0] * (config['max_vintage_weeks'] + 1)))
        
        prev_cumulative = 0
        for week in range(len(curve)):
            incremental_prob = curve.iloc[week] - prev_cumulative
            
            if incremental_prob > 0:
                created_date = deal['created_month'].to_timestamp()
                close_date = created_date + timedelta(weeks=week)
                
                closure_projections.append({
                    'deal_id': deal['deal_id'],
                    'market_segment': segment,
                    'created_month': deal['created_month'],
                    'close_month': close_date.to_period('M'),
                    'weeks_to_close': week,
                    'closure_probability': incremental_prob,
                    'projected_revenue': deal['projected_revenue'],
                    'expected_revenue': deal['projected_revenue'] * incremental_prob * deal['win_probability']
                })
            prev_cumulative = curve.iloc[week]
    
    return pd.DataFrame(closure_projections)

# =============================================================================
# FORECAST AGGREGATION
# =============================================================================

def aggregate_forecasts(active_pipeline_forecast, closure_projections, config):
    monthly_forecast = []
    segments = set(active_pipeline_forecast['market_segment'].unique()) | set(closure_projections['market_segment'].unique())
    
    for month_num in range(1, 13):
        forecast_month = pd.Period(f"{config['forecast_year']}-{month_num:02d}", freq='M')
        
        for segment in segments:
            active_seg = active_pipeline_forecast[active_pipeline_forecast['market_segment'] == segment]
            future_seg = closure_projections[
                (closure_projections['market_segment'] == segment) &
                (closure_projections['close_month'] == forecast_month)
            ]
            
            monthly_forecast.append({
                'forecast_month': str(forecast_month),
                'market_segment': segment,
                'open_pipeline_deal_count': len(active_seg),
                'open_pipeline_revenue': active_seg['net_revenue'].sum(),
                'expected_won_deal_count': active_seg['adjusted_probability'].sum() + future_seg['closure_probability'].sum(),
                'expected_won_revenue': active_seg['expected_revenue'].sum() + future_seg['expected_revenue'].sum(),
            })
    
    return pd.DataFrame(monthly_forecast)

# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(df, deal_summary, config, scenario, backtest_date, actuals_through):
    backtest_date = pd.to_datetime(backtest_date)
    actuals_through = pd.to_datetime(actuals_through)
    
    backtest_df = df[df['date_snapshot'] <= backtest_date].copy()
    backtest_deals = deal_summary[deal_summary['date_created'] <= backtest_date].copy()
    
    vintage_curves = calculate_vintage_curves(backtest_deals, config)
    stage_probs = calculate_stage_probabilities(backtest_deals, config)
    staleness_thresholds = calculate_staleness_thresholds(backtest_df, config)
    historical_patterns = calculate_historical_deal_patterns(backtest_deals, config)
    
    active_forecast = forecast_active_pipeline(backtest_df, backtest_deals, stage_probs, staleness_thresholds, backtest_date, config, scenario)
    
    temp_config = config.copy()
    temp_config['forecast_year'] = actuals_through.year
    
    future_pipeline = generate_future_pipeline(historical_patterns, temp_config, scenario)
    closure_projections = project_closure_timing(future_pipeline, vintage_curves, temp_config)
    closure_projections['close_date'] = closure_projections['close_month'].apply(lambda x: x.to_timestamp())
    
    future_in_period = closure_projections[
        (closure_projections['close_date'] > backtest_date) &
        (closure_projections['close_date'] <= actuals_through)
    ]
    
    actual_closed = deal_summary[
        (deal_summary['final_stage'] == 'Closed Won') &
        (deal_summary['date_closed'] > backtest_date) &
        (deal_summary['date_closed'] <= actuals_through)
    ]
    
    actual_by_segment = actual_closed.groupby('market_segment').agg({'deal_id': 'count', 'net_revenue': 'sum'}).reset_index()
    actual_by_segment.columns = ['market_segment', 'actual_won_deal_count', 'actual_won_revenue']
    
    active_by_segment = active_forecast.groupby('market_segment').agg({
        'adjusted_probability': 'sum',
        'expected_revenue': 'sum'
    }).reset_index()
    active_by_segment.columns = ['market_segment', 'active_expected_deals', 'active_expected_revenue']
    
    future_by_segment = future_in_period.groupby('market_segment').agg({
        'closure_probability': 'sum',
        'expected_revenue': 'sum'
    }).reset_index()
    future_by_segment.columns = ['market_segment', 'future_expected_deals', 'future_expected_revenue']
    
    comparison = pd.merge(actual_by_segment, active_by_segment, on='market_segment', how='outer')
    comparison = pd.merge(comparison, future_by_segment, on='market_segment', how='outer').fillna(0)
    comparison['forecast_won_deal_count'] = comparison['active_expected_deals'] + comparison['future_expected_deals']
    comparison['forecast_won_revenue'] = comparison['active_expected_revenue'] + comparison['future_expected_revenue']
    comparison['revenue_variance_pct'] = np.where(
        comparison['actual_won_revenue'] > 0,
        (comparison['forecast_won_revenue'] - comparison['actual_won_revenue']) / comparison['actual_won_revenue'] * 100,
        0
    )
    
    total_forecast = comparison['forecast_won_revenue'].sum()
    total_actual = actual_closed['net_revenue'].sum()
    
    return {
        'comparison': comparison,
        'total_forecast': total_forecast,
        'total_actual': total_actual,
        'overall_variance_pct': ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0
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
            'overall_variance_pct': backtest_results['overall_variance_pct']
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
    staleness_thresholds = calculate_staleness_thresholds(df, CONFIG)
    
    print("Forecasting active pipeline...")
    active_forecast = forecast_active_pipeline(df, deal_summary, stage_probs, staleness_thresholds, CONFIG['training_end'], CONFIG, scenario)
    
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
        print("Running backtest validation...")
        backtest_results = run_backtest(df, deal_summary, CONFIG, scenario, BACKTEST_DATE, ACTUALS_THROUGH)
    
    print("Exporting results...")
    export_results(monthly_forecast, assumptions, CONFIG, backtest_results)
    
    annual_total = monthly_forecast['expected_won_revenue'].sum()
    print(f"2026 Annual Forecast: ${annual_total:,.0f}")
    
    if backtest_results:
        print(f"Backtest Variance: {backtest_results['overall_variance_pct']:+.1f}%")
    
    print("Complete.")
    
    return {
        'monthly_forecast': monthly_forecast,
        'backtest_results': backtest_results,
        'assumptions': assumptions
    }


if __name__ == "__main__":
    run_forecast()