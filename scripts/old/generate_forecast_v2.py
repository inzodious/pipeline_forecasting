### 1. Imports and Configuration
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'data_path': 'data/fact_snapshots.csv',
    'validation_export_path': 'validation/',
    'active_stages': ['Qualified', 'Alignment', 'Solutioning', 'Verbal'],
    'min_deals_for_segment': 10,
    'trailing_months': 12,
    'future_conservatism': 0.85,
    'default_conversion_rate': 0.25,
}

SCENARIOS = {
    'base': {'volume_growth': 1.0, 'win_rate_adjustment': 1.0, 'deal_size_adjustment': 1.0},
    'growth': {'volume_growth': 1.15, 'win_rate_adjustment': 1.05, 'deal_size_adjustment': 1.0},
    'conservative': {'volume_growth': 0.85, 'win_rate_adjustment': 0.90, 'deal_size_adjustment': 1.0},
}

### 2. Data Loading
def load_data(config):
    df = pd.read_csv(config['data_path'], parse_dates=['date_snapshot', 'date_created', 'date_closed', 'date_implementation'])
    df['market_segment'] = df['market_segment'].fillna('Unknown')
    df['net_revenue'] = pd.to_numeric(df['net_revenue'], errors='coerce').fillna(0)
    df['impl_year'] = df['date_implementation'].dt.year
    return df

def get_latest_state(df, as_of_date=None):
    if as_of_date:
        df = df[df['date_snapshot'] <= pd.to_datetime(as_of_date)].copy()
    return df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()

def get_deal_outcomes(df):
    latest = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    latest['outcome'] = latest['stage'].apply(lambda x: 'Won' if x == 'Closed Won' else ('Lost' if x == 'Closed Lost' else 'Open'))
    return latest

### 3. Stage Conversion Rates
def calculate_stage_conversion_rates(df, as_of_date, target_impl_year, config):
    as_of = pd.to_datetime(as_of_date)
    closed_deals = df[(df['date_closed'].notna()) & (df['date_closed'] <= as_of) & (df['impl_year'] < target_impl_year)].copy()
    
    final_state = closed_deals.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    final_state = final_state[final_state['stage'].isin(['Closed Won', 'Closed Lost'])]
    
    deal_stages = closed_deals.groupby('deal_id')['stage'].apply(lambda x: set(x)).reset_index()
    deal_stages.columns = ['deal_id', 'stages_passed']
    
    final_state = final_state.merge(deal_stages, on='deal_id', how='left')
    final_state['is_won'] = final_state['stage'] == 'Closed Won'
    
    rates = {'_GLOBAL': {}}
    for stage in config['active_stages']:
        through_stage = final_state[final_state['stages_passed'].apply(lambda x: stage in x if x else False)]
        rates['_GLOBAL'][stage] = through_stage['is_won'].mean() if len(through_stage) >= 5 else config['default_conversion_rate']
    
    for segment in final_state['market_segment'].unique():
        seg_deals = final_state[final_state['market_segment'] == segment]
        rates[segment] = {}
        for stage in config['active_stages']:
            through_stage = seg_deals[seg_deals['stages_passed'].apply(lambda x: stage in x if x else False)]
            rates[segment][stage] = through_stage['is_won'].mean() if len(through_stage) >= config['min_deals_for_segment'] else rates['_GLOBAL'][stage]
    
    return rates

### 4. Historical Baseline
def calculate_historical_baseline(df, as_of_date, target_impl_year, config):
    as_of = pd.to_datetime(as_of_date)
    closed_won = df[(df['stage'] == 'Closed Won') & (df['date_closed'].notna()) & (df['date_closed'] <= as_of)].copy()
    won_deals = closed_won.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    baseline_deals = won_deals[won_deals['impl_year'] == (target_impl_year - 1)]
    if len(baseline_deals) == 0: return None
    
    total_revenue = baseline_deals['net_revenue'].sum()
    segment_totals = baseline_deals.groupby('market_segment').agg({'deal_id': 'count', 'net_revenue': 'sum'}).reset_index()
    segment_totals.columns = ['segment', 'deals', 'revenue']
    segment_totals['revenue_share'] = segment_totals['revenue'] / total_revenue if total_revenue > 0 else 0
    
    baseline = {'_GLOBAL': {'impl_year': target_impl_year - 1, 'total_deals': len(baseline_deals), 'total_revenue': total_revenue}}
    for _, row in segment_totals.iterrows():
        baseline[row['segment']] = {'deals': row['deals'], 'revenue': row['revenue'], 'revenue_share': row['revenue_share']}
    
    return baseline

### 5. Active Pipeline Forecast
def get_active_pipeline_for_impl_year(df, as_of_date, target_impl_year, config):
    latest = get_latest_state(df, as_of_date)
    return latest[(latest['stage'].isin(config['active_stages'])) & (latest['impl_year'] == target_impl_year)].copy()

def forecast_active_pipeline(active_pipeline, stage_rates, config, scenario):
    if active_pipeline.empty: return pd.DataFrame()
    results = []
    for _, deal in active_pipeline.iterrows():
        segment, stage, revenue = deal['market_segment'], deal['stage'], deal['net_revenue']
        base_rate = stage_rates.get(segment, {}).get(stage, stage_rates['_GLOBAL'].get(stage, config['default_conversion_rate']))
        adjusted_rate = min(base_rate * scenario['win_rate_adjustment'], 1.0)
        
        results.append({
            'deal_id': deal['deal_id'], 'deal_name': deal.get('deal_name', ''), 'market_segment': segment,
            'stage': stage, 'impl_year': deal['impl_year'], 'net_revenue': revenue,
            'conversion_rate': base_rate, 'adjusted_rate': adjusted_rate,
            'expected_revenue': revenue * adjusted_rate * scenario['deal_size_adjustment']
        })
    return pd.DataFrame(results)

### 6. Future Pipeline Forecast
def forecast_future_pipeline(baseline, config, scenario):
    if baseline is None: return pd.DataFrame()
    results = []
    for segment, metrics in baseline.items():
        if segment == '_GLOBAL': continue
        expected_revenue = metrics['revenue'] * scenario['volume_growth'] * scenario['deal_size_adjustment'] * config.get('future_conservatism', 0.85)
        results.append({'market_segment': segment, 'baseline_revenue': metrics['revenue'], 'expected_revenue': expected_revenue, 'source': 'future_projection'})
    return pd.DataFrame(results)

### 7. Backtesting and Execution
def run_backtest(df, config, scenario, backtest_date, target_impl_year):
    backtest_date = pd.to_datetime(backtest_date)
    
    # Calculate components
    stage_rates = calculate_stage_conversion_rates(df, backtest_date, target_impl_year, config)
    baseline = calculate_historical_baseline(df, backtest_date, target_impl_year, config)
    
    active_pipeline = get_active_pipeline_for_impl_year(df, backtest_date, target_impl_year, config)
    active_forecast = forecast_active_pipeline(active_pipeline, stage_rates, config, scenario)
    active_expected = active_forecast['expected_revenue'].sum() if not active_forecast.empty else 0
    
    future_forecast = forecast_future_pipeline(baseline, config, scenario)
    future_expected = future_forecast['expected_revenue'].sum() if not future_forecast.empty else 0
    
    total_forecast = active_expected + future_expected
    
    # Calculate Actuals
    outcomes = get_deal_outcomes(df)
    actual_won = outcomes[(outcomes['outcome'] == 'Won') & (outcomes['impl_year'] == target_impl_year)]
    total_actual = actual_won['net_revenue'].sum()
    variance_pct = ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0
    
    # Segment Breakdown Calculation
    actual_by_seg = actual_won.groupby('market_segment').agg({'deal_id': 'count', 'net_revenue': 'sum'}).reset_index()
    actual_by_seg.columns = ['market_segment', 'actual_deals', 'actual_revenue']
    
    forecast_by_seg = []
    all_segments = set(actual_by_seg['market_segment'].tolist())
    if not active_forecast.empty: all_segments.update(active_forecast['market_segment'].unique())
    if not future_forecast.empty: all_segments.update(future_forecast['market_segment'].unique())
    
    for seg in all_segments:
        active_exp = active_forecast[active_forecast['market_segment'] == seg]['expected_revenue'].sum() if not active_forecast.empty else 0
        future_exp = future_forecast[future_forecast['market_segment'] == seg]['expected_revenue'].sum() if not future_forecast.empty else 0
        actual_rev = actual_by_seg[actual_by_seg['market_segment'] == seg]['actual_revenue'].sum() if len(actual_by_seg[actual_by_seg['market_segment'] == seg]) > 0 else 0
        var = ((active_exp + future_exp - actual_rev) / actual_rev * 100) if actual_rev > 0 else 0
        
        forecast_by_seg.append({
            'market_segment': seg, 'actual_revenue': actual_rev,
            'active_expected': active_exp, 'future_expected': future_exp,
            'total_expected': active_exp + future_exp, 'variance_pct': var
        })
    
    return {
        'total_forecast': total_forecast, 'total_actual': total_actual, 'variance_pct': variance_pct,
        'active_forecast': active_forecast, 'future_forecast': future_forecast,
        'stage_rates': stage_rates, 'baseline': baseline, 'comparison': pd.DataFrame(forecast_by_seg),
    }

def export_results(results, config, target_impl_year):
    validation_path = Path(config['validation_export_path'])
    validation_path.mkdir(parents=True, exist_ok=True)
    results['comparison'].to_csv(validation_path / f'backtest_impl{target_impl_year}.csv', index=False)
    
    summary = {
        'model_version': 'V3-ImplYear', 'target_impl_year': target_impl_year,
        'total_forecast': float(results['total_forecast']), 'total_actual': float(results['total_actual']),
        'variance_pct': float(results['variance_pct']), 'within_target': abs(results['variance_pct']) <= 10,
        'stage_conversion_rates': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results['stage_rates'].items()},
        'config': {'future_conservatism': config['future_conservatism'], 'min_deals_for_segment': config['min_deals_for_segment']},
        'generated_at': datetime.now().isoformat()
    }
    
    with open(validation_path / f'backtest_impl{target_impl_year}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

### 8. Main
def main():
    df = load_data(CONFIG)
    scenario = SCENARIOS['base']
    results = run_backtest(df=df, config=CONFIG, scenario=scenario, backtest_date='2025-01-01', target_impl_year=2025)
    export_results(results, CONFIG, target_impl_year=2025)
    return results

if __name__ == "__main__":
    main()