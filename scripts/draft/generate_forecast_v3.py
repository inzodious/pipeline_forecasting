import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
from pandas.tseries.offsets import MonthEnd

warnings.filterwarnings('ignore')

CONFIG = {
    'data_path': 'data/fact_snapshots.csv',
    'validation_export_path': 'validation/',
    'forecast_export_path': 'exports/',
    'active_stages': ['Qualified', 'Alignment', 'Solutioning', 'Verbal'],
    'min_deals_for_segment': 10,
    'trailing_months': 12,
    'future_conservatism': 0.85,
    'default_conversion_rate': 0.25,
    'forecast_basis': 'implementation', 
}

SCENARIOS = {
    'base': {'volume_growth': 1.0, 'win_rate_adjustment': 1.0, 'deal_size_adjustment': 1.0},
    'growth': {'volume_growth': 1.15, 'win_rate_adjustment': 1.05, 'deal_size_adjustment': 1.0},
    'conservative': {'volume_growth': 0.85, 'win_rate_adjustment': 0.90, 'deal_size_adjustment': 1.0},
}

def load_data(config):
    df = pd.read_csv(config['data_path'], parse_dates=['date_snapshot', 'date_created', 'date_closed', 'date_implementation'])
    df['market_segment'] = df['market_segment'].fillna('Unknown')
    df['net_revenue'] = pd.to_numeric(df['net_revenue'], errors='coerce').fillna(0)
    
    basis_col = 'date_implementation' if config['forecast_basis'] == 'implementation' else 'date_closed'
    
    df['forecast_dt'] = df[basis_col] + MonthEnd(0)
    df['forecast_year'] = df['forecast_dt'].dt.year
    
    return df

def get_latest_state(df, as_of_date=None):
    if as_of_date:
        df = df[df['date_snapshot'] <= pd.to_datetime(as_of_date)].copy()
    return df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()

def get_deal_outcomes(df):
    latest = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    latest['outcome'] = latest['stage'].apply(lambda x: 'Won' if x == 'Closed Won' else ('Lost' if x == 'Closed Lost' else 'Open'))
    return latest

def calculate_stage_conversion_rates(df, as_of_date, target_year, config):
    as_of = pd.to_datetime(as_of_date)
    closed_deals = df[(df['date_closed'].notna()) & (df['date_closed'] <= as_of) & (df['forecast_year'] < target_year)].copy()
    
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

def calculate_historical_baseline(df, as_of_date, target_year, config):
    as_of = pd.to_datetime(as_of_date)
    closed_won = df[(df['stage'] == 'Closed Won') & (df['date_closed'].notna()) & (df['date_closed'] <= as_of)].copy()
    won_deals = closed_won.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    baseline_deals = won_deals[won_deals['forecast_year'] == (target_year - 1)]
    if len(baseline_deals) == 0: return None
    
    total_revenue = baseline_deals['net_revenue'].sum()
    segment_totals = baseline_deals.groupby('market_segment').agg({'deal_id': 'count', 'net_revenue': 'sum'}).reset_index()
    segment_totals.columns = ['segment', 'deals', 'revenue']
    segment_totals['revenue_share'] = segment_totals['revenue'] / total_revenue if total_revenue > 0 else 0
    
    baseline = {'_GLOBAL': {'forecast_year': target_year - 1, 'total_deals': len(baseline_deals), 'total_revenue': total_revenue}}
    for _, row in segment_totals.iterrows():
        baseline[row['segment']] = {'deals': row['deals'], 'revenue': row['revenue'], 'revenue_share': row['revenue_share']}
    
    return baseline

def get_active_pipeline_for_year(df, as_of_date, target_year, config):
    latest = get_latest_state(df, as_of_date)
    return latest[(latest['stage'].isin(config['active_stages'])) & (latest['forecast_year'] == target_year)].copy()

def forecast_active_pipeline(active_pipeline, stage_rates, config, scenario):
    if active_pipeline.empty: return pd.DataFrame()
    results = []
    for _, deal in active_pipeline.iterrows():
        segment, stage, revenue = deal['market_segment'], deal['stage'], deal['net_revenue']
        base_rate = stage_rates.get(segment, {}).get(stage, stage_rates['_GLOBAL'].get(stage, config['default_conversion_rate']))
        adjusted_rate = min(base_rate * scenario['win_rate_adjustment'], 1.0)
        
        results.append({
            'deal_id': deal['deal_id'], 'market_segment': segment, 'stage': stage, 
            'forecast_dt': deal['forecast_dt'], 
            'net_revenue': revenue, 'expected_revenue': revenue * adjusted_rate * scenario['deal_size_adjustment'],
            'source': 'active_pipeline'
        })
    return pd.DataFrame(results)

def forecast_future_pipeline(baseline, config, scenario, target_year):
    if baseline is None: return pd.DataFrame()
    results = []
    for segment, metrics in baseline.items():
        if segment == '_GLOBAL': continue
        
        expected_revenue_annual = metrics['revenue'] * scenario['volume_growth'] * scenario['deal_size_adjustment'] * config.get('future_conservatism', 0.85)
        
        monthly_rev = expected_revenue_annual / 12
        for month in range(1, 13):
            eom_date = pd.Timestamp(year=target_year, month=month, day=1) + MonthEnd(0)
            results.append({
                'market_segment': segment, 
                'forecast_dt': eom_date,
                'expected_revenue': monthly_rev, 
                'source': 'future_projection'
            })
            
    return pd.DataFrame(results)

def run_backtest(df, config, scenario, backtest_date, target_year):
    backtest_date = pd.to_datetime(backtest_date)
    
    stage_rates = calculate_stage_conversion_rates(df, backtest_date, target_year, config)
    baseline = calculate_historical_baseline(df, backtest_date, target_year, config)
    
    active_pipeline = get_active_pipeline_for_year(df, backtest_date, target_year, config)
    active_forecast = forecast_active_pipeline(active_pipeline, stage_rates, config, scenario)
    active_expected = active_forecast['expected_revenue'].sum() if not active_forecast.empty else 0
    
    future_forecast = forecast_future_pipeline(baseline, config, scenario, target_year)
    future_expected = future_forecast['expected_revenue'].sum() if not future_forecast.empty else 0
    
    total_forecast = active_expected + future_expected
    
    outcomes = get_deal_outcomes(df)
    actual_won = outcomes[(outcomes['outcome'] == 'Won') & (outcomes['forecast_year'] == target_year)]
    total_actual = actual_won['net_revenue'].sum()
    variance_pct = ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0
    
    all_forecasts = pd.concat([
        active_forecast[['forecast_dt', 'expected_revenue']], 
        future_forecast[['forecast_dt', 'expected_revenue']]
    ]) if not active_forecast.empty or not future_forecast.empty else pd.DataFrame(columns=['forecast_dt', 'expected_revenue'])
    
    monthly_forecast = all_forecasts.groupby('forecast_dt')['expected_revenue'].sum().reset_index()
    monthly_actual = actual_won.groupby('forecast_dt')['net_revenue'].sum().reset_index()
    
    comparison_df = pd.merge(monthly_forecast, monthly_actual, on='forecast_dt', how='outer', suffixes=('_forecast', '_actual')).fillna(0)
    comparison_df['variance_pct'] = ((comparison_df['expected_revenue'] - comparison_df['net_revenue']) / comparison_df['net_revenue'] * 100).replace([np.inf, -np.inf], 0)

    return {
        'total_forecast': total_forecast, 'total_actual': total_actual, 'variance_pct': variance_pct,
        'comparison': comparison_df.sort_values('forecast_dt'),
    }

def run_forecast(df, config, scenario, target_year):
    as_of_date = df['date_snapshot'].max()
    
    stage_rates = calculate_stage_conversion_rates(df, as_of_date, target_year, config)
    baseline = calculate_historical_baseline(df, as_of_date, target_year, config)
    
    active_pipeline = get_active_pipeline_for_year(df, as_of_date, target_year, config)
    active_forecast = forecast_active_pipeline(active_pipeline, stage_rates, config, scenario)
    future_forecast = forecast_future_pipeline(baseline, config, scenario, target_year)
    
    total_forecast = pd.concat([active_forecast, future_forecast]) if not active_forecast.empty or not future_forecast.empty else pd.DataFrame()
    return total_forecast

def export_backtest_results(results, config, target_year):
    validation_path = Path(config['validation_export_path'])
    validation_path.mkdir(parents=True, exist_ok=True)
    results['comparison'].to_csv(validation_path / f"backtest_{config['forecast_basis']}_{target_year}.csv", index=False)

def export_forecast_results(forecast_df, config, target_year):
    export_path = Path(config['forecast_export_path'])
    export_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"forecast_{config['forecast_basis']}_{target_year}.csv"
    
    summary_df = forecast_df.groupby(['forecast_dt', 'market_segment', 'source'])['expected_revenue'].sum().reset_index()
    summary_df.to_csv(export_path / filename, index=False)

def main():
    df = load_data(CONFIG)
    scenario = SCENARIOS['base']
    
    backtest_results = run_backtest(df=df, config=CONFIG, scenario=scenario, backtest_date='2025-01-01', target_year=2025)
    export_backtest_results(backtest_results, CONFIG, target_year=2025)
    
    forecast_2026 = run_forecast(df=df, config=CONFIG, scenario=scenario, target_year=2026)
    export_forecast_results(forecast_2026, CONFIG, target_year=2026)

if __name__ == "__main__":
    main()