import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'data_path': 'data/fact_snapshots.csv',
    'validation_path': 'validation/',
    'export_path': 'exports/',
    'active_stages': ['Qualified', 'Alignment', 'Solutioning', 'Verbal'],
    'min_deals_for_segment': 10,
    'trailing_months': 12,
    'future_conservatism': 0.85,
    'default_conversion_rate': 0.25,
}

SCENARIOS = {
    'base': {
        'volume_growth': 1.0, 
        'win_rate_adjustment': 1.0, 
        'deal_size_adjustment': 1.0,
    },
    'growth': {
        'volume_growth': 1.15, 
        'win_rate_adjustment': 1.05, 
        'deal_size_adjustment': 1.0,
    },
    'conservative': {
        'volume_growth': 0.85, 
        'win_rate_adjustment': 0.90, 
        'deal_size_adjustment': 1.0,
    },
}

def load_data(config):
    df = pd.read_csv(
        config['data_path'], 
        parse_dates=['date_snapshot', 'date_created', 'date_closed', 'date_implementation']
    )
    
    df['market_segment'] = df['market_segment'].fillna('Unknown')
    df['net_revenue'] = pd.to_numeric(df['net_revenue'], errors='coerce').fillna(0)
    df['impl_year'] = df['date_implementation'].dt.year
    
    return df

def get_latest_state(df, as_of_date=None):
    if as_of_date:
        df = df[df['date_snapshot'] <= pd.to_datetime(as_of_date)].copy()
    
    latest = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    return latest

def get_deal_outcomes(df):
    latest = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    def classify_outcome(stage):
        if stage == 'Closed Won':
            return 'Won'
        elif stage == 'Closed Lost':
            return 'Lost'
        return 'Open'
    
    latest['outcome'] = latest['stage'].apply(classify_outcome)
    return latest

def calculate_stage_conversion_rates(df, as_of_date, target_impl_year, config):
    as_of = pd.to_datetime(as_of_date)
    
    closed_deals = df[
        (df['date_closed'].notna()) &
        (df['date_closed'] <= as_of) &
        (df['impl_year'] < target_impl_year)
    ].copy()
    
    final_state = closed_deals.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    final_state = final_state[final_state['stage'].isin(['Closed Won', 'Closed Lost'])]
    
    deal_stages = closed_deals.groupby('deal_id')['stage'].apply(lambda x: set(x)).reset_index()
    deal_stages.columns = ['deal_id', 'stages_passed']
    
    final_state = final_state.merge(deal_stages, on='deal_id', how='left')
    final_state['is_won'] = final_state['stage'] == 'Closed Won'
    
    rates = {'_GLOBAL': {}}
    
    for stage in config['active_stages']:
        through_stage = final_state[final_state['stages_passed'].apply(lambda x: stage in x if x else False)]
        if len(through_stage) >= 5:
            rate = through_stage['is_won'].mean()
        else:
            rate = config['default_conversion_rate']
        rates['_GLOBAL'][stage] = rate
    
    for segment in final_state['market_segment'].unique():
        seg_deals = final_state[final_state['market_segment'] == segment]
        rates[segment] = {}
        
        for stage in config['active_stages']:
            through_stage = seg_deals[seg_deals['stages_passed'].apply(lambda x: stage in x if x else False)]
            
            if len(through_stage) >= config['min_deals_for_segment']:
                rates[segment][stage] = through_stage['is_won'].mean()
            else:
                rates[segment][stage] = rates['_GLOBAL'][stage]
    
    return rates

def calculate_historical_baseline(df, as_of_date, target_impl_year, config):
    as_of = pd.to_datetime(as_of_date)
    
    closed_won = df[
        (df['stage'] == 'Closed Won') &
        (df['date_closed'].notna()) &
        (df['date_closed'] <= as_of)
    ].copy()
    
    won_deals = closed_won.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    baseline_impl_year = target_impl_year - 1
    
    baseline_deals = won_deals[won_deals['impl_year'] == baseline_impl_year]
    
    if len(baseline_deals) == 0:
        print(f"  WARNING: No closed deals found with impl_year={baseline_impl_year}")
        return None
    
    total_revenue = baseline_deals['net_revenue'].sum()
    total_deals = len(baseline_deals)
    
    segment_totals = baseline_deals.groupby('market_segment').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    }).reset_index()
    segment_totals.columns = ['segment', 'deals', 'revenue']
    segment_totals['revenue_share'] = segment_totals['revenue'] / total_revenue if total_revenue > 0 else 0
    
    baseline = {
        '_GLOBAL': {
            'impl_year': baseline_impl_year,
            'total_deals': total_deals,
            'total_revenue': total_revenue,
        }
    }
    
    for _, row in segment_totals.iterrows():
        baseline[row['segment']] = {
            'deals': row['deals'],
            'revenue': row['revenue'],
            'revenue_share': row['revenue_share'],
        }
    
    return baseline

def get_active_pipeline_for_impl_year(df, as_of_date, target_impl_year, config):
    latest = get_latest_state(df, as_of_date)
    
    active = latest[
        (latest['stage'].isin(config['active_stages'])) &
        (latest['impl_year'] == target_impl_year)
    ].copy()
    
    return active

def forecast_active_pipeline(active_pipeline, stage_rates, config, scenario):
    if active_pipeline.empty:
        return pd.DataFrame()
    
    results = []
    for _, deal in active_pipeline.iterrows():
        segment = deal['market_segment']
        stage = deal['stage']
        revenue = deal['net_revenue']
        
        base_rate = stage_rates.get(segment, {}).get(stage)
        if base_rate is None or pd.isna(base_rate):
            base_rate = stage_rates['_GLOBAL'].get(stage, config['default_conversion_rate'])
        
        adjusted_rate = min(base_rate * scenario['win_rate_adjustment'], 1.0)
        expected_revenue = revenue * adjusted_rate * scenario['deal_size_adjustment']
        
        results.append({
            'deal_id': deal['deal_id'],
            'deal_name': deal.get('deal_name', ''),
            'market_segment': segment,
            'stage': stage,
            'impl_year': deal['impl_year'],
            'net_revenue': revenue,
            'conversion_rate': base_rate,
            'adjusted_rate': adjusted_rate,
            'expected_revenue': expected_revenue,
        })
    
    return pd.DataFrame(results)

def forecast_future_pipeline(baseline, config, scenario):
    if baseline is None:
        return pd.DataFrame()
    
    conservatism = config.get('future_conservatism', 0.85)
    
    results = []
    for segment, metrics in baseline.items():
        if segment == '_GLOBAL':
            continue
        
        expected_revenue = (
            metrics['revenue'] * scenario['volume_growth'] * scenario['deal_size_adjustment'] * conservatism
        )
        
        results.append({
            'market_segment': segment,
            'baseline_revenue': metrics['revenue'],
            'expected_revenue': expected_revenue,
            'source': 'future_projection'
        })
    
    return pd.DataFrame(results)

def run_backtest(df, config, scenario, backtest_date, target_impl_year):
    backtest_date = pd.to_datetime(backtest_date)
    
    print(f"\n{'='*70}")
    print(f"RUNNING MODEL: Implementation Year {target_impl_year}")
    print(f"Data Cut-off Date: {backtest_date.date()}")
    print(f"{'='*70}")
    
    stage_rates = calculate_stage_conversion_rates(df, backtest_date, target_impl_year, config)
    baseline = calculate_historical_baseline(df, backtest_date, target_impl_year, config)
    
    active_pipeline = get_active_pipeline_for_impl_year(df, backtest_date, target_impl_year, config)
    active_forecast = forecast_active_pipeline(active_pipeline, stage_rates, config, scenario)
    
    active_value = active_pipeline['net_revenue'].sum() if not active_pipeline.empty else 0
    active_expected = active_forecast['expected_revenue'].sum() if not active_forecast.empty else 0
    active_count = len(active_pipeline)
    
    print(f"\nActive Pipeline (impl_year={target_impl_year}):")
    print(f"  Deals: {active_count}")
    print(f"  Total Value: ${active_value:,.0f}")
    print(f"  Expected Value: ${active_expected:,.0f}")
    
    future_forecast = forecast_future_pipeline(baseline, config, scenario)
    future_expected = future_forecast['expected_revenue'].sum() if not future_forecast.empty else 0
    
    print(f"\nFuture Pipeline Projection:")
    print(f"  Expected Revenue: ${future_expected:,.0f}")
    
    total_forecast = active_expected + future_expected
    
    print(f"\n{'─'*40}")
    print(f"TOTAL FORECAST: ${total_forecast:,.0f}")
    print(f"  Active Pipeline: ${active_expected:,.0f}")
    print(f"  Future Deals: ${future_expected:,.0f}")
    print(f"{'─'*40}")
    
    outcomes = get_deal_outcomes(df)
    actual_won = outcomes[
        (outcomes['outcome'] == 'Won') &
        (outcomes['impl_year'] == target_impl_year)
    ]
    
    total_actual = actual_won['net_revenue'].sum()
    actual_deals = len(actual_won)
    
    if total_actual > 0:
        print(f"\nACTUALS (impl_year={target_impl_year}):")
        print(f"  Total Won: {actual_deals} deals, ${total_actual:,.0f}")
        
        variance_pct = ((total_forecast - total_actual) / total_actual * 100)
        
        print(f"\n{'='*70}")
        print(f"VARIANCE: {variance_pct:+.1f}%")
        print(f"  Forecast: ${total_forecast:,.0f}")
        print(f"  Actual:   ${total_actual:,.0f}")
        print(f"  Delta:    ${total_forecast - total_actual:+,.0f}")
        
        if abs(variance_pct) <= 10:
            print(f"  Status:   ✓ WITHIN TARGET (±10%)")
        else:
            print(f"  Status:   ✗ OUTSIDE TARGET (±10%)")
        print(f"{'='*70}")
    else:
        print(f"\nNOTE: No actuals available yet for {target_impl_year} (Forward Forecast).")

    actual_by_seg = actual_won.groupby('market_segment').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    }).reset_index()
    actual_by_seg.columns = ['market_segment', 'actual_deals', 'actual_revenue']
    
    forecast_by_seg = []
    all_segments = set(actual_by_seg['market_segment'].tolist())
    
    if not active_forecast.empty:
        all_segments.update(active_forecast['market_segment'].unique())
    if not future_forecast.empty:
        all_segments.update(future_forecast['market_segment'].unique())
    
    for seg in all_segments:
        active_exp = active_forecast[active_forecast['market_segment'] == seg]['expected_revenue'].sum() if not active_forecast.empty else 0
        future_exp = future_forecast[future_forecast['market_segment'] == seg]['expected_revenue'].sum() if not future_forecast.empty else 0
        actual_rev = actual_by_seg[actual_by_seg['market_segment'] == seg]['actual_revenue'].sum() if len(actual_by_seg[actual_by_seg['market_segment'] == seg]) > 0 else 0
        
        var = ((active_exp + future_exp - actual_rev) / actual_rev * 100) if actual_rev > 0 else 0
        
        forecast_by_seg.append({
            'market_segment': seg,
            'actual_revenue': actual_rev,
            'active_expected': active_exp,
            'future_expected': future_exp,
            'total_expected': active_exp + future_exp,
            'variance_pct': var
        })
    
    return {
        'total_forecast': total_forecast,
        'total_actual': total_actual,
        'active_forecast': active_forecast,
        'future_forecast': future_forecast,
        'stage_rates': stage_rates,
        'baseline': baseline,
        'comparison': pd.DataFrame(forecast_by_seg),
    }

def export_results(results, config, target_impl_year, is_backtest=False):
    # Select path based on run type
    base_path = config['validation_path'] if is_backtest else config['export_path']
    export_path = Path(base_path)
    export_path.mkdir(parents=True, exist_ok=True)
    
    file_name = f'forecast_{target_impl_year}.csv'
    results['comparison'].to_csv(export_path / file_name, index=False)
    
    assumptions = {
        'target_impl_year': target_impl_year,
        'run_type': 'backtest' if is_backtest else 'actual_forecast',
        'model_version': 'V3-ImplYear',
        'generated_at': datetime.now().isoformat(),
        'config': {
            'future_conservatism': config['future_conservatism'],
            'min_deals_for_segment': config['min_deals_for_segment'],
        },
        'assumptions': {
            'stage_conversion_rates': {
                k: {kk: float(vv) for kk, vv in v.items()} 
                for k, v in results['stage_rates'].items()
            },
            'historical_baseline': results['baseline']
        }
    }
    
    with open(export_path / f'assumptions_{target_impl_year}.json', 'w') as f:
        json.dump(assumptions, f, indent=2, default=str)
    
    run_label = "Backtest" if is_backtest else "Actual Forecast"
    print(f"\n{run_label} for {target_impl_year} exported to {export_path}/{file_name}")
    print(f"Assumptions for {target_impl_year} exported to {export_path}/assumptions_{target_impl_year}.json")

def main():
    print("="*70)
    print("Pipeline Revenue Forecasting Model V3")
    print("Forecasting by IMPLEMENTATION YEAR")
    print("="*70)
    
    df = load_data(CONFIG)
    latest_snapshot = df['date_snapshot'].max()
    print(f"\nLoaded {len(df):,} snapshot records")
    print(f"Latest Data Snapshot: {latest_snapshot.date()}")
    
    scenario = SCENARIOS['base']
    
    print("\n>>> STAGE 1: RUNNING 2025 BACKTEST")
    results_2025 = run_backtest(
        df=df,
        config=CONFIG,
        scenario=scenario,
        backtest_date='2025-01-01',
        target_impl_year=2025
    )
    # is_backtest=True directs to /validation/
    export_results(results_2025, CONFIG, target_impl_year=2025, is_backtest=True)

    print("\n>>> STAGE 2: RUNNING 2026 FORWARD FORECAST")
    results_2026 = run_backtest(
        df=df,
        config=CONFIG,
        scenario=scenario,
        backtest_date=latest_snapshot,
        target_impl_year=2026
    )
    # is_backtest=False (default) directs to /exports/
    export_results(results_2026, CONFIG, target_impl_year=2026, is_backtest=False)
    
    return results_2026

if __name__ == "__main__":
    main()