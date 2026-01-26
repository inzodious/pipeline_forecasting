"""
Pipeline Revenue Forecasting Model V3
=====================================

FUNDAMENTAL CHANGE: Forecasts by IMPLEMENTATION YEAR, not close date.

When management asks "What's our 2025 forecast?", they mean:
"How much revenue will we win that has a 2025 implementation date?"

This is different from "How much will close in 2025?" because:
- A deal closing in Dec 2024 might have Jan 2025 implementation
- A deal closing in Nov 2025 might have Jan 2026 implementation

Model Components:
1. Active Pipeline: Deals in active stages with target implementation year
2. Future Pipeline: Projected new deals that will have target implementation year

Author: Pipeline Forecasting System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_path': 'data/fact_snapshots.csv',
    'validation_export_path': 'validation/',
    
    # Active stages (deals we can still win)
    'active_stages': ['Qualified', 'Alignment', 'Solutioning', 'Verbal'],
    
    # Minimum deals needed for segment-specific rates
    'min_deals_for_segment': 10,
    
    # Historical lookback for baseline calculation
    'trailing_months': 12,
    
    # Conservatism factor for future pipeline (accounts for YoY variance)
    'future_conservatism': 0.85,
    
    # Default conversion rate when insufficient data
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

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config):
    """Load and parse snapshot data."""
    df = pd.read_csv(
        config['data_path'], 
        parse_dates=['date_snapshot', 'date_created', 'date_closed', 'date_implementation']
    )
    
    df['market_segment'] = df['market_segment'].fillna('Unknown')
    df['net_revenue'] = pd.to_numeric(df['net_revenue'], errors='coerce').fillna(0)
    df['impl_year'] = df['date_implementation'].dt.year
    
    return df


def get_latest_state(df, as_of_date=None):
    """
    Get the latest state of each deal as of a given date.
    This is the "point-in-time" view for backtesting.
    """
    if as_of_date:
        df = df[df['date_snapshot'] <= pd.to_datetime(as_of_date)].copy()
    
    latest = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    return latest


def get_deal_outcomes(df):
    """
    Get final outcome for each deal (using all available data).
    Used for calculating actual results in backtesting.
    """
    latest = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    def classify_outcome(stage):
        if stage == 'Closed Won':
            return 'Won'
        elif stage == 'Closed Lost':
            return 'Lost'
        return 'Open'
    
    latest['outcome'] = latest['stage'].apply(classify_outcome)
    return latest

# =============================================================================
# STAGE CONVERSION RATES (BY IMPLEMENTATION YEAR)
# =============================================================================

def calculate_stage_conversion_rates(df, as_of_date, target_impl_year, config):
    """
    Calculate P(Won | stage, implementation_year).
    
    CRITICAL: Only use deals with implementation years in the PAST
    to avoid data leakage. We train on impl_year < target_impl_year.
    """
    as_of = pd.to_datetime(as_of_date)
    
    # Get deals that closed BEFORE as_of_date with PAST implementation years
    closed_deals = df[
        (df['date_closed'].notna()) &
        (df['date_closed'] <= as_of) &
        (df['impl_year'] < target_impl_year)  # Only past implementation years
    ].copy()
    
    # Get final state of these deals
    final_state = closed_deals.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    final_state = final_state[final_state['stage'].isin(['Closed Won', 'Closed Lost'])]
    
    # Get ALL stages each deal passed through
    deal_stages = closed_deals.groupby('deal_id')['stage'].apply(lambda x: set(x)).reset_index()
    deal_stages.columns = ['deal_id', 'stages_passed']
    
    final_state = final_state.merge(deal_stages, on='deal_id', how='left')
    final_state['is_won'] = final_state['stage'] == 'Closed Won'
    
    rates = {'_GLOBAL': {}}
    
    # Calculate global rates
    for stage in config['active_stages']:
        through_stage = final_state[final_state['stages_passed'].apply(lambda x: stage in x if x else False)]
        if len(through_stage) >= 5:
            rate = through_stage['is_won'].mean()
        else:
            rate = config['default_conversion_rate']
        rates['_GLOBAL'][stage] = rate
    
    # Calculate segment-specific rates
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

# =============================================================================
# HISTORICAL BASELINE (BY IMPLEMENTATION YEAR)
# =============================================================================

def calculate_historical_baseline(df, as_of_date, target_impl_year, config):
    """
    Calculate historical closure patterns for deals with specific implementation years.
    
    Key insight: For forecasting 2025 implementation year revenue, we look at
    how much 2024 implementation year revenue was won, 2023 impl year, etc.
    """
    as_of = pd.to_datetime(as_of_date)
    
    # Get closed won deals before as_of_date
    closed_won = df[
        (df['stage'] == 'Closed Won') &
        (df['date_closed'].notna()) &
        (df['date_closed'] <= as_of)
    ].copy()
    
    # Get unique deals (latest snapshot)
    won_deals = closed_won.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    # Look at the PREVIOUS implementation year for baseline
    # If forecasting 2025, look at 2024 impl year deals
    baseline_impl_year = target_impl_year - 1
    
    baseline_deals = won_deals[won_deals['impl_year'] == baseline_impl_year]
    
    if len(baseline_deals) == 0:
        print(f"  WARNING: No closed deals found with impl_year={baseline_impl_year}")
        return None
    
    total_revenue = baseline_deals['net_revenue'].sum()
    total_deals = len(baseline_deals)
    
    # Segment breakdown
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

# =============================================================================
# ACTIVE PIPELINE FORECAST
# =============================================================================

def get_active_pipeline_for_impl_year(df, as_of_date, target_impl_year, config):
    """
    Get deals in active stages with the target implementation year.
    
    This is the key filter: only count pipeline where impl_year matches target.
    """
    latest = get_latest_state(df, as_of_date)
    
    active = latest[
        (latest['stage'].isin(config['active_stages'])) &
        (latest['impl_year'] == target_impl_year)
    ].copy()
    
    return active


def forecast_active_pipeline(active_pipeline, stage_rates, config, scenario):
    """
    Calculate expected value of active pipeline.
    """
    if active_pipeline.empty:
        return pd.DataFrame()
    
    results = []
    for _, deal in active_pipeline.iterrows():
        segment = deal['market_segment']
        stage = deal['stage']
        revenue = deal['net_revenue']
        
        # Get conversion rate (segment-specific or global fallback)
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

# =============================================================================
# FUTURE PIPELINE FORECAST
# =============================================================================

def forecast_future_pipeline(baseline, config, scenario):
    """
    Project future deals based on historical baseline.
    
    Logic: If we won $X of 2024 impl year deals, we expect to win 
    approximately $X * conservatism * growth of 2025 impl year deals
    from pipeline not yet created.
    """
    if baseline is None:
        return pd.DataFrame()
    
    conservatism = config.get('future_conservatism', 0.85)
    
    results = []
    for segment, metrics in baseline.items():
        if segment == '_GLOBAL':
            continue
        
        # Project based on prior year, adjusted for scenario and conservatism
        expected_revenue = (
            metrics['revenue'] * 
            scenario['volume_growth'] * 
            scenario['deal_size_adjustment'] * 
            conservatism
        )
        
        results.append({
            'market_segment': segment,
            'baseline_revenue': metrics['revenue'],
            'expected_revenue': expected_revenue,
            'source': 'future_projection'
        })
    
    return pd.DataFrame(results)

# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(df, config, scenario, backtest_date, target_impl_year):
    """
    Backtest the model by forecasting a past implementation year.
    
    Example: On 2025-01-01, forecast impl_year=2025, then compare to actuals.
    """
    backtest_date = pd.to_datetime(backtest_date)
    
    print(f"\n{'='*70}")
    print(f"BACKTEST: Implementation Year {target_impl_year}")
    print(f"Forecast Date: {backtest_date.date()}")
    print(f"{'='*70}")
    
    # === STAGE CONVERSION RATES ===
    stage_rates = calculate_stage_conversion_rates(df, backtest_date, target_impl_year, config)
    
    print(f"\nStage Conversion Rates (from impl_year < {target_impl_year}):")
    for stage in config['active_stages']:
        rate = stage_rates['_GLOBAL'].get(stage, 0)
        print(f"  {stage}: {rate*100:.1f}%")
    
    # === HISTORICAL BASELINE ===
    baseline = calculate_historical_baseline(df, backtest_date, target_impl_year, config)
    
    if baseline:
        g = baseline['_GLOBAL']
        print(f"\nHistorical Baseline (impl_year={g['impl_year']}):")
        print(f"  Total Won: {g['total_deals']} deals, ${g['total_revenue']:,.0f}")
        print(f"  Conservatism: {config['future_conservatism']:.0%}")
    
    # === ACTIVE PIPELINE ===
    active_pipeline = get_active_pipeline_for_impl_year(df, backtest_date, target_impl_year, config)
    active_forecast = forecast_active_pipeline(active_pipeline, stage_rates, config, scenario)
    
    active_value = active_pipeline['net_revenue'].sum() if not active_pipeline.empty else 0
    active_expected = active_forecast['expected_revenue'].sum() if not active_forecast.empty else 0
    active_count = len(active_pipeline)
    
    print(f"\nActive Pipeline (impl_year={target_impl_year}):")
    print(f"  Deals: {active_count}")
    print(f"  Total Value: ${active_value:,.0f}")
    print(f"  Expected Value: ${active_expected:,.0f}")
    
    if not active_pipeline.empty:
        print(f"\n  By Segment:")
        for seg in active_pipeline['market_segment'].unique():
            seg_deals = active_pipeline[active_pipeline['market_segment'] == seg]
            seg_forecast = active_forecast[active_forecast['market_segment'] == seg] if not active_forecast.empty else pd.DataFrame()
            seg_exp = seg_forecast['expected_revenue'].sum() if not seg_forecast.empty else 0
            print(f"    {seg}: {len(seg_deals)} deals, ${seg_deals['net_revenue'].sum():,.0f} value, ${seg_exp:,.0f} expected")
    
    # === FUTURE PIPELINE ===
    future_forecast = forecast_future_pipeline(baseline, config, scenario)
    future_expected = future_forecast['expected_revenue'].sum() if not future_forecast.empty else 0
    
    print(f"\nFuture Pipeline Projection:")
    print(f"  Expected Revenue: ${future_expected:,.0f}")
    
    # === TOTAL FORECAST ===
    total_forecast = active_expected + future_expected
    
    print(f"\n{'─'*40}")
    print(f"TOTAL FORECAST: ${total_forecast:,.0f}")
    print(f"  Active Pipeline: ${active_expected:,.0f}")
    print(f"  Future Deals: ${future_expected:,.0f}")
    print(f"{'─'*40}")
    
    # === ACTUALS ===
    # Get all deals with target implementation year that ultimately closed won
    outcomes = get_deal_outcomes(df)
    actual_won = outcomes[
        (outcomes['outcome'] == 'Won') &
        (outcomes['impl_year'] == target_impl_year)
    ]
    
    total_actual = actual_won['net_revenue'].sum()
    actual_deals = len(actual_won)
    
    # Split: which were in active pipeline vs created later
    active_deal_ids = set(active_pipeline['deal_id']) if not active_pipeline.empty else set()
    actual_from_active = actual_won[actual_won['deal_id'].isin(active_deal_ids)]
    actual_from_future = actual_won[~actual_won['deal_id'].isin(active_deal_ids)]
    
    print(f"\nACTUALS (impl_year={target_impl_year}):")
    print(f"  Total Won: {actual_deals} deals, ${total_actual:,.0f}")
    print(f"  From Active Pipeline: {len(actual_from_active)} deals, ${actual_from_active['net_revenue'].sum():,.0f}")
    print(f"  From Future Deals: {len(actual_from_future)} deals, ${actual_from_future['net_revenue'].sum():,.0f}")
    
    # === VARIANCE ===
    variance_pct = ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0
    
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
    
    # === SEGMENT BREAKDOWN ===
    print(f"\nSegment Breakdown:")
    
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
        
        print(f"  {seg}:")
        print(f"    Actual: ${actual_rev:,.0f}")
        print(f"    Forecast: ${active_exp + future_exp:,.0f} (active: ${active_exp:,.0f}, future: ${future_exp:,.0f})")
        print(f"    Variance: {var:+.1f}%")
        
        forecast_by_seg.append({
            'market_segment': seg,
            'actual_revenue': actual_rev,
            'active_expected': active_exp,
            'future_expected': future_exp,
            'total_expected': active_exp + future_exp,
            'variance_pct': var
        })
    
    # === ACTIVE PIPELINE CONVERSION ANALYSIS ===
    if not active_pipeline.empty and len(actual_from_active) > 0:
        print(f"\nActive Pipeline Conversion Analysis:")
        
        for seg in active_pipeline['market_segment'].unique():
            seg_active = active_pipeline[active_pipeline['market_segment'] == seg]
            seg_won = actual_from_active[actual_from_active['market_segment'] == seg]
            
            if len(seg_active) > 0:
                conv_rate = len(seg_won) / len(seg_active) * 100
                value_conv = seg_won['net_revenue'].sum() / seg_active['net_revenue'].sum() * 100 if seg_active['net_revenue'].sum() > 0 else 0
                
                print(f"  {seg}:")
                print(f"    Pipeline: {len(seg_active)} deals, ${seg_active['net_revenue'].sum():,.0f}")
                print(f"    Won: {len(seg_won)} deals, ${seg_won['net_revenue'].sum():,.0f}")
                print(f"    Deal Conversion: {conv_rate:.1f}%")
                print(f"    Value Conversion: {value_conv:.1f}%")
    
    return {
        'total_forecast': total_forecast,
        'total_actual': total_actual,
        'variance_pct': variance_pct,
        'active_forecast': active_forecast,
        'future_forecast': future_forecast,
        'stage_rates': stage_rates,
        'baseline': baseline,
        'comparison': pd.DataFrame(forecast_by_seg),
    }


def export_results(results, config, target_impl_year):
    """Export validation results."""
    validation_path = Path(config['validation_export_path'])
    validation_path.mkdir(parents=True, exist_ok=True)
    
    results['comparison'].to_csv(validation_path / f'backtest_impl{target_impl_year}.csv', index=False)
    
    summary = {
        'model_version': 'V3-ImplYear',
        'target_impl_year': target_impl_year,
        'total_forecast': float(results['total_forecast']),
        'total_actual': float(results['total_actual']),
        'variance_pct': float(results['variance_pct']),
        'within_target': abs(results['variance_pct']) <= 10,
        'stage_conversion_rates': {
            k: {kk: float(vv) for kk, vv in v.items()} 
            for k, v in results['stage_rates'].items()
        },
        'config': {
            'future_conservatism': config['future_conservatism'],
            'min_deals_for_segment': config['min_deals_for_segment'],
        },
        'generated_at': datetime.now().isoformat()
    }
    
    with open(validation_path / f'backtest_impl{target_impl_year}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults exported to {validation_path}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("Pipeline Revenue Forecasting Model V3")
    print("Forecasting by IMPLEMENTATION YEAR")
    print("="*70)
    
    # Load data
    df = load_data(CONFIG)
    print(f"\nLoaded {len(df):,} snapshot records")
    print(f"Date range: {df['date_snapshot'].min().date()} to {df['date_snapshot'].max().date()}")
    print(f"Implementation years: {sorted(df['impl_year'].dropna().unique())}")
    
    # Backtest: Forecast 2025 implementation year as of 2025-01-01
    scenario = SCENARIOS['base']
    
    results = run_backtest(
        df=df,
        config=CONFIG,
        scenario=scenario,
        backtest_date='2025-01-01',
        target_impl_year=2025
    )
    
    export_results(results, CONFIG, target_impl_year=2025)
    
    return results


if __name__ == "__main__":
    main()