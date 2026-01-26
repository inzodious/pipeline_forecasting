"""
Pipeline Revenue Forecasting Model V2
=====================================

A two-component model for forecasting pipeline revenue:
1. Active Pipeline Conversion: Stage-based probability model
2. Future Pipeline Projection: Historical closure baseline with conservatism adjustment

Methodology Summary:
- Uses historical stage conversion rates to value active pipeline
- Projects future closures based on trailing 12-month closure patterns
- Applies a conservatism factor (default 10%) to future projections
- Validates via backtesting against historical actuals

Target Accuracy: ±10% variance on aggregate forecasts

Author: Generated for EAP Revenue Forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PARAMETERS (Override these in Fabric pipeline)
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
    'forecast_year': 2026,
    
    'active_stages': ['Qualified', 'Alignment', 'Solutioning'],
    'min_deals_for_segment': 10,
    'trailing_months': 12,
    
    # Conservatism factor: Applied to future pipeline projections
    # Rationale: Historical YoY variance typically 10-15% 
    # 0.90 = 10% haircut on baseline projection
    'future_conservatism': 0.90,
    
    # Fallback conversion rate if insufficient historical data
    'default_conversion_rate': 0.35,
}

SCENARIOS = {
    'base': {
        'volume_growth': 1.0, 
        'win_rate_adjustment': 1.0, 
        'deal_size_adjustment': 1.0,
        'description': 'Baseline forecast using historical patterns'
    },
    'growth': {
        'volume_growth': 1.15, 
        'win_rate_adjustment': 1.05, 
        'deal_size_adjustment': 1.05,
        'description': '15% volume growth, 5% win rate improvement'
    },
    'conservative': {
        'volume_growth': 0.90, 
        'win_rate_adjustment': 0.95, 
        'deal_size_adjustment': 1.0,
        'description': '10% volume decline, 5% win rate decline'
    },
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config):
    """Load and validate snapshot data."""
    df = pd.read_csv(config['data_path'], parse_dates=['date_created', 'date_closed', 'date_snapshot'])
    df['market_segment'] = df['market_segment'].fillna('Unknown')
    df['net_revenue'] = pd.to_numeric(df['net_revenue'], errors='coerce').fillna(0)
    return df


def build_deal_summary(df, as_of_date=None):
    """
    Build deal-level summary with outcomes.
    
    If as_of_date provided, only uses snapshots up to that date (prevents data leakage).
    """
    if as_of_date:
        df = df[df['date_snapshot'] <= pd.to_datetime(as_of_date)].copy()
    
    deal_summary = df.groupby('deal_id').agg({
        'date_created': 'first',
        'date_closed': 'first',
        'net_revenue': 'last',
        'market_segment': 'first',
        'stage': lambda x: list(x.unique())
    }).reset_index()
    deal_summary.columns = ['deal_id', 'date_created', 'date_closed', 'net_revenue', 
                           'market_segment', 'stages_observed']
    
    def get_outcome(stages):
        if 'Closed Won' in stages:
            return 'Closed Won'
        elif 'Closed Lost' in stages:
            return 'Closed Lost'
        return 'Open'
    
    deal_summary['outcome'] = deal_summary['stages_observed'].apply(get_outcome)
    return deal_summary

# =============================================================================
# COMPONENT 1: STAGE CONVERSION RATES
# =============================================================================

def calculate_stage_conversion_rates(deal_summary, config):
    """
    Calculate P(Won | deal passed through stage) for each stage.
    
    Uses closed deals only to avoid bias from open deals.
    Returns segment-specific rates with global fallback.
    """
    closed = deal_summary[deal_summary['outcome'].isin(['Closed Won', 'Closed Lost'])]
    
    rates = {'_GLOBAL': {}}
    
    for stage in config['active_stages']:
        stage_deals = closed[closed['stages_observed'].apply(lambda x: stage in x)]
        won = stage_deals[stage_deals['outcome'] == 'Closed Won']
        rate = len(won) / len(stage_deals) if len(stage_deals) > 0 else config['default_conversion_rate']
        rates['_GLOBAL'][stage] = rate
    
    for segment in deal_summary['market_segment'].unique():
        seg_closed = closed[closed['market_segment'] == segment]
        rates[segment] = {}
        
        for stage in config['active_stages']:
            stage_deals = seg_closed[seg_closed['stages_observed'].apply(lambda x: stage in x)]
            
            if len(stage_deals) < config['min_deals_for_segment']:
                rates[segment][stage] = rates['_GLOBAL'][stage]
            else:
                won = stage_deals[stage_deals['outcome'] == 'Closed Won']
                rates[segment][stage] = len(won) / len(stage_deals)
    
    return rates

# =============================================================================
# COMPONENT 2: HISTORICAL CLOSURE BASELINE
# =============================================================================

def calculate_historical_closure_baseline(deal_summary, training_end_date, config):
    """
    Calculate trailing N-month closure baseline for revenue projection.
    
    Key features:
    - Uses global totals distributed by segment share (prevents thin segment amplification)
    - Only counts deals created within trailing period (avoids double-counting carryover)
    """
    training_end = pd.to_datetime(training_end_date)
    trailing_start = training_end - pd.DateOffset(months=config['trailing_months'])
    
    won = deal_summary[
        (deal_summary['outcome'] == 'Closed Won') &
        (deal_summary['date_closed'] >= trailing_start) &
        (deal_summary['date_closed'] <= training_end) &
        (deal_summary['date_created'] >= trailing_start)  # New deals only
    ].copy()
    
    if len(won) == 0:
        return None
    
    won['close_month'] = won['date_closed'].dt.to_period('M')
    n_months = max(won['close_month'].nunique(), 1)
    
    total_deals = len(won)
    total_revenue = won['net_revenue'].sum()
    
    global_monthly_deals = total_deals / n_months
    global_monthly_revenue = total_revenue / n_months
    
    segment_totals = won.groupby('market_segment').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    }).reset_index()
    segment_totals.columns = ['segment', 'deals', 'revenue']
    
    if total_deals > 0:
        segment_totals['deal_share'] = segment_totals['deals'] / total_deals
        segment_totals['revenue_share'] = segment_totals['revenue'] / total_revenue
    else:
        segment_totals['deal_share'] = 0
        segment_totals['revenue_share'] = 0
    
    baseline = {
        '_GLOBAL': {
            'avg_monthly_deals': global_monthly_deals,
            'avg_monthly_revenue': global_monthly_revenue,
            'total_deals': total_deals,
            'total_revenue': total_revenue,
            'n_months': n_months,
        }
    }
    
    for _, row in segment_totals.iterrows():
        baseline[row['segment']] = {
            'avg_monthly_deals': global_monthly_deals * row['deal_share'],
            'avg_monthly_revenue': global_monthly_revenue * row['revenue_share'],
            'deal_share': row['deal_share'],
            'revenue_share': row['revenue_share'],
        }
    
    return baseline

# =============================================================================
# ACTIVE PIPELINE FORECAST
# =============================================================================

def get_active_pipeline(df, deal_summary, as_of_date, config):
    """Get all deals in active stages as of given date."""
    as_of = pd.to_datetime(as_of_date)
    
    snapshot_dates = df[df['date_snapshot'] <= as_of]['date_snapshot'].unique()
    if len(snapshot_dates) == 0:
        return pd.DataFrame()
    
    latest_snapshot_date = max(snapshot_dates)
    latest = df[df['date_snapshot'] == latest_snapshot_date].copy()
    active = latest[latest['stage'].isin(config['active_stages'])].copy()
    
    closed_dates = deal_summary.set_index('deal_id')['date_closed'].to_dict()
    
    def is_actually_open(deal_id):
        close_date = closed_dates.get(deal_id)
        if pd.isna(close_date):
            return True
        return pd.to_datetime(close_date) > as_of
    
    active = active[active['deal_id'].apply(is_actually_open)]
    return active


def forecast_active_pipeline(active_pipeline, stage_rates, config, scenario):
    """
    Calculate expected value of active pipeline.
    
    Expected Value = Sum(Deal Value × Stage Conversion Rate × Adjustments)
    """
    if active_pipeline.empty:
        return pd.DataFrame()
    
    results = []
    for _, deal in active_pipeline.iterrows():
        segment = deal['market_segment']
        stage = deal['stage']
        revenue = deal['net_revenue']
        
        base_rate = stage_rates.get(segment, {}).get(stage)
        if base_rate is None:
            base_rate = stage_rates['_GLOBAL'].get(stage, config['default_conversion_rate'])
        
        adjusted_rate = min(base_rate * scenario['win_rate_adjustment'], 1.0)
        expected_revenue = revenue * adjusted_rate * scenario['deal_size_adjustment']
        
        results.append({
            'deal_id': deal['deal_id'],
            'market_segment': segment,
            'stage': stage,
            'net_revenue': revenue,
            'conversion_rate': base_rate,
            'adjusted_rate': adjusted_rate,
            'expected_revenue': expected_revenue,
        })
    
    return pd.DataFrame(results)

# =============================================================================
# FUTURE PIPELINE FORECAST
# =============================================================================

def forecast_future_pipeline(baseline, forecast_months, config, scenario):
    """
    Project future closures based on historical baseline.
    
    Applies:
    - Scenario volume/deal size adjustments
    - Conservatism factor (default 10% haircut)
    """
    if baseline is None:
        return pd.DataFrame()
    
    conservatism = config.get('future_conservatism', 1.0)
    
    results = []
    for month in forecast_months:
        for segment, metrics in baseline.items():
            if segment == '_GLOBAL':
                continue
            
            expected_deals = (metrics['avg_monthly_deals'] * 
                           scenario['volume_growth'] * conservatism)
            expected_revenue = (metrics['avg_monthly_revenue'] * 
                              scenario['volume_growth'] * 
                              scenario['deal_size_adjustment'] * conservatism)
            
            results.append({
                'forecast_month': str(month),
                'market_segment': segment,
                'expected_deals': expected_deals,
                'expected_revenue': expected_revenue,
            })
    
    return pd.DataFrame(results)

# =============================================================================
# AGGREGATION & OUTPUT
# =============================================================================

def aggregate_forecast(active_forecast, future_forecast):
    """Combine active and future forecasts by segment."""
    results = []
    
    active_by_seg = pd.DataFrame()
    if not active_forecast.empty:
        active_by_seg = active_forecast.groupby('market_segment').agg({
            'expected_revenue': 'sum',
            'deal_id': 'count'
        }).reset_index()
        active_by_seg.columns = ['market_segment', 'active_expected_revenue', 'active_deal_count']
    
    future_by_seg = pd.DataFrame()
    if not future_forecast.empty:
        future_by_seg = future_forecast.groupby('market_segment').agg({
            'expected_revenue': 'sum',
            'expected_deals': 'sum'
        }).reset_index()
        future_by_seg.columns = ['market_segment', 'future_expected_revenue', 'future_expected_deals']
    
    all_segments = set()
    if not active_by_seg.empty:
        all_segments.update(active_by_seg['market_segment'].tolist())
    if not future_by_seg.empty:
        all_segments.update(future_by_seg['market_segment'].tolist())
    
    for segment in all_segments:
        active_rev = 0
        active_count = 0
        future_rev = 0
        future_deals = 0
        
        if not active_by_seg.empty:
            seg_data = active_by_seg[active_by_seg['market_segment'] == segment]
            if len(seg_data) > 0:
                active_rev = seg_data['active_expected_revenue'].iloc[0]
                active_count = seg_data['active_deal_count'].iloc[0]
        
        if not future_by_seg.empty:
            seg_data = future_by_seg[future_by_seg['market_segment'] == segment]
            if len(seg_data) > 0:
                future_rev = seg_data['future_expected_revenue'].iloc[0]
                future_deals = seg_data['future_expected_deals'].iloc[0]
        
        results.append({
            'market_segment': segment,
            'active_pipeline_deals': active_count,
            'active_expected_revenue': active_rev,
            'future_expected_deals': future_deals,
            'future_expected_revenue': future_rev,
            'total_expected_revenue': active_rev + future_rev,
        })
    
    return pd.DataFrame(results)

# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(df, config, scenario, backtest_date, actuals_through):
    """
    Validate forecast accuracy against historical actuals.
    
    Key principle: Only use data available as of backtest_date.
    """
    backtest_date = pd.to_datetime(backtest_date)
    actuals_through = pd.to_datetime(actuals_through)
    
    print(f"\n{'='*70}")
    print(f"BACKTEST: Forecasting {backtest_date.year} using data through {backtest_date.date()}")
    print(f"{'='*70}")
    
    # === TRAINING DATA ===
    training_summary = build_deal_summary(df, as_of_date=backtest_date)
    training_df = df[df['date_snapshot'] <= backtest_date].copy()
    
    closed_won = len(training_summary[training_summary['outcome'] == 'Closed Won'])
    closed_lost = len(training_summary[training_summary['outcome'] == 'Closed Lost'])
    open_deals = len(training_summary[training_summary['outcome'] == 'Open'])
    
    print(f"\nTraining Data (as of {backtest_date.date()}):")
    print(f"  Closed Won: {closed_won}")
    print(f"  Closed Lost: {closed_lost}")
    print(f"  Open: {open_deals}")
    
    # === CALCULATE INPUTS ===
    stage_rates = calculate_stage_conversion_rates(training_summary, config)
    baseline = calculate_historical_closure_baseline(training_summary, backtest_date, config)
    
    print(f"\nStage Conversion Rates:")
    for stage in config['active_stages']:
        print(f"  {stage}: {stage_rates['_GLOBAL'][stage]*100:.1f}%")
    
    if baseline:
        g = baseline['_GLOBAL']
        print(f"\nHistorical Baseline (trailing {config['trailing_months']}mo):")
        print(f"  {g['avg_monthly_deals']:.1f} deals/month")
        print(f"  ${g['avg_monthly_revenue']:,.0f}/month")
        print(f"  Conservatism factor: {config['future_conservatism']:.0%}")
    
    # === ACTIVE PIPELINE ===
    active_pipeline = get_active_pipeline(training_df, training_summary, backtest_date, config)
    active_forecast = forecast_active_pipeline(active_pipeline, stage_rates, config, scenario)
    
    active_value = active_pipeline['net_revenue'].sum() if not active_pipeline.empty else 0
    active_expected = active_forecast['expected_revenue'].sum() if not active_forecast.empty else 0
    
    print(f"\nActive Pipeline:")
    print(f"  Deals: {len(active_pipeline)}")
    print(f"  Total Value: ${active_value:,.0f}")
    print(f"  Expected Value: ${active_expected:,.0f}")
    
    # === FUTURE PIPELINE ===
    forecast_months = pd.period_range(
        start=backtest_date.to_period('M'),
        end=actuals_through.to_period('M'),
        freq='M'
    )
    future_forecast = forecast_future_pipeline(baseline, forecast_months, config, scenario)
    future_expected = future_forecast['expected_revenue'].sum() if not future_forecast.empty else 0
    
    print(f"\nFuture Pipeline ({len(forecast_months)} months):")
    print(f"  Expected Revenue: ${future_expected:,.0f}")
    
    # === TOTAL FORECAST ===
    total_forecast = active_expected + future_expected
    print(f"\n{'─'*40}")
    print(f"TOTAL FORECAST: ${total_forecast:,.0f}")
    print(f"{'─'*40}")
    
    # === ACTUALS ===
    full_summary = build_deal_summary(df, as_of_date=actuals_through)
    actual_closed = full_summary[
        (full_summary['outcome'] == 'Closed Won') &
        (full_summary['date_closed'] > backtest_date) &
        (full_summary['date_closed'] <= actuals_through)
    ]
    
    carryover = actual_closed[actual_closed['date_created'] < backtest_date]
    new_deals = actual_closed[actual_closed['date_created'] >= backtest_date]
    
    total_actual = actual_closed['net_revenue'].sum()
    
    print(f"\nACTUALS:")
    print(f"  Total: {len(actual_closed)} deals, ${total_actual:,.0f}")
    print(f"  Carryover: {len(carryover)} deals, ${carryover['net_revenue'].sum():,.0f}")
    print(f"  New: {len(new_deals)} deals, ${new_deals['net_revenue'].sum():,.0f}")
    
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
    actual_by_seg = actual_closed.groupby('market_segment').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    }).reset_index()
    actual_by_seg.columns = ['market_segment', 'actual_deals', 'actual_revenue']
    
    forecast_summary = aggregate_forecast(active_forecast, future_forecast)
    
    comparison = pd.merge(actual_by_seg, forecast_summary, on='market_segment', how='outer').fillna(0)
    comparison['variance_pct'] = comparison.apply(
        lambda r: ((r['total_expected_revenue'] - r['actual_revenue']) / r['actual_revenue'] * 100)
        if r['actual_revenue'] > 0 else 0, axis=1
    )
    
    print(f"\nSegment Breakdown:")
    for _, row in comparison.iterrows():
        print(f"  {row['market_segment']}:")
        print(f"    Actual: ${row['actual_revenue']:,.0f}")
        print(f"    Forecast: ${row['total_expected_revenue']:,.0f}")
        print(f"    Variance: {row['variance_pct']:+.1f}%")
    
    return {
        'total_forecast': total_forecast,
        'total_actual': total_actual,
        'variance_pct': variance_pct,
        'comparison': comparison,
        'active_forecast': active_forecast,
        'future_forecast': future_forecast,
        'stage_rates': stage_rates,
        'baseline': baseline,
        'config_used': config,
        'scenario_used': scenario,
    }

# =============================================================================
# EXPORT
# =============================================================================

def export_results(results, config):
    """Export forecast and validation results."""
    validation_path = Path(config['validation_export_path'])
    validation_path.mkdir(parents=True, exist_ok=True)
    
    results['comparison'].to_csv(validation_path / 'backtest_results.csv', index=False)
    
    summary = {
        'model_version': 'V2',
        'total_forecast': float(results['total_forecast']),
        'total_actual': float(results['total_actual']),
        'variance_pct': float(results['variance_pct']),
        'within_target': abs(results['variance_pct']) <= 10,
        'stage_conversion_rates': {
            k: {kk: float(vv) for kk, vv in v.items()} 
            for k, v in results['stage_rates'].items()
        },
        'config': {
            'trailing_months': config['trailing_months'],
            'future_conservatism': config['future_conservatism'],
            'min_deals_for_segment': config['min_deals_for_segment'],
        },
        'generated_at': datetime.now().isoformat()
    }
    
    with open(validation_path / 'backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults exported to {validation_path}/")

# =============================================================================
# MAIN
# =============================================================================

def run_forecast():
    """Main entry point."""
    print("="*70)
    print("Pipeline Revenue Forecasting Model V2")
    print("="*70)
    
    scenario = SCENARIOS.get(SCENARIO, SCENARIOS['base'])
    print(f"\nScenario: {SCENARIO}")
    print(f"  {scenario.get('description', '')}")
    
    if GENERATE_MOCK:
        from generate_mock import generate_mock_data
        generate_mock_data(output_path=CONFIG['data_path'])
    
    df = load_data(CONFIG)
    print(f"\nLoaded {len(df):,} snapshot records")
    print(f"Date range: {df['date_snapshot'].min().date()} to {df['date_snapshot'].max().date()}")
    
    if RUN_BACKTEST:
        results = run_backtest(df, CONFIG, scenario, BACKTEST_DATE, ACTUALS_THROUGH)
        export_results(results, CONFIG)
        return results
    
    # Production forecast (non-backtest mode)
    deal_summary = build_deal_summary(df)
    stage_rates = calculate_stage_conversion_rates(deal_summary, CONFIG)
    training_end = df['date_snapshot'].max()
    baseline = calculate_historical_closure_baseline(deal_summary, training_end, CONFIG)
    
    active_pipeline = get_active_pipeline(df, deal_summary, training_end, CONFIG)
    active_forecast = forecast_active_pipeline(active_pipeline, stage_rates, CONFIG, scenario)
    
    forecast_months = pd.period_range(
        start=f"{CONFIG['forecast_year']}-01",
        end=f"{CONFIG['forecast_year']}-12",
        freq='M'
    )
    future_forecast = forecast_future_pipeline(baseline, forecast_months, CONFIG, scenario)
    
    total = (active_forecast['expected_revenue'].sum() + 
             future_forecast['expected_revenue'].sum())
    
    print(f"\n{CONFIG['forecast_year']} Forecast: ${total:,.0f}")
    
    return {
        'active_forecast': active_forecast,
        'future_forecast': future_forecast,
        'total_forecast': total,
        'stage_rates': stage_rates,
        'baseline': baseline,
    }


if __name__ == "__main__":
    run_forecast()