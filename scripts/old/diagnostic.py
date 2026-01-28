"""
Pipeline Data Diagnostic Script
================================

Run this BEFORE the forecast model to understand your data patterns.
This will help identify issues and calibrate the model parameters.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def run_diagnostics(data_path, backtest_date='2025-01-01', target_impl_year=2025):
    """
    Run comprehensive diagnostics on pipeline data.
    """
    print("="*70)
    print("PIPELINE DATA DIAGNOSTICS")
    print("="*70)
    
    # Load data
    df = pd.read_csv(
        data_path,
        parse_dates=['date_snapshot', 'date_created', 'date_closed', 'date_implementation']
    )
    
    df['impl_year'] = df['date_implementation'].dt.year
    backtest_dt = pd.to_datetime(backtest_date)
    
    print(f"\nData Overview:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique deals: {df['deal_id'].nunique():,}")
    print(f"  Snapshot range: {df['date_snapshot'].min().date()} to {df['date_snapshot'].max().date()}")
    
    # Get latest state of each deal (all time)
    latest_all = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    print(f"\n{'='*70}")
    print("1. STAGE DISTRIBUTION")
    print("="*70)
    print(latest_all['stage'].value_counts())
    
    print(f"\n{'='*70}")
    print("2. IMPLEMENTATION YEAR DISTRIBUTION")
    print("="*70)
    print(latest_all['impl_year'].value_counts().sort_index())
    
    print(f"\n{'='*70}")
    print(f"3. CLOSED WON BY IMPLEMENTATION YEAR")
    print("="*70)
    won = latest_all[latest_all['stage'] == 'Closed Won']
    won_by_impl = won.groupby('impl_year').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    })
    won_by_impl.columns = ['deals', 'revenue']
    print(won_by_impl)
    
    print(f"\n{'='*70}")
    print("4. IMPLEMENTATION YEAR vs CLOSE DATE YEAR CROSS-TAB")
    print("="*70)
    won = won.copy()
    won['close_year'] = won['date_closed'].dt.year
    cross = pd.crosstab(
        won['close_year'], 
        won['impl_year'], 
        values=won['net_revenue'], 
        aggfunc='sum',
        margins=True
    )
    print(cross.fillna(0).astype(int))
    
    print(f"\n{'='*70}")
    print(f"5. ACTIVE PIPELINE AT {backtest_date} BY IMPLEMENTATION YEAR")
    print("="*70)
    
    # Get state at backtest date
    df_at_backtest = df[df['date_snapshot'] <= backtest_dt]
    latest_at_backtest = df_at_backtest.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    active_stages = ['Qualified', 'Alignment', 'Solutioning', 'Verbal']
    active = latest_at_backtest[latest_at_backtest['stage'].isin(active_stages)]
    
    active_by_impl = active.groupby('impl_year').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    })
    active_by_impl.columns = ['deals', 'pipeline_value']
    print(active_by_impl)
    
    print(f"\n{'='*70}")
    print(f"6. ACTIVE PIPELINE OUTCOMES (impl_year={target_impl_year})")
    print("="*70)
    
    target_active = active[active['impl_year'] == target_impl_year]
    target_active_ids = set(target_active['deal_id'])
    
    # Get final outcomes for these deals
    final_outcomes = latest_all[latest_all['deal_id'].isin(target_active_ids)].copy()
    
    def classify(stage):
        if stage == 'Closed Won': return 'Won'
        elif stage == 'Closed Lost': return 'Lost'
        return 'Still Open'
    
    final_outcomes['outcome'] = final_outcomes['stage'].apply(classify)
    
    print(f"\nTarget Active Pipeline (impl_year={target_impl_year}):")
    print(f"  Total deals: {len(target_active)}")
    print(f"  Total value: ${target_active['net_revenue'].sum():,.0f}")
    
    print(f"\nFinal Outcomes:")
    outcome_summary = final_outcomes.groupby('outcome').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    })
    outcome_summary.columns = ['deals', 'revenue']
    outcome_summary['pct_deals'] = (outcome_summary['deals'] / len(final_outcomes) * 100).round(1)
    outcome_summary['pct_value'] = (outcome_summary['revenue'] / final_outcomes['net_revenue'].sum() * 100).round(1)
    print(outcome_summary)
    
    print(f"\n{'='*70}")
    print(f"7. SEGMENT BREAKDOWN FOR impl_year={target_impl_year}")
    print("="*70)
    
    for segment in target_active['market_segment'].unique():
        seg_active = target_active[target_active['market_segment'] == segment]
        seg_outcomes = final_outcomes[final_outcomes['market_segment'] == segment]
        
        seg_won = seg_outcomes[seg_outcomes['outcome'] == 'Won']
        seg_lost = seg_outcomes[seg_outcomes['outcome'] == 'Lost']
        seg_open = seg_outcomes[seg_outcomes['outcome'] == 'Still Open']
        
        deal_conv = len(seg_won) / len(seg_active) * 100 if len(seg_active) > 0 else 0
        value_conv = seg_won['net_revenue'].sum() / seg_active['net_revenue'].sum() * 100 if seg_active['net_revenue'].sum() > 0 else 0
        
        print(f"\n{segment}:")
        print(f"  Active at backtest: {len(seg_active)} deals, ${seg_active['net_revenue'].sum():,.0f}")
        print(f"  → Won: {len(seg_won)} deals, ${seg_won['net_revenue'].sum():,.0f}")
        print(f"  → Lost: {len(seg_lost)} deals, ${seg_lost['net_revenue'].sum():,.0f}")
        print(f"  → Still Open: {len(seg_open)} deals, ${seg_open['net_revenue'].sum():,.0f}")
        print(f"  Deal Conversion: {deal_conv:.1f}%")
        print(f"  Value Conversion: {value_conv:.1f}%")
    
    print(f"\n{'='*70}")
    print("8. HISTORICAL CONVERSION RATES BY STAGE")
    print("="*70)
    
    # Closed deals before backtest date with impl_year < target
    closed_for_training = df_at_backtest[
        (df_at_backtest['date_closed'].notna()) &
        (df_at_backtest['date_closed'] <= backtest_dt) &
        (df_at_backtest['impl_year'] < target_impl_year)
    ]
    
    closed_deals = closed_for_training.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    closed_deals = closed_deals[closed_deals['stage'].isin(['Closed Won', 'Closed Lost'])]
    
    # Get all stages each deal passed through
    deal_stages = closed_for_training.groupby('deal_id')['stage'].apply(lambda x: set(x)).reset_index()
    deal_stages.columns = ['deal_id', 'stages_passed']
    
    closed_deals = closed_deals.merge(deal_stages, on='deal_id', how='left')
    closed_deals['is_won'] = closed_deals['stage'] == 'Closed Won'
    
    print(f"\nTraining data: {len(closed_deals)} closed deals with impl_year < {target_impl_year}")
    
    for stage in ['Qualified', 'Alignment', 'Solutioning', 'Verbal']:
        through_stage = closed_deals[closed_deals['stages_passed'].apply(lambda x: stage in x if x else False)]
        if len(through_stage) > 0:
            rate = through_stage['is_won'].mean() * 100
            print(f"  {stage}: {rate:.1f}% ({through_stage['is_won'].sum()}/{len(through_stage)})")
        else:
            print(f"  {stage}: N/A (no deals)")
    
    print(f"\n{'='*70}")
    print("9. RECOMMENDED MODEL PARAMETERS")
    print("="*70)
    
    # Calculate observed conversion rate for target impl year
    if len(target_active) > 0:
        observed_deal_conv = len(final_outcomes[final_outcomes['outcome'] == 'Won']) / len(target_active)
        observed_value_conv = final_outcomes[final_outcomes['outcome'] == 'Won']['net_revenue'].sum() / target_active['net_revenue'].sum() if target_active['net_revenue'].sum() > 0 else 0
        
        print(f"\nObserved Active Pipeline Conversion (impl_year={target_impl_year}):")
        print(f"  Deal conversion: {observed_deal_conv*100:.1f}%")
        print(f"  Value conversion: {observed_value_conv*100:.1f}%")
    
    # Calculate YoY change
    prior_impl_year = target_impl_year - 1
    prior_won = latest_all[(latest_all['stage'] == 'Closed Won') & (latest_all['impl_year'] == prior_impl_year)]
    target_won = latest_all[(latest_all['stage'] == 'Closed Won') & (latest_all['impl_year'] == target_impl_year)]
    
    if prior_won['net_revenue'].sum() > 0:
        yoy_ratio = target_won['net_revenue'].sum() / prior_won['net_revenue'].sum()
        print(f"\nYear-over-Year Revenue Change:")
        print(f"  impl_year={prior_impl_year}: ${prior_won['net_revenue'].sum():,.0f}")
        print(f"  impl_year={target_impl_year}: ${target_won['net_revenue'].sum():,.0f}")
        print(f"  Ratio: {yoy_ratio:.2f} ({(yoy_ratio-1)*100:+.1f}%)")
        print(f"\nSuggested conservatism factor: {min(yoy_ratio, 1.0):.2f}")
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/fact_snapshots.csv'
    
    run_diagnostics(data_path)