import pandas as pd
import numpy as np

# Load and build deal facts
df = pd.read_csv('./data/fact_snapshots.csv', parse_dates=['date_created', 'date_closed', 'date_snapshot'])
df = df.sort_values(['deal_id', 'date_snapshot'])
df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])
df['is_closed_lost'] = df['stage'] == 'Closed Lost'
df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']

# Build deal facts for all data through 2025
backtest_end = pd.to_datetime('2025-12-31')
actual_full_data = df[df['date_snapshot'] <= backtest_end].copy()

first = actual_full_data.groupby('deal_id').first().reset_index()
closed = actual_full_data[actual_full_data['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed', 'stage']]
first = first.drop(columns=['stage', 'date_closed'], errors='ignore')
deals = first.merge(closed, on='deal_id', how='left')
deals['created_month'] = deals['date_created'].dt.to_period('M')
deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])
deals['volume_weight'] = 1.0

# Separate historical vs 2025
backtest_cutoff = pd.to_datetime('2025-01-01')
historical_deals = deals[deals['date_created'] < backtest_cutoff].copy()
deals_2025 = deals[
    (deals['date_created'] >= backtest_cutoff) &
    (deals['date_created'] <= backtest_end)
].copy()

print("HISTORICAL BASELINE (pre-2025)")
print("="*60)
hist_summary = historical_deals.groupby('market_segment').apply(
    lambda x: pd.Series({
        'total_deals': len(x),
        'won_deals': x['won'].sum(),
        'won_revenue': x[x['won']]['net_revenue'].sum(),
        'months': x['created_month'].nunique(),
        'avg_monthly_vol': x['volume_weight'].sum() / x['created_month'].nunique(),
        'avg_won_size': x[x['won']]['net_revenue'].sum() / x['won'].sum() if x['won'].sum() > 0 else 0,
        'win_rate': x['won'].mean()
    })
).reset_index()
print(hist_summary.to_string(index=False))

print("\n\n2025 ACTUALS (deals CREATED in 2025)")
print("="*60)
actual_2025 = deals_2025.groupby('market_segment').apply(
    lambda x: pd.Series({
        'total_deals': len(x),
        'won_deals': x['won'].sum(),
        'won_revenue': x[x['won']]['net_revenue'].sum(),
        'months': x['created_month'].nunique(),
        'avg_monthly_vol': x['volume_weight'].sum() / x['created_month'].nunique(),
        'avg_won_size': x[x['won']]['net_revenue'].sum() / x['won'].sum() if x['won'].sum() > 0 else 0,
        'win_rate': x['won'].mean()
    })
).reset_index()
print(actual_2025.to_string(index=False))

print("\n\nMULTIPLIERS (2025 vs Historical)")
print("="*60)
comparison = hist_summary.merge(actual_2025, on='market_segment', suffixes=('_hist', '_2025'))
comparison['vol_mult'] = comparison['avg_monthly_vol_2025'] / comparison['avg_monthly_vol_hist']
comparison['wr_mult'] = comparison['win_rate_2025'] / comparison['win_rate_hist']
comparison['size_mult'] = comparison['avg_won_size_2025'] / comparison['avg_won_size_hist']

print(comparison[['market_segment', 'vol_mult', 'wr_mult', 'size_mult']].to_string(index=False))

print("\n\nFORECAST MATH CHECK (Future Pipeline)")
print("="*60)
for _, row in comparison.iterrows():
    segment = row['market_segment']
    
    # Historical baseline per month
    hist_vol = row['avg_monthly_vol_hist']
    hist_size = row['avg_won_size_hist']
    hist_wr = row['win_rate_hist']
    
    # Apply multipliers
    vol = hist_vol * row['vol_mult']
    size = hist_size * row['size_mult']
    wr = hist_wr * row['wr_mult']
    
    # Expected per month and annual (12 months)
    expected_monthly = vol * size * wr
    expected_annual = expected_monthly * 12
    
    # Actual
    actual = row['won_revenue_2025']
    
    print(f"{segment}:")
    print(f"  Historical: {hist_vol:.1f} vol/mo * ${hist_size:,.0f} * {hist_wr:.1%} = ${hist_vol * hist_size * hist_wr:,.0f}/mo")
    print(f"  With multipliers: {vol:.1f} vol/mo * ${size:,.0f} * {wr:.1%} = ${expected_monthly:,.0f}/mo")
    print(f"  Expected annual (12mo): ${expected_annual:,.0f}")
    print(f"  Actual 2025: ${actual:,.0f}")
    print(f"  Variance: {((expected_annual - actual) / actual * 100):.1f}%")
    print()
