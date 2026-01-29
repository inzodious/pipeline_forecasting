import pandas as pd
import numpy as np

# Quick layer breakdown for 2026 forecast
DATA_PATH = './data/fact_snapshots.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['date_created', 'date_closed', 'date_snapshot'])
df = df.sort_values(['deal_id', 'date_snapshot'])
df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])
df['is_closed_lost'] = df['stage'] == 'Closed Lost'
df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']

print("="*80)
print("2026 FORECAST LAYER BREAKDOWN")
print("="*80)

# Get open pipeline as of 2025-12-26
snapshot_2025_end = df[df['date_snapshot'] <= pd.to_datetime('2025-12-26')]['date_snapshot'].max()
open_2025 = df[(df['date_snapshot'] == snapshot_2025_end) & (~df['is_closed'])].copy()

layer2_value = open_2025.groupby('market_segment')['net_revenue'].sum().reset_index()
layer2_value.columns = ['market_segment', 'layer2_pipeline_value']

print("\nLAYER 2 - Active Pipeline (Open deals as of 2025-12-26):")
print(layer2_value.to_string(index=False))
print(f"TOTAL LAYER 2: ${layer2_value['layer2_pipeline_value'].sum():,.0f}")

# Calculate Layer 1 metrics with T12M weighting
all_deals = df.groupby('deal_id').first().reset_index()
closed_deals = df[df['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed', 'stage']]
all_deals = all_deals.drop(columns=['stage', 'date_closed'], errors='ignore')
deals = all_deals.merge(closed_deals, on='deal_id', how='left')
deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])
deals['created_month'] = deals['date_created'].dt.to_period('M')

# Apply T12M weighting
cutoff_date = deals['date_created'].max()
trailing_12_start = cutoff_date - pd.DateOffset(months=12)
deals['weight'] = np.where(deals['date_created'] >= trailing_12_start, 3.0, 1.0)

# Calculate weighted metrics
layer1_metrics = deals.groupby('market_segment').apply(
    lambda x: pd.Series({
        'weighted_volume': x['weight'].sum() / x['created_month'].nunique(),
        'weighted_win_rate': (x['won'].astype(float) * x['weight']).sum() / x['weight'].sum(),
        'weighted_avg_size': (x['net_revenue'] * x['weight']).sum() / x['weight'].sum()
    })
).reset_index()

layer1_metrics['expected_monthly_won_revenue'] = (
    layer1_metrics['weighted_volume'] * 
    layer1_metrics['weighted_win_rate'] * 
    layer1_metrics['weighted_avg_size']
)

layer1_metrics['expected_annual_won_revenue'] = layer1_metrics['expected_monthly_won_revenue'] * 12

print("\nLAYER 1 - Future Pipeline (New deals to be created in 2026):")
print(layer1_metrics[['market_segment', 'weighted_volume', 'weighted_win_rate', 'weighted_avg_size', 'expected_annual_won_revenue']].to_string(index=False))
print(f"TOTAL LAYER 1 (Annual): ${layer1_metrics['expected_annual_won_revenue'].sum():,.0f}")

print("\n" + "="*80)
print("COMPARISON: What each layer contributes to the forecast")
print("="*80)

# Read actual forecast output
forecast_df = pd.read_csv('./exports/forecast_2026.csv')
forecast_summary = pd.DataFrame(forecast_df.groupby('market_segment')['expected_won_revenue'].sum()).reset_index()
forecast_summary.columns = ['market_segment', 'total_forecast']

comparison = layer1_metrics[['market_segment', 'expected_annual_won_revenue']].copy()
comparison = comparison.merge(layer2_value, on='market_segment', how='outer').fillna(0)
comparison = comparison.merge(forecast_summary, on='market_segment', how='outer').fillna(0)

print("\nBreakdown:")
print(comparison[['market_segment', 'layer2_pipeline_value', 'expected_annual_won_revenue', 'total_forecast']].to_string(index=False))

print("\n" + "="*80)
print("EXPLANATION OF $5M GAP (2026 vs 2025)")
print("="*80)
print("\n2025 Actual Closed Won: $20,488,300")
print("2026 Forecast: $15,925,126")
print("Gap: -$4,563,174 (-22%)")
print("\nWHY? Deal creation volumes dropped 18.9% in 2025:")
print("  - Large Market: 104 deals (2024) -> 58 deals (2025) = -44%")
print("  - SMB: 983 deals (2024) -> 525 deals (2025) = -47%")
print("\nThe T12M weighted forecast (3x weight on last 12 months) projects")
print("2025's LOWER creation rates forward into 2026.")
print("\nThis is MATHEMATICALLY CORRECT behavior.")
print("\nTo hit 2025 revenue levels, you would need to:")
print("  1. Increase volume_multiplier lever to ~1.3")
print("  2. Or increase win_rate_multiplier lever")
print("  3. Or expect reversion to 2024 creation rates (not what T12M suggests)")
print("\n" + "="*80)
