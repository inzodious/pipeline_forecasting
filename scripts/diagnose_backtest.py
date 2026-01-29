import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('./data/fact_snapshots.csv', parse_dates=['date_created', 'date_closed', 'date_snapshot'])
df = df.sort_values(['deal_id', 'date_snapshot'])
df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])

# Get last snapshot in 2025
last_2025 = df[df['date_snapshot'] <= '2025-12-31']['date_snapshot'].max()
final_snapshot = df[df['date_snapshot'] == last_2025]

# Find deals that closed won in 2025
wins_2025 = final_snapshot[
    (final_snapshot['stage'].isin(['Closed Won', 'Verbal'])) &
    (final_snapshot['date_closed'] >= '2025-01-01') &
    (final_snapshot['date_closed'] <= '2025-12-31') &
    (final_snapshot['date_closed'].notna())
].copy()

# Categorize by when they were created
wins_2025['created_before_2025'] = wins_2025['date_created'] < '2025-01-01'

# Summary
summary = wins_2025.groupby(['market_segment', 'created_before_2025']).agg(
    count=('deal_id', 'count'),
    revenue=('net_revenue', 'sum')
).reset_index()

print("2025 WINS BREAKDOWN BY CREATION DATE")
print("="*60)
print(summary.to_string(index=False))
print()

totals = wins_2025.groupby('created_before_2025').agg(
    count=('deal_id', 'count'),
    revenue=('net_revenue', 'sum')
).reset_index()

print("\nTOTAL BREAKDOWN:")
print(totals.to_string(index=False))
print()

# Check what was open at start of 2025
last_2024 = df[df['date_snapshot'] < '2025-01-01']['date_snapshot'].max()
open_2024 = df[
    (df['date_snapshot'] == last_2024) &
    (~df['stage'].isin(['Closed Won', 'Verbal', 'Closed Lost']))
].copy()

print(f"\nDEALS OPEN AT START OF 2025: {len(open_2024)}")
print(f"Total revenue in open pipeline: ${open_2024['net_revenue'].sum():,.2f}")
print(f"\nBy segment:")
print(open_2024.groupby('market_segment').agg(
    count=('deal_id', 'count'),
    revenue=('net_revenue', 'sum')
).to_string())
