import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('./data/fact_snapshots.csv', parse_dates=['date_created', 'date_closed', 'date_snapshot'])

# Get deals that were created AND closed in 2025
backtest_end = pd.to_datetime('2025-12-31')
backtest_start = pd.to_datetime('2025-01-01')

# Get final snapshot
last_2025 = df[df['date_snapshot'] <= backtest_end]['date_snapshot'].max()
final = df[df['date_snapshot'] == last_2025]

# Deals created in 2025
created_2025 = final[
    (final['date_created'] >= backtest_start) &
    (final['date_created'] <= backtest_end)
].copy()

print(f"DEALS CREATED IN 2025: {len(created_2025)}")
print(f"Won: {len(created_2025[created_2025['stage'].isin(['Closed Won', 'Verbal'])])}")
print(f"Lost: {len(created_2025[created_2025['stage'] == 'Closed Lost'])}")
print(f"Still Open: {len(created_2025[~created_2025['stage'].isin(['Closed Won', 'Verbal', 'Closed Lost'])])}")

won_2025_created = created_2025[created_2025['stage'].isin(['Closed Won', 'Verbal'])].copy()
print(f"\nWON DEALS CREATED IN 2025: {len(won_2025_created)}")
print(f"Closed in 2025: {len(won_2025_created[won_2025_created['date_closed'] <= backtest_end])}")
print(f"Not closed yet or closed after 2025: {len(won_2025_created[won_2025_created['date_closed'] > backtest_end])}")

# Calculate timing for those that closed in 2025
closed_in_2025 = won_2025_created[
    (won_2025_created['date_closed'].notna()) &
    (won_2025_created['date_closed'] <= backtest_end)
].copy()

if len(closed_in_2025) > 0:
    closed_in_2025['months_to_close'] = ((closed_in_2025['date_closed'] - closed_in_2025['date_created']).dt.days / 30).round()
    
    print(f"\nTIMING DISTRIBUTION FOR DEALS CREATED & CLOSED IN 2025:")
    timing = closed_in_2025.groupby('market_segment')['months_to_close'].describe()
    print(timing)
    
    print(f"\nREVENUE BY SEGMENT:")
    revenue_summary = closed_in_2025.groupby('market_segment').agg(
        count=('deal_id', 'count'),
        revenue=('net_revenue', 'sum')
    )
    print(revenue_summary)
    print(f"\nTotal revenue from deals created & closed in 2025: ${closed_in_2025['net_revenue'].sum():,.2f}")

# Now check ALL wins in 2025 regardless of when created
all_wins_2025 = final[
    (final['stage'].isin(['Closed Won', 'Verbal'])) &
    (final['date_closed'] >= backtest_start) &
    (final['date_closed'] <= backtest_end) &
    (final['date_closed'].notna())
].copy()

print(f"\n{'='*60}")
print(f"ALL WINS IN 2025 (regardless of creation date):")
print(f"Total: {len(all_wins_2025)}")
print(f"Total Revenue: ${all_wins_2025['net_revenue'].sum():,.2f}")

print(f"\nBREAKDOWN BY CREATION PERIOD:")
all_wins_2025['created_in_2025'] = all_wins_2025['date_created'] >= backtest_start
breakdown = all_wins_2025.groupby('created_in_2025').agg(
    count=('deal_id', 'count'),
    revenue=('net_revenue', 'sum')
)
print(breakdown)
