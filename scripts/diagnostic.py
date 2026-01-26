# Diagnostic: What happened to active pipeline deals?
import pandas as pd

df = pd.read_csv('data/fact_snapshots.csv', parse_dates=['date_created', 'date_closed', 'date_snapshot'])

backtest_date = pd.to_datetime('2025-01-01')

# Get active pipeline at backtest
snapshot = df[df['date_snapshot'] == df[df['date_snapshot'] <= backtest_date]['date_snapshot'].max()]
active = snapshot[snapshot['stage'].isin(['Qualified', 'Alignment', 'Solutioning'])]

# Get outcomes for these deals
outcomes = df.groupby('deal_id').agg({
    'date_closed': 'first',
    'stage': lambda x: 'Won' if 'Closed Won' in list(x) else ('Lost' if 'Closed Lost' in list(x) else 'Open'),
    'net_revenue': 'last',
    'market_segment': 'first'
}).reset_index()

# Merge
active_outcomes = active[['deal_id', 'stage', 'net_revenue', 'market_segment']].merge(
    outcomes[['deal_id', 'date_closed', 'stage']].rename(columns={'stage': 'final_outcome'}),
    on='deal_id'
)

# Show what happened to Large Market deals
lm = active_outcomes[active_outcomes['market_segment'] == 'Large Market']
print("Large Market Active Pipeline Outcomes:")
print(lm['final_outcome'].value_counts())
print(f"\nTotal value: ${lm['net_revenue'].sum():,.0f}")
print(f"Won value: ${lm[lm['final_outcome'] == 'Won']['net_revenue'].sum():,.0f}")