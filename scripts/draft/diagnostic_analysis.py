import pandas as pd
import numpy as np
from datetime import datetime

# Load data
DATA_PATH = './data/fact_snapshots.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['date_created', 'date_closed', 'date_snapshot'])
df = df.sort_values(['deal_id', 'date_snapshot'])
df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])
df['is_closed_lost'] = df['stage'] == 'Closed Lost'
df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']

print("="*80)
print("DIAGNOSTIC ANALYSIS: Why is 2026 Forecast $5M below 2025 Actuals?")
print("="*80)

# ============================================================================
# 1. COMPARE STARTING OPEN PIPELINE (Layer 2)
# ============================================================================
print("\n1. STARTING OPEN PIPELINE COMPARISON (Layer 2)")
print("-"*80)

# Open pipeline going into 2025 (as of 2024-12-31)
snapshot_2024_end = df[df['date_snapshot'] <= pd.to_datetime('2024-12-31')]['date_snapshot'].max()
open_2024 = df[(df['date_snapshot'] == snapshot_2024_end) & (~df['is_closed'])].copy()
open_2024_summary = open_2024.groupby('market_segment').agg(
    deal_count=('deal_id', 'count'),
    total_revenue=('net_revenue', 'sum')
).reset_index()

print(f"\nOpen Pipeline as of {snapshot_2024_end.date()} (going into 2025):")
print(open_2024_summary.to_string(index=False))
print(f"TOTAL OPEN DEALS: {open_2024['deal_id'].count()}")
print(f"TOTAL OPEN VALUE: ${open_2024['net_revenue'].sum():,.0f}")

# Open pipeline going into 2026 (as of 2025-12-26)
snapshot_2025_end = df[df['date_snapshot'] <= pd.to_datetime('2025-12-26')]['date_snapshot'].max()
open_2025 = df[(df['date_snapshot'] == snapshot_2025_end) & (~df['is_closed'])].copy()
open_2025_summary = open_2025.groupby('market_segment').agg(
    deal_count=('deal_id', 'count'),
    total_revenue=('net_revenue', 'sum')
).reset_index()

print(f"\nOpen Pipeline as of {snapshot_2025_end.date()} (going into 2026):")
print(open_2025_summary.to_string(index=False))
print(f"TOTAL OPEN DEALS: {open_2025['deal_id'].count()}")
print(f"TOTAL OPEN VALUE: ${open_2025['net_revenue'].sum():,.0f}")

# Comparison
comparison = open_2024_summary.merge(open_2025_summary, on='market_segment', suffixes=('_2024', '_2025'), how='outer').fillna(0)
comparison['revenue_change'] = comparison['total_revenue_2025'] - comparison['total_revenue_2024']
comparison['revenue_change_pct'] = ((comparison['total_revenue_2025'] / comparison['total_revenue_2024'].replace(0, 1)) - 1) * 100

print("\nChange in Open Pipeline (2024-end vs 2025-end):")
print(comparison[['market_segment', 'total_revenue_2024', 'total_revenue_2025', 'revenue_change', 'revenue_change_pct']].to_string(index=False))

# ============================================================================
# 2. COMPARE DEAL CREATION RATES (Layer 1)
# ============================================================================
print("\n\n2. DEAL CREATION RATES COMPARISON (Layer 1)")
print("-"*80)

# Deals created in 2024
deals_2024 = df[df['date_created'].dt.year == 2024].groupby('deal_id').first().reset_index()
creation_2024 = deals_2024.groupby('market_segment').agg(
    deal_count=('deal_id', 'count'),
    total_value=('net_revenue', 'sum')
).reset_index()
creation_2024['monthly_avg'] = creation_2024['deal_count'] / 12

print("\nDeals Created in 2024:")
print(creation_2024.to_string(index=False))

# Deals created in 2025
deals_2025 = df[df['date_created'].dt.year == 2025].groupby('deal_id').first().reset_index()
creation_2025 = deals_2025.groupby('market_segment').agg(
    deal_count=('deal_id', 'count'),
    total_value=('net_revenue', 'sum')
).reset_index()
creation_2025['monthly_avg'] = creation_2025['deal_count'] / 12

print("\nDeals Created in 2025:")
print(creation_2025.to_string(index=False))

# ============================================================================
# 3. COMPARE WIN RATES
# ============================================================================
print("\n\n3. WIN RATES COMPARISON")
print("-"*80)

# Build deal facts for each year
def get_year_facts(year):
    year_df = df[df['date_created'].dt.year == year].copy()
    first = year_df.groupby('deal_id').first().reset_index()
    closed = year_df[year_df['is_closed']].groupby('deal_id').first().reset_index()[['deal_id', 'date_closed', 'stage']]
    first = first.drop(columns=['stage', 'date_closed'], errors='ignore')
    deals = first.merge(closed, on='deal_id', how='left')
    deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])
    return deals

deals_2024 = get_year_facts(2024)
deals_2025 = get_year_facts(2025)

wr_2024 = deals_2024.groupby('market_segment').agg(
    total=('deal_id', 'count'),
    won=('won', 'sum')
).reset_index()
wr_2024['win_rate'] = wr_2024['won'] / wr_2024['total']

wr_2025 = deals_2025.groupby('market_segment').agg(
    total=('deal_id', 'count'),
    won=('won', 'sum')
).reset_index()
wr_2025['win_rate'] = wr_2025['won'] / wr_2025['total']

print("\n2024 Win Rates:")
print(wr_2024[['market_segment', 'win_rate']].to_string(index=False))

print("\n2025 Win Rates:")
print(wr_2025[['market_segment', 'win_rate']].to_string(index=False))

# ============================================================================
# 4. WHAT CLOSED IN EACH YEAR
# ============================================================================
print("\n\n4. CLOSED DEALS COMPARISON")
print("-"*80)

# Get last snapshot for each year
last_2024 = df[df['date_snapshot'].dt.year == 2024]['date_snapshot'].max()
last_2025 = df[df['date_snapshot'].dt.year == 2025]['date_snapshot'].max()

closed_2024 = df[
    (df['date_snapshot'] == last_2024) &
    (df['stage'].isin(['Closed Won', 'Verbal'])) &
    (df['date_closed'].notna()) &
    (df['date_closed'].dt.year == 2024)
].copy()

closed_2025 = df[
    (df['date_snapshot'] == last_2025) &
    (df['stage'].isin(['Closed Won', 'Verbal'])) &
    (df['date_closed'].notna()) &
    (df['date_closed'].dt.year == 2025)
].copy()

closed_2024_summary = closed_2024.groupby('market_segment').agg(
    deal_count=('deal_id', 'count'),
    total_revenue=('net_revenue', 'sum')
).reset_index()

closed_2025_summary = closed_2025.groupby('market_segment').agg(
    deal_count=('deal_id', 'count'),
    total_revenue=('net_revenue', 'sum')
).reset_index()

print("\nClosed Won in 2024:")
print(closed_2024_summary.to_string(index=False))
print(f"TOTAL: ${closed_2024_summary['total_revenue'].sum():,.0f}")

print("\nClosed Won in 2025:")
print(closed_2025_summary.to_string(index=False))
print(f"TOTAL: ${closed_2025_summary['total_revenue'].sum():,.0f}")

# ============================================================================
# 5. KEY INSIGHTS
# ============================================================================
print("\n\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

total_open_2024 = open_2024['net_revenue'].sum()
total_open_2025 = open_2025['net_revenue'].sum()
open_delta = total_open_2025 - total_open_2024

total_created_2024 = creation_2024['deal_count'].sum()
total_created_2025 = creation_2025['deal_count'].sum()
created_delta = total_created_2025 - total_created_2024

print(f"\n1. OPEN PIPELINE CARRYOVER:")
print(f"   - Going into 2025: ${total_open_2024:,.0f}")
print(f"   - Going into 2026: ${total_open_2025:,.0f}")
print(f"   - Delta: ${open_delta:,.0f} ({(open_delta/total_open_2024)*100:.1f}%)")

print(f"\n2. DEAL CREATION VOLUME:")
print(f"   - 2024: {total_created_2024} deals")
print(f"   - 2025: {total_created_2025} deals")
print(f"   - Delta: {created_delta} deals ({(created_delta/total_created_2024)*100:.1f}%)")

print(f"\n3. CLOSED WON REVENUE:")
print(f"   - 2024: ${closed_2024_summary['total_revenue'].sum():,.0f}")
print(f"   - 2025: ${closed_2025_summary['total_revenue'].sum():,.0f}")

print("\n4. FORECAST LOGIC:")
print("   - Layer 2 (Active Pipeline): Uses open deals as of 2025-12-26")
print("   - Layer 1 (Future Pipeline): Uses ALL HISTORICAL data for avg monthly volume/win rate/size")
print("   - With T12M weighting (3x), recent 2025 data has more influence")

print("\n5. HYPOTHESIS:")
if open_delta < 0:
    print(f"   YES - Smaller open pipeline carryover explains part of the gap")
    print(f"   The open pipeline going into 2026 is ${abs(open_delta):,.0f} LOWER than into 2025")
if created_delta < 0:
    print(f"   If 2025 had fewer deal creations than 2024, the T12M weighted averages")
    print(f"   would project lower future pipeline for 2026")

print("\n" + "="*80)
