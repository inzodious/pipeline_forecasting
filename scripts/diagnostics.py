import pandas as pd
import numpy as np
import os

"""
DIAGNOSTIC: Validate data before running forecast.
Uses correct column names: date_snapshot, date_created, date_closed, 
net_revenue, market_segment, stage
"""

BASE_DIR = "data"
PATH_RAW = os.path.join(BASE_DIR, "fact_snapshots.csv")  # <-- Your filename

# Stage classifications (same as forecast)
WON_STAGES = ["Closed Won", "Verbal"]
LOST_PATTERNS = ["Closed Lost", "Declined to Bid"]

def is_won(stage):
    return stage in WON_STAGES if pd.notna(stage) else False

def is_lost(stage):
    if pd.isna(stage):
        return False
    return any(p.lower() in stage.lower() for p in LOST_PATTERNS)

def run_diagnostic():
    print("=" * 70)
    print("DATA DIAGNOSTIC - Validate Before Forecasting")
    print("=" * 70)
    
    if not os.path.exists(PATH_RAW):
        print(f"\nERROR: File not found: {PATH_RAW}")
        print("Update PATH_RAW to match your actual filename.")
        return
    
    df = pd.read_csv(PATH_RAW)
    df['date_snapshot'] = pd.to_datetime(df['date_snapshot'])
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['date_closed'] = pd.to_datetime(df['date_closed'])
    
    latest = df['date_snapshot'].max()
    
    print(f"\n1. DATA OVERVIEW")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique deals:  {df['deal_id'].nunique():,}")
    print(f"   Snapshot range: {df['date_snapshot'].min().date()} to {latest.date()}")
    print(f"   Columns: {df.columns.tolist()}")
    
    print(f"\n2. STAGE DISTRIBUTION (latest snapshot)")
    df_latest = df[df['date_snapshot'] == latest].copy()
    stage_counts = df_latest['stage'].value_counts()
    for stage, count in stage_counts.items():
        marker = ""
        if is_won(stage):
            marker = " [WON]"
        elif is_lost(stage):
            marker = " [LOST]"
        print(f"   {stage:30s}: {count:>5,}{marker}")
    
    print(f"\n3. OPEN PIPELINE (latest snapshot)")
    df_latest['is_closed'] = df_latest['stage'].apply(lambda x: is_won(x) or is_lost(x))
    df_open = df_latest[~df_latest['is_closed']]
    print(f"   Open deals: {len(df_open):,}")
    print(f"   Open revenue: ${df_open['net_revenue'].sum():,.0f}")
    
    # Creation date distribution for open deals
    df_open['create_month'] = df_open['date_created'].dt.to_period('M')
    create_dist = df_open['create_month'].value_counts().sort_index()
    print(f"\n   Open deals by creation month (recent):")
    for period, count in create_dist.tail(6).items():
        print(f"     {period}: {count:>3} deals")
    
    print(f"\n4. HISTORICAL CLOSINGS (for DSO calculation)")
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    df_final['is_won'] = df_final['stage'].apply(is_won)
    df_final['is_lost'] = df_final['stage'].apply(is_lost)
    
    df_won = df_final[df_final['is_won']].copy()
    df_lost = df_final[df_final['is_lost']].copy()
    
    print(f"   Won deals:  {len(df_won):,} (${df_won['net_revenue'].sum():,.0f})")
    print(f"   Lost deals: {len(df_lost):,} (${df_lost['net_revenue'].sum():,.0f})")
    
    # DSO calculation
    df_won['cycle_days'] = (df_won['date_closed'] - df_won['date_created']).dt.days
    df_won_valid = df_won[(df_won['cycle_days'] >= 7) & (df_won['cycle_days'] <= 365)]
    
    print(f"\n   DSO (Won deals, 7-365 days):")
    print(f"     Mean:   {df_won_valid['cycle_days'].mean():.0f} days")
    print(f"     Median: {df_won_valid['cycle_days'].median():.0f} days")
    print(f"     Std:    {df_won_valid['cycle_days'].std():.0f} days")
    
    print(f"\n   DSO by Segment:")
    for seg in df['market_segment'].unique():
        seg_data = df_won_valid[df_won_valid['market_segment'] == seg]
        if len(seg_data) >= 5:
            print(f"     {seg:20s}: median={seg_data['cycle_days'].median():.0f} days ({len(seg_data)} deals)")
    
    print(f"\n5. WIN RATE CHECK")
    for seg in df['market_segment'].unique():
        seg_won = df_won[df_won['market_segment'] == seg]
        seg_lost = df_lost[df_lost['market_segment'] == seg]
        
        won_rev = seg_won['net_revenue'].sum()
        lost_rev = seg_lost['net_revenue'].sum()
        total_rev = won_rev + lost_rev
        
        wr = (won_rev / total_rev * 100) if total_rev > 0 else 0
        vol_wr = (len(seg_won) / (len(seg_won) + len(seg_lost)) * 100) if (len(seg_won) + len(seg_lost)) > 0 else 0
        
        print(f"   {seg:20s}: {wr:>5.1f}% (rev-weighted), {vol_wr:>5.1f}% (vol-weighted)")
    
    print(f"\n6. ACTUAL MONTHLY CLOSINGS (last 12 months)")
    df_won['close_month'] = df_won['date_closed'].dt.to_period('M')
    monthly_wins = df_won.groupby('close_month').agg({
        'deal_id': 'count',
        'net_revenue': 'sum'
    }).tail(12)
    
    for period, row in monthly_wins.iterrows():
        print(f"   {period}: {row['deal_id']:>3} deals, ${row['net_revenue']:>12,.0f}")
    
    avg_monthly_deals = monthly_wins['deal_id'].mean()
    avg_monthly_rev = monthly_wins['net_revenue'].mean()
    
    print(f"\n   Average: {avg_monthly_deals:.0f} deals/month, ${avg_monthly_rev:,.0f}/month")
    print(f"   Implied annual: {avg_monthly_deals * 12:.0f} deals, ${avg_monthly_rev * 12:,.0f}")
    
    print(f"\n7. VELOCITY CHECK (Deal first appearances)")
    df_first = df.sort_values('date_snapshot').groupby('deal_id').first().reset_index()
    df_first['appear_year'] = df_first['date_snapshot'].dt.year
    df_first['appear_month'] = df_first['date_snapshot'].dt.month
    
    entry_stages = df_first['stage'].value_counts(normalize=True)
    print(f"   Entry stage distribution:")
    for stage, pct in entry_stages.head(6).items():
        print(f"     {stage:30s}: {pct:>6.1%}")
    
    # Monthly velocity by year
    for year in sorted(df_first['appear_year'].unique())[-2:]:
        year_data = df_first[df_first['appear_year'] == year]
        print(f"\n   {year} monthly deal appearances:")
        monthly = year_data.groupby('appear_month').size()
        for m, count in monthly.items():
            print(f"     Month {m:>2}: {count:>3} deals")
        print(f"     Total: {len(year_data)} deals")
    
    print(f"\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE - Review above before running forecast")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()