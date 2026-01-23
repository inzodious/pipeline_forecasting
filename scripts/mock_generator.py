import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
"""
UPDATED: High-volume mock data generator for proper forecast validation.
- 360 deals/year (~30/month) provides statistical stability
- Realistic segment distribution and seasonality
- Simple naming: 'Mock Deal ' + deal_id
"""

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fact_snapshots.csv")

FISCAL_YEARS = [2024, 2025]
DEALS_PER_YEAR = 360
DEALS_PER_MONTH = 30

# 2023 Carryover Pipeline
CARRYOVER_CONFIG = {
    'enabled': True,
    'count': 45,
    'days_before_start': (30, 120),
    'segment_distribution': {
        'Large Market': 5,
        'Mid Market': 15,
        'Small Market': 25
    }
}

# Segment configuration - realistic enterprise sales mix
SEGMENT_CONFIG = {
    'Large Market': {
        'pct_of_deals': 0.10,           # 10% of deals
        'revenue_min': 150000,
        'revenue_max': 500000,
        'win_rate': 0.15,               # Lower win rate, higher value
        'dso_mean': 90,                 # Longer sales cycle
        'dso_std': 25
    },
    'Mid Market': {
        'pct_of_deals': 0.35,           # 35% of deals
        'revenue_min': 50000,
        'revenue_max': 149000,
        'win_rate': 0.22,
        'dso_mean': 60,
        'dso_std': 20
    },
    'Small Market': {
        'pct_of_deals': 0.55,           # 55% of deals
        'revenue_min': 15000,
        'revenue_max': 49000,
        'win_rate': 0.28,               # Higher win rate, lower value
        'dso_mean': 45,                 # Shorter sales cycle
        'dso_std': 15
    }
}

# Seasonality - monthly multipliers (Q4 heavy, Q1 slow)
SEASONALITY = {
    1: 0.70,    # January - slow start
    2: 0.80,
    3: 0.95,
    4: 1.00,
    5: 1.00,
    6: 1.10,    # Q2 push
    7: 0.85,    # Summer slowdown
    8: 0.85,
    9: 1.05,    # Q3 ramp
    10: 1.15,
    11: 1.20,   # Q4 push
    12: 1.35    # Year-end close
}

STAGES = ['Qualified', 'Solutioning', 'Alignment']
CLOSED_WON = 'Closed Won'
CLOSED_LOST = 'Closed Lost'

DEAL_OWNER = "Joshua Biondo"

SNAPSHOT_INTERVAL_DAYS = 7


# ==========================================
# DEAL GENERATION
# ==========================================

def generate_deal_name(deal_id):
    """Simple unique deal name."""
    return f"Mock Deal {deal_id}"


def generate_deal_id(year, index):
    """Generate unique deal ID."""
    return f"DEAL-{year}-{index:04d}"


def get_segment_for_deal():
    """Randomly assign segment based on distribution."""
    r = random.random()
    cumulative = 0
    for segment, config in SEGMENT_CONFIG.items():
        cumulative += config['pct_of_deals']
        if r <= cumulative:
            return segment
    return 'Small Market'


def get_monthly_deal_count(base_count, month):
    """Apply seasonality to monthly deal count."""
    multiplier = SEASONALITY.get(month, 1.0)
    # Add some randomness
    adjusted = base_count * multiplier * random.uniform(0.85, 1.15)
    return max(1, int(round(adjusted)))


def generate_carryover_deals(deal_counter):
    """Generate deals from late 2023 that carry over into 2024."""
    if not CARRYOVER_CONFIG.get('enabled', False):
        return [], deal_counter
    
    carryover_deals = []
    start_date = datetime(2024, 1, 1)
    days_min, days_max = CARRYOVER_CONFIG['days_before_start']
    
    for segment, count in CARRYOVER_CONFIG['segment_distribution'].items():
        config = SEGMENT_CONFIG[segment]
        
        for _ in range(count):
            deal_id = generate_deal_id(2023, deal_counter)
            deal_name = generate_deal_name(deal_id)
            
            days_before = random.randint(days_min, days_max)
            date_created = start_date - timedelta(days=days_before)
            
            # Use segment-specific DSO
            close_days = max(14, int(random.gauss(config['dso_mean'], config['dso_std'])))
            date_closed = date_created + timedelta(days=close_days)
            
            impl_days = close_days + random.randint(30, 90)
            date_implementation = date_created + timedelta(days=impl_days)
            
            revenue = random.randint(config['revenue_min'], config['revenue_max'])
            
            is_won = random.random() < config['win_rate']
            final_stage = CLOSED_WON if is_won else CLOSED_LOST
            
            carryover_deals.append({
                'deal_id': deal_id,
                'deal_name': deal_name,
                'deal_owner': DEAL_OWNER,
                'market_segment': segment,
                'net_revenue': revenue,
                'date_created': date_created,
                'date_closed': date_closed,
                'date_implementation': date_implementation,
                'final_stage': final_stage,
                'close_days': close_days
            })
            
            deal_counter += 1
    
    return carryover_deals, deal_counter


def generate_deal(deal_id, deal_name, year, month, day, segment, config):
    """Generate a single deal with all its attributes."""
    date_created = datetime(year, month, day)
    
    # Use segment-specific DSO distribution
    close_days = max(14, int(random.gauss(config['dso_mean'], config['dso_std'])))
    date_closed = date_created + timedelta(days=close_days)
    
    impl_days = close_days + random.randint(30, 90)
    date_implementation = date_created + timedelta(days=impl_days)
    
    revenue = random.randint(config['revenue_min'], config['revenue_max'])
    
    is_won = random.random() < config['win_rate']
    final_stage = CLOSED_WON if is_won else CLOSED_LOST
    
    return {
        'deal_id': deal_id,
        'deal_name': deal_name,
        'deal_owner': DEAL_OWNER,
        'market_segment': segment,
        'net_revenue': revenue,
        'date_created': date_created,
        'date_closed': date_closed,
        'date_implementation': date_implementation,
        'final_stage': final_stage,
        'close_days': close_days
    }


def generate_year_deals(year, deal_counter):
    """Generate all deals for a given year with seasonality."""
    deals = []
    
    for month in range(1, 13):
        # Get seasonality-adjusted deal count for month
        monthly_count = get_monthly_deal_count(DEALS_PER_MONTH, month)
        
        # Get days in month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        days_in_month = (next_month - datetime(year, month, 1)).days
        
        for _ in range(monthly_count):
            segment = get_segment_for_deal()
            config = SEGMENT_CONFIG[segment]
            
            deal_id = generate_deal_id(year, deal_counter)
            deal_name = generate_deal_name(deal_id)
            
            # Random day within month
            day = random.randint(1, days_in_month)
            
            deal = generate_deal(deal_id, deal_name, year, month, day, segment, config)
            deals.append(deal)
            
            deal_counter += 1
    
    return deals, deal_counter


# ==========================================
# SNAPSHOT SIMULATION
# ==========================================

def calculate_stage_transitions(deal):
    """Calculate when a deal transitions through each stage."""
    close_days = deal['close_days']
    
    # Random stage durations
    stage_durations = {
        'Qualified': random.uniform(0.2, 0.4),
        'Solutioning': random.uniform(0.2, 0.4),
        'Alignment': random.uniform(0.2, 0.4)
    }
    
    total = sum(stage_durations.values())
    stage_durations = {k: v / total for k, v in stage_durations.items()}
    
    transitions = []
    cumulative_days = 0
    
    for stage in STAGES:
        stage_days = int(close_days * stage_durations[stage])
        transition_date = deal['date_created'] + timedelta(days=cumulative_days)
        transitions.append((transition_date, stage))
        cumulative_days += stage_days
    
    transitions.append((deal['date_closed'], deal['final_stage']))
    
    return transitions


def generate_snapshots_for_deal(deal, snapshot_dates):
    """Generate snapshot records for a deal across all snapshot dates."""
    close_year = deal['date_closed'].year
    is_future_close = close_year >= 2026
    
    transitions = calculate_stage_transitions(deal)
    
    snapshots = []
    
    for snap_date in snapshot_dates:
        if snap_date < deal['date_created']:
            continue
        
        if is_future_close:
            # Deal closes in future - keep as open pipeline
            current_stage = None
            for trans_date, stage in transitions:
                if stage in [CLOSED_WON, CLOSED_LOST]:
                    break
                if snap_date >= trans_date:
                    current_stage = stage
            if current_stage is None:
                current_stage = 'Qualified'
            
            snapshots.append({
                'deal_id': deal['deal_id'],
                'deal_name': deal['deal_name'],
                'deal_owner': deal['deal_owner'],
                'market_segment': deal['market_segment'],
                'net_revenue': deal['net_revenue'],
                'date_created': deal['date_created'],
                'date_closed': pd.NaT,
                'date_implementation': pd.NaT,
                'date_snapshot': snap_date,
                'stage': current_stage
            })
        else:
            # Deal closes in 2024/2025 - backfill date_closed
            current_stage = None
            for trans_date, stage in transitions:
                if snap_date >= trans_date:
                    current_stage = stage
                else:
                    break
            if current_stage is None:
                current_stage = 'Qualified'
            
            snapshots.append({
                'deal_id': deal['deal_id'],
                'deal_name': deal['deal_name'],
                'deal_owner': deal['deal_owner'],
                'market_segment': deal['market_segment'],
                'net_revenue': deal['net_revenue'],
                'date_created': deal['date_created'],
                'date_closed': deal['date_closed'],
                'date_implementation': deal['date_implementation'],
                'date_snapshot': snap_date,
                'stage': current_stage
            })
    
    return snapshots


def generate_snapshot_dates(start_year, end_year):
    """Generate weekly snapshot dates."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=SNAPSHOT_INTERVAL_DAYS)
    
    return dates


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_history_generator():
    print("=" * 70)
    print("MOCK DATA GENERATOR V2 - HIGH VOLUME")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_deals = []
    deal_counter = 1
    
    print("\n--- 1. Generating Deals ---")
    
    # Generate 2023 carryover pipeline
    if CARRYOVER_CONFIG.get('enabled', False):
        print(f"\n  2023 Carryover Pipeline:")
        carryover_deals, deal_counter = generate_carryover_deals(deal_counter)
        all_deals.extend(carryover_deals)
        
        carryover_stats = {seg: 0 for seg in SEGMENT_CONFIG.keys()}
        carryover_won = {seg: 0 for seg in SEGMENT_CONFIG.keys()}
        carryover_rev = {seg: 0 for seg in SEGMENT_CONFIG.keys()}
        
        for deal in carryover_deals:
            carryover_stats[deal['market_segment']] += 1
            if deal['final_stage'] == CLOSED_WON:
                carryover_won[deal['market_segment']] += 1
                carryover_rev[deal['market_segment']] += deal['net_revenue']
        
        for seg in SEGMENT_CONFIG.keys():
            actual_wr = (carryover_won[seg] / carryover_stats[seg] * 100) if carryover_stats[seg] > 0 else 0
            print(f"    {seg:15s}: {carryover_stats[seg]:>3} deals, {carryover_won[seg]:>2} won ({actual_wr:.0f}%), ${carryover_rev[seg]:>10,.0f}")
        
        print(f"    Total carryover: {len(carryover_deals)} deals")
    
    # Generate FY 2024 and 2025 deals
    for year in FISCAL_YEARS:
        print(f"\n  FY {year}:")
        
        year_deals, deal_counter = generate_year_deals(year, deal_counter)
        all_deals.extend(year_deals)
        
        year_stats = {seg: 0 for seg in SEGMENT_CONFIG.keys()}
        year_won = {seg: 0 for seg in SEGMENT_CONFIG.keys()}
        year_rev = {seg: 0 for seg in SEGMENT_CONFIG.keys()}
        
        for deal in year_deals:
            year_stats[deal['market_segment']] += 1
            if deal['final_stage'] == CLOSED_WON:
                year_won[deal['market_segment']] += 1
                year_rev[deal['market_segment']] += deal['net_revenue']
        
        for seg in SEGMENT_CONFIG.keys():
            actual_wr = (year_won[seg] / year_stats[seg] * 100) if year_stats[seg] > 0 else 0
            print(f"    {seg:15s}: {year_stats[seg]:>3} deals, {year_won[seg]:>2} won ({actual_wr:.0f}%), ${year_rev[seg]:>10,.0f}")
        
        total_year = sum(year_stats.values())
        total_won = sum(year_won.values())
        total_rev = sum(year_rev.values())
        print(f"    {'TOTAL':15s}: {total_year:>3} deals, {total_won:>2} won ({total_won/total_year*100:.0f}%), ${total_rev:>10,.0f}")
    
    print(f"\n  Grand total deals generated: {len(all_deals)}")
    
    print("\n--- 2. Generating Snapshots ---")
    
    snapshot_dates = generate_snapshot_dates(min(FISCAL_YEARS), max(FISCAL_YEARS))
    print(f"  Snapshot date range: {snapshot_dates[0].date()} to {snapshot_dates[-1].date()}")
    print(f"  Total snapshot dates: {len(snapshot_dates)}")
    
    all_snapshots = []
    
    for deal in all_deals:
        deal_snapshots = generate_snapshots_for_deal(deal, snapshot_dates)
        all_snapshots.extend(deal_snapshots)
    
    print(f"  Total snapshot records: {len(all_snapshots):,}")
    
    print("\n--- 3. Building DataFrame ---")
    
    df = pd.DataFrame(all_snapshots)
    
    df['date_snapshot'] = pd.to_datetime(df['date_snapshot'])
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['date_closed'] = pd.to_datetime(df['date_closed'])
    df['date_implementation'] = pd.to_datetime(df['date_implementation'])
    
    df = df.sort_values(['date_snapshot', 'deal_id']).reset_index(drop=True)
    
    print(f"\n--- 4. Data Summary ---")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Shape: {df.shape}")
    print(f"  Unique deals: {df['deal_id'].nunique()}")
    print(f"  Date range: {df['date_snapshot'].min().date()} to {df['date_snapshot'].max().date()}")
    
    # Latest snapshot summary
    latest = df['date_snapshot'].max()
    df_latest = df[df['date_snapshot'] == latest]
    
    print(f"\n  Stage distribution (latest snapshot {latest.date()}):")
    for stage, count in df_latest['stage'].value_counts().items():
        print(f"    {stage:20s}: {count:>4}")
    
    print(f"\n  Segment distribution (latest snapshot):")
    for seg, count in df_latest['market_segment'].value_counts().items():
        print(f"    {seg:15s}: {count:>4}")
    
    # Historical win summary
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    df_won = df_final[df_final['stage'] == CLOSED_WON]
    df_lost = df_final[df_final['stage'] == CLOSED_LOST]
    
    print(f"\n  Historical outcomes:")
    print(f"    Won:  {len(df_won):>4} deals, ${df_won['net_revenue'].sum():>12,.0f}")
    print(f"    Lost: {len(df_lost):>4} deals, ${df_lost['net_revenue'].sum():>12,.0f}")
    print(f"    Open: {len(df_final) - len(df_won) - len(df_lost):>4} deals (pipeline)")
    
    # Monthly velocity check
    df_first = df.sort_values('date_snapshot').groupby('deal_id').first().reset_index()
    df_first['appear_month'] = df_first['date_snapshot'].dt.to_period('M')
    monthly_vel = df_first.groupby('appear_month').size()
    
    print(f"\n  Monthly deal creation velocity (last 12 months):")
    for period, count in monthly_vel.tail(12).items():
        print(f"    {period}: {count:>3} deals")
    
    print(f"\n--- 5. Saving Output ---")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}")
    
    print(f"\n  Sample records:")
    print(df[['deal_id', 'market_segment', 'stage', 'net_revenue', 'date_snapshot']].head(10).to_string())
    
    print("\n" + "=" * 70)
    print("MOCK DATA GENERATION COMPLETE")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    run_history_generator()