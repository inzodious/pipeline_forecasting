import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fact_snapshots.csv")

FISCAL_YEARS = [2024, 2025]
DEALS_PER_YEAR = 36
DEALS_PER_MONTH = 3

# 2023 Carryover Pipeline (deals created 30-90 days before 2024)
CARRYOVER_CONFIG = {
    'enabled': True,
    'count': 12,  # Slightly less than monthly run rate
    'days_before_start': (30, 90),  # Created 30-90 days before Jan 1, 2024
    'segment_distribution': {
        'Large Market': 1,
        'Mid Market': 4,
        'Small Market': 7
    }
}

SEGMENT_CONFIG = {
    'Large Market': {
        'count_per_year': 4,
        'revenue_min': 100000,
        'revenue_max': 300000,
        'win_rate': 0.15
    },
    'Mid Market': {
        'count_per_year': 12,
        'revenue_min': 50000,
        'revenue_max': 99000,
        'win_rate': 0.20
    },
    'Small Market': {
        'count_per_year': 20,
        'revenue_min': 25000,
        'revenue_max': 49999,
        'win_rate': 0.25
    }
}

STAGES = ['Qualified', 'Solutioning', 'Alignment']
CLOSED_WON = 'Closed Won'
CLOSED_LOST = 'Closed Lost'

DEAL_OWNER = "Joshua Biondo"

CLOSE_DAYS_MIN = 30
CLOSE_DAYS_MAX = 90

IMPL_DAYS_MIN = 60
IMPL_DAYS_MAX = 180

SNAPSHOT_INTERVAL_DAYS = 7

COMPANY_PREFIXES = [
    "Apex", "Summit", "Pinnacle", "Horizon", "Velocity", "Catalyst", "Synergy",
    "Vertex", "Elevate", "Quantum", "Nexus", "Vanguard", "Prism", "Compass",
    "Eclipse", "Momentum", "Phoenix", "Atlas", "Fusion", "Sterling", "Ember",
    "Cobalt", "Crimson", "Nova", "Orion", "Titan", "Pulse", "Radiant", "Vector",
    "Zenith", "Aurora", "Cascade", "Delta", "Epoch", "Flux", "Granite", "Helix"
]

COMPANY_SUFFIXES = [
    "Industries", "Solutions", "Technologies", "Systems", "Group", "Corp",
    "Enterprises", "Partners", "Holdings", "Services", "Dynamics", "Labs",
    "Innovations", "Networks", "Ventures", "Global", "Digital", "Analytics"
]


# ==========================================
# DEAL GENERATION
# ==========================================

def generate_deal_name(used_names):
    """Generate a unique company/deal name."""
    for _ in range(1000):
        prefix = random.choice(COMPANY_PREFIXES)
        suffix = random.choice(COMPANY_SUFFIXES)
        name = f"{prefix} {suffix}"
        if name not in used_names:
            used_names.add(name)
            return name
    raise ValueError("Could not generate unique deal name")


def generate_deal_id(year, index):
    """Generate unique deal ID."""
    return f"DEAL-{year}-{index:04d}"


def distribute_deals_across_months(year, segment_counts):
    """
    Distribute deals evenly across 12 months (3 per month).
    Returns list of (month, segment) tuples.
    """
    deals_by_month = {m: [] for m in range(1, 13)}
    
    all_deals = []
    for segment, count in segment_counts.items():
        all_deals.extend([segment] * count)
    
    random.shuffle(all_deals)
    
    month = 1
    for segment in all_deals:
        deals_by_month[month].append(segment)
        month = (month % 12) + 1
    
    result = []
    for month in range(1, 13):
        for segment in deals_by_month[month]:
            result.append((month, segment))
    
    return result


def generate_carryover_deals(deal_counter, used_names):
    """
    Generate deals from late 2023 that carry over into 2024.
    These represent the open pipeline at the start of our simulation.
    """
    if not CARRYOVER_CONFIG.get('enabled', False):
        return [], deal_counter
    
    carryover_deals = []
    start_date = datetime(2024, 1, 1)
    days_min, days_max = CARRYOVER_CONFIG['days_before_start']
    
    for segment, count in CARRYOVER_CONFIG['segment_distribution'].items():
        config = SEGMENT_CONFIG[segment]
        
        for _ in range(count):
            deal_id = generate_deal_id(2023, deal_counter)
            deal_name = generate_deal_name(used_names)
            
            days_before = random.randint(days_min, days_max)
            date_created = start_date - timedelta(days=days_before)
            
            close_days = random.randint(CLOSE_DAYS_MIN, CLOSE_DAYS_MAX)
            date_closed = date_created + timedelta(days=close_days)
            
            impl_days = random.randint(IMPL_DAYS_MIN, IMPL_DAYS_MAX)
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


def generate_deal(deal_id, deal_name, year, month, segment, config):
    """Generate a single deal with all its attributes."""
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0:
        days_in_month[1] = 29
    
    create_day = random.randint(1, days_in_month[month - 1])
    date_created = datetime(year, month, create_day)
    
    close_days = random.randint(CLOSE_DAYS_MIN, CLOSE_DAYS_MAX)
    date_closed = date_created + timedelta(days=close_days)
    
    impl_days = random.randint(IMPL_DAYS_MIN, IMPL_DAYS_MAX)
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


# ==========================================
# SNAPSHOT SIMULATION
# ==========================================

def calculate_stage_transitions(deal):
    """
    Calculate when a deal transitions through each stage.
    Stages: Qualified -> Solutioning -> Alignment -> Closed Won/Lost
    """
    close_days = deal['close_days']
    
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
    """
    Generate snapshot records for a deal across all snapshot dates.
    Only includes snapshots from deal creation onward.
    
    date_closed logic:
    - If deal closes in 2024 or 2025: backfill date_closed on ALL rows
    - If deal closes in 2026: keep as open pipeline (null date_closed, open stage)
    """
    close_year = deal['date_closed'].year
    is_future_close = close_year >= 2026
    
    transitions = calculate_stage_transitions(deal)
    
    snapshots = []
    
    for snap_date in snapshot_dates:
        if snap_date < deal['date_created']:
            continue
        
        if is_future_close:
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
    """Generate weekly snapshot dates covering the simulation period."""
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
    print("HISTORY GENERATOR - Mock Deal Snapshot Data")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    used_names = set()
    all_deals = []
    deal_counter = 1
    
    print("\n--- 1. Generating Deals ---")
    
    # Generate 2023 carryover pipeline
    if CARRYOVER_CONFIG.get('enabled', False):
        print(f"\n  2023 Carryover Pipeline:")
        carryover_deals, deal_counter = generate_carryover_deals(deal_counter, used_names)
        all_deals.extend(carryover_deals)
        
        carryover_stats = {'Large Market': 0, 'Mid Market': 0, 'Small Market': 0}
        carryover_won = {'Large Market': 0, 'Mid Market': 0, 'Small Market': 0}
        for deal in carryover_deals:
            carryover_stats[deal['market_segment']] += 1
            if deal['final_stage'] == CLOSED_WON:
                carryover_won[deal['market_segment']] += 1
        
        for seg in SEGMENT_CONFIG.keys():
            actual_wr = (carryover_won[seg] / carryover_stats[seg] * 100) if carryover_stats[seg] > 0 else 0
            print(f"    {seg:15s}: {carryover_stats[seg]:>2} deals, {carryover_won[seg]:>2} won ({actual_wr:.0f}%)")
        
        print(f"    Created: {min(d['date_created'] for d in carryover_deals).date()} to {max(d['date_created'] for d in carryover_deals).date()}")
    
    # Generate FY 2024 and 2025 deals
    for year in FISCAL_YEARS:
        print(f"\n  FY {year}:")
        
        segment_counts = {seg: cfg['count_per_year'] for seg, cfg in SEGMENT_CONFIG.items()}
        deal_distribution = distribute_deals_across_months(year, segment_counts)
        
        year_deals = {'Large Market': 0, 'Mid Market': 0, 'Small Market': 0}
        year_won = {'Large Market': 0, 'Mid Market': 0, 'Small Market': 0}
        
        for month, segment in deal_distribution:
            deal_id = generate_deal_id(year, deal_counter)
            deal_name = generate_deal_name(used_names)
            
            deal = generate_deal(
                deal_id=deal_id,
                deal_name=deal_name,
                year=year,
                month=month,
                segment=segment,
                config=SEGMENT_CONFIG[segment]
            )
            
            all_deals.append(deal)
            year_deals[segment] += 1
            if deal['final_stage'] == CLOSED_WON:
                year_won[segment] += 1
            
            deal_counter += 1
        
        for seg in SEGMENT_CONFIG.keys():
            actual_wr = (year_won[seg] / year_deals[seg] * 100) if year_deals[seg] > 0 else 0
            print(f"    {seg:15s}: {year_deals[seg]:>2} deals, {year_won[seg]:>2} won ({actual_wr:.0f}%)")
    
    print(f"\n  Total deals generated: {len(all_deals)}")
    
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
    
    print(f"\n  Stage distribution (latest snapshot):")
    latest = df['date_snapshot'].max()
    df_latest = df[df['date_snapshot'] == latest]
    for stage, count in df_latest['stage'].value_counts().items():
        print(f"    {stage:20s}: {count}")
    
    print(f"\n  Segment distribution:")
    for seg, count in df_latest['market_segment'].value_counts().items():
        print(f"    {seg:15s}: {count}")
    
    print(f"\n--- 5. Saving Output ---")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}")
    
    print(f"\n  Sample records:")
    print(df[['deal_id', 'market_segment', 'stage', 'net_revenue', 'date_snapshot']].head(10).to_string())
    
    print("\n" + "=" * 70)
    print("HISTORY GENERATION COMPLETE")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    run_history_generator()