import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION & ASSUMPTIONS
# ==========================================
ASSUMPTIONS = {
    "volume_growth_multiplier": 1.10,   # 10% Increase in deal volume
    "win_rate_uplift_multiplier": 1.05,  # 5% Increase in win efficiency
    "deal_size_inflation": 1.03,        # 3% Increase in deal value
    # If not None, overrides historical lag. 
    "manual_cycle_lag_months": None     
}

# Paths
BASE_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(BASE_DIR, "fact_pipeline_snapshot.csv")

# Output Paths (For validation)
OUTPUT_DRIVER_WIN = os.path.join(BASE_DIR, "driver_win_rates.csv")
OUTPUT_DRIVER_VEL = os.path.join(BASE_DIR, "driver_velocity.csv")
OUTPUT_DETAILED = os.path.join(BASE_DIR, "forecast_detailed_2026.csv")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "forecast_summary_2026.csv")

class Deal:
    def __init__(self, deal_id, deal_name, segment, revenue, create_date, close_date, is_won, source="Ghost"):
        self.deal_id = deal_id
        self.deal_name = deal_name
        self.segment = segment
        self.revenue = revenue
        self.create_date = create_date
        self.target_close_date = close_date
        self.is_won = is_won
        self.source = source # 'Existing' or 'Ghost'
        self.status = "Qualified" # Default start state
        self.actual_close_date = None

    def update_status(self, current_date):
        # If we passed the close date, flip to final status
        if current_date >= self.target_close_date:
            self.status = "Closed Won" if self.is_won else "Closed Lost"
            self.actual_close_date = self.target_close_date
        else:
            self.status = "Qualified" 

# ==========================================
# PART 1: DRIVER CALCULATIONS (The "Silver Layer")
# ==========================================

def calculate_win_rates(df):
    """Calculates historical win rates by cohort and segment."""
    print("  > Calculating Win Rates...")
    df['date_created'] = pd.to_datetime(df['date_created'])
    
    # Get final state of every deal
    df_final = df.sort_values('snapshot_date', ascending=False).groupby('deal_id').first().reset_index()
    df_final['cohort_month'] = df_final['date_created'].dt.to_period('M').dt.to_timestamp()
    
    # Filter for relevant history (2024-2025)
    df_final = df_final[df_final['date_created'].dt.year.isin([2024, 2025])]
    
    # Group by Segment
    # We calculate the overall weighted win rate per segment
    segment_stats = []
    win_rates_dict = {}
    
    for segment, group in df_final.groupby('market_segment'):
        total_created = group['revenue'].sum()
        won_rev = group[group['status'] == 'Closed Won']['revenue'].sum()
        
        win_rate = (won_rev / total_created) if total_created > 0 else 0
        win_rates_dict[segment] = win_rate
        
        segment_stats.append({
            'market_segment': segment,
            'total_created_revenue': total_created,
            'total_won_revenue': won_rev,
            'win_rate_pct': win_rate
        })
    
    # Export for validation
    pd.DataFrame(segment_stats).to_csv(OUTPUT_DRIVER_WIN, index=False)
    return win_rates_dict

def calculate_velocity(df):
    """Calculates monthly deal creation velocity and avg size per segment."""
    print("  > Calculating Velocity...")
    # Filter for Qualified events
    df_qual = df[df['status'] == 'Qualified'].copy()
    
    # Find first Qualified date per deal
    df_vel = df_qual.groupby(['deal_id', 'market_segment'])['snapshot_date'].min().reset_index()
    df_vel.rename(columns={'snapshot_date': 'qualified_date'}, inplace=True)
    
    # Filter for Last FY (2025) to model seasonality/run-rate
    df_vel = df_vel[df_vel['qualified_date'].dt.year == 2025]
    
    # Join back to get Revenue at time of qualification
    # (Simplified: using latest revenue from raw df for that deal)
    # Ideally we'd join on snapshot date, but let's grab the deal's size from the main df
    df_sizes = df.groupby('deal_id')['revenue'].mean().reset_index()
    df_vel = pd.merge(df_vel, df_sizes, on='deal_id', how='left')
    
    df_vel['month_num'] = df_vel['qualified_date'].dt.month
    
    # Build Dictionary: {Segment: {MonthNum: {'vol': X, 'size': Y}}}
    velocity_dict = {}
    
    # Validation Export List
    export_rows = []
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_vel[df_vel['market_segment'] == seg]
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['month_num'] == m]
            
            if not m_data.empty:
                vol = len(m_data)
                size = m_data['revenue'].mean()
            else:
                # Fallback to annual average if month is missing
                vol = len(seg_data) / 12 if not seg_data.empty else 0
                size = seg_data['revenue'].mean() if not seg_data.empty else 0
            
            velocity_dict[seg][m] = {'vol': vol, 'size': size}
            
            export_rows.append({
                'market_segment': seg,
                'month': m,
                'avg_volume': vol,
                'avg_size': size
            })
            
    pd.DataFrame(export_rows).to_csv(OUTPUT_DRIVER_VEL, index=False)
    return velocity_dict

def calculate_lags(df):
    """Calculates average days to close per segment."""
    print("  > Calculating Lags...")
    df_won = df[df['status'] == 'Closed Won'].copy()
    df_won['date_created'] = pd.to_datetime(df_won['date_created'])
    df_won['target_implementation_date'] = pd.to_datetime(df_won['target_implementation_date'])
    
    df_won['cycle_days'] = (df_won['target_implementation_date'] - df_won['date_created']).dt.days
    
    cycle_lags = df_won.groupby('market_segment')['cycle_days'].mean().to_dict()
    # Convert to months (min 1)
    return {k: int(max(1, v/30)) for k, v in cycle_lags.items()}

# ==========================================
# PART 2: INITIALIZATION
# ==========================================

def initialize_active_deals(df, win_rates):
    """Loads existing open deals from the final snapshot."""
    print("  > Initializing Open Pipeline...")
    latest_date = df['snapshot_date'].max()
    print(f"    Latest Snapshot: {latest_date.date()}")
    
    df_latest = df[df['snapshot_date'] == latest_date].copy()
    excluded = ['Closed Won', 'Closed Lost', 'Initiated', 'Verbal', 'Declined to Bid']
    df_open = df_latest[~df_latest['status'].isin(excluded)].copy()
    
    active_deals = []
    
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        
        # Determine Fate based on Win Rate
        base_wr = win_rates.get(seg, 0.2)
        adj_wr = min(base_wr * ASSUMPTIONS['win_rate_uplift_multiplier'], 1.0)
        is_won = random.random() < adj_wr
        
        close_date = pd.to_datetime(row['target_implementation_date'])
        if close_date <= latest_date: close_date = latest_date + timedelta(days=30)
            
        deal = Deal(
            deal_id=row['deal_id'],
            deal_name=row['deal_name'], # Real Name
            segment=seg,
            revenue=row['revenue'],
            create_date=pd.to_datetime(row['date_created']),
            close_date=close_date,
            is_won=is_won,
            source="Existing"
        )
        deal.status = row['status'] 
        active_deals.append(deal)
        
    return active_deals

# ==========================================
# PART 3: MAIN EXECUTION
# ==========================================

def run_simulation():
    print("--- 1. Ingesting Data & Calculating Drivers ---")
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print("Error: snapshot file not found.")
        return

    df_raw = pd.read_csv(PATH_RAW_SNAPSHOTS)
    df_raw['snapshot_date'] = pd.to_datetime(df_raw['snapshot_date'])

    # Calc Drivers
    win_rates = calculate_win_rates(df_raw)
    velocity = calculate_velocity(df_raw)
    lags = calculate_lags(df_raw)

    print(f"    Drivers exported to {BASE_DIR}")

    print("--- 2. Setting up Simulation ---")
    active_deals = initialize_active_deals(df_raw, win_rates)
    
    # Sim dates: Weekly 2026
    sim_dates = pd.date_range(start='2026-01-01', end='2026-12-31', freq='W-FRI')
    snapshot_rows = []
    ghost_counter = 1
    
    print(f"--- 3. Running Simulation ({len(sim_dates)} weeks) ---")
    
    for current_date in sim_dates:
        curr_month = current_date.month
        
        # A. GENERATE NEW DEALS (Ghost)
        for seg, monthly_data in velocity.items():
            if curr_month not in monthly_data: continue
            
            stats = monthly_data[curr_month]
            # Weekly Target = (Monthly Vol * Growth) / 4.3
            weekly_vol_target = (stats['vol'] * ASSUMPTIONS['volume_growth_multiplier']) / 4.3
            num_to_create = np.random.poisson(weekly_vol_target)
            
            if num_to_create > 0:
                avg_size = stats['size'] * ASSUMPTIONS['deal_size_inflation']
                base_wr = win_rates.get(seg, 0.2)
                adj_wr = min(base_wr * ASSUMPTIONS['win_rate_uplift_multiplier'], 1.0)
                lag_months = ASSUMPTIONS['manual_cycle_lag_months'] if ASSUMPTIONS['manual_cycle_lag_months'] else lags.get(seg, 4)
                
                for _ in range(num_to_create):
                    is_won = random.random() < adj_wr
                    actual_rev = int(avg_size * random.uniform(0.9, 1.1))
                    
                    target_close = current_date + timedelta(days=int(lag_months * 30))
                    target_close += timedelta(days=random.randint(-15, 15))
                    
                    ghost_id = f"Ghost-{seg[:3].upper()}-{ghost_counter}"
                    ghost_name = f"Ghost Deal {ghost_counter} ({seg})"
                    
                    new_deal = Deal(
                        deal_id=ghost_id,
                        deal_name=ghost_name,
                        segment=seg,
                        revenue=actual_rev,
                        create_date=current_date,
                        close_date=target_close,
                        is_won=is_won,
                        source="Ghost"
                    )
                    active_deals.append(new_deal)
                    ghost_counter += 1

        # B. UPDATE STATUS & SNAPSHOT
        for deal in active_deals:
            deal.update_status(current_date)
            
            snapshot_rows.append({
                "snapshot_date": current_date.strftime('%Y-%m-%d'),
                "deal_id": deal.deal_id,
                "deal_name": deal.deal_name,
                "market_segment": deal.segment,
                "status": deal.status,
                "revenue": deal.revenue,
                "date_created": deal.create_date.strftime('%Y-%m-%d'),
                "target_implementation_date": deal.target_close_date.strftime('%Y-%m-%d'),
                "source": deal.source
            })

    # Save Detailed
    df_detail = pd.DataFrame(snapshot_rows)
    df_detail.to_csv(OUTPUT_DETAILED, index=False)
    print(f"    Detailed Forecast Saved: {len(df_detail)} rows")
    
    print("--- 4. Aggregating Summary ---")
    df_detail['snapshot_date'] = pd.to_datetime(df_detail['snapshot_date'])
    df_detail['forecast_month'] = df_detail['snapshot_date'].dt.to_period('M').dt.to_timestamp()
    
    # Take last snapshot of month per deal
    df_monthly = df_detail.sort_values('snapshot_date').groupby(['forecast_month', 'deal_id']).tail(1)
    
    summary_rows = []
    
    for month, group in df_monthly.groupby('forecast_month'):
        for seg, seg_group in group.groupby('market_segment'):
            
            # Created (New Ghost Deals only)
            created_mask = (pd.to_datetime(seg_group['date_created']).dt.to_period('M') == month.to_period('M')) & (seg_group['source'] == 'Ghost')
            deals_created_count = seg_group[created_mask]['deal_id'].nunique()
            
            # Won (Closed Won in this month)
            won_mask = (seg_group['status'] == 'Closed Won') & \
                       (pd.to_datetime(seg_group['target_implementation_date']).dt.to_period('M') == month.to_period('M'))
            won_rev = seg_group[won_mask]['revenue'].sum()
            won_vol = seg_group[won_mask]['deal_id'].nunique()
            
            # Stock (Open Qualified)
            open_mask = (seg_group['status'] == 'Qualified')
            open_count = seg_group[open_mask]['deal_id'].nunique()
            open_val = seg_group[open_mask]['revenue'].sum()
            
            summary_rows.append({
                "forecast_month": month,
                "market_segment": seg,
                "deals_created": deals_created_count,
                "forecasted_won_volume": won_vol,
                "forecasted_won_rev": won_rev,
                "open_pipeline_count": open_count,
                "open_pipeline_value": open_val
            })
            
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"    Summary Saved: {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    run_simulation()