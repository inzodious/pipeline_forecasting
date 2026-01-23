import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_crm_data():
    # --- Configuration ---
    # REPORTING Start Date (When you want the snapshots to begin)
    reporting_start_date = datetime(2024, 1, 1)
    
    # CREATION Start Date (Backdated 6 months to "Warm up" the pipeline)
    # This ensures Jan 2024 has deals closing immediately.
    creation_start_date = reporting_start_date - timedelta(days=180)
    
    # Generate the Reporting Snapshots (Jan 2024 - Jan 2026)
    snapshot_dates = pd.date_range(start=reporting_start_date, periods=105, freq='W-FRI')
    
    # Generate the Deal Creation Weeks (July 2023 - Jan 2026)
    creation_weeks = pd.date_range(start=creation_start_date, end=snapshot_dates[-1], freq='W-FRI')
    
    market_segments = [
        {"name": "Small Market", "pop_range": (50, 500), "rev_range": (10000, 45000)},
        {"name": "Mid-Market", "pop_range": (501, 2500), "rev_range": (45001, 120000)},
        {"name": "Large Market", "pop_range": (2501, 15000), "rev_range": (120001, 300000)},
        {"name": "Global Market", "pop_range": (15001, 100000), "rev_range": (300001, 500000)}
    ]

    deal_names_pool = ["CloudScale","DataVantage","Nexus","Apex","Synergy","Vertex","Quantum","Beacon","IronGate","SilverLine","CoreTech","Stratosphere","OmniCore","Fluxion","Zenith","Horizon","Pinnacle","Aether","Vanguard","Catalyst","BlueShift","Hyperion","NovaCore","PrimeAxis","Ironclad","Summit","Parallax","ClearPath","NextGen","Helix","Atlas","Fusion","EdgePoint","Sentinel","BlackRocked","Optimum","HighMark","Ascendant","Continuum","NorthStar","Pulse","Keystone","Vector","Skyward","DeepCore","Elevate","Fortress","Stronghold","Frontier","Overwatch","PrimeWave","DataForge","SignalOne","TrueNorth","Gatekeeper","SteelPeak","Command","Pathfinder","IronPeak","BlueCore","Momentum","Everest","CrossPoint","Foundation","HawkEye","Longview","Powerhouse","Brightline","Redstone","Stonebridge","Northbound","Unity","AxisOne","Skyline","Trailhead","Bedrock","Cornerstone","SummitOne","PrimeLine","Clearwater"]
    suffixes = ["Logistics", "Enterprises", "Networks", "Industries", "Group", "Global", "Technologies", "Corp", "Systems", "Solutions"]

    # --- Step 1: Generate Base Deals (Using the Extended Timeline) ---
    all_deals = []
    current_deal_id = 1001
    used_deal_names = set() 

    # We iterate through creation_weeks, not snapshot_dates, to fill the funnel early
    for week_date in creation_weeks:
        year = week_date.year
        
        # FY25+ boost logic
        base_rate = random.randint(5, 10)
        if year >= 2025:
            base_rate = int(base_rate * 1.25)
        
        for _ in range(base_rate):
            segment = random.choice(market_segments)
            pop = random.randint(*segment['pop_range'])
            initial_revenue = random.randint(*segment['rev_range'])
            
            unique_name_found = False
            while not unique_name_found:
                new_name = f"{random.choice(deal_names_pool)} {random.choice(suffixes)}"
                new_name = f"{new_name} {random.randint(10, 999)}" 
                if new_name not in used_deal_names:
                    used_deal_names.add(new_name)
                    unique_name_found = True
            
            creation_date = week_date
            
            # Staging Logic
            reaches_qualified = random.random() < 0.65
            qualified_delay = random.randint(14, 28)
            solutioning_delay = qualified_delay + random.randint(28, 56)
            
            # --- NEW LOGIC: "Fast Track" Deals to prevent empty months ---
            # 20% of deals are "Fast Track" and close in 15-45 days
            is_fast_track = random.random() < 0.20
            
            if is_fast_track:
                days_to_close = random.randint(15, 45)
            else:
                # Normal distribution for standard deals
                days_to_close = int(np.random.normal(180, 90))
                if reaches_qualified:
                    days_to_close = max(solutioning_delay + 7, days_to_close)
                else:
                    days_to_close = max(30, days_to_close)
            
            closed_date = creation_date + timedelta(days=days_to_close)
            
            # Implementation date logic
            imp_date = creation_date + timedelta(days=random.randint(90, 180))
            
            status = "Closed Won" if random.random() < 0.15 else "Closed Lost"
            
            all_deals.append({
                "deal_id": current_deal_id,
                "deal_name": new_name,
                "market_segment": segment['name'],
                "employee_count": pop, 
                "date_created": creation_date,
                "target_implementation_date": imp_date,
                "date_closed": closed_date,
                "final_status": status,
                "reaches_qualified": reaches_qualified,
                "qualified_date": creation_date + timedelta(days=qualified_delay),
                "solutioning_date": creation_date + timedelta(days=solutioning_delay),
                "base_revenue": initial_revenue
            })
            current_deal_id += 1

    # --- Step 2: Generate Snapshot Records (Using ONLY Reporting Dates) ---
    rows = []
    
    # We only output rows for the requested 2 years (2024-2025)
    # But the deals from 2023 will naturally flow into these snapshots as "Active" or "Closed"
    for snapshot in snapshot_dates:
        # Include deals created before this snapshot
        active_deals = [d for d in all_deals if d['date_created'] <= snapshot]
        
        for d in active_deals:
            is_closed = snapshot >= d['date_closed']
            
            if is_closed:
                current_status = d['final_status']
                current_imp_date = d['target_implementation_date']
                actual_close_date = d['date_closed'].strftime('%Y-%m-%d')
            else:
                actual_close_date = None
                
                if d['reaches_qualified']:
                    if snapshot >= d['solutioning_date']:
                        current_status = "Solutioning"
                    elif snapshot >= d['qualified_date']:
                        current_status = "Qualified"
                    else:
                        current_status = "Initiated"
                else:
                    current_status = "Initiated"
                
                # Implementation date shift logic
                current_imp_date = d['target_implementation_date']
                if random.random() < 0.10:
                    shift = random.randint(-7, 21) 
                    current_imp_date += timedelta(days=shift)
                    d['target_implementation_date'] = current_imp_date

            current_rev = d['base_revenue']
            if not is_closed and random.random() < 0.20:
                fluctuation = random.uniform(0.98, 1.02) 
                current_rev = int(current_rev * fluctuation)
                d['base_revenue'] = current_rev

            rows.append({
                "snapshot_date": snapshot.strftime('%Y-%m-%d'),
                "deal_id": d['deal_id'],
                "deal_name": d['deal_name'],
                "market_segment": d['market_segment'],
                "population": d['employee_count'], 
                "status": current_status,
                "revenue": current_rev,
                "date_created": d['date_created'].strftime('%Y-%m-%d'),
                "target_implementation_date": current_imp_date.strftime('%Y-%m-%d'),
                "actual_close_date": actual_close_date,
                "is_closed": 1 if is_closed else 0
            })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    print("Generating B2B CRM Snapshots with Backdated Pipeline Warm-up...")
    df_snapshots = generate_crm_data()
    
    output_dir = "data"
    filename = "fact_pipeline_snapshot.csv"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_path = os.path.join(output_dir, filename)
    df_snapshots.to_csv(file_path, index=False)
    
    print(f"Successfully generated {len(df_snapshots)} rows.")
    print(f"File saved as: {file_path}")