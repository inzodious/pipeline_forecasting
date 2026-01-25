import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import random

def generate_mock_data(output_path='data/fact_snapshots.csv', seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    segments = {
        'Large Market': {
            'deals_per_month': 6,
            'avg_revenue': 180000,
            'revenue_std': 60000,
            'win_rate': 0.32,
            'avg_days': 140,
            'days_std': 50,
            'qualified_entry_rate': 0.70,
        },
        'Mid Market/SMB': {
            'deals_per_month': 20,
            'avg_revenue': 40000,
            'revenue_std': 18000,
            'win_rate': 0.42,
            'avg_days': 65,
            'days_std': 28,
            'qualified_entry_rate': 0.68,
        },
        'Indirect': {
            'deals_per_month': 22,
            'avg_revenue': 55000,
            'revenue_std': 35000,
            'win_rate': 0.48,
            'avg_days': 50,
            'days_std': 22,
            'qualified_entry_rate': 0.65,
        },
    }
    
    stage_durations = {'Qualified': (14, 35), 'Alignment': (7, 28), 'Solutioning': (14, 50)}
    
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2025-12-31')
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    deals = []
    deal_counter = 1
    
    for month_start in months:
        for segment, seg in segments.items():
            num_deals = max(1, int(np.random.poisson(seg['deals_per_month'])))
            
            for _ in range(num_deals):
                created_date = month_start + timedelta(days=random.randint(0, 27))
                if created_date > end_date:
                    continue
                
                enters_qualified = random.random() < seg['qualified_entry_rate']
                will_win = random.random() < seg['win_rate']
                days_to_close = max(7, int(np.random.normal(seg['avg_days'], seg['days_std'])))
                revenue = max(5000, np.random.normal(seg['avg_revenue'], seg['revenue_std']))
                
                lifecycle = []
                
                if not enters_qualified and not will_win:
                    close_date = created_date + timedelta(days=random.randint(1, 14))
                    lifecycle.append({'stage': 'Closed Lost', 'start_date': created_date, 'end_date': close_date})
                elif not enters_qualified and will_win:
                    close_date = created_date + timedelta(days=random.randint(7, 30))
                    lifecycle.append({'stage': 'Closed Won', 'start_date': close_date, 'end_date': None})
                else:
                    current_date = created_date
                    num_stages = random.choices([1, 2, 3], weights=[0.15, 0.30, 0.55] if will_win else [0.40, 0.35, 0.25])[0]
                    remaining_days = days_to_close
                    
                    for i, stage in enumerate(['Qualified', 'Alignment', 'Solutioning'][:num_stages]):
                        min_dur, max_dur = stage_durations[stage]
                        duration = max(7, remaining_days if i == num_stages - 1 else random.randint(min_dur, max_dur))
                        remaining_days -= duration
                        end_dt = current_date + timedelta(days=duration)
                        lifecycle.append({'stage': stage, 'start_date': current_date, 'end_date': end_dt})
                        current_date = end_dt
                    
                    lifecycle.append({'stage': 'Closed Won' if will_win else 'Closed Lost', 'start_date': current_date, 'end_date': None})
                    close_date = current_date
                
                if close_date and close_date > end_date:
                    close_date = None
                    lifecycle = [s for s in lifecycle if s['stage'] not in ['Closed Won', 'Closed Lost']]
                    if not lifecycle:
                        lifecycle = [{'stage': 'Qualified', 'start_date': created_date, 'end_date': None}]
                    else:
                        lifecycle[-1]['end_date'] = None
                
                deals.append({
                    'deal_id': f"DEAL_{deal_counter:05d}",
                    'date_created': created_date,
                    'date_closed': close_date,
                    'net_revenue': round(revenue, 2),
                    'market_segment': segment,
                    'lifecycle': lifecycle
                })
                deal_counter += 1
    
    snapshot_dates = pd.date_range(start=start_date, end=end_date, freq='7D')
    snapshots = []
    
    for snapshot_date in snapshot_dates:
        for deal in deals:
            if pd.to_datetime(deal['date_created']) > snapshot_date:
                continue
            
            current_stage = None
            for stage_info in deal['lifecycle']:
                stage_start = pd.to_datetime(stage_info['start_date'])
                stage_end = pd.to_datetime(stage_info['end_date']) if stage_info['end_date'] else None
                
                if stage_end is None and stage_start <= snapshot_date:
                    current_stage = stage_info['stage']
                    break
                elif stage_end and stage_start <= snapshot_date < stage_end:
                    current_stage = stage_info['stage']
                    break
                elif stage_info['stage'] in ['Closed Won', 'Closed Lost'] and stage_start <= snapshot_date:
                    current_stage = stage_info['stage']
                    break
            
            if current_stage:
                snapshots.append({
                    'deal_id': deal['deal_id'],
                    'date_created': deal['date_created'],
                    'date_closed': deal['date_closed'] if deal['date_closed'] and pd.to_datetime(deal['date_closed']) <= snapshot_date else None,
                    'date_snapshot': snapshot_date,
                    'stage': current_stage,
                    'net_revenue': deal['net_revenue'],
                    'market_segment': deal['market_segment']
                })
    
    df = pd.DataFrame(snapshots)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(deals)} deals, {len(df)} snapshot records")
    print(f"Saved to {output_path}")
    
    return df


if __name__ == "__main__":
    generate_mock_data()