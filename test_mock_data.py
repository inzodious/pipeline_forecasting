"""
Simple mock data test to verify forecasting logic
Creates a controlled dataset with known outcomes to test against
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_simple_test_data():
    """
    Generate a simple, predictable dataset:
    - 10 deals per month in 2024-2025 for Large Market
    - Each deal worth $100k
    - 50% win rate (5 wins, 5 losses per month)
    - Deals close exactly 12 weeks after creation
    
    Expected 2025 actuals: 10 deals/month × $100k × 50% = $500k/month × 12 = $6M
    """
    
    rows = []
    deal_id = 1
    
    # Generate deals for 2024-2025
    for year in [2024, 2025]:
        for month in range(1, 13):
            created_date = datetime(year, month, 1)
            
            # Create 10 deals per month
            for i in range(10):
                deal_created = created_date
                
                # First 5 deals win, next 5 lose
                if i < 5:
                    outcome = 'Closed Won'
                    close_date = deal_created + timedelta(weeks=12)
                else:
                    outcome = 'Closed Lost'
                    close_date = deal_created + timedelta(weeks=12)
                
                # Generate weekly snapshots from creation to close
                current_date = deal_created
                while current_date <= close_date:
                    if current_date < close_date:
                        stage = 'Qualified'  # Simple - just one stage before close
                    else:
                        stage = outcome
                    
                    rows.append({
                        'deal_id': f'DEAL-{deal_id}',
                        'date_created': deal_created,
                        'date_closed': close_date if current_date == close_date else None,
                        'date_snapshot': current_date,
                        'stage': stage,
                        'net_revenue': 100000,
                        'market_segment': 'Large Market'
                    })
                    
                    current_date += timedelta(weeks=1)
                
                deal_id += 1
    
    df = pd.DataFrame(rows)
    
    # Save to test location
    df.to_csv('./data/test_snapshots.csv', index=False)
    
    print(f"Generated {len(df)} snapshot rows for {deal_id-1} deals")
    
    # Calculate expected actuals
    closed = df[df['stage'].isin(['Closed Won', 'Closed Lost'])].copy()
    closed = closed.drop_duplicates(subset=['deal_id'])
    
    won = closed[closed['stage'] == 'Closed Won']
    won['close_month'] = pd.to_datetime(won['date_closed']).dt.to_period('M')
    
    actuals = won.groupby('close_month').agg(
        deals=('deal_id', 'count'),
        revenue=('net_revenue', 'sum')
    )
    
    print("\nExpected Actuals by Month:")
    print(actuals)
    print(f"\nTotal 2024 Revenue: ${actuals[actuals.index.year == 2024]['revenue'].sum():,.0f}")
    print(f"Total 2025 Revenue: ${actuals[actuals.index.year == 2025]['revenue'].sum():,.0f}")
    
    return df

if __name__ == "__main__":
    generate_simple_test_data()
