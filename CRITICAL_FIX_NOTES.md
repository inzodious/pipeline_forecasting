# Critical Issue Found in Vintage Curve Logic

## The Problem

The vintage curve is being calculated INCORRECTLY. Currently:

```python
won = deals[deals['won'] & deals['date_closed'].notna()].copy()
# ... calculate curve from ONLY won deals
curve['cum_win_pct'] = curve['won_count'] / curve['total_wins']
```

This creates a curve that shows: **"Of all deals that will WIN, what % have won by week N"**

This curve always asymptotes to **100%** because we're only looking at winning deals!

## What It Should Be

The curve should show: **"Of all deals CREATED, what % have closed WON by week N"**

This curve would asymptote to the **actual win rate** (e.g., 20-30%), not 100%.

## Why This Causes Massive Over-Forecasting

When we forecast future cohorts:
1. We calculate total cohort value = volume × deal size
2. We multiply by the "max maturity rate" from the curve
3. But max maturity rate = 100% (since we only looked at won deals!)
4. So we're forecasting that 100% of deals will win!

## The Fix

Change the vintage curve to be based on ALL deals, not just won deals:

```python
def build_vintage_curves_CORRECT(deals):
    # Start with ALL deals
    all_deals = deals.copy()
    all_deals['weeks_from_create'] = (
        (all_deals['date_closed'] - all_deals['date_created']).dt.days // 7
    ).fillna(999)  # Open deals get high value
    
    # For each segment and age, calculate what % of ALL created deals have won
    rows = []
    for segment in all_deals['market_segment'].unique():
        segment_deals = all_deals[all_deals['market_segment'] == segment]
        total_deals = len(segment_deals)
        
        won_deals = segment_deals[segment_deals['won']]
        
        for week in range(0, int(won_deals['weeks_from_create'].max()) + 1):
            # Count how many deals have WON by this week
            wins_by_week = len(won_deals[won_deals['weeks_from_create'] <= week])
            
            # As % of ALL deals (not just won deals)
            cum_win_pct = wins_by_week / total_deals
            
            rows.append({
                'market_segment': segment,
                'weeks_to_close': week,
                'cum_win_pct': cum_win_pct
            })
    
    return pd.DataFrame(rows)
```

This would give realistic asymptotes like 20-30% instead of 100%.
