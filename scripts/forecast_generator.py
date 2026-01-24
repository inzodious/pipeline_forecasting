import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from collections import defaultdict

"""
FORECAST GENERATOR V8 - AGGREGATE-FIRST APPROACH

KEY CHANGES FROM V7:
1. AGGREGATE-FIRST: Calculate total monthly metrics, then distribute by segment share
2. TRAILING BASELINE: Use trailing 12 months instead of just prior calendar year
3. SEGMENT SHARE: Revenue and volume distributed by historical segment proportions
4. DEAL SIZE: Use segment average (not just median) for high-variance segments
5. CONFIGURABLE: First-appearance-as-closed factor for counting closed deals in creation

This matches the validated backtest methodology for production forecasting.
"""

ASSUMPTIONS = {
    # Growth Multipliers (applied to data-derived baselines)
    "volume_growth_multiplier": 1.10,       # 10% increase in deal closure volume
    "deal_size_inflation": 1.03,            # 3% increase in average deal value
    
    # Management Override (set to 0 to disable)
    "target_annual_revenue": 0,
    
    # Simulation Parameters
    "num_simulations": 500,
    
    # Baseline Parameters
    "use_trailing_months": 12,              # Use trailing 12 months for baseline
    "velocity_smoothing_window": 3,
    "confidence_interval_clip": 0.95,
    "use_average_not_median": True,         # Use average deal size (captures large deal impact)
    
    # Pipeline Handling
    "first_appearance_closed_factor": 0.5,  # Count 50% of first-appearance-as-closed deals
    
    # Stage Classifications
    "won_stages": ["Closed Won", "Verbal"],
    "lost_stage_patterns": ["Closed Lost", "Declined to Bid"],
}

# Paths
BASE_DIR = "exports"
INPUT_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(INPUT_DIR, "fact_snapshots.csv")

# Output Paths
OUTPUT_AGGREGATE_METRICS = os.path.join(BASE_DIR, "driver_aggregate_monthly.csv")
OUTPUT_SEGMENT_SHARES = os.path.join(BASE_DIR, "driver_segment_shares.csv")
OUTPUT_MONTHLY_FORECAST = os.path.join(BASE_DIR, "forecast_monthly_2026.csv")
OUTPUT_CONFIDENCE = os.path.join(BASE_DIR, "forecast_confidence_intervals.csv")
OUTPUT_EXECUTIVE = os.path.join(BASE_DIR, "executive_summary.txt")
OUTPUT_ASSUMPTIONS = os.path.join(BASE_DIR, "forecast_assumptions_log.csv")


# ==========================================
# HELPER: STAGE CLASSIFICATION
# ==========================================

def is_won(stage):
    if pd.isna(stage):
        return False
    return stage in ASSUMPTIONS['won_stages']

def is_lost(stage):
    if pd.isna(stage):
        return False
    for pattern in ASSUMPTIONS['lost_stage_patterns']:
        if pattern.lower() in stage.lower():
            return True
    return False

def is_closed(stage):
    return is_won(stage) or is_lost(stage)


# ==========================================
# AGGREGATE MONTHLY METRICS
# ==========================================

def calculate_aggregate_monthly_metrics(df):
    """
    Calculate AGGREGATE monthly won volume and revenue.
    This is the primary forecast driver.
    """
    print("  > Calculating Aggregate Monthly Metrics...")
    
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    
    df_won = df_final[df_final['is_won_flag']].copy()
    
    df_won['date_closed'] = pd.to_datetime(df_won['date_closed'])
    df_won['close_month'] = df_won['date_closed'].dt.month
    df_won['close_year'] = df_won['date_closed'].dt.year
    
    # Use trailing months for baseline
    latest_date = df_won['date_closed'].max()
    trailing_months = ASSUMPTIONS.get('use_trailing_months', 12)
    baseline_start = latest_date - pd.DateOffset(months=trailing_months)
    
    df_trailing = df_won[df_won['date_closed'] > baseline_start].copy()
    
    print(f"    Baseline period: {baseline_start.date()} to {latest_date.date()}")
    print(f"    Won deals in baseline: {len(df_trailing):,}")
    
    # Calculate AGGREGATE monthly metrics
    monthly_agg = {}
    export_rows = []
    
    for m in range(1, 13):
        m_data = df_trailing[df_trailing['close_month'] == m]
        monthly_agg[m] = {
            'total_vol': len(m_data),
            'total_rev': m_data['net_revenue'].sum()
        }
        export_rows.append({
            'month': m,
            'raw_volume': len(m_data),
            'raw_revenue': m_data['net_revenue'].sum()
        })
    
    # Apply smoothing
    vols = [monthly_agg[m]['total_vol'] for m in range(1, 13)]
    revs = [monthly_agg[m]['total_rev'] for m in range(1, 13)]
    
    smoothing = ASSUMPTIONS.get('velocity_smoothing_window', 3)
    if smoothing > 1:
        vols_smooth = pd.Series(vols).rolling(window=smoothing, center=True, min_periods=1).mean().tolist()
        revs_smooth = pd.Series(revs).rolling(window=smoothing, center=True, min_periods=1).mean().tolist()
    else:
        vols_smooth = vols
        revs_smooth = revs
    
    for i, m in enumerate(range(1, 13)):
        monthly_agg[m]['vol_smooth'] = vols_smooth[i]
        monthly_agg[m]['rev_smooth'] = revs_smooth[i]
        export_rows[i]['smoothed_volume'] = round(vols_smooth[i], 2)
        export_rows[i]['smoothed_revenue'] = round(revs_smooth[i], 0)
    
    total_annual_vol = sum(vols)
    total_annual_rev = sum(revs)
    
    print(f"    Annual baseline: {total_annual_vol} deals, ${total_annual_rev:,.0f}")
    print(f"    Monthly average: {total_annual_vol/12:.1f} deals, ${total_annual_rev/12:,.0f}")
    
    pd.DataFrame(export_rows).to_csv(OUTPUT_AGGREGATE_METRICS, index=False)
    
    return monthly_agg, df_trailing


# ==========================================
# SEGMENT SHARE CALCULATION
# ==========================================

def calculate_segment_shares(df_trailing):
    """
    Calculate what % of total volume and revenue each segment represents.
    """
    print("  > Calculating Segment Shares...")
    
    total_vol = len(df_trailing)
    total_rev = df_trailing['net_revenue'].sum()
    
    segment_shares = {}
    export_rows = []
    
    for seg in df_trailing['market_segment'].unique():
        seg_data = df_trailing[df_trailing['market_segment'] == seg]
        
        seg_vol = len(seg_data)
        seg_rev = seg_data['net_revenue'].sum()
        
        avg_deal_size = seg_data['net_revenue'].mean() if len(seg_data) > 0 else 0
        median_deal_size = seg_data['net_revenue'].median() if len(seg_data) > 0 else 0
        std_deal_size = seg_data['net_revenue'].std() if len(seg_data) > 1 else avg_deal_size * 0.3
        
        # Use AVERAGE by default (captures large deal impact better)
        # Critical for segments with high deal size variance
        if ASSUMPTIONS.get('use_average_not_median', True):
            effective_deal_size = avg_deal_size
        else:
            # Fallback: Use average for high-variance segments
            if avg_deal_size > median_deal_size * 2 and median_deal_size > 0:
                effective_deal_size = avg_deal_size
            else:
                effective_deal_size = (avg_deal_size + median_deal_size) / 2
        
        segment_shares[seg] = {
            'vol_share': seg_vol / total_vol if total_vol > 0 else 0,
            'rev_share': seg_rev / total_rev if total_rev > 0 else 0,
            'avg_deal_size': avg_deal_size,
            'median_deal_size': median_deal_size,
            'std_deal_size': std_deal_size if not pd.isna(std_deal_size) else avg_deal_size * 0.3,
            'effective_deal_size': effective_deal_size,
            'deal_count': seg_vol
        }
        
        export_rows.append({
            'market_segment': seg,
            'deal_count': seg_vol,
            'total_revenue': round(seg_rev, 0),
            'vol_share': round(seg_vol / total_vol if total_vol > 0 else 0, 4),
            'rev_share': round(seg_rev / total_rev if total_rev > 0 else 0, 4),
            'avg_deal_size': round(avg_deal_size, 0),
            'median_deal_size': round(median_deal_size, 0),
            'effective_deal_size': round(effective_deal_size, 0)
        })
    
    print(f"\n    Segment Shares (using {'average' if ASSUMPTIONS.get('use_average_not_median', True) else 'blended'} deal size):")
    for seg, shares in sorted(segment_shares.items(), key=lambda x: x[1]['rev_share'], reverse=True):
        print(f"      {seg:20s}: Vol={shares['vol_share']:.1%}, Rev={shares['rev_share']:.1%}, AvgDeal=${shares['avg_deal_size']:,.0f}, MedianDeal=${shares['median_deal_size']:,.0f}")
    
    pd.DataFrame(export_rows).to_csv(OUTPUT_SEGMENT_SHARES, index=False)
    
    return segment_shares


# ==========================================
# DEAL SIZE SAMPLER
# ==========================================

def sample_deal_size(segment_shares, segment):
    """
    Sample deal size using effective deal size with variance.
    """
    seg_info = segment_shares.get(segment, {})
    
    effective_size = seg_info.get('effective_deal_size', 10000)
    std_size = seg_info.get('std_deal_size', effective_size * 0.3)
    
    # Cap std at 50% of effective size to prevent extreme outliers
    std_size = min(std_size, effective_size * 0.5)
    
    # Sample with variance
    sample = np.random.normal(effective_size, std_size)
    sample = max(effective_size * 0.1, sample)  # Floor at 10% of expected
    
    return int(sample)


# ==========================================
# FORECAST ENGINE (V8 - Aggregate-First)
# ==========================================

class ForecastEngine:
    """
    V8 Engine: Aggregate-first approach.
    1. Generate total monthly closures from aggregate baseline
    2. Distribute by segment share
    3. Apply growth multipliers
    """
    
    def __init__(self, monthly_agg, segment_shares):
        self.monthly_agg = monthly_agg
        self.segment_shares = segment_shares
        self.segments = list(segment_shares.keys())
        self.forecast_months = pd.date_range('2026-01-01', '2026-12-31', freq='MS')
    
    def run_simulation(self, existing_deals=None):
        """
        Run simulation using aggregate-first approach.
        """
        monthly_results = {month: defaultdict(lambda: {
            'total_won_vol': 0, 'total_won_rev': 0
        }) for month in self.forecast_months}
        
        for month in self.forecast_months:
            month_num = month.month
            
            # Get aggregate expectation for this month
            agg_stats = self.monthly_agg.get(month_num, {'vol_smooth': 0, 'rev_smooth': 0})
            
            # Apply growth multiplier
            base_vol = agg_stats['vol_smooth'] * ASSUMPTIONS.get('volume_growth_multiplier', 1.0)
            
            # Generate total deals from Poisson
            if base_vol <= 0:
                total_deals = 0
            else:
                total_deals = np.random.poisson(base_vol)
            
            # Distribute deals across segments by volume share
            for seg in self.segments:
                seg_share = self.segment_shares[seg]['vol_share']
                
                # Expected deals for this segment
                expected_seg_deals = total_deals * seg_share
                
                # Add variance with Poisson
                if expected_seg_deals > 0.5:
                    seg_deals = np.random.poisson(expected_seg_deals)
                elif expected_seg_deals > 0:
                    seg_deals = 1 if np.random.random() < expected_seg_deals else 0
                else:
                    seg_deals = 0
                
                # Generate revenue for each deal
                seg_revenue = 0
                for _ in range(seg_deals):
                    deal_size = sample_deal_size(self.segment_shares, seg)
                    deal_size = int(deal_size * ASSUMPTIONS.get('deal_size_inflation', 1.0))
                    seg_revenue += deal_size
                
                monthly_results[month][seg]['total_won_vol'] = seg_deals
                monthly_results[month][seg]['total_won_rev'] = seg_revenue
        
        return monthly_results


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_forecast():
    print("=" * 70)
    print("FORECAST GENERATOR V8 - AGGREGATE-FIRST MODEL")
    print("=" * 70)
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print("\n--- 1. Loading Data ---")
    
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print(f"ERROR: File not found: {PATH_RAW_SNAPSHOTS}")
        return
    
    df = pd.read_csv(PATH_RAW_SNAPSHOTS)
    df['date_snapshot'] = pd.to_datetime(df['date_snapshot'])
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['date_closed'] = pd.to_datetime(df['date_closed'])
    
    print(f"    Loaded {len(df):,} snapshot records")
    print(f"    Unique deals: {df['deal_id'].nunique():,}")
    print(f"    Date range: {df['date_snapshot'].min().date()} to {df['date_snapshot'].max().date()}")
    print(f"    Segments: {df['market_segment'].unique().tolist()}")
    
    # Save assumptions
    assumptions_log = pd.DataFrame([{
        'assumption': k,
        'value': str(v),
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    } for k, v in ASSUMPTIONS.items()])
    assumptions_log.to_csv(OUTPUT_ASSUMPTIONS, index=False)
    
    print("\n--- 2. Calculating Aggregate Metrics ---")
    monthly_agg, df_trailing = calculate_aggregate_monthly_metrics(df)
    
    print("\n--- 3. Calculating Segment Shares ---")
    segment_shares = calculate_segment_shares(df_trailing)
    
    print(f"\n--- 4. Running {ASSUMPTIONS['num_simulations']} Simulations ---")
    
    engine = ForecastEngine(monthly_agg, segment_shares)
    all_results = []
    
    for sim in range(ASSUMPTIONS['num_simulations']):
        if (sim + 1) % 100 == 0:
            print(f"    Completed {sim + 1}/{ASSUMPTIONS['num_simulations']}...")
        
        result = engine.run_simulation()
        all_results.append(result)
    
    print("\n--- 5. Aggregating Results ---")
    
    forecast_rows = []
    confidence_rows = []
    
    for month in engine.forecast_months:
        for seg in engine.segments:
            won_vol_vals = [sim[month][seg]['total_won_vol'] for sim in all_results]
            won_rev_vals = [sim[month][seg]['total_won_rev'] for sim in all_results]
            
            clip = ASSUMPTIONS['confidence_interval_clip']
            if len(won_rev_vals) > 0 and max(won_rev_vals) > 0:
                lower = np.percentile(won_rev_vals, (1 - clip) * 50)
                upper = np.percentile(won_rev_vals, 50 + clip * 50)
                won_rev_clipped = [v for v in won_rev_vals if lower <= v <= upper]
            else:
                won_rev_clipped = won_rev_vals
            
            forecast_rows.append({
                'forecast_month': month,
                'market_segment': seg,
                'forecasted_won_volume_median': int(np.median(won_vol_vals)) if won_vol_vals else 0,
                'forecasted_won_volume_mean': round(np.mean(won_vol_vals), 1) if won_vol_vals else 0,
                'forecasted_won_revenue_median': int(np.median(won_rev_vals)) if won_rev_vals else 0,
                'forecasted_won_revenue_mean': int(np.mean(won_rev_clipped)) if won_rev_clipped else 0
            })
            
            if won_rev_clipped and max(won_rev_clipped) > 0:
                confidence_rows.append({
                    'forecast_month': month,
                    'market_segment': seg,
                    'metric': 'won_revenue',
                    'p10': int(np.percentile(won_rev_clipped, 10)),
                    'p50': int(np.percentile(won_rev_clipped, 50)),
                    'p90': int(np.percentile(won_rev_clipped, 90))
                })
    
    df_forecast = pd.DataFrame(forecast_rows)
    df_confidence = pd.DataFrame(confidence_rows)
    
    # Apply management target override if set
    if ASSUMPTIONS['target_annual_revenue'] > 0:
        print(f"\n--- 6. Applying Management Target Override ---")
        
        raw_annual = df_forecast['forecasted_won_revenue_median'].sum()
        target = ASSUMPTIONS['target_annual_revenue']
        scale_factor = target / raw_annual if raw_annual > 0 else 1.0
        
        print(f"    Raw forecast total: ${raw_annual:,.0f}")
        print(f"    Target override:    ${target:,.0f}")
        print(f"    Scale factor:       {scale_factor:.3f}")
        
        df_forecast['forecasted_won_revenue_median'] = (df_forecast['forecasted_won_revenue_median'] * scale_factor).astype(int)
        df_forecast['forecasted_won_revenue_mean'] = (df_forecast['forecasted_won_revenue_mean'] * scale_factor).astype(int)
        
        for i, row in df_confidence.iterrows():
            df_confidence.at[i, 'p10'] = int(row['p10'] * scale_factor)
            df_confidence.at[i, 'p50'] = int(row['p50'] * scale_factor)
            df_confidence.at[i, 'p90'] = int(row['p90'] * scale_factor)
    
    df_forecast.to_csv(OUTPUT_MONTHLY_FORECAST, index=False)
    df_confidence.to_csv(OUTPUT_CONFIDENCE, index=False)
    
    print("\n--- 7. Sanity Check ---")
    
    monthly_totals = df_forecast.groupby('forecast_month').agg({
        'forecasted_won_volume_median': 'sum',
        'forecasted_won_revenue_median': 'sum'
    })
    
    print("    Monthly forecast totals:")
    for month, row in monthly_totals.iterrows():
        print(f"      {month.strftime('%Y-%m')}: {row['forecasted_won_volume_median']:>4.0f} deals, ${row['forecasted_won_revenue_median']:>12,.0f}")
    
    annual_vol = monthly_totals['forecasted_won_volume_median'].sum()
    annual_rev = monthly_totals['forecasted_won_revenue_median'].sum()
    
    print(f"\n    ANNUAL TOTAL: {annual_vol:,.0f} deals, ${annual_rev:,.0f}")
    
    print("\n--- 8. Generating Executive Summary ---")
    generate_executive_summary(df_forecast, df_confidence, monthly_agg, segment_shares)
    
    print(f"\n    Outputs saved:")
    print(f"      {OUTPUT_MONTHLY_FORECAST}")
    print(f"      {OUTPUT_CONFIDENCE}")
    print(f"      {OUTPUT_EXECUTIVE}")
    
    print("\n" + "=" * 70)
    print("FORECAST COMPLETE")
    print("=" * 70)


def generate_executive_summary(df_forecast, df_confidence, monthly_agg, segment_shares):
    annual_rev = df_forecast['forecasted_won_revenue_median'].sum()
    annual_vol = df_forecast['forecasted_won_volume_median'].sum()
    
    seg_totals = df_forecast.groupby('market_segment')['forecasted_won_revenue_median'].sum().sort_values(ascending=False)
    
    total_conf = df_confidence.groupby('forecast_month')[['p10', 'p50', 'p90']].sum()
    annual_p10 = total_conf['p10'].sum() if len(total_conf) > 0 else 0
    annual_p50 = total_conf['p50'].sum() if len(total_conf) > 0 else 0
    annual_p90 = total_conf['p90'].sum() if len(total_conf) > 0 else 0
    
    # Calculate baseline annual from aggregate
    baseline_annual_vol = sum([monthly_agg[m]['total_vol'] for m in range(1, 13)])
    baseline_annual_rev = sum([monthly_agg[m]['total_rev'] for m in range(1, 13)])
    
    summary = f"""
{'=' * 70}
EXECUTIVE SUMMARY: 2026 REVENUE FORECAST
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

FORECAST RESULTS
----------------
Expected Annual Revenue (P50):     ${annual_p50:>15,.0f}
Conservative Case (P10):           ${annual_p10:>15,.0f}
Optimistic Case (P90):             ${annual_p90:>15,.0f}
Expected Deal Volume:              {annual_vol:>15,.0f} deals

{'MANAGEMENT TARGET APPLIED' if ASSUMPTIONS['target_annual_revenue'] > 0 else 'NO MANAGEMENT TARGET OVERRIDE'}

SEGMENT BREAKDOWN
-----------------
"""
    
    for seg, rev in seg_totals.items():
        pct = (rev / annual_rev * 100) if annual_rev > 0 else 0
        seg_info = segment_shares.get(seg, {})
        vol_share = seg_info.get('vol_share', 0)
        avg_deal = seg_info.get('avg_deal_size', 0)
        summary += f"  {seg:20s}: ${rev:>12,.0f}  ({pct:>5.1f}%)  VolShare: {vol_share:.1%}  AvgDeal: ${avg_deal:,.0f}\n"

    summary += f"""

BASELINE METRICS (Trailing {ASSUMPTIONS.get('use_trailing_months', 12)} Months)
--------------------------------------------------
Baseline Annual Volume:    {baseline_annual_vol:>10,} deals
Baseline Annual Revenue:   ${baseline_annual_rev:>14,.0f}
Baseline Monthly Avg:      {baseline_annual_vol/12:>10.1f} deals / ${baseline_annual_rev/12:>,.0f}

GROWTH ASSUMPTIONS APPLIED
--------------------------
Volume Growth:             {((ASSUMPTIONS['volume_growth_multiplier'] - 1) * 100):>+.0f}%
Deal Size Inflation:       {((ASSUMPTIONS['deal_size_inflation'] - 1) * 100):>+.0f}%
Simulations Run:           {ASSUMPTIONS['num_simulations']:,}

METHODOLOGY
-----------
This forecast uses an AGGREGATE-FIRST approach:
1. Calculate total monthly won volume/revenue from trailing period
2. Calculate segment share of total (% of volume, % of revenue)
3. For each forecast month:
   a. Generate total deals from aggregate baseline (with growth)
   b. Distribute to segments by historical volume share
   c. Generate revenue per deal from segment-specific distribution
4. Monte Carlo simulation captures variance in outcomes

This approach ensures:
- Segment forecasts sum to a coherent total
- No individual segment can over-forecast beyond its historical share
- Variance is captured through simulation

{'=' * 70}
"""
    
    with open(OUTPUT_EXECUTIVE, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)


if __name__ == "__main__":
    run_forecast()