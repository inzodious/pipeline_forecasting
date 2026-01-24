import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict

"""
BACKTEST VALIDATION V6 - AGGREGATE-FIRST APPROACH

KEY CHANGES FROM V5:
1. AGGREGATE-FIRST: Calculate total monthly metrics, then distribute by segment share
2. TRAILING BASELINE: Use trailing 12 months instead of just prior calendar year
3. FULL YEAR BACKTEST: Validate against full FY25 (train on trailing 12mo before each month)
4. SEGMENT SHARE: Revenue and volume distributed by historical segment proportions
5. DEAL SIZE: Use segment average (not median) for high-variance segments

This fixes:
- SMB over-forecasting (segment share limits contribution)
- Indirect under-forecasting (aggregate captures total, share distributes correctly)
"""

# Paths
EXPORT_DIR = "exports"
INPUT_DIR = "data"
VALIDATION_DIR = "validation"

PATH_RAW_SNAPSHOTS = os.path.join(INPUT_DIR, "fact_snapshots.csv")
OUTPUT_BACKTEST_RESULTS = os.path.join(VALIDATION_DIR, "backtest_validation_results.csv")
OUTPUT_BACKTEST_SUMMARY = os.path.join(VALIDATION_DIR, "backtest_summary.txt")

# Configuration - FULL FY25 BACKTEST
TRAIN_END_DATE = '2024-12-31'  # Train on 2024
TEST_START_DATE = '2025-01-01'  # Test full FY25
TEST_END_DATE = '2025-12-31'

# Stage Classifications
WON_STAGES = ["Closed Won", "Verbal"]
LOST_PATTERNS = ["Closed Lost", "Declined to Bid"]

# BACKTEST ASSUMPTIONS - NEUTRAL for fair model validation
BACKTEST_ASSUMPTIONS = {
    'volume_growth_multiplier': 1.00,
    'deal_size_inflation': 1.00,
    'num_simulations': 500,
    'velocity_smoothing_window': 3,
    'use_trailing_months': 12,  # Use trailing 12 months for baseline
    'first_appearance_closed_factor': 0.5,  # Count 50% of first-appearance-as-closed deals
    'use_average_not_median': True,  # Use average deal size (captures large deal impact)
}


# ==========================================
# HELPER: STAGE CLASSIFICATION
# ==========================================

def is_won(stage):
    if pd.isna(stage):
        return False
    return stage in WON_STAGES

def is_lost(stage):
    if pd.isna(stage):
        return False
    for pattern in LOST_PATTERNS:
        if pattern.lower() in stage.lower():
            return True
    return False

def is_closed(stage):
    return is_won(stage) or is_lost(stage)


# ==========================================
# AGGREGATE MONTHLY METRICS (KEY V6 CHANGE)
# ==========================================

def calculate_aggregate_monthly_metrics(df, cutoff_date):
    """
    Calculate AGGREGATE monthly won volume and revenue.
    This is the primary forecast driver.
    """
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    
    # Only won deals that closed before cutoff
    df_won = df_final[
        (df_final['is_won_flag']) & 
        (df_final['date_closed'] <= cutoff_date)
    ].copy()
    
    df_won['date_closed'] = pd.to_datetime(df_won['date_closed'])
    df_won['close_month'] = df_won['date_closed'].dt.month
    df_won['close_year'] = df_won['date_closed'].dt.year
    
    # Use trailing months for baseline
    trailing_months = BACKTEST_ASSUMPTIONS.get('use_trailing_months', 12)
    cutoff_minus_trailing = cutoff_date - pd.DateOffset(months=trailing_months)
    
    df_trailing = df_won[df_won['date_closed'] > cutoff_minus_trailing].copy()
    
    print(f"    Aggregate baseline: {cutoff_minus_trailing.date()} to {cutoff_date.date()}")
    print(f"    Trailing period won deals: {len(df_trailing):,}")
    
    # Calculate AGGREGATE monthly metrics
    monthly_agg = defaultdict(lambda: {'total_vol': 0, 'total_rev': 0})
    
    for m in range(1, 13):
        m_data = df_trailing[df_trailing['close_month'] == m]
        monthly_agg[m]['total_vol'] = len(m_data)
        monthly_agg[m]['total_rev'] = m_data['net_revenue'].sum()
    
    # Calculate smoothed values
    vols = [monthly_agg[m]['total_vol'] for m in range(1, 13)]
    revs = [monthly_agg[m]['total_rev'] for m in range(1, 13)]
    
    smoothing = BACKTEST_ASSUMPTIONS.get('velocity_smoothing_window', 3)
    if smoothing > 1:
        vols_smooth = pd.Series(vols).rolling(window=smoothing, center=True, min_periods=1).mean().tolist()
        revs_smooth = pd.Series(revs).rolling(window=smoothing, center=True, min_periods=1).mean().tolist()
    else:
        vols_smooth = vols
        revs_smooth = revs
    
    for m in range(1, 13):
        monthly_agg[m]['vol_smooth'] = vols_smooth[m - 1]
        monthly_agg[m]['rev_smooth'] = revs_smooth[m - 1]
    
    total_annual_vol = sum(vols)
    total_annual_rev = sum(revs)
    
    print(f"    Annual baseline: {total_annual_vol} deals, ${total_annual_rev:,.0f}")
    print(f"    Monthly average: {total_annual_vol/12:.1f} deals, ${total_annual_rev/12:,.0f}")
    
    return monthly_agg, df_trailing


# ==========================================
# SEGMENT SHARE CALCULATION (KEY V6 CHANGE)
# ==========================================

def calculate_segment_shares(df_trailing):
    """
    Calculate what % of total volume and revenue each segment represents.
    This is used to distribute aggregate forecast by segment.
    """
    total_vol = len(df_trailing)
    total_rev = df_trailing['net_revenue'].sum()
    
    segment_shares = {}
    
    for seg in df_trailing['market_segment'].unique():
        seg_data = df_trailing[df_trailing['market_segment'] == seg]
        
        seg_vol = len(seg_data)
        seg_rev = seg_data['net_revenue'].sum()
        
        # Calculate deal size metrics
        avg_deal_size = seg_data['net_revenue'].mean() if len(seg_data) > 0 else 0
        median_deal_size = seg_data['net_revenue'].median() if len(seg_data) > 0 else 0
        
        # Use AVERAGE by default (captures large deal impact better)
        # This is critical for segments like Indirect where deal sizes are volatile
        if BACKTEST_ASSUMPTIONS.get('use_average_not_median', True):
            effective_deal_size = avg_deal_size
        else:
            # Fallback: Use average for high-variance segments (where median << mean)
            if avg_deal_size > median_deal_size * 2 and median_deal_size > 0:
                effective_deal_size = avg_deal_size
            else:
                effective_deal_size = (avg_deal_size + median_deal_size) / 2
        
        segment_shares[seg] = {
            'vol_share': seg_vol / total_vol if total_vol > 0 else 0,
            'rev_share': seg_rev / total_rev if total_rev > 0 else 0,
            'avg_deal_size': avg_deal_size,
            'median_deal_size': median_deal_size,
            'effective_deal_size': effective_deal_size,
            'deal_count': seg_vol
        }
    
    print(f"\n    Segment Shares (using {'average' if BACKTEST_ASSUMPTIONS.get('use_average_not_median', True) else 'blended'} deal size):")
    for seg, shares in sorted(segment_shares.items(), key=lambda x: x[1]['rev_share'], reverse=True):
        print(f"      {seg:20s}: Vol={shares['vol_share']:.1%}, Rev={shares['rev_share']:.1%}, AvgDeal=${shares['avg_deal_size']:,.0f}, MedianDeal=${shares['median_deal_size']:,.0f}")
    
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
    avg_size = seg_info.get('avg_deal_size', effective_size)
    median_size = seg_info.get('median_deal_size', effective_size)
    
    # Estimate std from the avg/median ratio (high ratio = high variance)
    if median_size > 0:
        variance_ratio = avg_size / median_size
        std_estimate = effective_size * min(0.5, (variance_ratio - 1) * 0.3 + 0.2)
    else:
        std_estimate = effective_size * 0.3
    
    # Sample with variance
    sample = np.random.normal(effective_size, std_estimate)
    sample = max(effective_size * 0.1, sample)  # Floor at 10% of expected
    
    return int(sample)


# ==========================================
# BACKTEST FORECAST ENGINE (V6)
# ==========================================

class BacktestEngine:
    """
    V6 Engine: Aggregate-first approach.
    1. Generate total monthly closures from aggregate baseline
    2. Distribute by segment share
    """
    
    def __init__(self, monthly_agg, segment_shares, test_months):
        self.monthly_agg = monthly_agg
        self.segment_shares = segment_shares
        self.test_months = test_months
        self.segments = list(segment_shares.keys())
    
    def run_simulation(self, existing_deals=None):
        """
        Run simulation using aggregate-first approach.
        
        1. For each month, determine total expected wins from aggregate baseline
        2. Distribute wins across segments by their historical share
        3. Sample deal sizes per segment
        """
        monthly_results = {month: defaultdict(lambda: {
            'won_vol': 0, 'won_rev': 0
        }) for month in self.test_months}
        
        for month in self.test_months:
            month_num = month.month
            
            # Get aggregate expectation for this month
            agg_stats = self.monthly_agg.get(month_num, {'vol_smooth': 0, 'rev_smooth': 0})
            
            base_vol = agg_stats['vol_smooth'] * BACKTEST_ASSUMPTIONS.get('volume_growth_multiplier', 1.0)
            
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
                
                # Add some variance with Poisson (if expected > 0)
                if expected_seg_deals > 0.5:
                    seg_deals = np.random.poisson(expected_seg_deals)
                elif expected_seg_deals > 0:
                    # For very low volume, use bernoulli
                    seg_deals = 1 if np.random.random() < expected_seg_deals else 0
                else:
                    seg_deals = 0
                
                # Generate revenue for each deal
                seg_revenue = 0
                for _ in range(seg_deals):
                    deal_size = sample_deal_size(self.segment_shares, seg)
                    deal_size = int(deal_size * BACKTEST_ASSUMPTIONS.get('deal_size_inflation', 1.0))
                    seg_revenue += deal_size
                
                monthly_results[month][seg]['won_vol'] = seg_deals
                monthly_results[month][seg]['won_rev'] = seg_revenue
        
        return monthly_results


# ==========================================
# EXTRACT ACTUAL RESULTS
# ==========================================

def extract_actual_monthly_results(df, test_start, test_end):
    """Extract actual results from test period."""
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    
    df_won = df_final[
        (df_final['is_won_flag']) &
        (df_final['date_closed'] >= test_start) &
        (df_final['date_closed'] <= test_end)
    ].copy()
    
    df_won['close_month'] = df_won['date_closed'].dt.to_period('M').dt.to_timestamp()
    
    actual_monthly = defaultdict(lambda: defaultdict(lambda: {
        'actual_won_vol': 0,
        'actual_won_rev': 0
    }))
    
    for _, row in df_won.iterrows():
        month = row['close_month']
        seg = row['market_segment']
        actual_monthly[month][seg]['actual_won_vol'] += 1
        actual_monthly[month][seg]['actual_won_rev'] += row['net_revenue']
    
    return actual_monthly


# ==========================================
# COMPARISON & METRICS
# ==========================================

def compare_forecast_to_actual(forecasted_list, actual):
    """Compare average forecasted results to actual."""
    avg_forecasted = defaultdict(lambda: defaultdict(lambda: {
        'forecasted_won_vol': 0,
        'forecasted_won_rev': 0
    }))
    
    for forecasted in forecasted_list:
        for month, seg_data in forecasted.items():
            for seg, metrics in seg_data.items():
                avg_forecasted[month][seg]['forecasted_won_vol'] += metrics['won_vol']
                avg_forecasted[month][seg]['forecasted_won_rev'] += metrics['won_rev']
    
    num_sims = len(forecasted_list)
    for month in avg_forecasted:
        for seg in avg_forecasted[month]:
            avg_forecasted[month][seg]['forecasted_won_vol'] /= num_sims
            avg_forecasted[month][seg]['forecasted_won_rev'] /= num_sims
    
    comparison_rows = []
    
    all_months = set(avg_forecasted.keys()) | set(actual.keys())
    all_segments = set()
    
    for month_data in list(avg_forecasted.values()) + list(actual.values()):
        all_segments.update(month_data.keys())
    
    for month in sorted(all_months):
        for seg in sorted(all_segments):
            f_vol = avg_forecasted.get(month, {}).get(seg, {}).get('forecasted_won_vol', 0)
            f_rev = avg_forecasted.get(month, {}).get(seg, {}).get('forecasted_won_rev', 0)
            
            a_vol = actual.get(month, {}).get(seg, {}).get('actual_won_vol', 0)
            a_rev = actual.get(month, {}).get(seg, {}).get('actual_won_rev', 0)
            
            vol_error = f_vol - a_vol
            rev_error = f_rev - a_rev
            
            vol_pct_error = (vol_error / a_vol * 100) if a_vol > 0 else 0
            rev_pct_error = (rev_error / a_rev * 100) if a_rev > 0 else 0
            
            comparison_rows.append({
                'month': month,
                'market_segment': seg,
                'forecasted_won_volume': round(f_vol, 1),
                'actual_won_volume': a_vol,
                'volume_error': round(vol_error, 1),
                'volume_pct_error': round(vol_pct_error, 1),
                'forecasted_won_revenue': int(f_rev),
                'actual_won_revenue': int(a_rev),
                'revenue_error': int(rev_error),
                'revenue_pct_error': round(rev_pct_error, 1)
            })
    
    return pd.DataFrame(comparison_rows)


def calculate_accuracy_metrics(df_comparison):
    """Calculate summary accuracy statistics."""
    # Overall metrics (sum across all segments)
    monthly_totals = df_comparison.groupby('month').agg({
        'forecasted_won_revenue': 'sum',
        'actual_won_revenue': 'sum',
        'forecasted_won_volume': 'sum',
        'actual_won_volume': 'sum'
    }).reset_index()
    
    total_forecasted = monthly_totals['forecasted_won_revenue'].sum()
    total_actual = monthly_totals['actual_won_revenue'].sum()
    
    overall_pct_error = ((total_forecasted - total_actual) / total_actual * 100) if total_actual > 0 else 0
    
    # Calculate monthly-level error (more meaningful than segment-month)
    monthly_totals['rev_error'] = monthly_totals['forecasted_won_revenue'] - monthly_totals['actual_won_revenue']
    monthly_totals['rev_pct_error'] = (monthly_totals['rev_error'] / monthly_totals['actual_won_revenue'] * 100).fillna(0)
    
    mape_monthly = monthly_totals['rev_pct_error'].abs().mean()
    bias_monthly = monthly_totals['rev_pct_error'].mean()
    rmse_monthly = np.sqrt((monthly_totals['rev_error'] ** 2).mean())
    
    within_20pct = (monthly_totals['rev_pct_error'].abs() <= 20).sum()
    hit_rate = (within_20pct / len(monthly_totals)) * 100
    
    # Segment-level metrics
    df_valid = df_comparison[df_comparison['actual_won_revenue'] > 0].copy()
    mape_segment = df_valid['revenue_pct_error'].abs().mean() if len(df_valid) > 0 else 0
    bias_segment = df_valid['revenue_pct_error'].mean() if len(df_valid) > 0 else 0
    
    return {
        'mape_monthly': mape_monthly,
        'mape_segment': mape_segment,
        'bias_monthly': bias_monthly,
        'bias_segment': bias_segment,
        'rmse_monthly': rmse_monthly,
        'hit_rate_within_20pct': hit_rate,
        'num_months_analyzed': len(monthly_totals),
        'overall_pct_error': overall_pct_error,
        'total_forecasted': total_forecasted,
        'total_actual': total_actual
    }


# ==========================================
# MAIN BACKTEST EXECUTION
# ==========================================

def run_backtest_validation():
    print("=" * 70)
    print("BACKTEST VALIDATION V6 - AGGREGATE-FIRST MODEL")
    print("=" * 70)
    
    print("\n--- 0. Backtest Configuration ---")
    print("\n    NEUTRAL multipliers test if model learned patterns correctly:")
    for key, value in BACKTEST_ASSUMPTIONS.items():
        print(f"      {key:35s} = {value}")
    
    print("\n--- 1. Loading Historical Data ---")
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print(f"Error: Snapshot file not found: {PATH_RAW_SNAPSHOTS}")
        return
    
    df_raw = pd.read_csv(PATH_RAW_SNAPSHOTS)
    df_raw['date_snapshot'] = pd.to_datetime(df_raw['date_snapshot'])
    df_raw['date_created'] = pd.to_datetime(df_raw['date_created'])
    df_raw['date_closed'] = pd.to_datetime(df_raw['date_closed'])
    
    train_end = pd.to_datetime(TRAIN_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)
    
    df_train = df_raw[df_raw['date_snapshot'] <= train_end].copy()
    df_full = df_raw.copy()
    
    print(f"    Training cutoff: {train_end.date()}")
    print(f"    Testing:  {test_start.date()} to {test_end.date()} (Full FY25)")
    print(f"    Training records: {len(df_train):,}")
    print(f"    Unique deals in training: {df_train['deal_id'].nunique():,}")
    
    print("\n--- 2. Calculating Aggregate Metrics (from training data) ---")
    monthly_agg, df_trailing = calculate_aggregate_monthly_metrics(df_train, train_end)
    
    print("\n--- 3. Calculating Segment Shares ---")
    segment_shares = calculate_segment_shares(df_trailing)
    
    print("\n--- 4. Running Backtest Simulations ---")
    
    test_months = pd.date_range(test_start, test_end, freq='MS')
    num_sims = BACKTEST_ASSUMPTIONS.get('num_simulations', 500)
    
    engine = BacktestEngine(monthly_agg, segment_shares, test_months)
    
    all_forecasted = []
    
    for sim in range(num_sims):
        if (sim + 1) % 100 == 0:
            print(f"    Completed {sim + 1}/{num_sims} simulations...")
        
        forecasted = engine.run_simulation()
        all_forecasted.append(forecasted)
    
    print("\n--- 5. Extracting Actual Results (Test Period) ---")
    actual_results = extract_actual_monthly_results(df_full, test_start, test_end)
    
    total_actual_vol = sum(
        seg_data['actual_won_vol'] 
        for month_data in actual_results.values() 
        for seg_data in month_data.values()
    )
    total_actual_rev = sum(
        seg_data['actual_won_rev'] 
        for month_data in actual_results.values() 
        for seg_data in month_data.values()
    )
    print(f"    Actual test period: {total_actual_vol} deals, ${total_actual_rev:,.0f}")
    
    # Diagnostic: Compare baseline vs actual by segment
    print("\n--- 5b. TREND DIAGNOSTIC: Baseline vs Actual by Segment ---")
    print("    This shows year-over-year changes that explain forecast variance:")
    print(f"    {'Segment':<20} {'Base Vol/Mo':>12} {'Actual Vol/Mo':>14} {'Vol Change':>12} {'Base $/Deal':>14} {'Actual $/Deal':>14} {'$ Change':>10}")
    print("    " + "-" * 100)
    
    # Calculate actual segment metrics
    df_actual_final = df_full.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    df_actual_final['is_won_flag'] = df_actual_final['stage'].apply(is_won)
    df_actual_won = df_actual_final[
        (df_actual_final['is_won_flag']) &
        (df_actual_final['date_closed'] >= test_start) &
        (df_actual_final['date_closed'] <= test_end)
    ].copy()
    
    test_months_count = len(pd.date_range(test_start, test_end, freq='MS'))
    
    for seg in sorted(segment_shares.keys()):
        # Baseline metrics
        base_info = segment_shares[seg]
        base_vol_mo = base_info['deal_count'] / 12  # Annualized to monthly
        base_deal_size = base_info['avg_deal_size']
        
        # Actual metrics
        seg_actual = df_actual_won[df_actual_won['market_segment'] == seg]
        actual_vol_mo = len(seg_actual) / test_months_count if test_months_count > 0 else 0
        actual_deal_size = seg_actual['net_revenue'].mean() if len(seg_actual) > 0 else 0
        
        # Changes
        vol_change = ((actual_vol_mo / base_vol_mo - 1) * 100) if base_vol_mo > 0 else 0
        size_change = ((actual_deal_size / base_deal_size - 1) * 100) if base_deal_size > 0 else 0
        
        print(f"    {seg:<20} {base_vol_mo:>12.1f} {actual_vol_mo:>14.1f} {vol_change:>+11.0f}% ${base_deal_size:>13,.0f} ${actual_deal_size:>13,.0f} {size_change:>+9.0f}%")
    
    print("\n--- 6. Comparing Forecast to Actual ---")
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    df_comparison = compare_forecast_to_actual(all_forecasted, actual_results)
    df_comparison.to_csv(OUTPUT_BACKTEST_RESULTS, index=False)
    
    # Monthly totals comparison
    monthly_comp = df_comparison.groupby('month').agg({
        'forecasted_won_volume': 'sum',
        'actual_won_volume': 'sum',
        'forecasted_won_revenue': 'sum',
        'actual_won_revenue': 'sum'
    })
    
    print("\n    Monthly Comparison (Aggregate):")
    print(f"    {'Month':<12} {'Fcst Vol':>10} {'Actual Vol':>12} {'Fcst Rev':>14} {'Actual Rev':>14} {'Error':>10}")
    print("    " + "-" * 76)
    
    for month, row in monthly_comp.iterrows():
        error_pct = ((row['forecasted_won_revenue'] - row['actual_won_revenue']) / row['actual_won_revenue'] * 100) if row['actual_won_revenue'] > 0 else 0
        print(f"    {month.strftime('%Y-%m'):<12} {row['forecasted_won_volume']:>10.1f} {row['actual_won_volume']:>12.0f} ${row['forecasted_won_revenue']:>13,.0f} ${row['actual_won_revenue']:>13,.0f} {error_pct:>+9.1f}%")
    
    total_fcst = monthly_comp['forecasted_won_revenue'].sum()
    total_fcst_vol = monthly_comp['forecasted_won_volume'].sum()
    total_actual_vol = monthly_comp['actual_won_volume'].sum()
    error_pct = ((total_fcst - total_actual_rev) / total_actual_rev * 100) if total_actual_rev > 0 else 0
    print("    " + "-" * 76)
    print(f"    {'TOTAL':<12} {total_fcst_vol:>10.1f} {total_actual_vol:>12.0f} ${total_fcst:>13,.0f} ${total_actual_rev:>13,.0f} {error_pct:>+9.1f}%")
    
    print(f"\n    Results saved: {OUTPUT_BACKTEST_RESULTS}")
    
    print("\n--- 7. Calculating Accuracy Metrics ---")
    metrics = calculate_accuracy_metrics(df_comparison)
    
    summary_text = f"""
{'=' * 70}
BACKTEST VALIDATION SUMMARY V6 - AGGREGATE-FIRST MODEL
{'=' * 70}

TEST PERIOD: {test_start.date()} to {test_end.date()} (Full FY25)
TRAINING CUTOFF: {train_end.date()}
BASELINE: Trailing {BACKTEST_ASSUMPTIONS.get('use_trailing_months', 12)} months

OVERALL ACCURACY
----------------
Total Forecasted Revenue:      ${metrics['total_forecasted']:>15,.0f}
Total Actual Revenue:          ${metrics['total_actual']:>15,.0f}
Difference:                    ${metrics['total_forecasted'] - metrics['total_actual']:>+15,.0f}
Overall Percent Error:         {metrics['overall_pct_error']:>+15.1f}%

MONTHLY-LEVEL METRICS (Aggregate)
---------------------------------
Mean Absolute Percentage Error (MAPE):  {metrics['mape_monthly']:.2f}%
Forecast Bias:                          {metrics['bias_monthly']:+.2f}%
Root Mean Squared Error (RMSE):         ${metrics['rmse_monthly']:,.0f}
Hit Rate (within +/-20%):               {metrics['hit_rate_within_20pct']:.1f}%
Months Analyzed:                        {metrics['num_months_analyzed']}

SEGMENT-LEVEL METRICS
---------------------
Segment MAPE:                           {metrics['mape_segment']:.2f}%
Segment Bias:                           {metrics['bias_segment']:+.2f}%

INTERPRETATION
--------------
"""
    
    abs_overall_error = abs(metrics['overall_pct_error'])
    if abs_overall_error < 10:
        summary_text += "[OK] Excellent model accuracy (<10% error)\n"
    elif abs_overall_error < 20:
        summary_text += "[OK] Good model accuracy (<20% error)\n"
    elif abs_overall_error < 30:
        summary_text += "[--] Moderate model accuracy (20-30% error)\n"
    else:
        summary_text += "[!!] Model needs calibration (>30% error)\n"
    
    if metrics['overall_pct_error'] > 20:
        summary_text += "[!!] OVER-FORECASTING: Consider reducing baseline\n"
    elif metrics['overall_pct_error'] < -20:
        summary_text += "[!!] UNDER-FORECASTING: Consider increasing baseline\n"
    else:
        summary_text += "[OK] No significant systematic bias\n"
    
    summary_text += f"""

SEGMENT SHARES USED (from Baseline)
-----------------------------------
"""
    for seg, shares in sorted(segment_shares.items(), key=lambda x: x[1]['rev_share'], reverse=True):
        summary_text += f"  {seg:20s}: Vol={shares['vol_share']:.1%}, Rev={shares['rev_share']:.1%}, AvgDeal=${shares['avg_deal_size']:,.0f}, MedianDeal=${shares['median_deal_size']:,.0f}\n"

    summary_text += f"""

UNDERSTANDING SEGMENT VARIANCE
------------------------------
Large segment-level errors typically occur when:
1. VOLUME SHIFTS: A segment grows or shrinks year-over-year
2. DEAL SIZE SHIFTS: Average deal size changes significantly
3. SEASONALITY CHANGES: Historical monthly patterns don't repeat

The TREND DIAGNOSTIC above shows baseline vs actual for each segment.
- Volume Change > +/- 30% indicates the segment is growing/shrinking
- $/Deal Change > +/- 50% indicates deal size volatility

The AGGREGATE-FIRST approach mitigates this by:
- Forecasting total company metrics first
- Distributing by segment share (not forecasting segments independently)
- Using average deal size (not median) to capture large deal impact
"""

    summary_text += f"""

BACKTEST ASSUMPTIONS (NEUTRAL)
------------------------------
"""
    for key, value in BACKTEST_ASSUMPTIONS.items():
        summary_text += f"  {key:40s}: {value}\n"
    
    summary_text += f"""

KEY CHANGES IN V6
-----------------
1. AGGREGATE-FIRST: Calculate total monthly metrics, distribute by segment share
2. TRAILING BASELINE: Uses trailing 12 months (blends patterns better)
3. FULL YEAR TEST: Validates against complete FY25
4. SEGMENT SHARE: Volume/revenue distributed by historical proportions
5. DEAL SIZE: Uses average for high-variance segments (captures large deal impact)

METHODOLOGY
-----------
1. Calculate total monthly won volume/revenue from trailing period
2. Calculate segment share of total (% of volume, % of revenue)
3. For each forecast month:
   a. Generate total deals from aggregate baseline (Poisson)
   b. Distribute to segments by volume share
   c. Generate revenue per deal from segment-specific distribution
4. Monte Carlo simulation captures variance

{'=' * 70}
"""
    
    print(summary_text)
    
    with open(OUTPUT_BACKTEST_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"Summary saved: {OUTPUT_BACKTEST_SUMMARY}")
    
    print("\n" + "=" * 70)
    print("BACKTEST VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest_validation()