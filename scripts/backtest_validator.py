import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict

# ==========================================
# BACKTEST VALIDATION FRAMEWORK V4
# ==========================================
"""
PURPOSE:
Validates the forecast model by training on early data and testing against 
known outcomes in later periods.

PHILOSOPHY:
- Uses NEUTRAL multipliers (1.0) for backtest to test model accuracy
- All baselines derived from data (same logic as forecast)
- No artificial floors/ceilings
- A good backtest means the model learned the patterns correctly
- Production forecast then applies growth assumptions on top
"""

# Paths
EXPORT_DIR = "exports"
INPUT_DIR = "data"
VALIDATION_DIR = "validation"

PATH_RAW_SNAPSHOTS = os.path.join(INPUT_DIR, "fact_snapshots.csv")
OUTPUT_BACKTEST_RESULTS = os.path.join(VALIDATION_DIR, "backtest_validation_results.csv")
OUTPUT_BACKTEST_SUMMARY = os.path.join(VALIDATION_DIR, "backtest_summary.txt")

# Configuration
TRAIN_END_DATE = '2025-06-30'
TEST_START_DATE = '2025-07-01'
TEST_END_DATE = '2025-12-31'

# Stage Classifications (must match forecast_generator)
WON_STAGES = ["Closed Won", "Verbal"]
LOST_PATTERNS = ["Closed Lost", "Declined to Bid"]

# BACKTEST ASSUMPTIONS - NEUTRAL for fair model validation
# These are 1.0 because we're testing if the model learned the patterns correctly
BACKTEST_ASSUMPTIONS = {
    'volume_growth_multiplier': 1.00,       # NEUTRAL
    'win_rate_uplift_multiplier': 1.00,     # NEUTRAL
    'deal_size_inflation': 1.00,            # NEUTRAL
    'num_simulations': 500,
    'velocity_smoothing_window': 3,
    'confidence_interval_clip': 0.95
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
# DSO CALCULATION (from data)
# ==========================================

def calculate_dso_backtest(df, cutoff_date):
    """Calculate DSO from training data only."""
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_closed = df_final[df_final['is_won_flag'] | df_final['is_lost_flag']].copy()
    
    df_closed = df_closed[df_closed['date_closed'] <= cutoff_date].copy()
    
    df_closed['cycle_days'] = (df_closed['date_closed'] - df_closed['date_created']).dt.days
    df_valid = df_closed[df_closed['cycle_days'] > 0]
    
    # Overall fallback
    if len(df_valid) > 0:
        overall_mean = df_valid['cycle_days'].mean()
        overall_std = df_valid['cycle_days'].std()
        overall_median = df_valid['cycle_days'].median()
    else:
        overall_mean, overall_std, overall_median = 60, 20, 60
    
    dso_dict = {}
    
    for seg in df['market_segment'].unique():
        seg_data = df_valid[df_valid['market_segment'] == seg]
        
        if len(seg_data) >= 3:
            dso_dict[seg] = {
                'mean': seg_data['cycle_days'].mean(),
                'std': max(seg_data['cycle_days'].std(), 1),
                'median': seg_data['cycle_days'].median()
            }
        else:
            dso_dict[seg] = {
                'mean': overall_mean,
                'std': max(overall_std, 1),
                'median': overall_median
            }
    
    return dso_dict


# ==========================================
# WIN RATE CALCULATION (from data)
# ==========================================

def calculate_win_rates_backtest(df, cutoff_date):
    """Calculate win rates from training data only."""
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_final['is_closed_flag'] = df_final['is_won_flag'] | df_final['is_lost_flag']
    
    df_closed = df_final[
        (df_final['is_closed_flag']) & 
        (df_final['date_closed'] <= cutoff_date)
    ].copy()
    
    # Overall fallback
    overall_won = df_closed['is_won_flag'].sum()
    overall_total = len(df_closed)
    overall_wr = overall_won / overall_total if overall_total > 0 else 0.20
    
    win_rates = {}
    
    for seg in df['market_segment'].unique():
        seg_data = df_closed[df_closed['market_segment'] == seg]
        
        won_count = seg_data['is_won_flag'].sum()
        total_count = len(seg_data)
        
        # Use actual rate from data
        if total_count > 0:
            win_rates[seg] = won_count / total_count
        else:
            win_rates[seg] = overall_wr
    
    return win_rates


# ==========================================
# VELOCITY CALCULATION (from data)
# ==========================================

def calculate_velocity_backtest(df, cutoff_date):
    """Calculate velocity from training data only."""
    df_train = df[df['date_snapshot'] <= cutoff_date].copy()
    
    df_first = df_train.sort_values('date_snapshot').groupby('deal_id').first().reset_index()
    df_first['appear_year'] = df_first['date_snapshot'].dt.year
    df_first['appear_month'] = df_first['date_snapshot'].dt.month
    
    cutoff_year = cutoff_date.year
    cutoff_month = cutoff_date.month
    
    if cutoff_month < 12:
        baseline_year = cutoff_year - 1
    else:
        baseline_year = cutoff_year
    
    if baseline_year not in df_first['appear_year'].values:
        baseline_year = df_first['appear_year'].max()
    
    df_baseline = df_first[df_first['appear_year'] == baseline_year].copy()
    
    velocity_dict = {}
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_baseline[df_baseline['market_segment'] == seg]
        
        annual_count = len(seg_data)
        annual_avg_size = seg_data['net_revenue'].mean() if len(seg_data) > 0 else df_baseline['net_revenue'].mean()
        
        if pd.isna(annual_avg_size):
            annual_avg_size = df['net_revenue'].mean()
        
        monthly_vols = []
        monthly_sizes = []
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['appear_month'] == m]
            
            # Use actual counts (including 0)
            monthly_vols.append(len(m_data))
            
            if len(m_data) > 0:
                monthly_sizes.append(m_data['net_revenue'].mean())
            else:
                monthly_sizes.append(annual_avg_size)
        
        # Optional smoothing
        smoothing = BACKTEST_ASSUMPTIONS.get('velocity_smoothing_window', 3)
        if smoothing > 1:
            monthly_vols_smooth = pd.Series(monthly_vols).rolling(
                window=smoothing, center=True, min_periods=1
            ).mean().tolist()
        else:
            monthly_vols_smooth = monthly_vols
        
        for m in range(1, 13):
            velocity_dict[seg][m] = {
                'vol': monthly_vols_smooth[m - 1],
                'size': monthly_sizes[m - 1]
            }
    
    return velocity_dict


# ==========================================
# DATA PROFILE (from training data)
# ==========================================

def profile_data_backtest(df, cutoff_date):
    """Profile deal size ranges from training data."""
    df_train = df[df['date_snapshot'] <= cutoff_date].copy()
    df_final = df_train.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    profile = {}
    
    for seg in df['market_segment'].unique():
        seg_data = df_final[df_final['market_segment'] == seg]
        
        if len(seg_data) > 0:
            profile[seg] = {
                'min_deal_size': seg_data['net_revenue'].min(),
                'max_deal_size': seg_data['net_revenue'].max(),
                'avg_deal_size': seg_data['net_revenue'].mean()
            }
        else:
            # Fallback to overall
            profile[seg] = {
                'min_deal_size': df_final['net_revenue'].min(),
                'max_deal_size': df_final['net_revenue'].max(),
                'avg_deal_size': df_final['net_revenue'].mean()
            }
    
    return profile


# ==========================================
# BACKTEST FORECAST ENGINE
# ==========================================

class BacktestEngine:
    """Forecast engine for backtest - mirrors production logic with neutral assumptions."""
    
    def __init__(self, win_rates, velocity, dso_dict, data_profile, test_months):
        self.win_rates = win_rates
        self.velocity = velocity
        self.dso_dict = dso_dict
        self.data_profile = data_profile
        self.test_months = test_months
    
    def generate_monthly_deals(self, month, segment):
        """Generate deals using data-derived baselines with NEUTRAL multipliers."""
        month_num = month.month
        
        if segment not in self.velocity or month_num not in self.velocity[segment]:
            return []
        
        stats = self.velocity[segment][month_num]
        
        # Apply NEUTRAL multiplier
        base_vol = stats['vol'] * BACKTEST_ASSUMPTIONS.get('volume_growth_multiplier', 1.0)
        
        if base_vol <= 0:
            num_deals = 0
        else:
            num_deals = np.random.poisson(base_vol)
        
        deals = []
        
        # Apply NEUTRAL inflation
        avg_size = stats['size'] * BACKTEST_ASSUMPTIONS.get('deal_size_inflation', 1.0)
        
        # Apply NEUTRAL win rate uplift
        base_wr = self.win_rates.get(segment, 0.20)
        adj_wr = base_wr * BACKTEST_ASSUMPTIONS.get('win_rate_uplift_multiplier', 1.0)
        
        seg_dso = self.dso_dict.get(segment, {'mean': 60, 'std': 20})
        
        # Get deal size range from data
        seg_profile = self.data_profile.get(segment, {})
        min_size = seg_profile.get('min_deal_size', avg_size * 0.5)
        max_size = seg_profile.get('max_deal_size', avg_size * 1.5)
        
        for _ in range(num_deals):
            is_won = np.random.random() < adj_wr
            
            # Sample from actual deal size range
            actual_rev = int(np.random.uniform(min_size, max_size))
            
            days_in_month = (month + pd.DateOffset(months=1) - timedelta(days=1)).day
            create_day = np.random.randint(1, days_in_month + 1)
            create_date = month + timedelta(days=create_day - 1)
            
            cycle_days = max(1, int(np.random.normal(seg_dso['mean'], seg_dso['std'])))
            close_date = create_date + timedelta(days=cycle_days)
            
            deals.append({
                'segment': segment,
                'revenue': actual_rev,
                'create_date': create_date,
                'close_date': close_date,
                'is_won': is_won,
                'create_month': month
            })
        
        return deals
    
    def run_simulation(self, existing_deals):
        """Run monthly simulation."""
        monthly_results = {month: defaultdict(lambda: {
            'won_vol': 0, 'won_rev': 0
        }) for month in self.test_months}
        
        all_deals = list(existing_deals)
        
        # Generate new deals for test period
        for month in self.test_months:
            for segment in self.velocity.keys():
                new_deals = self.generate_monthly_deals(month, segment)
                all_deals.extend(new_deals)
        
        # Aggregate by close month
        for deal in all_deals:
            close_month = pd.Period(deal['close_date'], freq='M').to_timestamp()
            
            if close_month in monthly_results and deal['is_won']:
                seg = deal['segment']
                monthly_results[close_month][seg]['won_vol'] += 1
                monthly_results[close_month][seg]['won_rev'] += deal['revenue']
        
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
    total_forecasted = df_comparison['forecasted_won_revenue'].sum()
    total_actual = df_comparison['actual_won_revenue'].sum()
    
    overall_pct_error = ((total_forecasted - total_actual) / total_actual * 100) if total_actual > 0 else 0
    
    df_valid = df_comparison[df_comparison['actual_won_revenue'] > 0].copy()
    
    if len(df_valid) == 0:
        return {
            'mape_revenue': 0, 
            'bias_revenue': 0, 
            'rmse_revenue': 0,
            'hit_rate_within_20pct': 0, 
            'num_periods_analyzed': 0,
            'overall_pct_error': overall_pct_error,
            'total_forecasted': total_forecasted,
            'total_actual': total_actual
        }
    
    mape_rev = df_valid['revenue_pct_error'].abs().mean()
    bias_rev = df_valid['revenue_pct_error'].mean()
    rmse_rev = np.sqrt((df_valid['revenue_error'] ** 2).mean())
    
    within_20pct = (df_valid['revenue_pct_error'].abs() <= 20).sum()
    hit_rate = (within_20pct / len(df_valid)) * 100
    
    return {
        'mape_revenue': mape_rev,
        'bias_revenue': bias_rev,
        'rmse_revenue': rmse_rev,
        'hit_rate_within_20pct': hit_rate,
        'num_periods_analyzed': len(df_valid),
        'overall_pct_error': overall_pct_error,
        'total_forecasted': total_forecasted,
        'total_actual': total_actual
    }


# ==========================================
# MAIN BACKTEST EXECUTION
# ==========================================

def run_backtest_validation():
    print("=" * 70)
    print("BACKTEST VALIDATION V4 - DATA-DRIVEN WITH NEUTRAL ASSUMPTIONS")
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
    
    print(f"    Training: {df_train['date_snapshot'].min().date()} to {train_end.date()}")
    print(f"    Testing:  {test_start.date()} to {test_end.date()}")
    print(f"    Training records: {len(df_train):,}")
    print(f"    Unique deals in training: {df_train['deal_id'].nunique():,}")
    
    print("\n--- 2. Calculating Drivers (from training data) ---")
    dso_dict = calculate_dso_backtest(df_train, train_end)
    win_rates = calculate_win_rates_backtest(df_train, train_end)
    velocity = calculate_velocity_backtest(df_train, train_end)
    data_profile = profile_data_backtest(df_train, train_end)
    
    print(f"    Drivers calculated for {len(win_rates)} segments:")
    for seg, wr in win_rates.items():
        dso = dso_dict.get(seg, {}).get('median', 60)
        seg_vel = velocity.get(seg, {})
        avg_monthly = np.mean([seg_vel.get(m, {}).get('vol', 0) for m in range(1, 13)])
        print(f"      {seg}: WR={wr:.1%}, DSO={dso:.0f}d, Vel={avg_monthly:.2f}/month")
    
    print("\n--- 3. Initializing Pipeline (as of training cutoff) ---")
    
    latest_train_snapshot = df_train['date_snapshot'].max()
    df_latest = df_train[df_train['date_snapshot'] == latest_train_snapshot].copy()
    
    df_latest['is_closed_flag'] = df_latest['stage'].apply(is_closed)
    df_open = df_latest[~df_latest['is_closed_flag']].copy()
    
    print(f"    Latest training snapshot: {latest_train_snapshot.date()}")
    print(f"    Open deals at cutoff: {len(df_open):,}")
    
    existing_deals = []
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        create_date = pd.to_datetime(row['date_created'])
        
        base_wr = win_rates.get(seg, 0.20)
        # NEUTRAL multiplier
        adj_wr = base_wr * BACKTEST_ASSUMPTIONS.get('win_rate_uplift_multiplier', 1.0)
        is_won = np.random.random() < adj_wr
        
        seg_dso = dso_dict.get(seg, {'mean': 60, 'std': 20})
        days_open = (latest_train_snapshot - create_date).days
        
        total_cycle = int(np.random.normal(seg_dso['mean'], seg_dso['std']))
        total_cycle = max(days_open + np.random.randint(7, 30), total_cycle)
        
        close_date = create_date + timedelta(days=total_cycle)
        
        existing_deals.append({
            'segment': seg,
            'revenue': row['net_revenue'],
            'create_date': create_date,
            'close_date': close_date,
            'is_won': is_won
        })
    
    print("\n--- 4. Running Backtest Simulations ---")
    
    test_months = pd.date_range(test_start, test_end, freq='MS')
    num_sims = BACKTEST_ASSUMPTIONS.get('num_simulations', 500)
    
    engine = BacktestEngine(win_rates, velocity, dso_dict, data_profile, test_months)
    
    all_forecasted = []
    
    for sim in range(num_sims):
        if (sim + 1) % 100 == 0:
            print(f"    Completed {sim + 1}/{num_sims} simulations...")
        
        forecasted = engine.run_simulation(existing_deals)
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
    
    print("\n--- 6. Comparing Forecast to Actual ---")
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    df_comparison = compare_forecast_to_actual(all_forecasted, actual_results)
    df_comparison.to_csv(OUTPUT_BACKTEST_RESULTS, index=False)
    
    monthly_comp = df_comparison.groupby('month').agg({
        'forecasted_won_volume': 'sum',
        'actual_won_volume': 'sum',
        'forecasted_won_revenue': 'sum',
        'actual_won_revenue': 'sum'
    })
    
    print("\n    Monthly Comparison:")
    print(f"    {'Month':<12} {'Fcst Vol':>10} {'Actual Vol':>12} {'Fcst Rev':>14} {'Actual Rev':>14}")
    print("    " + "-" * 66)
    
    for month, row in monthly_comp.iterrows():
        print(f"    {month.strftime('%Y-%m'):<12} {row['forecasted_won_volume']:>10.1f} {row['actual_won_volume']:>12.0f} ${row['forecasted_won_revenue']:>13,.0f} ${row['actual_won_revenue']:>13,.0f}")
    
    total_fcst = monthly_comp['forecasted_won_revenue'].sum()
    print("    " + "-" * 66)
    print(f"    {'TOTAL':<12} {monthly_comp['forecasted_won_volume'].sum():>10.1f} {monthly_comp['actual_won_volume'].sum():>12.0f} ${total_fcst:>13,.0f} ${total_actual_rev:>13,.0f}")
    
    print(f"\n    Results saved: {OUTPUT_BACKTEST_RESULTS}")
    
    print("\n--- 7. Calculating Accuracy Metrics ---")
    metrics = calculate_accuracy_metrics(df_comparison)
    
    # Generate summary
    summary_text = f"""
{'=' * 70}
BACKTEST VALIDATION SUMMARY V4 - DATA-DRIVEN MODEL
{'=' * 70}

TEST PERIOD: {test_start.date()} to {test_end.date()} (6 months)
TRAINING PERIOD: {df_train['date_snapshot'].min().date()} to {train_end.date()}

OVERALL ACCURACY
----------------
Total Forecasted Revenue:      ${metrics['total_forecasted']:>15,.0f}
Total Actual Revenue:          ${metrics['total_actual']:>15,.0f}
Difference:                    ${metrics['total_forecasted'] - metrics['total_actual']:>+15,.0f}
Overall Percent Error:         {metrics['overall_pct_error']:>+15.1f}%

SEGMENT-LEVEL METRICS
---------------------
Mean Absolute Percentage Error (MAPE):  {metrics['mape_revenue']:.2f}%
Forecast Bias:                          {metrics['bias_revenue']:+.2f}%
Root Mean Squared Error (RMSE):         ${metrics['rmse_revenue']:,.0f}
Hit Rate (within +/-20%):               {metrics['hit_rate_within_20pct']:.1f}%

Periods Analyzed:                       {metrics['num_periods_analyzed']}

INTERPRETATION
--------------
"""
    
    abs_overall_error = abs(metrics['overall_pct_error'])
    if abs_overall_error < 25:
        summary_text += "[OK] Model learned historical patterns well (<25% error)\n"
    elif abs_overall_error < 50:
        summary_text += "[--] Model shows moderate accuracy (25-50% error)\n"
    else:
        summary_text += "[!!] Model needs calibration (>50% error)\n"
    
    if metrics['overall_pct_error'] > 25:
        summary_text += "[!!] OVER-FORECASTING: Check velocity calculation\n"
    elif metrics['overall_pct_error'] < -25:
        summary_text += "[!!] UNDER-FORECASTING: Check DSO distribution\n"
    else:
        summary_text += "[OK] No significant systematic bias\n"
    
    summary_text += f"""

DATA-DERIVED BASELINES USED
---------------------------
"""
    for seg in win_rates.keys():
        wr = win_rates.get(seg, 0)
        dso = dso_dict.get(seg, {}).get('mean', 60)
        seg_vel = velocity.get(seg, {})
        avg_monthly = np.mean([seg_vel.get(m, {}).get('vol', 0) for m in range(1, 13)])
        summary_text += f"  {seg:20s}: WR={wr:.1%}, DSO={dso:.0f}d, Vel={avg_monthly:.2f}/mo\n"

    summary_text += f"""

BACKTEST ASSUMPTIONS (NEUTRAL)
------------------------------
"""
    for key, value in BACKTEST_ASSUMPTIONS.items():
        summary_text += f"  {key:40s}: {value}\n"
    
    summary_text += f"""

RECOMMENDATION
--------------
"""
    
    if abs_overall_error < 30:
        summary_text += "[OK] Model is suitable for production forecasting.\n"
        summary_text += "     Apply growth multipliers for 2026 projections.\n"
    else:
        summary_text += "[!!] Review velocity and DSO calculations.\n"
        summary_text += "     Model may not have learned patterns correctly.\n"
    
    summary_text += f"\n{'=' * 70}\n"
    
    print(summary_text)
    
    with open(OUTPUT_BACKTEST_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"Summary saved: {OUTPUT_BACKTEST_SUMMARY}")
    
    print("\n" + "=" * 70)
    print("BACKTEST VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest_validation()