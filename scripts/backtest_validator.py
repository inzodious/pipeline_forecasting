import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict

# ==========================================
# BACKTEST VALIDATION FRAMEWORK
# ==========================================
"""
PURPOSE:
Validates the forecast model by training on early data and testing against 
known outcomes in later periods. Operates at MONTHLY AGGREGATION level.

METHODOLOGY:
1. Split data: Train on Jan 2024 - Jun 2025, Test on Jul - Dec 2025
2. Calculate drivers from training period only
3. Generate monthly forecast for test period
4. Compare forecasted monthly totals to actual monthly totals
5. Calculate accuracy metrics (MAPE, bias, hit rate)

This proves the model works before trusting it with future forecasts.
"""

# Paths
BASE_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(BASE_DIR, "fact_pipeline_snapshot.csv")
OUTPUT_BACKTEST_RESULTS = os.path.join(BASE_DIR, "backtest_validation_results.csv")
OUTPUT_BACKTEST_SUMMARY = os.path.join(BASE_DIR, "backtest_summary.txt")

# Configuration
TRAIN_END_DATE = '2025-06-30'
TEST_START_DATE = '2025-07-01'
TEST_END_DATE = '2025-12-31'

PATH_ASSUMPTIONS_LOG = os.path.join(BASE_DIR, "forecast_assumptions_log.csv")

# Load assumptions from forecast run
def load_forecast_assumptions():
    """
    Loads the exact assumptions used in the production forecast.
    This ensures backtest uses identical parameters for fair validation.
    """
    if os.path.exists(PATH_ASSUMPTIONS_LOG):
        print("  > Loading assumptions from production forecast run...")
        df_assumptions = pd.read_csv(PATH_ASSUMPTIONS_LOG)
        
        assumptions = {}
        for _, row in df_assumptions.iterrows():
            key = row['assumption']
            value = row['value']
            
            # Convert string values to appropriate types
            if value == 'None':
                assumptions[key] = None
            elif isinstance(value, str) and value.replace('.', '').isdigit():
                assumptions[key] = float(value) if '.' in value else int(value)
            else:
                assumptions[key] = value
        
        print(f"    Loaded {len(assumptions)} assumptions from forecast run")
        return assumptions
    else:
        print("  > WARNING: No forecast assumptions log found.")
        print("    Using default backtest assumptions (no growth for pure model test)")
        return {
            'volume_growth_multiplier': 1.00,
            'win_rate_uplift_multiplier': 1.00,
            'deal_size_inflation': 1.00,
            'num_simulations': 200,
            'min_monthly_deals_floor': 3,
            'max_monthly_deals_ceiling': 100,
            'min_win_rate_floor': 0.05,
            'max_win_rate_ceiling': 0.50,
            'deal_size_variance_cap': 0.25,
            'velocity_smoothing_window': 3,
            'confidence_interval_clip': 0.95
        }

# Load assumptions at module level
BACKTEST_ASSUMPTIONS = load_forecast_assumptions()

# ==========================================
# DRIVER CALCULATIONS (TRAINING DATA ONLY)
# ==========================================

def calculate_win_rates_backtest(df, cutoff_date):
    """Calculate win rates using only training data."""
    df = df.sort_values(['deal_id', 'snapshot_date'])
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['target_implementation_date'] = pd.to_datetime(df['target_implementation_date'])
    
    df_final = df.groupby('deal_id').last().reset_index()
    
    df_closed = df_final[df_final['status'].isin(['Closed Won', 'Closed Lost'])].copy()
    df_closed['cycle_days'] = (df_closed['target_implementation_date'] - df_closed['date_created']).dt.days
    
    avg_cycle_days = df_closed.groupby('market_segment')['cycle_days'].median().to_dict()
    default_cycle = df_closed['cycle_days'].median() if len(df_closed) > 0 else 120
    
    df_final['expected_close'] = df_final.apply(
        lambda row: row['date_created'] + timedelta(days=avg_cycle_days.get(row['market_segment'], default_cycle)),
        axis=1
    )
    
    mask_mature = (df_final['status'].isin(['Closed Won', 'Closed Lost'])) | (df_final['expected_close'] <= cutoff_date)
    df_mature = df_final[mask_mature].copy()
    
    if len(df_mature) == 0:
        return {seg: 0.20 for seg in df['market_segment'].unique()}
    
    win_rates_dict = {}
    for segment, group in df_mature.groupby('market_segment'):
        total_created = group['revenue'].sum()
        won_rev = group[group['status'] == 'Closed Won']['revenue'].sum()
        raw_wr = (won_rev / total_created) if total_created > 0 else 0.20
        
        win_rates_dict[segment] = np.clip(
            raw_wr,
            BACKTEST_ASSUMPTIONS['min_win_rate_floor'],
            BACKTEST_ASSUMPTIONS['max_win_rate_ceiling']
        )
    
    return win_rates_dict


def calculate_velocity_backtest(df):
    """Calculate velocity using only training data."""
    df_qual = df[df['status'] == 'Qualified'].copy()
    df_qual['snapshot_date'] = pd.to_datetime(df_qual['snapshot_date'])
    
    df_first_qual = df_qual.sort_values('snapshot_date').groupby('deal_id').first().reset_index()
    df_first_qual['qualified_year'] = df_first_qual['snapshot_date'].dt.year
    df_first_qual['qualified_month'] = df_first_qual['snapshot_date'].dt.month
    
    # Use 2024 data only for backtest
    df_train = df_first_qual[df_first_qual['qualified_year'] == 2024].copy()
    
    if len(df_train) == 0:
        df_train = df_first_qual.copy()
    
    velocity_dict = {}
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_train[df_train['market_segment'] == seg]
        
        annual_avg_vol = max(BACKTEST_ASSUMPTIONS['min_monthly_deals_floor'], len(seg_data) / 12)
        annual_avg_size = seg_data['revenue'].mean() if len(seg_data) > 0 else 50000
        
        monthly_vols = []
        monthly_sizes = []
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['qualified_month'] == m]
            
            if len(m_data) > 0:
                monthly_vols.append(len(m_data))
                monthly_sizes.append(m_data['revenue'].mean())
            else:
                monthly_vols.append(annual_avg_vol)
                monthly_sizes.append(annual_avg_size)
        
        # Apply smoothing
        if BACKTEST_ASSUMPTIONS['velocity_smoothing_window'] > 1:
            monthly_vols_smooth = pd.Series(monthly_vols).rolling(
                window=BACKTEST_ASSUMPTIONS['velocity_smoothing_window'],
                center=True,
                min_periods=1
            ).mean().tolist()
        else:
            monthly_vols_smooth = monthly_vols
        
        for m in range(1, 13):
            vol = np.clip(
                monthly_vols_smooth[m - 1],
                BACKTEST_ASSUMPTIONS['min_monthly_deals_floor'],
                BACKTEST_ASSUMPTIONS['max_monthly_deals_ceiling']
            )
            velocity_dict[seg][m] = {'vol': vol, 'size': monthly_sizes[m - 1]}
    
    return velocity_dict


def calculate_lags_backtest(df):
    """Calculate cycle lags using only training data."""
    df_won = df[df['status'] == 'Closed Won'].copy()
    df_won['date_created'] = pd.to_datetime(df_won['date_created'])
    df_won['target_implementation_date'] = pd.to_datetime(df_won['target_implementation_date'])
    
    df_won['cycle_days'] = (df_won['target_implementation_date'] - df_won['date_created']).dt.days
    df_won = df_won[(df_won['cycle_days'] > 0) & (df_won['cycle_days'] < 730)]
    
    cycle_stats = df_won.groupby('market_segment')['cycle_days'].median().reset_index()
    cycle_stats['lag_months'] = (cycle_stats['cycle_days'] / 30).apply(lambda x: int(max(1, x)))
    
    return dict(zip(cycle_stats['market_segment'], cycle_stats['lag_months']))


# ==========================================
# MONTHLY FORECAST ENGINE (BACKTEST VERSION)
# ==========================================

class BacktestMonthlyEngine:
    """Optimized monthly forecast engine for backtest."""
    
    def __init__(self, win_rates, velocity, lags, test_months):
        self.win_rates = win_rates
        self.velocity = velocity
        self.lags = lags
        self.test_months = test_months
    
    def generate_monthly_deals(self, month, segment):
        """Generate deals for a specific month/segment."""
        month_num = month.month
        
        if month_num not in self.velocity[segment]:
            return []
        
        stats = self.velocity[segment][month_num]
        
        base_vol = stats['vol'] * BACKTEST_ASSUMPTIONS['volume_growth_multiplier']
        base_vol = np.clip(
            base_vol,
            BACKTEST_ASSUMPTIONS['min_monthly_deals_floor'],
            BACKTEST_ASSUMPTIONS['max_monthly_deals_ceiling']
        )
        
        num_deals = max(
            BACKTEST_ASSUMPTIONS['min_monthly_deals_floor'],
            np.random.poisson(base_vol)
        )
        
        deals = []
        avg_size = stats['size'] * BACKTEST_ASSUMPTIONS['deal_size_inflation']
        
        base_wr = self.win_rates.get(segment, 0.20)
        adj_wr = np.clip(
            base_wr * BACKTEST_ASSUMPTIONS['win_rate_uplift_multiplier'],
            BACKTEST_ASSUMPTIONS['min_win_rate_floor'],
            BACKTEST_ASSUMPTIONS['max_win_rate_ceiling']
        )
        
        lag_months = self.lags.get(segment, 4)
        
        for i in range(num_deals):
            is_won = np.random.random() < adj_wr
            
            size_mult = 1.0 + np.random.uniform(
                -BACKTEST_ASSUMPTIONS['deal_size_variance_cap'],
                BACKTEST_ASSUMPTIONS['deal_size_variance_cap']
            )
            actual_rev = int(avg_size * size_mult)
            
            days_in_month = (month + pd.DateOffset(months=1) - timedelta(days=1)).day
            create_day = np.random.randint(1, days_in_month + 1)
            create_date = month + timedelta(days=create_day - 1)
            
            close_date = create_date + timedelta(days=int(lag_months * 30))
            close_date += timedelta(days=np.random.randint(-10, 10))
            
            deals.append({
                'segment': segment,
                'revenue': actual_rev,
                'create_date': create_date,
                'close_date': close_date,
                'is_won': is_won,
                'create_month': month
            })
        
        return deals
    
    def run_monthly_simulation(self, existing_deals):
        """Run monthly aggregated simulation."""
        monthly_results = {month: defaultdict(lambda: {
            'won_vol': 0, 'won_rev': 0
        }) for month in self.test_months}
        
        all_deals = list(existing_deals)
        
        # Generate new deals for each test month
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
# ACTUAL RESULTS EXTRACTION (MONTHLY)
# ==========================================

def extract_actual_monthly_results(df, test_start, test_end):
    """Extract actual monthly results from test period."""
    df['target_implementation_date'] = pd.to_datetime(df['target_implementation_date'])
    
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    
    # Get final state of each deal
    df_final = df.sort_values('snapshot_date').groupby('deal_id').last().reset_index()
    
    # Filter to deals closed in test period
    df_closed = df_final[
        (df_final['status'] == 'Closed Won') &
        (df_final['target_implementation_date'] >= test_start) &
        (df_final['target_implementation_date'] <= test_end)
    ].copy()
    
    df_closed['close_month'] = df_closed['target_implementation_date'].dt.to_period('M').dt.to_timestamp()
    
    # Aggregate by month and segment
    actual_monthly = defaultdict(lambda: defaultdict(lambda: {
        'actual_won_vol': 0,
        'actual_won_rev': 0
    }))
    
    for _, row in df_closed.iterrows():
        month = row['close_month']
        seg = row['market_segment']
        actual_monthly[month][seg]['actual_won_vol'] += 1
        actual_monthly[month][seg]['actual_won_rev'] += row['revenue']
    
    return actual_monthly


# ==========================================
# COMPARISON & METRICS
# ==========================================

def compare_forecast_to_actual(forecasted_list, actual):
    """Compare average forecasted results to actual."""
    # Average across simulations
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
    
    # Build comparison
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
    df_valid = df_comparison[df_comparison['actual_won_revenue'] > 0].copy()
    
    if len(df_valid) == 0:
        return {
            'mape_revenue': 0, 'bias_revenue': 0, 'rmse_revenue': 0,
            'hit_rate_within_20pct': 0, 'num_periods_analyzed': 0
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
        'num_periods_analyzed': len(df_valid)
    }


# ==========================================
# MAIN BACKTEST EXECUTION
# ==========================================

def run_backtest_validation():
    """Main backtest validation workflow."""
    print("=" * 70)
    print("MONTHLY BACKTEST VALIDATION - MODEL ACCURACY TEST")
    print("=" * 70)
    
    print("\n--- 0. Loading Production Forecast Assumptions ---")
    # Reload to show user what was loaded
    if os.path.exists(PATH_ASSUMPTIONS_LOG):
        df_show = pd.read_csv(PATH_ASSUMPTIONS_LOG)
        print("\n    Production forecast used these assumptions:")
        for _, row in df_show.iterrows():
            print(f"      {row['assumption']:30s} = {row['value']}")
        print("\n    Backtest will use IDENTICAL assumptions for fair comparison.")
    else:
        print("\n    No forecast log found - using default assumptions.")
        print("    Run forecast_generator_v2.py first to generate assumptions log.")
    
    print("\n--- 1. Loading Historical Data ---")
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print("Error: Snapshot file not found.")
        return
    
    df_raw = pd.read_csv(
        PATH_RAW_SNAPSHOTS,
        parse_dates=['snapshot_date', 'date_created', 'target_implementation_date']
    )
    
    train_end = pd.to_datetime(TRAIN_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)
    
    df_train = df_raw[df_raw['snapshot_date'] <= train_end].copy()
    df_test = df_raw[df_raw['snapshot_date'] <= test_end].copy()
    
    print(f"    Training: {df_train['snapshot_date'].min().date()} to {train_end.date()}")
    print(f"    Testing:  {test_start.date()} to {test_end.date()}")
    print(f"    Training records: {len(df_train):,}")
    
    print("\n--- 2. Calculating Drivers (Training Data Only) ---")
    win_rates = calculate_win_rates_backtest(df_train, train_end)
    velocity = calculate_velocity_backtest(df_train)
    lags = calculate_lags_backtest(df_train)
    
    print(f"    Win rates: {len(win_rates)} segments")
    print(f"    Velocity patterns established")
    
    print("\n--- 3. Running Backtest Simulations (Monthly Aggregation) ---")
    
    test_months = pd.date_range(test_start, test_end, freq='MS')
    engine = BacktestMonthlyEngine(win_rates, velocity, lags, test_months)
    
    # Initialize pipeline
    latest_train_date = df_train['snapshot_date'].max()
    df_latest = df_train[df_train['snapshot_date'] == latest_train_date].copy()
    
    excluded = ['Closed Won', 'Closed Lost', 'Initiated', 'Verbal', 'Declined to Bid']
    df_open = df_latest[~df_latest['status'].isin(excluded)].copy()
    
    existing_deals = []
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        base_wr = win_rates.get(seg, 0.20)
        is_won = np.random.random() < base_wr
        
        close_date = pd.to_datetime(row['target_implementation_date'])
        if close_date <= latest_train_date:
            close_date = latest_train_date + timedelta(days=30)
        
        existing_deals.append({
            'segment': seg,
            'revenue': row['revenue'],
            'create_date': pd.to_datetime(row['date_created']),
            'close_date': close_date,
            'is_won': is_won
        })
    
    # Run simulations
    all_forecasted = []
    
    for sim in range(BACKTEST_ASSUMPTIONS['num_simulations']):
        if (sim + 1) % 50 == 0:
            print(f"    Completed {sim + 1}/{BACKTEST_ASSUMPTIONS['num_simulations']} simulations...")
        
        forecasted = engine.run_monthly_simulation(existing_deals)
        all_forecasted.append(forecasted)
    
    print("\n--- 4. Extracting Actual Monthly Results ---")
    actual_results = extract_actual_monthly_results(df_test, test_start, test_end)
    
    print("\n--- 5. Comparing Forecast to Actual (Monthly Totals) ---")
    df_comparison = compare_forecast_to_actual(all_forecasted, actual_results)
    df_comparison.to_csv(OUTPUT_BACKTEST_RESULTS, index=False)
    
    print(f"    Results saved: {OUTPUT_BACKTEST_RESULTS}")
    
    print("\n--- 6. Calculating Accuracy Metrics ---")
    metrics = calculate_accuracy_metrics(df_comparison)
    
    # Generate summary
    summary_text = f"""
{'=' * 70}
MONTHLY BACKTEST VALIDATION SUMMARY
{'=' * 70}

TEST PERIOD: {test_start.date()} to {test_end.date()} (6 months)
TRAINING PERIOD: {df_train['snapshot_date'].min().date()} to {train_end.date()}
AGGREGATION LEVEL: Monthly totals by segment

ACCURACY METRICS
----------------
Mean Absolute Percentage Error (MAPE):  {metrics['mape_revenue']:.2f}%
Forecast Bias:                          {metrics['bias_revenue']:+.2f}%
Root Mean Squared Error (RMSE):         ${metrics['rmse_revenue']:,.0f}
Hit Rate (within ±20%):                 {metrics['hit_rate_within_20pct']:.1f}%

Month-Segment Periods Analyzed:         {metrics['num_periods_analyzed']}

INTERPRETATION
--------------
"""
    
    if metrics['mape_revenue'] < 15:
        summary_text += "✓ EXCELLENT: Model demonstrates strong predictive accuracy (MAPE < 15%)\n"
    elif metrics['mape_revenue'] < 25:
        summary_text += "✓ GOOD: Model shows acceptable predictive accuracy (MAPE < 25%)\n"
    else:
        summary_text += "⚠ REVIEW NEEDED: Model accuracy below target (MAPE > 25%)\n"
    
    if abs(metrics['bias_revenue']) < 5:
        summary_text += "✓ UNBIASED: No systematic over/under-forecasting\n"
    elif metrics['bias_revenue'] > 5:
        summary_text += "⚠ UPWARD BIAS: Model tends to over-forecast\n"
    else:
        summary_text += "⚠ DOWNWARD BIAS: Model tends to under-forecast\n"
    
    if metrics['hit_rate_within_20pct'] > 70:
        summary_text += "✓ HIGH PRECISION: Most forecasts within ±20% of actual\n"
    else:
        summary_text += "⚠ MODERATE PRECISION: Consider tightening variance constraints\n"
    
    summary_text += f"\n"
    summary_text += f"RECOMMENDATION\n"
    summary_text += f"--------------\n"
    
    if metrics['mape_revenue'] < 20 and abs(metrics['bias_revenue']) < 10:
        summary_text += "✓ Model is suitable for production forecasting.\n"
        summary_text += "  Proceed with 2026 forecast with confidence.\n"
    else:
        summary_text += "⚠ Model requires calibration.\n"
        summary_text += "  Review driver calculations and constraint parameters.\n"
    
    summary_text += f"\n"
    summary_text += f"ASSUMPTIONS USED IN BACKTEST\n"
    summary_text += f"----------------------------\n"
    summary_text += f"NOTE: These assumptions were loaded from the production forecast run\n"
    summary_text += f"      to ensure fair validation (same parameters, different time period).\n\n"
    
    if os.path.exists(PATH_ASSUMPTIONS_LOG):
        df_assumptions = pd.read_csv(PATH_ASSUMPTIONS_LOG)
        for _, row in df_assumptions.iterrows():
            summary_text += f"{row['assumption']:35s}: {row['value']}\n"
    else:
        summary_text += f"No assumptions log found - used defaults.\n"
    
    summary_text += f"\n{'=' * 70}\n"
    
    print(summary_text)
    
    with open(OUTPUT_BACKTEST_SUMMARY, 'w') as f:
        f.write(summary_text)
    
    print(f"Summary saved: {OUTPUT_BACKTEST_SUMMARY}")
    
    print("\n" + "=" * 70)
    print("MONTHLY BACKTEST VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest_validation()