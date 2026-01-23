import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt

# ==========================================
# BACKTEST VALIDATION FRAMEWORK
# ==========================================
"""
PURPOSE:
This script validates the forecast model by training on early data and 
testing against known outcomes in later periods.

METHODOLOGY:
1. Split historical data into training and test periods
2. Use training period to calculate drivers (win rates, velocity, lags)
3. Generate forecast for test period using same logic as production forecast
4. Compare forecasted values to actual realized values
5. Calculate accuracy metrics (MAPE, bias, hit rate)

This proves the model works before trusting it with future forecasts.
"""

# Paths
BASE_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(BASE_DIR, "fact_pipeline_snapshot.csv")
OUTPUT_BACKTEST_RESULTS = os.path.join(BASE_DIR, "backtest_validation_results.csv")
OUTPUT_BACKTEST_SUMMARY = os.path.join(BASE_DIR, "backtest_summary.txt")
OUTPUT_BACKTEST_CHART = os.path.join(BASE_DIR, "backtest_accuracy_chart.png")

# Configuration
TRAIN_END_DATE = '2025-06-30'  # Train on data through June 2025
TEST_START_DATE = '2025-07-01'  # Test on July-Dec 2025
TEST_END_DATE = '2025-12-31'

# Simplified assumptions for backtest (no growth, just test model mechanics)
BACKTEST_ASSUMPTIONS = {
    'volume_growth_multiplier': 1.00,
    'win_rate_uplift_multiplier': 1.00,
    'deal_size_inflation': 1.00,
    'num_simulations': 500
}

# ==========================================
# IMPORT DRIVER CALCULATION FUNCTIONS
# ==========================================

def calculate_win_rates_backtest(df, cutoff_date):
    """Calculate win rates using only training data."""
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['target_implementation_date'] = pd.to_datetime(df['target_implementation_date'])
    
    df_final = df.sort_values('snapshot_date', ascending=False).groupby('deal_id').first().reset_index()
    
    # Calculate expected cycle times
    df_closed = df_final[df_final['status'].isin(['Closed Won', 'Closed Lost'])].copy()
    df_closed['cycle_days'] = (df_closed['target_implementation_date'] - df_closed['date_created']).dt.days
    
    avg_cycle_days = {}
    for seg in df_closed['market_segment'].unique():
        seg_cycle = df_closed[df_closed['market_segment'] == seg]['cycle_days'].median()
        avg_cycle_days[seg] = seg_cycle if pd.notna(seg_cycle) else 120
    
    # Filter to mature deals
    df_mature = []
    for _, row in df_final.iterrows():
        seg = row['market_segment']
        expected_close = row['date_created'] + timedelta(days=avg_cycle_days.get(seg, 120))
        
        if row['status'] in ['Closed Won', 'Closed Lost'] or expected_close <= cutoff_date:
            df_mature.append(row)
    
    df_mature = pd.DataFrame(df_mature)
    
    if len(df_mature) == 0:
        return {seg: 0.20 for seg in df['market_segment'].unique()}
    
    # Calculate win rates
    win_rates_dict = {}
    for segment, group in df_mature.groupby('market_segment'):
        total_created = group['revenue'].sum()
        won_rev = group[group['status'] == 'Closed Won']['revenue'].sum()
        win_rate = (won_rev / total_created) if total_created > 0 else 0.20
        win_rates_dict[segment] = win_rate
    
    return win_rates_dict


def calculate_velocity_backtest(df, train_year):
    """Calculate velocity using only training data."""
    df_qual = df[df['status'] == 'Qualified'].copy()
    df_qual['snapshot_date'] = pd.to_datetime(df_qual['snapshot_date'])
    
    df_first_qual = df_qual.sort_values('snapshot_date').groupby('deal_id').first().reset_index()
    df_first_qual['qualified_year'] = df_first_qual['snapshot_date'].dt.year
    df_first_qual['qualified_month'] = df_first_qual['snapshot_date'].dt.month
    
    # Use training year only
    df_train = df_first_qual[df_first_qual['qualified_year'] == train_year].copy()
    
    if len(df_train) == 0:
        df_train = df_first_qual.copy()
    
    velocity_dict = {}
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_train[df_train['market_segment'] == seg]
        
        annual_avg_vol = len(seg_data) / 12 if len(seg_data) > 0 else 1
        annual_avg_size = seg_data['revenue'].mean() if len(seg_data) > 0 else 50000
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['qualified_month'] == m]
            
            if len(m_data) > 0:
                vol = len(m_data)
                size = m_data['revenue'].mean()
            else:
                vol = annual_avg_vol
                size = annual_avg_size
            
            velocity_dict[seg][m] = {'vol': vol, 'size': size}
    
    return velocity_dict


def calculate_lags_backtest(df):
    """Calculate cycle lags using only training data."""
    df_won = df[df['status'] == 'Closed Won'].copy()
    df_won['date_created'] = pd.to_datetime(df_won['date_created'])
    df_won['target_implementation_date'] = pd.to_datetime(df_won['target_implementation_date'])
    
    df_won['cycle_days'] = (df_won['target_implementation_date'] - df_won['date_created']).dt.days
    df_won = df_won[(df_won['cycle_days'] > 0) & (df_won['cycle_days'] < 730)]
    
    cycle_lags = {}
    for seg, group in df_won.groupby('market_segment'):
        median_days = group['cycle_days'].median()
        lag_months = int(max(1, median_days / 30))
        cycle_lags[seg] = lag_months
    
    return cycle_lags


# ==========================================
# BACKTEST SIMULATION ENGINE
# ==========================================

def run_backtest_simulation(df_train, win_rates, velocity, lags, test_start, test_end):
    """
    Runs forecast simulation for test period using training-derived parameters.
    Returns monthly aggregated results.
    """
    # Initialize with open pipeline at end of training
    latest_train_date = df_train['snapshot_date'].max()
    df_latest = df_train[df_train['snapshot_date'] == latest_train_date].copy()
    
    excluded = ['Closed Won', 'Closed Lost', 'Initiated', 'Verbal', 'Declined to Bid']
    df_open = df_latest[~df_latest['status'].isin(excluded)].copy()
    
    # Simulate test period
    sim_dates = pd.date_range(start=test_start, end=test_end, freq='W-FRI')
    
    forecasted_monthly = defaultdict(lambda: defaultdict(lambda: {
        'forecasted_won_vol': 0,
        'forecasted_won_rev': 0
    }))
    
    # Simplified simulation (no new deals generated, just track existing pipeline)
    # This is conservative - in reality we'd also forecast new deal creation
    
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        close_date = pd.to_datetime(row['target_implementation_date'])
        
        if close_date < test_start:
            close_date = test_start + timedelta(days=30)
        
        if close_date <= test_end:
            # Determine if won
            base_wr = win_rates.get(seg, 0.20)
            is_won = np.random.random() < base_wr
            
            if is_won:
                close_month = pd.Period(close_date, freq='M').to_timestamp()
                forecasted_monthly[close_month][seg]['forecasted_won_vol'] += 1
                forecasted_monthly[close_month][seg]['forecasted_won_rev'] += row['revenue']
    
    # Also simulate new deals created during test period
    for current_date in sim_dates:
        curr_month = current_date.month
        
        for seg, monthly_data in velocity.items():
            if curr_month not in monthly_data:
                continue
            
            stats = monthly_data[curr_month]
            weekly_vol_target = stats['vol'] / 4.3
            num_to_create = np.random.poisson(weekly_vol_target)
            
            if num_to_create > 0:
                avg_size = stats['size']
                base_wr = win_rates.get(seg, 0.20)
                lag_months = lags.get(seg, 4)
                
                for _ in range(num_to_create):
                    is_won = np.random.random() < base_wr
                    actual_rev = int(avg_size * np.random.uniform(0.9, 1.1))
                    
                    target_close = current_date + timedelta(days=int(lag_months * 30))
                    
                    if target_close <= test_end and is_won:
                        close_month = pd.Period(target_close, freq='M').to_timestamp()
                        forecasted_monthly[close_month][seg]['forecasted_won_vol'] += 1
                        forecasted_monthly[close_month][seg]['forecasted_won_rev'] += actual_rev
    
    return forecasted_monthly


# ==========================================
# ACTUAL RESULTS EXTRACTION
# ==========================================

def extract_actual_results(df, test_start, test_end):
    """
    Extracts actual realized performance during test period.
    """
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['target_implementation_date'] = pd.to_datetime(df['target_implementation_date'])
    
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    
    # Get deals that closed during test period
    df_closed = df[df['status'] == 'Closed Won'].copy()
    df_closed['close_month'] = df_closed['target_implementation_date'].dt.to_period('M').dt.to_timestamp()
    
    # Filter to test period
    df_test_closed = df_closed[
        (df_closed['target_implementation_date'] >= test_start) &
        (df_closed['target_implementation_date'] <= test_end)
    ]
    
    # Aggregate by month and segment
    actual_monthly = defaultdict(lambda: defaultdict(lambda: {
        'actual_won_vol': 0,
        'actual_won_rev': 0
    }))
    
    for _, row in df_test_closed.iterrows():
        month = row['close_month']
        seg = row['market_segment']
        actual_monthly[month][seg]['actual_won_vol'] += 1
        actual_monthly[month][seg]['actual_won_rev'] += row['revenue']
    
    return actual_monthly


# ==========================================
# COMPARISON & ACCURACY METRICS
# ==========================================

def compare_forecast_to_actual(forecasted, actual):
    """
    Compares forecasted vs actual and calculates accuracy metrics.
    """
    comparison_rows = []
    
    # Get all month-segment combinations
    all_months = set(forecasted.keys()) | set(actual.keys())
    all_segments = set()
    
    for month_data in list(forecasted.values()) + list(actual.values()):
        all_segments.update(month_data.keys())
    
    for month in sorted(all_months):
        for seg in sorted(all_segments):
            # Get forecasted values
            f_vol = forecasted.get(month, {}).get(seg, {}).get('forecasted_won_vol', 0)
            f_rev = forecasted.get(month, {}).get(seg, {}).get('forecasted_won_rev', 0)
            
            # Get actual values
            a_vol = actual.get(month, {}).get(seg, {}).get('actual_won_vol', 0)
            a_rev = actual.get(month, {}).get(seg, {}).get('actual_won_rev', 0)
            
            # Calculate errors
            vol_error = f_vol - a_vol
            rev_error = f_rev - a_rev
            
            vol_pct_error = (vol_error / a_vol * 100) if a_vol > 0 else 0
            rev_pct_error = (rev_error / a_rev * 100) if a_rev > 0 else 0
            
            comparison_rows.append({
                'month': month,
                'market_segment': seg,
                'forecasted_won_volume': f_vol,
                'actual_won_volume': a_vol,
                'volume_error': vol_error,
                'volume_pct_error': vol_pct_error,
                'forecasted_won_revenue': f_rev,
                'actual_won_revenue': a_rev,
                'revenue_error': rev_error,
                'revenue_pct_error': rev_pct_error
            })
    
    return pd.DataFrame(comparison_rows)


def calculate_accuracy_metrics(df_comparison):
    """
    Calculates summary accuracy statistics.
    """
    # Filter out rows with zero actuals (can't measure error)
    df_valid = df_comparison[df_comparison['actual_won_revenue'] > 0].copy()
    
    if len(df_valid) == 0:
        return {
            'mape_revenue': 0,
            'bias_revenue': 0,
            'rmse_revenue': 0,
            'hit_rate_within_20pct': 0
        }
    
    # MAPE (Mean Absolute Percentage Error)
    mape_rev = df_valid['revenue_pct_error'].abs().mean()
    
    # Bias (are we systematically over/under forecasting?)
    bias_rev = df_valid['revenue_pct_error'].mean()
    
    # RMSE (Root Mean Squared Error)
    rmse_rev = np.sqrt((df_valid['revenue_error'] ** 2).mean())
    
    # Hit Rate (% of forecasts within 20% of actual)
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
# VISUALIZATION
# ==========================================

def create_accuracy_chart(df_comparison):
    """
    Creates visual comparison of forecasted vs actual revenue.
    """
    df_plot = df_comparison.groupby('month').agg({
        'forecasted_won_revenue': 'sum',
        'actual_won_revenue': 'sum'
    }).reset_index()
    
    df_plot['month'] = pd.to_datetime(df_plot['month'])
    df_plot = df_plot.sort_values('month')
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df_plot['month'], df_plot['forecasted_won_revenue'], 
             marker='o', linewidth=2, label='Forecasted', color='#2E86AB')
    plt.plot(df_plot['month'], df_plot['actual_won_revenue'], 
             marker='s', linewidth=2, label='Actual', color='#A23B72')
    
    plt.title('Backtest Validation: Forecasted vs Actual Revenue', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_BACKTEST_CHART, dpi=150)
    print(f"    Chart saved: {OUTPUT_BACKTEST_CHART}")


# ==========================================
# MAIN BACKTEST EXECUTION
# ==========================================

def run_backtest_validation():
    """
    Main backtest validation workflow.
    """
    print("=" * 70)
    print("BACKTEST VALIDATION - FORECAST MODEL ACCURACY TEST")
    print("=" * 70)
    
    print("\n--- 1. Loading Historical Data ---")
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print("Error: Snapshot file not found.")
        return
    
    df_raw = pd.read_csv(PATH_RAW_SNAPSHOTS)
    df_raw['snapshot_date'] = pd.to_datetime(df_raw['snapshot_date'])
    
    # Split into train and test
    train_end = pd.to_datetime(TRAIN_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)
    
    df_train = df_raw[df_raw['snapshot_date'] <= train_end].copy()
    df_test = df_raw[df_raw['snapshot_date'] <= test_end].copy()
    
    print(f"    Training period: {df_train['snapshot_date'].min().date()} to {train_end.date()}")
    print(f"    Test period:     {test_start.date()} to {test_end.date()}")
    print(f"    Training records: {len(df_train):,}")
    
    print("\n--- 2. Calculating Drivers from Training Data ---")
    win_rates = calculate_win_rates_backtest(df_train, train_end)
    velocity = calculate_velocity_backtest(df_train, train_year=2024)  # Use 2024 as baseline
    lags = calculate_lags_backtest(df_train)
    
    print(f"    Win rates calculated for {len(win_rates)} segments")
    print(f"    Velocity patterns established")
    
    print("\n--- 3. Running Backtest Simulations ---")
    
    # Run multiple simulations and average
    all_forecasted = []
    
    for sim in range(BACKTEST_ASSUMPTIONS['num_simulations']):
        if (sim + 1) % 100 == 0:
            print(f"    Completed {sim + 1}/{BACKTEST_ASSUMPTIONS['num_simulations']} simulations...")
        
        forecasted = run_backtest_simulation(df_train, win_rates, velocity, lags, test_start, test_end)
        all_forecasted.append(forecasted)
    
    # Average across simulations
    avg_forecasted = defaultdict(lambda: defaultdict(lambda: {
        'forecasted_won_vol': 0,
        'forecasted_won_rev': 0
    }))
    
    for forecasted in all_forecasted:
        for month, seg_data in forecasted.items():
            for seg, metrics in seg_data.items():
                avg_forecasted[month][seg]['forecasted_won_vol'] += metrics['forecasted_won_vol']
                avg_forecasted[month][seg]['forecasted_won_rev'] += metrics['forecasted_won_rev']
    
    # Divide by num sims
    for month in avg_forecasted:
        for seg in avg_forecasted[month]:
            avg_forecasted[month][seg]['forecasted_won_vol'] /= BACKTEST_ASSUMPTIONS['num_simulations']
            avg_forecasted[month][seg]['forecasted_won_rev'] /= BACKTEST_ASSUMPTIONS['num_simulations']
    
    print("\n--- 4. Extracting Actual Results from Test Period ---")
    actual_results = extract_actual_results(df_test, test_start, test_end)
    
    print("\n--- 5. Comparing Forecast to Actual ---")
    df_comparison = compare_forecast_to_actual(avg_forecasted, actual_results)
    df_comparison.to_csv(OUTPUT_BACKTEST_RESULTS, index=False)
    
    print(f"    Comparison saved: {OUTPUT_BACKTEST_RESULTS}")
    
    print("\n--- 6. Calculating Accuracy Metrics ---")
    metrics = calculate_accuracy_metrics(df_comparison)
    
    # Print summary
    summary_text = f"""
{'=' * 70}
BACKTEST VALIDATION SUMMARY
{'=' * 70}

TEST PERIOD: {test_start.date()} to {test_end.date()}
TRAINING PERIOD: {df_train['snapshot_date'].min().date()} to {train_end.date()}

ACCURACY METRICS
----------------
Mean Absolute Percentage Error (MAPE):  {metrics['mape_revenue']:.2f}%
Forecast Bias:                          {metrics['bias_revenue']:.2f}%
Root Mean Squared Error (RMSE):         ${metrics['rmse_revenue']:,.0f}
Hit Rate (within ±20%):                 {metrics['hit_rate_within_20pct']:.1f}%

Periods Analyzed:                       {metrics['num_periods_analyzed']}

INTERPRETATION
--------------
"""
    
    if metrics['mape_revenue'] < 15:
        summary_text += "✓ EXCELLENT: Model demonstrates strong predictive accuracy (MAPE < 15%)\n"
    elif metrics['mape_revenue'] < 25:
        summary_text += "✓ GOOD: Model shows acceptable predictive accuracy (MAPE < 25%)\n"
    else:
        summary_text += "⚠ REVIEW NEEDED: Model accuracy is below target (MAPE > 25%)\n"
    
    if abs(metrics['bias_revenue']) < 5:
        summary_text += "✓ UNBIASED: No systematic over/under-forecasting detected\n"
    elif metrics['bias_revenue'] > 5:
        summary_text += "⚠ UPWARD BIAS: Model tends to over-forecast (systematically high)\n"
    else:
        summary_text += "⚠ DOWNWARD BIAS: Model tends to under-forecast (systematically low)\n"
    
    summary_text += f"\n"
    summary_text += f"RECOMMENDATION\n"
    summary_text += f"--------------\n"
    
    if metrics['mape_revenue'] < 20 and abs(metrics['bias_revenue']) < 10:
        summary_text += "Model is suitable for production forecasting. Proceed with confidence.\n"
    else:
        summary_text += "Model requires calibration. Review driver calculations and assumptions.\n"
    
    summary_text += f"\n{'=' * 70}\n"
    
    print(summary_text)
    
    with open(OUTPUT_BACKTEST_SUMMARY, 'w') as f:
        f.write(summary_text)
    
    print(f"Summary saved: {OUTPUT_BACKTEST_SUMMARY}")
    
    print("\n--- 7. Creating Visualization ---")
    create_accuracy_chart(df_comparison)
    
    print("\n" + "=" * 70)
    print("BACKTEST VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest_validation()