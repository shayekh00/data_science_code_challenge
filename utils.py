def check_missing(df, name):
    null_counts = df.isnull().sum()
    total_missing = null_counts.sum()
    if total_missing == 0:
        print(f"✅ {name}: No missing values.")
    else:
        print(f"❌ {name}: {total_missing} missing values found.")
        print(null_counts[null_counts > 0])


def create_forecasting_dataset(df, window_size=30, target_col='snow'):
    all_X, all_y, all_dates = [], [], []

    df = df.sort_values('date').reset_index(drop=True)

    feature_cols = [col for col in df.columns if col not in ['date', target_col]]

    for i in range(len(df) - window_size - 1):
        window = df.iloc[i:i + window_size]
        target_row = df.iloc[i + window_size]

        # Avoid using target_col in input
        X = window[feature_cols].values
        y = int(target_row[target_col])

        all_X.append(X)
        all_y.append(y)
        all_dates.append(target_row['date'])

    return np.array(all_X), np.array(all_y), all_dates
