import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if 'datetime' not in df.columns:
        if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        else:
            raise ValueError("No 'datetime' column and no date columns to construct it.")
    return df

def plot_pm25_trend(df: pd.DataFrame):
    df['Date'] = pd.to_datetime(df['Date'])
    daily_avg = df.groupby(df['Date'].dt.date)['Close'].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(daily_avg.index, daily_avg.values)
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title("Daily Average Close")
    plt.tight_layout()
    plt.savefig("eda_pm25_trend.pdf")
    plt.close()

def plot_correlation(df: pd.DataFrame):
    # Select only numeric columns for correlation
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.pdf")
    plt.close()

def plot_histogram_pm25(df: pd.DataFrame):
    df['Date'] = pd.to_datetime(df['Date'])
    daily_avg = df.groupby(df['Date'].dt.date)['Close'].mean()
    plt.figure(figsize=(8, 5))
    plt.hist(daily_avg.values, bins=30, edgecolor='black')
    plt.xlabel("Daily Average Close")
    plt.ylabel("Frequency")
    plt.title("Distribution of Daily Average Close")
    plt.tight_layout()
    plt.savefig("eda_pm25_histogram.pdf")
    plt.close()

if __name__ == '__main__':
    df = pd.read_csv("air_quality_cleaned.csv")
    print("Columns in air_quality_cleaned.csv:", df.columns.tolist())
    plot_pm25_trend(df)
    plot_correlation(df)
    plot_histogram_pm25(df)