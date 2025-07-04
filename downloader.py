import yfinance as yf
import pandas as pd

# Define ticker and date range
ticker = "GE"
start_date = "2023-01-01"
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

# Download historical daily data
df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
df.to_csv("GE_history.csv")
print("✅ Data downloaded and saved as GE_history.csv:", df.shape)
