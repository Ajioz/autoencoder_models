import pandas as pd
import time
import os
from datetime import datetime

# Define start and end UNIX timestamps
period1 = 1672531200  # Jan 1, 2023
period2 = int(time.time())  # current time

# Yahoo Finance download URL for GE stock
url = (
    "https://query1.finance.yahoo.com/v7/finance/download/GE"
    f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
)

# Read data from the URL
try:
    df = pd.read_csv(url)
except Exception as e:
    print("❌ Failed to download CSV:", e)
    exit()

# Create a filename with timestamp for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"GE_history_{timestamp}.csv"

# Save the file in the current working directory
output_path = os.path.join(os.getcwd(), filename)
df.to_csv(output_path, index=False)

print(f"✅ Downloaded and saved: {filename} ({df.shape[0]} rows, {df.shape[1]} columns)")

