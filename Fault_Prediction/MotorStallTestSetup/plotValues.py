import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv

filename = 'data/motor_data.csv'

x_data, y_data = [], []

with open(filename, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header

    for row in reader:
        try:
            t = float(row[0])      # time (µs or ms)
            current = float(row[1]) * 1000  # convert to mA

            x_data.append(t)
            y_data.append(current)

        except:
            pass

# --- Normalize time (start from 0, convert to ms if needed) ---
x_data = np.array(x_data)
x_data = (x_data - x_data[0]) / 1000.0  # µs → ms

y_data = np.array(y_data)

# --- Downsample (VERY IMPORTANT for readability) ---
DOWNSAMPLE = 10   # change to 5–50 depending on density
x_data = x_data[::DOWNSAMPLE]
y_data = y_data[::DOWNSAMPLE]

# --- smooth signal (moving average) ---
WINDOW = 5
y_smooth = np.convolve(y_data, np.ones(WINDOW)/WINDOW, mode='same')

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x_data, y_smooth, linewidth=1)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Current (mA)")
ax.set_title("Motor Current vs Time")

ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
ax.yaxis.set_major_locator(ticker.MaxNLocator(8))

ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plots/plot.png')
plt.show()
