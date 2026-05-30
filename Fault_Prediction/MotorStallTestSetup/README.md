# Motor Stall Test Setup

A simple test setup to measure current draw of a 5V DC motor using a low-side shunt resistor. Data is sampled at ~2 kHz and streamed over serial for logging and visualization.

---

## Required Components

- Arduino (Uno / compatible)
- Shunt Resistor (≈ 0.1Ω – 1Ω recommended)
- 5V DC Motor
- Flyback Diode (1N4007 or similar)
- Jumper Wires
- External 5V Power Supply
- USB Cable (for Arduino)

---

## Circuit Overview

![Circuit Schematic](/Fault_Prediction/MotorStallTestSetup/circuit_layout.png)

**Key Notes:**

- The shunt resistor is placed on the **low side** (between motor and GND)
- Arduino GND **must** be shared with the motor power supply GND
- The flyback diode **must** be wired in parallel with the motor and the stripe always faces the positive side of the power supply
- The stripe from the flyback diode represents the arrow that shows which way the current is allowed to flow through
- The ADC pin measures the voltage across the shunt resistor

---

## Setup Instructions

### 1. Wire the Circuit

- Double-check all connections before powering on
- Ensure the power supply is set to **≤ 5V** before connecting

---

### 2. Upload Arduino Code

Upload `motorStallTestSetup.ino` using Arduino IDE

**Troubleshooting Upload Errors:**

- If you see any upload errors (e.g., Exit 74):
  - Double-press the RESET button to enter bootloader mode
  - Retry upload
- Close any Serial Monitor before uploading

---

### 3. Configure Data Logging Script

Open `saveValues.py`

- Update the COM port (line 6) to match your Arduino
  - Check Device Manager (Windows) or `/dev/tty*` (Linux/Mac)
- Ensure no other program is using the serial port

---

### 4. Run the Experiment

- Start the Python script to begin logging
- Data will be saved to: `data/`

---

## Output

- CSV file with timestamped current measurements
- Real-time plot (if enabled in script)

Note: Data is transmitted in batches of 500 samples. During transmission over UART, sampling is temporarily paused, resulting in gaps in the recorded timestamps between batches. These gaps appear as sudden jumps in time in the plotted data.

---

## Safety Notes

- Do **not** exceed the motor’s rated voltage
- Be careful when stalling the motor — it can draw high current
- Avoid overheating the shunt resistor

---
