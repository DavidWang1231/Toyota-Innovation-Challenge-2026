import serial
import matplotlib.pyplot as plt
import csv

filename = 'data/motor_data.csv'
ser = serial.Serial('COM10', 115200)
ser.reset_input_buffer()

times = []  
currents = []

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time_us', 'current_A'])

    plt.ion()
    fig, ax = plt.subplots()

    while True:
        line = ser.readline().decode(errors='ignore').strip()

        if line == "END":
            # Plot after full burst
            ax.clear()
            ax.plot(times, currents)
            ax.set_xlabel("Time (µs)")
            ax.set_ylabel("Current (A)")
            ax.set_title("High-Speed Motor Current")

            plt.pause(0.01)

            print("Batch plotted. Samples:", len(times))
            continue

        if line:
            try:
                t, current = line.split(',')
                t = int(t)
                current = float(current)

                times.append(t)
                currents.append(current)

                writer.writerow([t, current])

            except Exception as e:
                print("Parse error:", line)