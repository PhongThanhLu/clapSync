import os
from avro.datafile import DataFileReader
from avro.io import DatumReader
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json

# Path to the folder containing Avro files
folder_path = "/Users/levent/Documents/Research_Yu_Sun/2024-05-21/21"

# Target clapping time
clap_time_str = "10:58"
clap_time_obj = datetime.strptime(clap_time_str, "%H:%M")
clap_time_seconds = clap_time_obj.hour * 3600 + clap_time_obj.minute * 60

# Time window
time_window = 2
min_event_gap = 0.3

# List of .avro files
file_list = [f for f in os.listdir(folder_path) if f.endswith('.avro')]

found = False  # Flag to break from both loops

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    print(f"Processing file: {file_path}")

    try:
        reader = DataFileReader(open(file_path, "rb"), DatumReader())
        schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))
        data = [datum for datum in reader]
        reader.close()

        acc = data[-1]["rawData"]["accelerometer"]

        delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
        delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]

        x_g = [val * delta_physical / delta_digital for val in acc["x"]]
        y_g = [val * delta_physical / delta_digital for val in acc["y"]]
        z_g = [val * delta_physical / delta_digital for val in acc["z"]]

        magnitude = np.sqrt(np.array(x_g)**2 + np.array(y_g)**2 + np.array(z_g)**2)
        normalized_magnitude = 2 * ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())) - 1

        start_seconds = acc["timestampStart"] / 1_000_000
        time_seconds = [t / acc["samplingFrequency"] for t in range(len(x_g))]
        time_unix = [t + start_seconds for t in time_seconds]
        datetime_time = [datetime.fromtimestamp(t) for t in time_unix]

        filtered_indices = [
            i for i, t in enumerate(time_unix)
            if clap_time_seconds - time_window <= datetime.fromtimestamp(t).hour * 3600 + datetime.fromtimestamp(t).minute * 60 <= clap_time_seconds + time_window
        ]

        if not filtered_indices:
            print(f"No data points found in window for {file_name}")
            continue

        filtered_magnitude = [normalized_magnitude[i] for i in filtered_indices]
       # Define a threshold factor relative to the global max
        threshold_factor = -0.2 

# Set threshold as a fraction of the maximum
        acc_threshold = max(filtered_magnitude) * threshold_factor

        filtered_time = [datetime_time[i] for i in filtered_indices]

        detected_indices = [
            i for i, val in enumerate(filtered_magnitude) if val > acc_threshold
        ]

        final_detected_indices = []
        last_detected_time = None
        for i in detected_indices:
            if last_detected_time is None or (filtered_time[i] - last_detected_time).total_seconds() > min_event_gap:
                final_detected_indices.append(i)
                last_detected_time = filtered_time[i]
                # STOP right here on first valid detection
                detected_time = filtered_time[i]
                detected_acceleration = filtered_magnitude[i]
                print(f"First clapping event at: {detected_time}, Magnitude: {detected_acceleration:.2f}")

                # Plot the result
                plt.figure(figsize=(12, 6))
                plt.plot(filtered_time, filtered_magnitude, label="Normalized Magnitude", color='blue')
                plt.scatter(detected_time, detected_acceleration, color="red", s=100, label="First Clapping")
                plt.axvline(detected_time, color="red", linestyle="--")
                plt.title(f"First Clapping Event in {file_name}")
                plt.xlabel("Time")
                plt.ylabel("Normalized Acceleration")
                plt.grid(True)
                plt.legend()
                plt.xticks(rotation=45)
                plt.show()

                found = True
                break  # Exit the detection loop

        if found:
            break  # Exit the file loop

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
