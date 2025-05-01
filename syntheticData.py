import numpy as np
import matplotlib.pyplot as plt


def generate_accelerometer_data(num_samples, seq_length, frequency=10):
    """
    Creates synthetic accelerometer data with three sine waves.
    
    Args:
        num_samples: Number of samples to generate
        seq_length: Length of each time series
        frequency: Frequency of the sine waves
    
    Returns:
        NumPy array of shape (num_samples, seq_length, 3)
    """
    data = []
    for _ in range(num_samples):
        t = np.linspace(0, 2 * np.pi, seq_length)
        # Generate sine waves for 3 axes (x, y, z) with phase differences
        x = np.sin(frequency * t)
        y = np.sin(frequency * t + np.pi / 4)
        z = np.sin(frequency * t + np.pi / 2)
        signal = np.stack([x, y, z], axis=-1)  # shape: (seq_length, 3)
        
        data.append(signal)
            
    return np.array(data)

def plot_accelerometer_signal(signal, title="Synthetic Accelerometer Signal"):
    """
    Plot the three acceleration components over time.
    
    Args:
        signal: NumPy array of shape (seq_length, 3) containing X, Y, Z accelerations
        title: Title for the plot
    """
    seq_length = signal.shape[0]
    t = np.linspace(0, 1, seq_length)  # Normalized time
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].set_title(title)
    for i, axis in enumerate(["X", "Y", "Z"]):
        axes[i].plot(t, signal[:, i])
        axes[i].set_ylabel(f"{axis}-axis")
    axes[-1].set_xlabel("Normalized Time")
    plt.tight_layout()
    plt.show()


data = generate_accelerometer_data(3, 200, 5)
f = open("accelerometerData.txt", "w")
f.write(str(data))
print(data)
plot_accelerometer_signal(data[0], title="Sample 1: Synthetic Accelerometer Signal")


