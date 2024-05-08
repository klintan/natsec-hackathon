import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
FS = 44100  # Sampling frequency
DURATION = 5  # Duration of recording in seconds
MIC_POSITIONS = np.array(
    [
        [0, 0],  # Microphone 1 position (x1, y1)
        [1, 0],  # Microphone 2 position (x2, y2)
        [1, 1],  # Microphone 3 position (x3, y3)
        [0, 1],  # Microphone 4 position (x4, y4)
    ]
)  # Coordinates in meters
SPEED_OF_SOUND = 343  # Speed of sound in air (m/s)


def record_audio():
    """Records audio from multiple microphones simultaneously."""
    device_info = sd.query_devices(None, "input")
    num_channels = min(device_info["max_input_channels"], len(MIC_POSITIONS))
    print(f"Recording from {num_channels} channels.")

    return sd.rec(
        int(FS * DURATION), samplerate=FS, channels=num_channels, dtype="float32"
    )


def detect_leading_edge(signals):
    """Detects the leading edge in each microphone's signal."""
    peaks_indices = [
        find_peaks(signal, height=0.2)[0] for signal in signals.T
    ]  # Adjust height as necessary
    first_peaks = [
        indices[0] if len(indices) > 0 else None for indices in peaks_indices
    ]
    return np.array(first_peaks)


def multilateration(tdoas, mic_positions, speed_of_sound):
    def loss(x):
        estimated_distances = np.sqrt(((mic_positions - x) ** 2).sum(axis=1))
        estimated_tdoas = estimated_distances / speed_of_sound
        return np.sum((estimated_tdoas - tdoas) ** 2)

    initial_guess = mic_positions.mean(axis=0)
    result = minimize(loss, initial_guess)
    return result.x


def visualize_positions(mic_positions, source_position):
    plt.figure(figsize=(6, 6))
    plt.scatter(
        mic_positions[:, 0],
        mic_positions[:, 1],
        color="red",
        s=100,
        label="Microphones",
    )
    plt.scatter(
        source_position[0],
        source_position[1],
        color="blue",
        s=100,
        label="Sound Source",
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel("X coordinate (meters)")
    plt.ylabel("Y coordinate (meters)")
    plt.title("Microphone and Sound Source Positions")
    plt.axis("equal")
    plt.show()


def main():
    print("Starting recording...")
    audio_data = record_audio()
    sd.wait()
    print("Recording finished. Processing data...")

    leading_edges = detect_leading_edge(audio_data)
    tdoas = leading_edges / FS

    position = multilateration(tdoas)
    print(f"Estimated source position: {position}")

    visualize_positions(MIC_POSITIONS, position)


if __name__ == "__main__":
    main()
