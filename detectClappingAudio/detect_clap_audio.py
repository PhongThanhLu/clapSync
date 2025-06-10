import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.audio.io.AudioFileClip import AudioFileClip
from scipy.signal import find_peaks
import os

def extract_audio_from_video(video_path, audio_output="audio.wav"):
    """Extracts audio from a video file."""
    audio_clip = AudioFileClip(video_path)
    audio_clip.write_audiofile(audio_output, codec="pcm_s16le")
    return audio_output

def detect_first_loud_change(audio_path, sr=22050, window_size=2048, step_size=512, threshold=2.0):
    """
    Detect the first sudden loud sound that significantly differs from the previous average.
    
    - window_size: length of each analysis window.
    - step_size: how much to slide forward each time.
    - threshold: how many times louder the amplitude should be than the rolling average.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    y = y / np.max(np.abs(y))  # Normalize

    num_windows = (len(y) - window_size) // step_size
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window = y[start:end]
        mean_before = np.mean(np.abs(y[max(0, start - window_size):start])) + 1e-6  # Avoid division by 0
        current_mean = np.mean(np.abs(window))

        # If sudden jump compared to recent background
        if current_mean > threshold * mean_before:
            time_of_event = start / sr
            print(f"Detected sudden loud change at: {time_of_event:.4f} seconds")
            return time_of_event

    return None


def visualize_audio_with_first_clap(audio_path, first_clap_time, offset=1):
    """Plots waveform and highlights the first detected clap."""
    y, sr = librosa.load(audio_path, sr=22050, offset=offset)  # Load entire audio starting from 1 second
    time = np.linspace(offset, len(y) / sr + offset, num=len(y))  # Adjust time axis based on offset

    plt.figure(figsize=(12, 4))
    plt.plot(time, y, label="Audio Waveform")

    # Highlight the first detected clap
    if first_clap_time is not None:
        plt.axvline(x=first_clap_time, color='r', linestyle='--', label=f"First Clap at {first_clap_time:.2f} s")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"Clap Detection in Audio File")
    plt.legend()
    plt.show()

# Usage

video_path = "/Users/levent/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/clips/30.MP4"
audio_path = extract_audio_from_video(video_path)

# Detect the first clap in the audio, starting from 1 second
first_loud_time = detect_first_loud_change(audio_path)

# Visualize the audio with the first clap marked
visualize_audio_with_first_clap(audio_path, first_loud_time)