from typing import Any
import numpy as np
from audio_plotting import show_waveforms
from common_audio_processing import apply_audio_differentiation, normalize_symmetrical_float, read_audio_of_files
from time_conversion_util import sample_index_to_timestamp, sec_to_sample_index

# Release delay/time of impulse detection


def filter_close_samples(sample_indices, window_size_samples):

    filtered_indices = []

    last_sample_index = - window_size_samples
    for sample_index in sample_indices:
        if (last_sample_index + window_size_samples <= sample_index):
            filtered_indices.append(sample_index)
            last_sample_index = sample_index

    return filtered_indices


def extract_impulse_sample_indices(audio_data: np.ndarray, sample_rate: float, impulse_ramp_threshold: float = 0.4, detect_release_threshold_sec: float = 0.1, show_plots: bool = False):

    audio_channel = audio_data
    if (audio_data.shape[1] > 1):
        audio_channel = audio_data[:, 0]

    # We are focused on the speed of the change for detecting a short impulse
    audio_amplitude_sobel = apply_audio_differentiation(audio_channel)
    audio_amplitude_sobel = normalize_symmetrical_float(audio_amplitude_sobel)

    # audio_sample_bit_depth: int = 32
    # audio_sample_bit_depth: int = np.iinfo(original_data.dtype).max
    # threshold_float = 0.4
    # sample_threshold_int: int = int(pow(2, audio_sample_bit_depth) * threshold_float)
    # print(f"sample_threshold_int = {sample_threshold_int}")

    # selected_key_samples_indices, _ = np.where(audio_wav_data >= threshold_float)
    # print(len(selected_key_samples_indices))
    # print(selected_key_samples_indices)

    selected_key_samples_indices = np.where(audio_amplitude_sobel >= impulse_ramp_threshold)[0]

    # print(len(selected_key_samples_indices))
    # print(selected_key_samples_indices)

    filter_window_sample_indices_width = sec_to_sample_index(detect_release_threshold_sec, sample_rate)
    selected_key_samples_indices = filter_close_samples(selected_key_samples_indices, filter_window_sample_indices_width)

    if (show_plots):

        normalized_audio_data = normalize_symmetrical_float(audio_channel)
        show_waveforms(normalized_audio_data, sample_rate, marker_indices=selected_key_samples_indices, fill_down=False, block=False)
        # plotting_audio = np.concatenate(downsampled_audio, downsampled_audio_fine)
        show_waveforms(audio_amplitude_sobel, sample_rate, marker_indices=selected_key_samples_indices, fill_down=False, block=True)
        # show_waveforms(downsampled_audio_fine, downsampled_rate_fine, marker_indices=fine_rough_song_range, marked_ranges=[final_song_range], fill_down=True, block=False)

    # detected_timestamps: list[str] = list(map(lambda detected_sample_index: sample_index_to_time(detected_sample_index, sample_rate), selected_key_samples_indices))

    # print(detected_timestamps)
    # print(selected_key_samples_indices)

    return selected_key_samples_indices


def extract_impulse_timestamps(audio_wav_data, sample_rate, impulse_ramp_threshold: float = 0.4, detect_release_threshold_sec: float = 0.1, show_plots: bool = False):

    selected_key_samples_indices = extract_impulse_sample_indices(audio_wav_data, sample_rate, impulse_ramp_threshold, detect_release_threshold_sec, show_plots)

    detected_timestamps: list[str] = list(map(lambda detected_sample_index: sample_index_to_timestamp(detected_sample_index, sample_rate), selected_key_samples_indices))

    # print(detected_timestamps)
    # print(selected_key_samples_indices)

    return detected_timestamps


def extract_impulse_indices_of_files(file_paths, impulse_ramp_threshold: float = 0.4, detect_release_threshold_sec: float = 0.1, show_plots: bool = False):

    audio_data_list, sample_rates_list = read_audio_of_files(file_paths)

    impulses_per_file_list = []

    for audio_data, sample_rate in zip(audio_data_list, sample_rates_list):
        detected_impulse_sample_indices = extract_impulse_sample_indices(audio_data, sample_rate, impulse_ramp_threshold, detect_release_threshold_sec, show_plots)
        impulses_per_file_list.append(detected_impulse_sample_indices)
        sample_rates_list.append(sample_rate)

    return impulses_per_file_list, sample_rates_list
