from scipy import ndimage, datasets
# import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join
from scipy.io import wavfile
import scipy.io
from skimage.measure import block_reduce

sample_rate, original_data = wavfile.read("230412_0072S12.wav")
print(f"number of channels = {original_data.shape[1]}")
length = original_data.shape[0] / sample_rate
print(f"samplerate = {sample_rate}")
print(f"length = {length}s")
print(original_data.dtype)


def msec_to_samples(msec, sample_rate):
    return int((msec / 1000) * sample_rate)


msec_precision_digits = 3


def sample_index_to_time(sample_index, sample_rate):
    time_per_sample_frame_sec = 1 / sample_rate

    time_at_sample_sec = time_per_sample_frame_sec * sample_index

    time_at_sample_sec = round(time_at_sample_sec, 3)

    print(time_at_sample_sec)

    msec = round((time_at_sample_sec * pow(10, msec_precision_digits)) % pow(10, msec_precision_digits))
    sec = int(time_at_sample_sec % 60)
    min = int(time_at_sample_sec / 60)

    return f"{min:0>2}:{sec:0>2}." + str(msec).zfill(msec_precision_digits)


def normalize_to_float(data):

    # max_numpy_array_int_value = np.iinfo(data.dtype).max - np.iinfo(data.dtype).min
    # print(max_numpy_array_int_value)
    max_data_value = np.max(data)
    return np.asarray((data / max_data_value), dtype=np.float64)


def downsample_audio(downsample_block_size, sample_rate):
    return block_reduce(original_data, block_size=downsample_block_size, func=np.mean, cval=np.mean(original_data)), int(sample_rate / downsample_block_size)


def extract_clap_sample_indices(audio_wav_data, sample_rate, detection_threshold: float, detect_release_time_msec=100):
    float_wav_array = normalize_to_float(audio_wav_data)

    # We are focused on the speed of the change for detecting a short impulse
    audio_amplitude_sobel = normalize_to_float(ndimage.sobel(float_wav_array, axis=0))

    # audio_sample_bit_depth: int = 32
    # audio_sample_bit_depth: int = np.iinfo(original_data.dtype).max
    # threshold_float = 0.4
    # sample_threshold_int: int = int(pow(2, audio_sample_bit_depth) * threshold_float)
    # print(f"sample_threshold_int = {sample_threshold_int}")

    # selected_key_samples_indices, _ = np.where(audio_wav_data >= threshold_float)
    # print(len(selected_key_samples_indices))
    # print(selected_key_samples_indices)

    selected_key_samples_indices, _ = np.where(audio_amplitude_sobel >= detection_threshold)

    def filter_close_samples(sample_indices, window_size_samples):

        filtered_indices = []

        last_sample_index = - window_size_samples
        for sample_index in sample_indices:
            if (last_sample_index + window_size_samples <= sample_index):
                filtered_indices.append(sample_index)
                last_sample_index = sample_index

        return filtered_indices

    print(len(selected_key_samples_indices))
    print(selected_key_samples_indices)

    filter_window_size_samples = msec_to_samples(detect_release_time_msec, sample_rate)

    selected_key_samples_indices = filter_close_samples(selected_key_samples_indices, filter_window_size_samples)

    detected_timestamps: list[str] = list(map(lambda detected_sample_index: sample_index_to_time(detected_sample_index, sample_rate), selected_key_samples_indices))

    print(detected_timestamps)
    print(selected_key_samples_indices)

    return detected_timestamps


def extract_clap_timestamps(audio_wav_data, sample_rate, detection_threshold: float, detect_release_time_msec=100):

    selected_key_samples_indices = extract_clap_sample_indices(audio_wav_data, sample_rate, detection_threshold, detect_release_time_msec)

    detected_timestamps: list[str] = list(map(lambda detected_sample_index: sample_index_to_time(detected_sample_index, sample_rate), selected_key_samples_indices))

    print(detected_timestamps)
    print(selected_key_samples_indices)

    return detected_timestamps


clap_sound_attack_time_msec = 1
# clap_sound_attack_time_samples = msec_to_samples(3, sample_rate)

clap_sound_attack_time_samples = 4
downsampled_audio, downsampled_sample_rate = downsample_audio(clap_sound_attack_time_samples, sample_rate)

detection_threshold = 0.2

print("Downsampled")
detected_timestamps: list[str] = extract_clap_timestamps(downsampled_audio, downsampled_sample_rate, detection_threshold)
print(detected_timestamps)

print("Orig sampled")
detected_timestamps: list[str] = extract_clap_timestamps(original_data, sample_rate, detection_threshold)
print(detected_timestamps)


# All 14.7, 16.6, 18.3, 20.8, 21.4, 22.06, 24.1, (28.470)
