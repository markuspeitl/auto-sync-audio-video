
import math
import re
from typing import Any
import numpy as np
from scipy import ndimage
from scipy.io import wavfile

from ffmpeg_processing import extract_audio_from_video


video_matcher_regex = re.compile(".+\.[mp4|mov|mkv|avi|webm|m4a]")
audio_file_cache: dict[str, Any] = {}


def normalize_to_float(data):

    # max_numpy_array_int_value = np.iinfo(data.dtype).max - np.iinfo(data.dtype).min
    # print(max_numpy_array_int_value)
    max_data_value = np.max(data)
    min_data_value = np.min(data)

    return np.asarray(((data - min_data_value) / (max_data_value - min_data_value)), dtype=np.float64)


def downsample_audio(audio_np_array, downsample_block_size, sample_rate):

    total_sample_count = audio_np_array.shape[0]

    max_down_blocks = math.ceil(total_sample_count / downsample_block_size)

    downsampled_audio = np.zeros((max_down_blocks, audio_np_array.shape[1]))

    for block_index in range(0, max_down_blocks):
        start_index = int(block_index * downsample_block_size)
        end_index = int((block_index + 1) * downsample_block_size)

        if (end_index >= total_sample_count):
            end_index = total_sample_count - 1

        # mean_value = np.mean(audio_np_array[start_index:end_index], axis=1)
        mean_value = np.sum(audio_np_array[start_index:end_index], axis=0) / downsample_block_size

        # print(mean_value)

        downsampled_audio[block_index] = mean_value

    return downsampled_audio, float(sample_rate / downsample_block_size)

    # return block_reduce(audio_np_array, downsample_block_size, np.mean), int(sample_rate / downsample_block_size)


def apply_audio_differentiation(audio_sample_data):

    float_wav_array = normalize_to_float(audio_sample_data)
    # We are focused on the speed of the change for detecting a short impulse
    # audio_amplitude_sobel = normalize_to_float(ndimage.sobel(float_wav_array, axis=0))
    audio_amplitude_sobel = ndimage.sobel(float_wav_array, axis=0)
    return audio_amplitude_sobel


def read_audio_of_file(file_path):
    global audio_file_cache

    audio_file_path = None
    if (file_path.endswith('.wav')):
        audio_file_path = file_path

    elif (video_matcher_regex.match(file_path)):
        audio_file_path = extract_audio_from_video(file_path)

    # Cache loaded audio files
    if (audio_file_path in audio_file_cache):
        return audio_file_cache[audio_file_path][0], audio_file_cache[audio_file_path][1]

    sample_rate, audio_data = wavfile.read(audio_file_path)

    audio_file_cache[audio_file_path] = (sample_rate, audio_data)

    return sample_rate, audio_data


def read_audio_of_files(file_paths):

    audio_data_list = []
    sample_rates_list = []

    for file_path in file_paths:
        sample_rate, audio_data = read_audio_of_file(file_path)

        audio_data_list.append(audio_data)
        sample_rates_list.append(sample_rate)

    return audio_data_list, sample_rates_list


def resample_audio(target_sample_width_sec, audio_data, sample_rate):
    block_size_samples = target_sample_width_sec * sample_rate

    downsampled_audio, downsampled_rate = downsample_audio(audio_data, block_size_samples, sample_rate)

    norm_float_audio = normalize_to_float(downsampled_audio)

    return norm_float_audio, downsampled_rate
