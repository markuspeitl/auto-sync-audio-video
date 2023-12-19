
import math
import os
import re
from typing import Any, Callable
import numpy as np
from scipy import ndimage
from ffmpeg_processing import read_audio_ffmpeg
from path_util import get_extracted_audio_path


video_matcher_regex = re.compile(".+\.[mp4|mov|mkv|avi|webm|m4a]")
audio_file_cache: dict[str, Any] = {}


def convert_dtype(data: np.ndarray, dtype=np.float32):
    if (data.dtype != dtype):
        return np.asarray(data, dtype=dtype)
    return data


"""def normalize_to_float(data: np.ndarray, dtype=np.float32):

    # max_numpy_array_int_value = np.iinfo(data.dtype).max - np.iinfo(data.dtype).min
    # print(max_numpy_array_int_value)

    float_data = convert_dtype(data, dtype)

    max_data_value = np.max(float_data)
    min_data_value = np.min(float_data)
    # scale symmetrical, by which max has less headroom

    # return (float_data - min_data_value) / (max_data_value - min_data_value)
"""


def normalize_per_channel(normalize_function: Callable, data: np.ndarray, dtype=np.float32):
    if (len(data.shape) > 1):
        for audio_channel_index in range(0, data.shape):
            data[audio_channel_index] = normalize_function(data[audio_channel_index], dtype=dtype)

    return normalize_function(data, dtype=dtype)


def normalize_to_interval(data: np.ndarray, interval: tuple[float] = (0, 1), dtype=np.float32):
    float_data = convert_dtype(data, dtype)

    max_data_value = np.max(float_data)
    min_data_value = np.min(float_data)

    interval_size = interval[1] - interval[0]
    zero_to_one_scaled = (float_data - min_data_value) / (max_data_value - min_data_value)

    if (interval == (0, 1)):
        return zero_to_one_scaled

    # Example (-1, 3) -> size = 4 -> scale to (0,4) and shift -1
    # 0.0 -> -1.0
    # 1.0 -> 3.0
    # 0.5 -> 1

    return (zero_to_one_scaled * interval_size) - interval[0]


def normalize_symmetrical_float(data: np.ndarray, dtype=np.float32):
    # max_numpy_array_int_value = np.iinfo(data.dtype).max - np.iinfo(data.dtype).min
    # print(max_numpy_array_int_value)

    float_data = convert_dtype(data, dtype)

    max_data_value = np.max(float_data)
    min_data_value = np.min(float_data)
    min_data_value_abs = np.abs(np.min(float_data))

    biggest_max_value = max(max_data_value, min_data_value_abs)
    return float_data / biggest_max_value

    # scale symmetrical, by which max has less headroom

    # return ((float_data - min_data_value) / (max_data_value - min_data_value) - 0.5) * 2


def downsample_audio(audio_np_array: np.ndarray, downsample_block_size: int, sample_rate: float):

    total_sample_count = audio_np_array.shape[0]

    max_down_blocks = math.ceil(total_sample_count / downsample_block_size)

    downsampled_audio = None

    if (len(audio_np_array.shape) > 1):
        downsampled_audio = np.zeros((max_down_blocks, audio_np_array.shape[1]), dtype=audio_np_array.dtype)
    else:
        downsampled_audio = np.zeros(max_down_blocks, dtype=audio_np_array.dtype)

    for block_index in range(0, max_down_blocks):
        start_index = int(block_index * downsample_block_size)
        end_index = int((block_index + 1) * downsample_block_size)

        if (end_index >= total_sample_count):
            end_index = total_sample_count - 1

        if (len(audio_np_array.shape) > 1):
            # mean_value = np.mean(audio_np_array[start_index:end_index], axis=1)
            mean_value = np.sum(audio_np_array[start_index:end_index], axis=0) / downsample_block_size
        else:
            # mean_value = np.mean(audio_np_array[start_index:end_index], axis=1)
            mean_value = np.sum(audio_np_array[start_index:end_index]) / downsample_block_size

        # print(mean_value)

        downsampled_audio[block_index] = mean_value

    return downsampled_audio, float(sample_rate / downsample_block_size)

    # return block_reduce(audio_np_array, downsample_block_size, np.mean), int(sample_rate / downsample_block_size)


def apply_audio_differentiation(audio_sample_data):

    float_wav_array = normalize_symmetrical_float(audio_sample_data)
    # We are focused on the speed of the change for detecting a short impulse
    # audio_amplitude_sobel = normalize_to_float(ndimage.sobel(float_wav_array, axis=0))
    audio_amplitude_sobel = ndimage.sobel(float_wav_array, axis=0)
    return audio_amplitude_sobel


def read_audio_of_file(file_path):
    global audio_file_cache

    # audio_file_path = None

    # Cache loaded audio files
    if (file_path in audio_file_cache):
        return audio_file_cache[file_path][0], audio_file_cache[file_path][1]

    sample_rate, audio_data = read_audio_ffmpeg(file_path)
    audio_file_cache[file_path] = (sample_rate, audio_data)
    return sample_rate, audio_data

    """if (file_path.endswith('.wav')):
        audio_file_path = file_path

    elif (video_matcher_regex.match(file_path)):

        read_audio_from_video

        audio_file_path = extract_audio_from_video(file_path)

    # Cache loaded audio files
    if (audio_file_path in audio_file_cache):
        return audio_file_cache[audio_file_path][0], audio_file_cache[audio_file_path][1]

    sample_rate, audio_data = wavfile.read(audio_file_path)

    audio_file_cache[audio_file_path] = (sample_rate, audio_data)

    return sample_rate, audio_data"""


def read_audio_of_files(file_paths):

    audio_data_list = []
    sample_rates_list = []

    for file_path in file_paths:
        sample_rate, audio_data = read_audio_of_file(file_path)

        audio_data_list.append(audio_data)
        sample_rates_list.append(sample_rate)

    return audio_data_list, sample_rates_list


def delete_video_extracted_audio(video_file_path: str):

    if (not video_file_path or not os.path.exists(video_file_path)):
        return

    audio_file_path = get_extracted_audio_path(video_file_path)
    if (not audio_file_path or not os.path.exists(audio_file_path)):
        return

    os.unlink(audio_file_path)


def resample_audio(target_sample_width_sec, audio_data, sample_rate):
    block_size_samples = target_sample_width_sec * sample_rate

    downsampled_audio, downsampled_rate = downsample_audio(audio_data, block_size_samples, sample_rate)

    norm_float_audio = normalize_symmetrical_float(downsampled_audio)

    return norm_float_audio, downsampled_rate
