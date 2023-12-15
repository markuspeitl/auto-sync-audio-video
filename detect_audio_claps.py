
import argparse
import shutil
import math
import re
from pathlib import PurePath
import os
import subprocess
import sys
import time
from typing import Any
from matplotlib import ticker
from scipy import ndimage, datasets
# import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join
from scipy.io import wavfile
import scipy.io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from apply_reaper_fx_chain import apply_audio_processing_chain
from path_util import add_name_suffix_path, get_extracted_audio_path

from transcribe_audio import transcribe_audio_file

# sample_rate, original_data = wavfile.read("230412_0072S12.wav")


def extract_audio_from_video(video_path):

    extracted_audio_path = get_extracted_audio_path(video_path)

    if (not os.path.exists(extracted_audio_path)):
        os.system(f"ffmpeg -i {video_path} -n -vn -acodec pcm_s24le {extracted_audio_path}")

    return extracted_audio_path


# print(f"number of channels = {original_data.shape[1]}")
# length = original_data.shape[0] / sample_rate
# print(f"samplerate = {sample_rate}")
# print(f"length = {length}s")
# print(original_data.dtype)


def msec_to_samples(msec, sample_rate):
    return int((msec / 1000) * sample_rate)


def sec_to_samples(sec, sample_rate):
    return int((sec) * sample_rate)


def samples_to_msec(sample_index, sample_rate):
    return int(samples_to_sec(sample_index, sample_rate) * 1000)


def samples_to_sec(sample_index, sample_rate):
    return sample_index / sample_rate


msec_precision_digits = 3


def msec_to_timestamp(sample_msec_time):
    return sec_to_timestamp(sample_msec_time / 1000)


def sec_to_timestamp(sample_sec_time):

    number_sign = ''
    if (sample_sec_time < 0):
        number_sign = '-'

    sample_sec_time = abs(sample_sec_time)

    sample_sec_time = round(sample_sec_time, 3)

    msec = round((sample_sec_time * pow(10, msec_precision_digits)) % pow(10, msec_precision_digits))
    sec = int(sample_sec_time) % 60
    min = int(sample_sec_time / 60) % 3600
    hours = int(sample_sec_time / 3600)

    return f"{number_sign}{hours:0>2}:{min:0>2}:{sec:0>2}." + str(msec).zfill(msec_precision_digits)


"""def sec_to_timestamp(sample_sec_time):

    timestamp_string_microseconds = time.strftime('%H:%M:%S.%f', time.gmtime(sample_sec_time))

    # Trim last 3 characters from string (discard microseconds)
    # timestamp_string_msec = timestamp_string_microseconds[:-3]

    return timestamp_string_microseconds
"""


def timestamp_to_sec(timestamp):

    time_and_msec = timestamp.split('.')
    time_parts = time_and_msec[0].split(':')

    index = len(time_parts) - 1

    seconds_sum: float = 0
    for i in range(0, time_parts[index]):
        factor = pow(60, index)
        reverse_index = len(time_parts) - 1 - i
        seconds_sum += time_parts[reverse_index] * factor

    return seconds_sum + time_and_msec / 1000


def sample_index_to_timestamp(sample_index, sample_rate):
    time_per_sample_frame_sec = 1 / sample_rate

    time_at_sample_sec = time_per_sample_frame_sec * sample_index

    return sec_to_timestamp(time_at_sample_sec)


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


def extract_clap_sample_indices(audio_wav_data, sample_rate, detection_threshold: float, detect_release_time_msec=100):

    # We are focused on the speed of the change for detecting a short impulse
    audio_amplitude_sobel = apply_audio_differentiation(audio_wav_data)

    # audio_sample_bit_depth: int = 32
    # audio_sample_bit_depth: int = np.iinfo(original_data.dtype).max
    # threshold_float = 0.4
    # sample_threshold_int: int = int(pow(2, audio_sample_bit_depth) * threshold_float)
    # print(f"sample_threshold_int = {sample_threshold_int}")

    # selected_key_samples_indices, _ = np.where(audio_wav_data >= threshold_float)
    # print(len(selected_key_samples_indices))
    # print(selected_key_samples_indices)

    selected_key_samples_indices, _ = np.where(audio_amplitude_sobel >= detection_threshold)

    # Release delay/time of impulse detection
    def filter_close_samples(sample_indices, window_size_samples):

        filtered_indices = []

        last_sample_index = - window_size_samples
        for sample_index in sample_indices:
            if (last_sample_index + window_size_samples <= sample_index):
                filtered_indices.append(sample_index)
                last_sample_index = sample_index

        return filtered_indices

    # print(len(selected_key_samples_indices))
    # print(selected_key_samples_indices)

    filter_window_size_samples = msec_to_samples(detect_release_time_msec, sample_rate)

    selected_key_samples_indices = filter_close_samples(selected_key_samples_indices, filter_window_size_samples)

    # detected_timestamps: list[str] = list(map(lambda detected_sample_index: sample_index_to_time(detected_sample_index, sample_rate), selected_key_samples_indices))

    # print(detected_timestamps)
    # print(selected_key_samples_indices)

    return selected_key_samples_indices


def extract_clap_timestamps(audio_wav_data, sample_rate, detection_threshold: float, detect_release_time_msec=100):

    selected_key_samples_indices = extract_clap_sample_indices(audio_wav_data, sample_rate, detection_threshold, detect_release_time_msec)

    detected_timestamps: list[str] = list(map(lambda detected_sample_index: sample_index_to_timestamp(detected_sample_index, sample_rate), selected_key_samples_indices))

    # print(detected_timestamps)
    # print(selected_key_samples_indices)

    return detected_timestamps


def find_sync_offset(reference_first_sample_time_msec, clap_sample_indices: list[int], sample_rate: int):

    selected_first_sample_time_msec = samples_to_msec(clap_sample_indices[0], sample_rate)
    first_clap_offset_msec = reference_first_sample_time_msec - selected_first_sample_time_msec
    return first_clap_offset_msec


def find_sync_offsets(claps_per_file_list: list[list[int]], sample_rates_list: list[int]):

    reference_file_clap_samples = claps_per_file_list[0]
    reference_first_sample_time_msec = samples_to_msec(reference_file_clap_samples[0], sample_rates_list[0])

    sync_offsets_msec = []

    for clap_samples, file_sample_rate in zip(claps_per_file_list[1:], sample_rates_list[1:]):
        sync_offsets_msec.append(find_sync_offset(reference_first_sample_time_msec, clap_samples, file_sample_rate))

    return sync_offsets_msec


video_matcher_regex = re.compile(".+\.[mp4|mov|mkv|avi|webm|m4a]")
audio_file_cache: dict[str, Any] = {}


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


def extract_clap_indices_of_files(file_paths):

    audio_data_list, sample_rates_list = read_audio_of_files(file_paths)

    claps_per_file_list = []

    detection_threshold = 0.5
    for audio_data, sample_rate in zip(audio_data_list, sample_rates_list):
        detected_clap_sample_indices = extract_clap_sample_indices(audio_data, sample_rate, detection_threshold, detect_release_time_msec=100)
        claps_per_file_list.append(detected_clap_sample_indices)
        sample_rates_list.append(sample_rate)

    return claps_per_file_list, sample_rates_list


def remux_video_audio_with_offset(video_file_path, audio_file_path, audio_offset_msec, output_range_timestamps: tuple):
    # remuxed_video_path = add_name_suffix_path(video_file_path, "_remuxed", ".mp4")

    # MKV container can hold both, normal pcm audio and hevc, h264, etc.
    # Which is great because we are not forced to reencode video audio, to mix our streams together into a container, like we would need to do with an mp4 container
    remuxed_video_path = add_name_suffix_path(video_file_path, "_remuxed", ".mkv")

    audio_offset_string = msec_to_timestamp(abs(audio_offset_msec))

    codec_args = [
        '-c:v',
        'copy',
        '-c:a',
        'copy',
    ]

    mixing_muxing_args = [
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        '-shortest',
    ]

    cmd_args = [
        'ffmpeg',
        '-i',
        video_file_path,
        '-i',
        audio_file_path,
        *codec_args,
        *mixing_muxing_args,
        '-ss',
        output_range_timestamps[0],
        '-to',
        output_range_timestamps[1],
        # '-vf',
        # f"trim={output_range_timestamps[0]}:{output_range_timestamps[1]}",
        remuxed_video_path
    ]

    if (audio_offset_msec < 0):

        cmd_args.insert(1, '-ss')
        cmd_args.insert(2, audio_offset_string)

        # remux_command = f"ffmpeg -ss {audio_offset_string} -i {video_file_path} -i {audio_file_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -ss {output_range_timestamps[0]} -to {output_range_timestamps[1]} {remuxed_video_path}"
    else:

        cmd_args.insert(3, '-ss')
        cmd_args.insert(4, audio_offset_string)

        # remux_command = f"ffmpeg -i {video_file_path}  -ss {audio_offset_string} -i {audio_file_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -ss {output_range_timestamps[0]} -to {output_range_timestamps[1]} {remuxed_video_path}"

    # Example (video was started before audio): ffmpeg  -ss 00:00:03.625 -i 20231204_204607.mp4 -i 230412_0072S12.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest remuxed_synced.mp4 -threads 6

    remux_command = " ".join(cmd_args)

    print(remux_command)

    os.system(remux_command)

    return remuxed_video_path

    # Audio starts before video -> sync clap is later in audio then in video ( -> audio needs to be trimmed at start, before syncing)


match_duration = re.compile('(?<=duration=).+')


def get_video_duration(video_file_path: str):
    cmd = f"ffprobe -i {video_file_path} -show_format"

    # | grep -Po \"(?<=duration=).+\""

    print(cmd)

    # Note: there are cases where this will fail
    args = cmd.split(' ')

    video_format_meta = subprocess.check_output(args).decode(sys.stdout.encoding)
    duration_match: re.Match = re.search(match_duration, video_format_meta)
    duration_string = duration_match.group(0)

    # duration_string = subprocess.check_output(args).decode(sys.stdout.encoding)
    return float(duration_string)


def pull_thumbnail(video_file_path: str, target_thumbnail_path: str, at_timestamp: str):
    # thumbnail_path = add_name_suffix_path(video_file_path, "_thumbnail", ".jpg")

    os.system(f"ffmpeg -ss {at_timestamp} -i {video_file_path} -y -vframes 1 {target_thumbnail_path}")

    return target_thumbnail_path


def pull_thumbnail_at_sec(video_file_path: str, target_thumbnail_path: str, seconds_time: float):
    return pull_thumbnail(video_file_path, target_thumbnail_path, sec_to_timestamp(seconds_time))


def pull_video_thumbnail_samples(video_file_path: str, thumbnail_cnt: int, clean_dir=False):

    video_duration = get_video_duration(video_file_path)

    thumbnail_sec_timepoints = np.linspace(0.5, video_duration - 0.5, thumbnail_cnt).tolist()

    video_dir = os.path.dirname(video_file_path)
    thumbnails_dir = join(video_dir, 'thumbnails')

    if (os.path.exists(thumbnails_dir) and clean_dir and 'thumbnails' in thumbnails_dir):
        shutil.rmtree(thumbnails_dir)

    os.makedirs(thumbnails_dir, exist_ok=True)

    # thumbnails_per_second = float(thumbnail_cnt / video_duration)
    # os.system(f"ffmpeg -i {video_file_path} -ss 00.500 -threads 6 -vf fps={thumbnails_per_second} {join(thumbnails_dir, 'thumbnail_at_%03d.jpg')}")

    thumbnail_paths = []
    for index, time_point_seconds in enumerate(thumbnail_sec_timepoints):
        time_label = str(int(time_point_seconds))
        thumbnail_path = join(thumbnails_dir, 'thumbnail_at_' + time_label + '.jpg')
        thumbnail_path = pull_thumbnail_at_sec(video_file_path, thumbnail_path, time_point_seconds)

        thumbnail_paths.append(thumbnail_path)

    return thumbnail_paths


def show_waveforms(audio_data, sample_rate, fill_down=False):

    data_sample_length = audio_data.shape[0]

    for audio_channel_index in range(0, audio_data.shape[1]):
        """time = np.linspace(
            0,  # start
            data_sample_length / sample_rate,
            num=data_sample_length
        )"""

        audio_channel_data = audio_data[:, audio_channel_index]

        timeline_data = np.linspace(
            0,  # start
            1,
            num=data_sample_length
        ) * data_sample_length / sample_rate

        time_labels = list(map(lambda time_sec: sec_to_timestamp(time_sec), timeline_data))

        plt.rcParams["figure.figsize"] = (16, 4)

        plt.figure(1)

        # title of the plot
        plt.title("Sound Wave")

        # import matplotlib.dates as md
        # min_sec_formatter = md.DateFormatter('%M:%S')
        # plt.set_major_formatter(xfmt)
        # plt.gcf().autofmt_xdate()
        # plt.gca().xaxis.set_major_formatter(min_sec_formatter)
        import time

        def sec_to_timestamp_format(x, pos):
            return time.strftime('%M:%S', time.gmtime(x))

        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(sec_to_timestamp_format))

        # major tick every 60 seconds/units
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(30))
        # plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))

        # plt.gca().tick_params(which='minor', length=10, color='r', )
        # plt.gca().locator_params(axis='both', nbins=10)
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # label of x-axis
        plt.xlabel("Time (seconds)")

        # plt.xticks(time, time_labels)
        plt.xticks()

        # plt.grid()
        plt.grid(which='major', linestyle='-', linewidth='1', color='black')
        plt.grid(which='minor', linestyle='-', linewidth='0.2', color='gray')
        # plt.grid(which='both', color='0.65', linestyle='-')

        plt.minorticks_on()
        # Needs to be after minor ticks on to adjust number of minor ticks
        plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(6))

        # actual plotting
        plt.plot(timeline_data, audio_channel_data)

        if (fill_down):
            plt.fill_between(timeline_data, audio_channel_data, color='blue', alpha=0.3)

        # shows the plot
        # in new window
        plt.show()


def merge_range_with_next_recursive(index, tuple_ranges, close_gap_threshold=1):

    if (index >= (len(tuple_ranges) - 1)):
        return tuple_ranges

    current_range = tuple_ranges[index]
    next_range = tuple_ranges[index + 1]

    if (next_range[0] - current_range[1] <= close_gap_threshold):
        tuple_ranges[0] = (current_range[0], next_range[1])

        return merge_range_with_next_recursive(index, tuple_ranges, close_gap_threshold)

    return tuple_ranges


def calc_range_length(range_tuple):
    return range_tuple[1] - range_tuple[0]


def select_long_ranges(tuple_ranges, long_range_threshold):

    filtered_tuple_ranges = filter(calc_range_length, tuple_ranges)
    return filtered_tuple_ranges


def sort_ranges_by_longest(tuple_ranges):
    return sorted(tuple_ranges, key=calc_range_length, reverse=True)

# ! side effect tuple_ranges modified


def merge_close_ranges(tuple_ranges, close_gap_threshold=1):

    for index in range(0, tuple_ranges - 1):
        merge_range_with_next_recursive(index, tuple_ranges, close_gap_threshold)

    return tuple_ranges

    """current_range = tuple_ranges[0]
    next_range = tuple_ranges[1]

    if (next_range[0] - current_range[1] <= close_gap_threshold):
        tuple_ranges[0] = (current_range[0], next_range[1])

        return merge_close_ranges(tuple_ranges, close_gap_threshold)

    return tuple_ranges

    for index in range(0, tuple_ranges - 1):
        current_range = tuple_ranges[index]
        next_range = tuple_ranges[index + 1]

        if(current_range)
    """


def detect_on_ranges(array_data, on_threshold=0.1):

    detected_on_range_tuples = []

    range_start = -1
    is_inside_range = False
    for index, sample in enumerate(array_data):

        if (any(sample >= on_threshold) and not is_inside_range):

            is_inside_range = True
            range_start = index

        elif (all(sample < on_threshold) and is_inside_range):

            is_inside_range = False
            detected_on_range_tuples.append((range_start, index))
            # detected_on_range_tuples.append((range_start, index - 1))

    if (is_inside_range):
        detected_on_range_tuples.append((range_start, len(array_data) - 1))

    return detected_on_range_tuples


def detect_song_range(array_data, on_threshold=0.1):

    detected_on_range_tuples = detect_on_ranges(array_data, on_threshold)
    detected_on_range_tuples = sort_ranges_by_longest(detected_on_range_tuples)

    return detected_on_range_tuples[0]


# audio_data = None
# sample_rate = None
# def extract_range_up_down_sample()


def resample_audio(target_sample_width_sec, audio_data, sample_rate):
    block_size_samples = target_sample_width_sec * sample_rate

    downsampled_audio, downsampled_rate = downsample_audio(audio_data, block_size_samples, sample_rate)

    norm_float_audio = normalize_to_float(downsampled_audio)

    return norm_float_audio, downsampled_rate

# find audio ramp start


def find_valley_index(data_array, start_index, direction=-1, valley_threshold=0.1):

    index = start_index
    while index >= 0 and index < len(data_array):

        current_sample_val = data_array[index]

        if (all(current_sample_val < valley_threshold)):
            return index

        index += direction

    return index


def convert_sample_index(src_sample_index: int, src_sample_rate: float, target_sample_rate: float, round_to_nearest=False):
    scale_factor = target_sample_rate / src_sample_rate

    scaled_index = src_sample_index * scale_factor

    if (round_to_nearest):
        return round(scaled_index)

    return int(scaled_index)


def convert_sample_tuple(range_tuple: tuple, src_sample_rate: float, target_sample_rate: float, round_to_nearest=False):

    scaled_index_list = []

    for index in range_tuple:
        scaled_index_list.append(convert_sample_index(index, src_sample_rate, target_sample_rate, round_to_nearest))

    return tuple(scaled_index_list)


def refine_range_towards_valleys(range_tuple: tuple, audio_data_samples: np.array, valley_thresholds=(0.05, 0.02)):
    start_valley_index = find_valley_index(audio_data_samples, range_tuple[0], direction=-1, valley_threshold=valley_thresholds[0])
    end_valley_index = find_valley_index(audio_data_samples, range_tuple[1], direction=1, valley_threshold=valley_thresholds[1])
    return (start_valley_index, end_valley_index)


def sample_range_to_timestamps(sample_range_tuple: tuple, sample_rate: float):
    start_timestamp = sample_index_to_timestamp(sample_range_tuple[0], sample_rate)
    end_timestamp = sample_index_to_timestamp(sample_range_tuple[1], sample_rate)

    return (start_timestamp, end_timestamp)


def print_sample_range_timestamps(sample_range_tuple: tuple, sample_rate: float):
    print(sample_range_tuple)

    sample_range_timestamps = sample_range_to_timestamps(sample_range_tuple, sample_rate)

    print(sample_range_timestamps[0])
    print(sample_range_timestamps[0])


def add_time_to_sample_range(sample_range_tuple: tuple, add_time_tuple_seconds: tuple[float], sample_rate: float):

    modified_index_list = []

    for index, range_index in enumerate(sample_range_tuple):
        modified_index_list.append(range_index + sec_to_samples(add_time_tuple_seconds[index], sample_rate))

    return tuple(modified_index_list)


def find_remux_sync_offset_msec(audio_file_path, video_file_path):

    claps_per_file_list, sample_rates_list = extract_clap_indices_of_files([audio_file_path, video_file_path])

    sync_offsets_msec = find_sync_offsets(claps_per_file_list, sample_rates_list)

    print(f"sync_offset_msec = {sync_offsets_msec[0]}")
    print(f"sync_offset_timestamp = {msec_to_timestamp(sync_offsets_msec[0])}")

    return sync_offsets_msec[0]


def apply_audio_effects(video_file_path):

    os.system(f"ffmpeg -i {video_file_path} -af acompression=threshold=-21dB:ratio=4:attack=200:release=1000 {video_file_path}")


def detect_song_time_range(video_file_path: str):
    extracted_audio_path = get_extracted_audio_path(video_file_path)
    sample_rate, audio_data = read_audio_of_file(video_file_path)

    # show_waveforms(audio_data, sample_rate)

    # abs data to make averaging operations cumulative -> more effect as positive and negative values do not cancel each
    downsampled_audio, downsampled_rate = resample_audio(4, np.abs(audio_data), sample_rate)

    selected_loud_samples_indices, _ = np.where(np.abs(downsampled_audio) >= 0.1)

    downsampled_audio[downsampled_audio >= 0.1] = 1.0
    downsampled_audio[downsampled_audio < 0.1] = 0.0

    rough_song_range = detect_song_range(downsampled_audio, 0.1)
    print_sample_range_timestamps(rough_song_range, downsampled_rate)

    fine_block_time_sec = 1.0

    downsampled_audio_fine, downsampled_rate_fine = resample_audio(fine_block_time_sec, np.abs(audio_data), sample_rate)
    differentiated_audio = apply_audio_differentiation(downsampled_audio_fine)
    # differentiated_audio = apply_audio_differentiation(differentiated_audio)

    fine_rough_song_range = convert_sample_tuple(rough_song_range, downsampled_rate, downsampled_rate_fine)

    fine_song_range = refine_range_towards_valleys(fine_rough_song_range, downsampled_audio_fine, valley_thresholds=(0.03, 0.005))
    print_sample_range_timestamps(fine_song_range, downsampled_rate_fine)

    final_song_range = add_time_to_sample_range(fine_song_range, (-1.0, 0.0), downsampled_rate_fine)
    print_sample_range_timestamps(final_song_range, downsampled_rate_fine)

    # show_waveforms(downsampled_audio_fine, downsampled_rate_fine)
    # show_waveforms(differentiated_audio, downsampled_rate_fine)

    # show_waveforms(downsampled_audio, downsampled_rate)
    # show_waveforms(downsampled_audio_left, downsampled_rate)

    # norm_float_audio = normalize_to_float(downsampled_audio_left)

    # print(downsampled_audio_left)

    # selected_loud_samples_indices, _ = np.where(np.abs(downsampled_audio_left) >= 30000)

    # print(selected_loud_samples_indices)

    return sample_range_to_timestamps(final_song_range, downsampled_rate_fine)

# pip3 install python-reapy


def normalize_audio():

    # Truepeak at -1.5dB, loudness normalization
    cmd = "ffmpeg -i input.mp3 -af loudnorm=TP=-1.5 output.mp3"


# remuxed_video_file_path = "20231204_204607_remuxed.mkv"
# pull_video_thumbnail_samples(remuxed_video_file_path, 30, clean_dir=True)


# transcription = transcribe_audio_file(audio_file_path)


# clap_sound_attack_time_msec = 1
# clap_sound_attack_time_samples = msec_to_samples(3, sample_rate)
# clap_sound_attack_time_samples = 4
# downsampled_audio, downsampled_sample_rate = downsample_audio(clap_sound_attack_time_samples, sample_rate)


# print("Downsampled")
# detected_timestamps: list[str] = extract_clap_timestamps(downsampled_audio, downsampled_sample_rate, detection_threshold)
# print(detected_timestamps)

# print("Orig sampled")
# detected_timestamps: list[str] = extract_clap_timestamps(original_data, sample_rate, detection_threshold)
# print(detected_timestamps)

# All 14.7, 16.6, 18.3, 20.8, 21.4, 22.06, 24.1, (28.470) -- Audio recording
# 18.325 -- Video recoding
# -> Video was started before audio
# video - audio = 00:00:03.625

def main():

    parser = argparse.ArgumentParser(
        description="Synchronize external audio with camera video, select only song range and apply sound processing effects in one go. Automatized postprocessing"
    )

    parser.add_argument('video_src_path', help="The video file that should be muxed with the audio_src_file, which audio stream is used for syncing and then discarded in favor if 'audio_src_path' file")
    parser.add_argument('audio_src_path', help="Audio file to be muxed/mixed and synced with the video data of the 'video_src_path' file")

    parser.add_argument('-ct', '--clap_loudness_threshold', '--clap_threshold', type=float, help="[0.0, 1.0]", default=0.4)
    parser.add_argument('-cr', '--clap_release_time', '--clap_release', type=float, help="[0.0, 1.0]", default=0.1)
    parser.add_argument('-st', '--song_on_threshold', '--song_threshold', type=float, help="[0.0, 1.0]", default=0.1)
    parser.add_argument('-sb', '--song_detection_block_size', '--song_block_size', type=float, help="", default=4.0)
    parser.add_argument('-rb', '--range_refine_block_size', '--refine_block_size', type=float, help="", default=1.0)
    parser.add_argument('-spr', '--song_start_prerun', '--prerun', type=float, help="Time value in seconds added to the final detected start position after song range detection finished", default=-1.0)
    parser.add_argument('-spo', '--song_end_postrun', '--postrun', type=float, help="Time value in seconds added to the final detected end position after song range detection finished", default=0)
    parser.add_argument('-sv', '--start_valley_threshold', '--start_valley', type=float, help="", default=0.03)
    parser.add_argument('-ev', '--end_valley_threshold', '--end_valley', type=float, help="", default=0.005)

    parser.add_argument('-k', '--keep_transient_files', '--keep', action="store_true", help="Keep files that were the results of processing or extraction, but can be recalculated from the source files if needed")
    parser.add_argument('-np', '--no_chain_processing', '--no_processing', action="store_true", help="Do not apply reaper FX chain onto the 'audio_src_path' before muxing/mixing")
    parser.add_argument('-nr', '--no_on_range_detection', '--no_on_range', action="store_true", help="Do not automatically detect where the song/audio action start, ends and cut the output video to that range when muxing")
    parser.add_argument('-ns', '--no_clap_sync', '--no_sync', action="store_true", help="Do not detect the positions of claps in the video and audio file for the purpose of synchronizing those 2 recordings/files")
    parser.add_argument('-p', '--play_on_finish', '--play', action="store_true", help="Play video when finished with processing")

    parser.add_argument('-range', '--on_range', nargs='+', help="Manually specifies the range of interested which should be selected from the files (relative to the video file)")
    parser.add_argument('-offset', '--sync_offset_msec', '--sync_offset', help="Manually set the offset between the 'video audio' and the 'external audio' for the purpose of syncing the 2")

    args: argparse.Namespace = parser.parse_args()

    video_file_path = args.video_src_path
    audio_file_path = args.audio_src_path
    # video_file_path = "20231204_204607.mp4"
    # audio_file_path = "230412_0072S12.wav"

    song_range_timestamps = detect_song_time_range(video_file_path)
    remux_sync_offset = find_remux_sync_offset_msec(audio_file_path, video_file_path)

    processed_audio_file_path = audio_file_path

    # male-voice-mix, warmness-booster-effects
    processed_audio_file_path = apply_audio_processing_chain(audio_file_path, 'male-voice-mix')

    # song_range_timestamps = ('00:03:33.000', '00:09:24.000')
    # remux_sync_offset = '-00:00:03.652'
    # remux_sync_offset = -3652

    # TODO song range timestamps have to consider the offset as we are truncating the output of the synced merge

    remuxed_video_file_path = remux_video_audio_with_offset(video_file_path, processed_audio_file_path, remux_sync_offset, song_range_timestamps)

    os.system(f'mpv {remuxed_video_file_path}')


if __name__ == '__main__':
    sys.exit(main())
