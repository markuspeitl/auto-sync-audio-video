from collections.abc import Iterable
from functools import reduce
from typing import Tuple, TypedDict
from typing import Any
import numpy as np
from audio_plotting import show_waveforms
from common_audio_processing import normalize_symmetrical_float, read_audio_of_file, resample_audio
from dot_dict_conversion import to_dot_dict_with_defaults
from time_conversion_util import print_sample_range_timestamps, sample_index_range_to_timestamps, sec_to_sample_index


# Convolve individual channels
def normalized_channel_convolve(data: np.ndarray, kernel: np.ndarray):

    kernel_sum = np.sum(kernel)
    norm_kernel = np.asarray(kernel, np.float32) / kernel_sum

    if (len(data.shape) > 1):
        for index in range(0, len(data.shape)):
            data[index] = np.convolve(data[index], norm_kernel, 'valid')
    else:
        data = np.convolve(data, norm_kernel, 'valid')

    return data


def merge_range_with_next_recursive(index: int, tuple_ranges: list[tuple], close_gap_samples_threshold: int = 1) -> list[tuple[int]]:

    if (index >= (len(tuple_ranges) - 1)):
        return tuple_ranges

    current_range = tuple_ranges[index]
    next_range = tuple_ranges[index + 1]

    if (next_range[0] - current_range[1] <= close_gap_samples_threshold and next_range[0] > current_range[1]):
        tuple_ranges[index] = (current_range[0], next_range[1])
        tuple_ranges.remove(next_range)

        return merge_range_with_next_recursive(index, tuple_ranges, close_gap_samples_threshold)

    return tuple_ranges


def calc_range_length(range_tuple: tuple[int]) -> int:
    return range_tuple[1] - range_tuple[0]


def select_long_ranges(tuple_ranges, long_range_threshold) -> list[tuple[int]]:

    filtered_tuple_ranges = filter(calc_range_length, tuple_ranges)
    return filtered_tuple_ranges


def sort_ranges_by_longest(tuple_ranges) -> list[tuple[int]]:
    return sorted(tuple_ranges, key=calc_range_length, reverse=True)


# ! side effect tuple_ranges modified
def merge_close_ranges(tuple_ranges: list[tuple], close_gap_samples_threshold: int = 1) -> list[tuple]:

    for index in range(0, len(tuple_ranges) - 1):
        merge_range_with_next_recursive(index, tuple_ranges, close_gap_samples_threshold)

    return tuple_ranges


def detect_on_ranges(array_data: np.ndarray, on_threshold: float = 0.1) -> list[tuple[int]]:

    detected_on_range_tuples: list[tuple[int]] = []

    range_start: int = -1
    is_inside_range: bool = False
    for index, sample in enumerate(array_data):

        # Handling single channel data
        if (not isinstance(sample, Iterable)):
            sample = np.array([sample])

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


def detect_active_range(array_data: np.ndarray, on_threshold: float = 0.1, close_gap_samples_threshold: int = 2) -> Tuple[tuple, list[tuple]]:

    all_detected_on_range_tuples = detect_on_ranges(array_data, on_threshold)
    detected_on_range_tuples = list(all_detected_on_range_tuples)
    merged_detected_on_range_tuples = merge_close_ranges(detected_on_range_tuples, close_gap_samples_threshold=close_gap_samples_threshold)
    merged_detected_on_range_tuples = sort_ranges_by_longest(merged_detected_on_range_tuples)

    return merged_detected_on_range_tuples[0], merged_detected_on_range_tuples, all_detected_on_range_tuples


def scale_sample_index(src_sample_index: int, src_sample_rate: float, target_sample_rate: float, round_to_nearest=False) -> int:
    scale_factor = target_sample_rate / src_sample_rate

    scaled_index = src_sample_index * scale_factor

    if (round_to_nearest):
        return round(scaled_index)

    return int(scaled_index)


def scale_sample_tuple(range_tuple: tuple, src_sample_rate: float, target_sample_rate: float, round_to_nearest=False) -> tuple:

    scaled_index_list = []

    for index in range_tuple:
        scaled_index_list.append(scale_sample_index(index, src_sample_rate, target_sample_rate, round_to_nearest))

    return tuple(scaled_index_list)


# find audio ramp start
def find_valley_index(data_array: np.ndarray, start_index: int, direction: int = -1, valley_threshold: float = 0.1) -> int:

    index = start_index
    while index >= 0 and index < len(data_array):

        current_sample_val = data_array[index]

        # Handling single channel data
        if (not isinstance(current_sample_val, Iterable)):
            current_sample_val = np.array([current_sample_val])

        if (all(current_sample_val < valley_threshold)):
            return index

        index += direction

    return index


def refine_range_towards_valleys(range_tuple: tuple, audio_data_samples: np.array, valley_thresholds=(0.05, 0.02)) -> tuple[int]:
    start_valley_index = find_valley_index(audio_data_samples, range_tuple[0], direction=-1, valley_threshold=valley_thresholds[0])
    end_valley_index = find_valley_index(audio_data_samples, range_tuple[1], direction=1, valley_threshold=valley_thresholds[1])
    return (start_valley_index, end_valley_index)


def add_time_to_sample_range(sample_range_tuple: tuple, add_time_tuple_seconds: tuple[float], sample_rate: float) -> tuple[int]:

    modified_index_list: list[int] = []

    for index, range_index in enumerate(sample_range_tuple):
        modified_index_list.append(range_index + sec_to_sample_index(add_time_tuple_seconds[index], sample_rate))

    return tuple(modified_index_list)


class ActivityDetectionOptions(TypedDict):
    rough_activity_block_size: float
    rough_active_threshold: float
    fine_activity_block_size: float
    start_valley_threshold: float
    end_valley_threshold: float
    activity_start_prerun: float
    activity_end_postrun: float
    show_plot: float


activity_detection_options_defaults: ActivityDetectionOptions = {
    'rough_activity_block_size': 3.25,
    'rough_active_threshold': 0.15,
    'fine_activity_block_size': 0.5,
    'start_valley_threshold': 0.03,
    'end_valley_threshold': 0.0175,
    'activity_start_prerun': -0.5,
    'activity_end_postrun': 0.25,
    'show_plot': False,
}


def detect_audio_resampled_active_range(audio_data: np.ndarray, sample_rate: float, resampled_block_size_sec: float, active_threshold: float, plot: bool = False, plot_block: bool = False):

    resampled_audio_blocks_data, resampled_audio_samplerate = resample_audio(resampled_block_size_sec, np.abs(audio_data), sample_rate)

    # sliding window for smoothing out data (simple averaging window)
    # normalized_channel_convolve(resampled_audio_blocks_data, np.ones(3, dtype=int))

    active_range_samples, all_merged_detected_ranges, all_detected_ranges = detect_active_range(resampled_audio_blocks_data, active_threshold)

    print_sample_range_timestamps(active_range_samples, resampled_audio_samplerate)

    if (plot):
        show_waveforms(resampled_audio_blocks_data, resampled_audio_samplerate, marker_indices=active_range_samples, marked_ranges=all_detected_ranges, fill_down=True, block=plot_block)

    return active_range_samples, resampled_audio_samplerate


def refine_active_audio_range(audio_data: np.ndarray, sample_rate: float, active_range_samples: tuple[int], active_range_samplerate: float, refine_block_size_sec: float, start_valley_threshold: float, end_valley_threshold: float, plot: bool = False, plot_block: bool = False):

    refine_audio_blocks_data, refine_audio_samplerate = resample_audio(refine_block_size_sec, np.abs(audio_data), sample_rate)

    to_refine_current_active_range: tuple[int] = scale_sample_tuple(active_range_samples, active_range_samplerate, refine_audio_samplerate)

    refined_song_range: tuple[int] = refine_range_towards_valleys(to_refine_current_active_range, refine_audio_blocks_data, valley_thresholds=(start_valley_threshold, end_valley_threshold))

    print_sample_range_timestamps(refined_song_range, refine_audio_samplerate)

    if (plot):
        show_waveforms(refine_audio_blocks_data, refine_audio_samplerate, marker_indices=refined_song_range, marked_ranges=[to_refine_current_active_range], fill_down=True, block=plot_block)

    return refined_song_range, refine_audio_samplerate


def detect_audio_activity_range_of_file(media_file_path: str, options: ActivityDetectionOptions = activity_detection_options_defaults):

    options: ActivityDetectionOptions = to_dot_dict_with_defaults(options, activity_detection_options_defaults)

    sample_rate, audio_data = read_audio_of_file(media_file_path)

    audio_channel = audio_data
    if (audio_data.shape[1] > 1):
        audio_channel = (audio_data[:, 0] + audio_data[:, 1]) / 2

    audio_channel = normalize_symmetrical_float(audio_channel)
    # show_waveforms(audio_channel, sample_rate, fill_down=False, block=True)

    print("Rough audio activity range estimate:")
    rough_song_range, rough_audio_samplerate = detect_audio_resampled_active_range(audio_channel, sample_rate, options.rough_activity_block_size, options.rough_active_threshold, options.show_plot)

    print("Refined activity range estimate:")
    fine_song_range, fine_audio_samplerate = refine_active_audio_range(audio_channel, sample_rate, rough_song_range, rough_audio_samplerate, options.fine_activity_block_size, options.start_valley_threshold, options.end_valley_threshold, options.show_plot, True)

    print("Final audio activity range estimate:")
    final_song_range: tuple[int] = add_time_to_sample_range(fine_song_range, (options.activity_start_prerun, options.activity_end_postrun), fine_audio_samplerate)
    print_sample_range_timestamps(final_song_range, fine_audio_samplerate)

    final_song_range_timestamps: tuple[str] = sample_index_range_to_timestamps(final_song_range, fine_audio_samplerate)

    return final_song_range_timestamps

    """fine_block_time_sec: float = options.fine_activity_block_size

    fine_audio_blocks_data, fine_audio_samplerate = resample_audio(fine_block_time_sec, np.abs(audio_channel), sample_rate)

    fine_rough_song_range: tuple[int] = scale_sample_tuple(rough_song_range, rough_audio_samplerate, fine_audio_samplerate)

    fine_song_range: tuple[int] = refine_range_towards_valleys(fine_rough_song_range, fine_audio_blocks_data, valley_thresholds=(options.start_valley_threshold, options.end_valley_threshold))
    print_sample_range_timestamps(fine_song_range, fine_audio_samplerate)"""

    """if (options.show_plot):

        show_waveforms(fine_audio_blocks_data, fine_audio_samplerate, marker_indices=fine_rough_song_range, marked_ranges=[final_song_range], fill_down=True, block=False)"""

    # differentiated_audio = apply_audio_differentiation(rough_audio_blocks_data_fine)
    # differentiated_audio = apply_audio_differentiation(differentiated_audio)

    # plotting_audio = np.concatenate(rough_audio_blocks_data, rough_audio_blocks_data_fine)
    # show_waveforms(rough_audio_blocks_data, rough_audio_samplerate, marker_indices=rough_song_range, marked_ranges=all_detected_ranges, fill_down=True, block=False)

    # extracted_audio_path = get_extracted_audio_path(video_file_path)
    # if (not options.keep_transient_files):
    #    delete_video_extracted_audio(video_file_path)

    # show_waveforms(audio_data, sample_rate)

    # abs data to make averaging operations cumulative -> more effect as positive and negative values do not cancel each
    """rough_audio_blocks_data, rough_audio_samplerate = resample_audio(options.rough_activity_block_size, np.abs(audio_data), sample_rate)

    # selected_loud_samples_indices, _ = np.where(np.abs(rough_audio_blocks_data) >= 0.1)

    # rough_audio_blocks_data[rough_audio_blocks_data >= options.rough_active_threshold] = 1.0
    # rough_audio_blocks_data[rough_audio_blocks_data < options.rough_active_threshold] = 0.0

    rough_song_range, all_detected_ranges = detect_active_range(rough_audio_blocks_data, options.rough_active_threshold)
    print_sample_range_timestamps(rough_song_range, rough_audio_samplerate)"""

    # show_waveforms(rough_audio_blocks_data_fine, rough_audio_samplerate_fine)
    # show_waveforms(differentiated_audio, rough_audio_samplerate_fine)

    # show_waveforms(rough_audio_blocks_data, rough_audio_samplerate)
    # show_waveforms(rough_audio_blocks_data_left, rough_audio_samplerate)

    # norm_float_audio = normalize_to_float(rough_audio_blocks_data_left)

    # print(rough_audio_blocks_data_left)

    # selected_loud_samples_indices, _ = np.where(np.abs(rough_audio_blocks_data_left) >= 30000)

    # print(selected_loud_samples_indices)
