from typing import Tuple, TypedDict
from typing import Any
import numpy as np
from audio_plotting import show_waveforms
from common_audio_processing import read_audio_of_file, resample_audio
from dot_dict_conversion import to_dot_dict_with_defaults
from time_conversion_util import print_sample_range_timestamps, sample_index_range_to_timestamps, sec_to_sample_index


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


def detect_on_ranges(array_data, on_threshold=0.1) -> list[tuple[int]]:

    detected_on_range_tuples: list[tuple[int]] = []

    range_start: int = -1
    is_inside_range: bool = False
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


def detect_song_range(array_data, on_threshold=0.1, close_gap_samples_threshold: int = 8) -> Tuple[tuple, list[tuple]]:

    detected_on_range_tuples = detect_on_ranges(array_data, on_threshold)
    detected_on_range_tuples = merge_close_ranges(detected_on_range_tuples, close_gap_samples_threshold=close_gap_samples_threshold)
    detected_on_range_tuples = sort_ranges_by_longest(detected_on_range_tuples)

    return detected_on_range_tuples[0], detected_on_range_tuples


def scale_sample_index(src_sample_index: int, src_sample_rate: float, target_sample_rate: float, round_to_nearest=False) -> int:
    scale_factor = target_sample_rate / src_sample_rate

    scaled_index = src_sample_index * scale_factor

    if (round_to_nearest):
        return round(scaled_index)

    return int(scaled_index)


def convert_sample_tuple(range_tuple: tuple, src_sample_rate: float, target_sample_rate: float, round_to_nearest=False) -> tuple:

    scaled_index_list = []

    for index in range_tuple:
        scaled_index_list.append(scale_sample_index(index, src_sample_rate, target_sample_rate, round_to_nearest))

    return tuple(scaled_index_list)


# find audio ramp start
def find_valley_index(data_array, start_index, direction=-1, valley_threshold=0.1) -> int:

    index = start_index
    while index >= 0 and index < len(data_array):

        current_sample_val = data_array[index]

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
    song_detection_block_size: float
    song_on_threshold: float
    range_refine_block_size: float
    start_valley_threshold: float
    end_valley_threshold: float
    song_start_prerun: float
    song_end_postrun: float
    show_plot: float


activity_detection_options_defaults: ActivityDetectionOptions = {
    'song_detection_block_size': 4.0,
    'song_on_threshold': 0.1,
    'range_refine_block_size': 1.0,
    'start_valley_threshold': 0.03,
    'end_valley_threshold': 0.005,
    'song_start_prerun': -1.0,
    'song_end_postrun': 0.0,
    'show_plot': False,
}


def detect_song_time_range(video_file_path: str, options: ActivityDetectionOptions = activity_detection_options_defaults):

    options: ActivityDetectionOptions = to_dot_dict_with_defaults(options, activity_detection_options_defaults)

    # extracted_audio_path = get_extracted_audio_path(video_file_path)
    sample_rate, audio_data = read_audio_of_file(video_file_path)

    # if (not options.keep_transient_files):
    #    delete_video_extracted_audio(video_file_path)

    # show_waveforms(audio_data, sample_rate)

    # abs data to make averaging operations cumulative -> more effect as positive and negative values do not cancel each
    downsampled_audio, downsampled_rate = resample_audio(options.song_detection_block_size, np.abs(audio_data), sample_rate)

    # selected_loud_samples_indices, _ = np.where(np.abs(downsampled_audio) >= 0.1)

    # downsampled_audio[downsampled_audio >= options.song_on_threshold] = 1.0
    # downsampled_audio[downsampled_audio < options.song_on_threshold] = 0.0

    rough_song_range, all_detected_ranges = detect_song_range(downsampled_audio, options.song_on_threshold)
    print_sample_range_timestamps(rough_song_range, downsampled_rate)

    fine_block_time_sec = options.range_refine_block_size

    downsampled_audio_fine, downsampled_rate_fine = resample_audio(fine_block_time_sec, np.abs(audio_data), sample_rate)
    # differentiated_audio = apply_audio_differentiation(downsampled_audio_fine)
    # differentiated_audio = apply_audio_differentiation(differentiated_audio)

    fine_rough_song_range = convert_sample_tuple(rough_song_range, downsampled_rate, downsampled_rate_fine)

    fine_song_range = refine_range_towards_valleys(fine_rough_song_range, downsampled_audio_fine, valley_thresholds=(options.start_valley_threshold, options.end_valley_threshold))
    print_sample_range_timestamps(fine_song_range, downsampled_rate_fine)

    final_song_range = add_time_to_sample_range(fine_song_range, (options.song_start_prerun, options.song_end_postrun), downsampled_rate_fine)
    print_sample_range_timestamps(final_song_range, downsampled_rate_fine)

    if (options.show_plot):

        # plotting_audio = np.concatenate(downsampled_audio, downsampled_audio_fine)

        show_waveforms(downsampled_audio, downsampled_rate, marker_indices=rough_song_range, marked_ranges=all_detected_ranges, fill_down=True, block=False)

        show_waveforms(downsampled_audio_fine, downsampled_rate_fine, marker_indices=fine_rough_song_range, marked_ranges=[final_song_range], fill_down=True, block=False)

    # show_waveforms(downsampled_audio_fine, downsampled_rate_fine)
    # show_waveforms(differentiated_audio, downsampled_rate_fine)

    # show_waveforms(downsampled_audio, downsampled_rate)
    # show_waveforms(downsampled_audio_left, downsampled_rate)

    # norm_float_audio = normalize_to_float(downsampled_audio_left)

    # print(downsampled_audio_left)

    # selected_loud_samples_indices, _ = np.where(np.abs(downsampled_audio_left) >= 30000)

    # print(selected_loud_samples_indices)

    final_song_range_timestamps = sample_index_range_to_timestamps(final_song_range, downsampled_rate_fine)

    return final_song_range_timestamps
