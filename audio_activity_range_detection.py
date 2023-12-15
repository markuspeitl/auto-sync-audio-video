
import numpy as np
from common_audio_processing import apply_audio_differentiation, read_audio_of_file, resample_audio
from path_util import get_extracted_audio_path
from time_conversion_util import sample_index_to_timestamp, sec_to_samples


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


# find audio ramp start
def find_valley_index(data_array, start_index, direction=-1, valley_threshold=0.1):

    index = start_index
    while index >= 0 and index < len(data_array):

        current_sample_val = data_array[index]

        if (all(current_sample_val < valley_threshold)):
            return index

        index += direction

    return index


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
    # differentiated_audio = apply_audio_differentiation(downsampled_audio_fine)
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
