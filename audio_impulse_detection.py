import numpy as np
from common_audio_processing import apply_audio_differentiation, read_audio_of_files
from time_conversion_util import msec_to_samples, msec_to_timestamp, sample_index_to_timestamp, samples_to_msec


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


def extract_clap_indices_of_files(file_paths):

    audio_data_list, sample_rates_list = read_audio_of_files(file_paths)

    claps_per_file_list = []

    detection_threshold = 0.5
    for audio_data, sample_rate in zip(audio_data_list, sample_rates_list):
        detected_clap_sample_indices = extract_clap_sample_indices(audio_data, sample_rate, detection_threshold, detect_release_time_msec=100)
        claps_per_file_list.append(detected_clap_sample_indices)
        sample_rates_list.append(sample_rate)

    return claps_per_file_list, sample_rates_list


def find_remux_sync_offset_msec(audio_file_path, video_file_path):

    claps_per_file_list, sample_rates_list = extract_clap_indices_of_files([audio_file_path, video_file_path])

    sync_offsets_msec = find_sync_offsets(claps_per_file_list, sample_rates_list)

    print(f"sync_offset_msec = {sync_offsets_msec[0]}")
    print(f"sync_offset_timestamp = {msec_to_timestamp(sync_offsets_msec[0])}")

    return sync_offsets_msec[0]
