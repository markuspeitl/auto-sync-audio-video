
import argparse
import os
import sys
from typing import Any, Callable
import numpy as np
from os.path import dirname, join
from audio_impulse_detection import extract_impulse_indices_of_files

from time_conversion_util import sample_index_to_sec


def simple_map(apply_function: Callable, *args):
    return list(map(lambda item: apply_function(item, *args[1:]), args[0]))


# TODO should be better than only matching up the first detected sample (maybe match up ranges and choose the result with most similar offset)
"""def find_sync_offset(reference_first_sample_time_msec: int, clap_sample_indices: list[int], sample_rate: float):

    selected_first_sample_time_msec = sample_index_to_sec(clap_sample_indices[0], sample_rate)
    first_clap_offset_msec = reference_first_sample_time_msec - selected_first_sample_time_msec
    return first_clap_offset_msec
"""


def impulse_sample_indices_to_sec(impulse_indices: list[int], sample_rate: float) -> list[float]:
    impulse_times_sec = simple_map(sample_index_to_sec, impulse_indices, sample_rate)
    return impulse_times_sec


def impulse_sample_indices_list_to_sec(impulse_indices_per_file_list: list[list[int]], sample_rates_list: list[float]) -> list[list[float]]:
    impulse_sec_times_per_file_list: list[list[float]] = []

    for impulse_indices, sample_rate in zip(impulse_indices_per_file_list, sample_rates_list):
        # impulse_times_sec = sample_index_to_sec(impulse_indices, sample_rate)
        # impulse_times_sec = simple_map(sample_index_to_sec, impulse_indices, sample_rate)
        impulse_times_sec = impulse_sample_indices_to_sec(impulse_indices, sample_rate)
        impulse_sec_times_per_file_list.append(impulse_times_sec)

    return impulse_sec_times_per_file_list

    # return simple_map(sample_index_to_sec, zip(impulse_indices_per_file_list, sample_rates_list))


def first_impulse_sync_offset_sec(file_impulse_times_sec_a: list[float], file_impulse_times_sec_b: list[float]) -> float:
    impulse_offset_sec = file_impulse_times_sec_b[0] - file_impulse_times_sec_a[0]
    return impulse_offset_sec


def extract_impulse_ranges(file_impulse_times_sec: list[float]):
    impulse_ranges = []

    last_impulse_time = file_impulse_times_sec[0]
    for impulse_time_sec in file_impulse_times_sec[1:]:

        range_tuple = ((impulse_time_sec - last_impulse_time), last_impulse_time)
        impulse_ranges.append(range_tuple)

        last_impulse_time = impulse_time_sec

    return impulse_ranges


time_match_precision_sec = 0.1


def find_impulse_list_sync_offset_sec(file_impulse_times_sec_a: list[float], file_impulse_times_sec_b: list[float]) -> float:
    # impulse_offset_sec = file_impulse_times_sec_a[0] - file_impulse_times_sec_b[0]

    if (len(file_impulse_times_sec_a) <= 0 or len(file_impulse_times_sec_b) <= 0):
        raise Exception("Error can not match impulses if one of the files does not have any impulses that were detected")

    if (len(file_impulse_times_sec_a) <= 1 or len(file_impulse_times_sec_b) <= 1):
        return first_impulse_sync_offset_sec(file_impulse_times_sec_a, file_impulse_times_sec_b)

    impulse_ranges_a = extract_impulse_ranges(file_impulse_times_sec_a)
    impulse_ranges_b = extract_impulse_ranges(file_impulse_times_sec_b)

    impulse_ranges_a = sorted(impulse_ranges_a, reverse=True)
    impulse_ranges_b = sorted(impulse_ranges_b, reverse=True)

    # Still not super robust as there could be similar wide ranges, between sounds by happenstance
    # but should provide a resistance against some truncated or weakly detected impulses in one of the files
    # Could be improved by giving wider ranges a higher significance (with bigger range matches getting a wrong match is less likely)
    for range_width_sec_a, range_offset_sec_a in impulse_ranges_a:
        for range_width_sec_b, range_offset_sec_b in impulse_ranges_b:

            if (abs(range_width_sec_b - range_width_sec_a) < time_match_precision_sec):
                # Average in case there are differences
                return range_offset_sec_b - range_offset_sec_a

    return first_impulse_sync_offset_sec(file_impulse_times_sec_a, file_impulse_times_sec_b)
    raise Exception("Error can not match impulses, could not match any of the ranges of file a to the impulse ranges of file b")

    # range-width, range-offset
    # Match range width between a and b
    # Check which range offset has the most votes
    # If there is no range/pair match -> fall back to first impulse

    # return first_impulse_sync_offset_sec(file_impulse_times_sec_a, file_impulse_times_sec_b)


def find_sync_offsets(impulse_indices_per_file_list: list[list[int]], sample_rates_list: list[float]):

    impulse_sec_times_per_file_list = impulse_sample_indices_list_to_sec(impulse_indices_per_file_list, sample_rates_list)

    reference_file_impulse_times_sec = impulse_sec_times_per_file_list[0]

    # sync_offsets_sec = simple_map(find_impulse_list_sync_offset_sec, file_impulse_times_sec, reference_file_impulse_times_sec)

    sync_offsets_sec = []
    for file_impulse_times_sec in impulse_sec_times_per_file_list[1:]:
        impulse_offset_sec = find_impulse_list_sync_offset_sec(reference_file_impulse_times_sec, file_impulse_times_sec)
        sync_offsets_sec.append(impulse_offset_sec)

    return sync_offsets_sec

    # reference_impulse_indices: list[int] = impulse_indices_per_file_list[0]
    # reference_sample_rate: float = sample_rates_list[0]

    # reference_first_sample_time_msec = samples_index_to_msec(reference_impulse_indices, reference_sample_rate)

    """for clap_samples, file_sample_rate in zip(impulse_indices_per_file_list[1:], sample_rates_list[1:]):
        sync_offsets_msec.append(find_sync_offset(reference_first_sample_time_msec, clap_samples, file_sample_rate))"""

    # return sync_offsets_msec


def calc_sync_offsets(audio_paths: list[str], impulse_ramp_threshold: float = 0.4, detect_release_threshold_sec: float = 0.1, show_plots: bool = False) -> int:
    impulse_indices_per_file_list, sample_rates_list = extract_impulse_indices_of_files(audio_paths, impulse_ramp_threshold, detect_release_threshold_sec, show_plots)

    sync_offsets_msec = find_sync_offsets(impulse_indices_per_file_list, sample_rates_list)

    # print(f"sync_offset_msec = {sync_offsets_msec[0]}")
    # print(f"sync_offset_timestamp = {msec_to_timestamp(sync_offsets_msec[0])}")

    return sync_offsets_msec


def calc_sync_offset(audio_path: str, impulse_ramp_threshold: float = 0.4, detect_release_threshold_sec: float = 0.1, show_plots: bool = False):
    return calc_sync_offsets([audio_path], impulse_ramp_threshold, detect_release_threshold_sec, show_plots)[0]


def add_generic_optional_parser_arguments(parser: argparse.ArgumentParser):

    # Impulse/clap detection options
    parser.add_argument('-it', '-ithreshold', '--impulse_ramp_threshold', '--impulse_threshold', type=float, help="Threshold of the loudness change that must have been at least occured for a sound/ramp to be marked as clap/impulse [0.0, 1.0]", default=0.4)
    parser.add_argument('-ir', '-irelease', '--impulse_release_delay', '--impulse_release', type=float, help="Time in seconds from which another impulse can be detected after detecting an impulse in the audio data", default=0.1)
    parser.add_argument('-plot', '--show_plot', '--plot_impulses', action="store_true", help="Show plots of the impulse/clap detection responses")


def main():

    parser = argparse.ArgumentParser(
        description="Synchronize external audio with camera video, select only song range and apply sound processing effects in one go. Automatized postprocessing"
    )

    parser.add_argument('audio_src_paths', nargs='+', help="The audio files where the synchronisation offset between them should be calculated (the first audio file is used as reference for the offset of the other ones)")
    # parser.add_argument('audio_src_path', help="Audio file to be muxed/mixed and synced with the video data of the 'video_src_path' file")
    add_generic_optional_parser_arguments(parser)

    args: argparse.Namespace = parser.parse_args()

    sync_offsets = calc_sync_offsets(args.audio_src_paths, args.impulse_ramp_threshold, args.impulse_release_delay, args.show_plot)
    print(sync_offsets)


if __name__ == '__main__':
    sys.exit(main())


"""def find_remux_sync_offset_msec(audio_file_path: str, video_file_path: str, options: dict[str, Any]):

    claps_per_file_list, sample_rates_list = extract_clap_indices_of_files([audio_file_path, video_file_path])

    sync_offsets_msec = find_sync_offsets(claps_per_file_list, sample_rates_list)

    # print(f"sync_offset_msec = {sync_offsets_msec[0]}")
    # print(f"sync_offset_timestamp = {msec_to_timestamp(sync_offsets_msec[0])}")

    if (not options.keep_transient_files):
        delete_video_extracted_audio(video_file_path)

    return sync_offsets_msec[0]
"""
