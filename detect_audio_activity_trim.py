
import argparse
import os
import sys
from audio_activity_range_detection import detect_audio_activity_range_of_file, activity_detection_options_defaults
from dot_dict_conversion import to_dict
# from os.path import dirname, join
from ffmpeg_processing import trim_media_output
from path_util import add_name_suffix_path
import shutil

from play_on_finish import add_play_on_finish_arguments, play_if_enabled


def move_overwrite(src: str, dst: str):
    shutil.copy2(src, dst)
    if (os.path.exists(dst) and os.path.samefile(src, dst)):
        os.unlink(src)
        return True
    return False


def detect_audio_activity_trim(audio_src_path: str, args: argparse.Namespace):

    options: dict = to_dict(args)

    active_range_timestamps = options.get('active_range_timestamps')
    overwrite_output = options.get('overwrite_output')

    if (active_range_timestamps and len(active_range_timestamps) > 2):
        raise Exception("--on_range can not have more than 2 timestamps: one for start and one for end")

    if (not active_range_timestamps):
        active_range_timestamps = detect_audio_activity_range_of_file(audio_src_path, options)

    if (options.get('no_file_trimming')):
        return active_range_timestamps

    trimmed_file_path = add_name_suffix_path(audio_src_path, f"_trimmmed_{active_range_timestamps[0]}-{active_range_timestamps[1]}")
    trimmed_file_path = trim_media_output(audio_src_path, trimmed_file_path, active_range_timestamps, overwrite_output=overwrite_output)

    if (options.get('trim_src_inplace') and move_overwrite(trimmed_file_path,  audio_src_path)):
        trimmed_file_path = audio_src_path

    print(trimmed_file_path)
    return trimmed_file_path


def add_generic_optional_parser_arguments(parser: argparse.ArgumentParser):

    # Song range/Audio activity range detection options
    parser.add_argument('-rt', '--rough_active_threshold', '--active_threshold',
                        type=float, help="Threshold of 'rough_activity_block_size' window where activity is detected [0.0, 1.0]",
                        default=activity_detection_options_defaults['rough_active_threshold'])
    parser.add_argument('-rb', '--rough_activity_block_size', '--activity_block_size',
                        type=float, help="Averaging window size for activity detection in seconds - all samples in each window are averaged to get an activity value over longer timescale",
                        default=activity_detection_options_defaults['rough_activity_block_size'])
    parser.add_argument('-fb', '--fine_activity_block_size', '--fine_block_size',
                        type=float, help="Averaging window size for refining the the ranges, detected with the 'song_block_size' window (accuracy of final range result)",
                        default=activity_detection_options_defaults['fine_activity_block_size'])
    parser.add_argument('-sv', '--start_valley_threshold', '--start_valley',
                        type=float, help="Song range refinement valley detection threshold, range start will walk towards start of song until this sample value is reached",
                        default=activity_detection_options_defaults['start_valley_threshold'])
    parser.add_argument('-ev', '--end_valley_threshold', '--end_valley',
                        type=float, help="Song range refinement valley detection threshold, range end will walk towards start of song until this sample value is reached",
                        default=activity_detection_options_defaults['end_valley_threshold'])
    parser.add_argument('-spr', '--activity_start_prerun', '--prerun',
                        type=float, help="Time value in seconds added to the final detected start position after song range detection finished",
                        default=activity_detection_options_defaults['activity_start_prerun'])
    parser.add_argument('-epo', '--activity_end_postrun', '--postrun',
                        type=float, help="Time value in seconds added to the final detected end position after song range detection finished",
                        default=activity_detection_options_defaults['activity_end_postrun'])

    # parser.add_argument('-nr', '--no_on_range_detection', '--no_on_range',          action="store_true", help="Do not automatically detect where the song/audio action start, ends and cut the output video to that range when muxing")

    parser.add_argument('-ntrim', '--no_file_trimming', '--no_trimming',
                        action="store_true", help="Disable trimming stage where the input file is trimmed down to activity range")
    parser.add_argument('-inplace', '--trim_src_inplace', '--trim_inplace',
                        action="store_true", help="Trim 'audio_src_path' file in place, without creating a new file")

    # parser.add_argument('-k', '--keep_transient_files', '--keep',                   action="store_true", help="Keep files that were the results of processing or extraction, but can be recalculated from the source files if needed")

    parser.add_argument('-range', '--active_range_timestamps', '--active_range', nargs='+', type=str, help="Manually specifies the range of interested which should be selected from the files (relative to the video file), 2 timestamps in the format HH:MM:SS.MSEC3DIGITS  ")


def main():

    parser = argparse.ArgumentParser(
        description="Detect high energy/intensity audio activity, find where this activity start/ends and trim the file to fit this range only"
    )

    parser.add_argument('audio_src_path', help="Audio file to detect active on range from, can be a audio file or a video container containing an audio stream")
    parser.add_argument('-target', '--trim_target', help="Set to the file you want to be trimmed to the detected range (only set if you want to trim a different file than the 'audio_src_path')", default=None)

    parser.add_argument('-plot', '--show_plot', '--plot_activity',
                        action="store_true", help="Show plots of the song/activity detection responses")

    parser.add_argument('-y', '--overwrite_output', '--overwrite',
                        action="store_true", help="Overwrite the target trimmed media file if one exists with the same name")

    add_play_on_finish_arguments(parser)

    add_generic_optional_parser_arguments(parser)

    # parser.add_argument('-offset', '--remux_sync_offset_msec', '--sync_offset_msec', '--sync_offset', type=int, help="Manually set the offset between the 'video audio' and the 'external audio' for the purpose of syncing the 2")

    args: argparse.Namespace = parser.parse_args()

    detect_trim_result = detect_audio_activity_trim(args.audio_src_path, args)
    print(detect_trim_result)

    play_if_enabled(detect_trim_result, args)


if __name__ == '__main__':
    sys.exit(main())
