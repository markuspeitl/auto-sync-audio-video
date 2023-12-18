
import argparse
import os
import sys
from typing import Any
from os.path import dirname, join
from audio_sync_offset_calc import add_generic_optional_parser_arguments as add_sync_calc_generic_optional_parser_arguments, calc_sync_offset, calc_sync_offsets
from ffmpeg_processing import remux_video_audio_with_offset


"""def sync_video_audio_remux(video_path: str, audio_path: str, sync_offset_msec: int = -1, impulse_detection_options):

    if (sync_offset_msec < 0):
        sync_offset_msec = calc_sync_offset(video_path, audio_path, args.impulse_ramp_threshold, args.impulse_release_delay, args.show_plot)


    if (not args.disable_video_remuxing):

        # TODO song range timestamps have to consider the offset as we are truncating the output of the synced merge
        # args.on_range_timestamps = ('00:03:33.000', '00:09:24.000')
        # args.remux_sync_offset_msec = -3652
        remuxed_video_file_path = remux_video_audio_with_offset(video_file_path, processed_audio_file_path, args.remux_sync_offset_msec, args.on_range_timestamps)

        if (not args.keep_transient_files and processed_audio_file_path != audio_file_path):
            os.unlink(processed_audio_file_path)
"""


def add_generic_optional_parser_arguments(parser: argparse.ArgumentParser):

    parser.add_argument('-nsync', '--no_impulse_sync', '--no_sync', action="store_true", help="Do not detect the positions of claps in the video and audio file for the purpose of synchronizing those 2 recordings/files")
    parser.add_argument('-nmux', '--no_video_remuxing', '--no_remux', action="store_true", help="Disable remuxing stage where the audio stream of the passed video is replaced by the processed audio")

    # parser.add_argument('-p', '--play_on_finish', '--play',                         action="store_true", help="Play video when finished with processing")

    parser.add_argument('-p', '--play_on_finish', '--play', action="store_true",  help="")
    parser.add_argument('-pbin', '--play_on_finish_binary', '--play_bin', help="Use this application to play back the processed audio, when the fx chains have been applied", default='/usr/bin/mpv')
    parser.add_argument('-k', '--keep_transient_files', '--keep',                   action="store_true", help="Keep files that were the results of processing or extraction, but can be recalculated from the source files if needed")


def main():

    parser = argparse.ArgumentParser(
        description="Synchronize external audio with camera video, select only song range and apply sound processing effects in one go. Automatized postprocessing"
    )

    parser.add_argument('video_src_path', help="The audio files where the synchronisation offset between them should be calculated (the first audio file is used as reference for the offset of the other ones)")
    parser.add_argument('audio_src_path', help="The audio files where the synchronisation offset between them should be calculated (the first audio file is used as reference for the offset of the other ones)")

    add_sync_calc_generic_optional_parser_arguments(parser)
    add_generic_optional_parser_arguments(parser)

    # parser.add_argument('-range', '--on_range_timestamps', '--on_range', nargs='+', type=str, help="Manually specifies the range of interested which should be selected from the files (relative to the video file), 2 timestamps in the format HH:MM:SS.MSEC3DIGITS  ")
    parser.add_argument('-offset', '--remux_sync_offset_sec', '--sync_offset_sec', '--sync_offset', type=float, help="Manually set the offset between the 'video audio' and the 'external audio' for the purpose of syncing the 2")

    args: argparse.Namespace = parser.parse_args()

    if (args.no_impulse_sync):
        args.remux_sync_offset_sec = 0

    elif (not args.remux_sync_offset_sec):
        args.remux_sync_offset_sec = calc_sync_offset(args.video_src_path, args.audio_src_path, args.impulse_ramp_threshold, args.impulse_release_delay, args.show_plot)

    remuxed_video_file_path = None
    if (not args.no_video_remuxing):
        remuxed_video_file_path = remux_video_audio_with_offset(args.video_src_path, args.audio_src_path, args.remux_sync_offset_sec)

    if (args.play_on_finish and args.play_on_finish_binary and remuxed_video_file_path):
        os.system(f'{args.play_on_finish_binary} {remuxed_video_file_path}')


if __name__ == '__main__':
    sys.exit(main())
