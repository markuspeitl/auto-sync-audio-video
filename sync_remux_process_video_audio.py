
import argparse
import os
import sys
import numpy as np
from os.path import dirname, join
import matplotlib.pyplot as plt
from apply_reaper_fx_chain import apply_audio_fx_chains, apply_audio_processing_chain
from audio_activity_range_detection import detect_song_time_range
from audio_impulse_detection import find_remux_sync_offset_msec
from ffmpeg_processing import pull_video_thumbnail_samples, remux_video_audio_with_offset
from transcribe_audio import transcribe_audio_file

video_player_binary = "mpv"


def add_generic_optional_parser_arguments(parser: argparse.ArgumentParser):

    # Impulse/clap detection options
    parser.add_argument('-ct', '--clap_loudness_threshold', '--clap_threshold',     type=float, help="Threshold of the loudness change that must have been at least occured for a sound/ramp to be marked as clap/impulse [0.0, 1.0]", default=0.4)
    parser.add_argument('-cr', '--clap_release_time', '--clap_release',             type=float, help="Time in seconds from which another impulse can be detected after detecting an impulse in the audi", default=0.1)
    parser.add_argument('-ploti', '--plot_impulse_detection', '--plot_impulses',    action="store_true", help="Show plots of the impulse/clap detection responses")
    parser.add_argument('-ns', '--no_clap_sync', '--no_sync',                       action="store_true", help="Do not detect the positions of claps in the video and audio file for the purpose of synchronizing those 2 recordings/files")

    # Song range/Audio activity range detection options
    parser.add_argument('-st', '--song_on_threshold', '--song_threshold',           type=float, help="Threshold of 'song_detection_block_size' window where activity is detected [0.0, 1.0]", default=0.1)
    parser.add_argument('-sb', '--song_detection_block_size', '--song_block_size',  type=float, help="Averaging window size for activity detection in seconds - all samples in each window are averaged to get an activity value over longer timescale", default=4.0)
    parser.add_argument('-rb', '--range_refine_block_size', '--refine_block_size',  type=float, help="Averaging window size for refining the the ranges, detected with the 'song_block_size' window (accuracy of final range result)", default=1.0)
    parser.add_argument('-sv', '--start_valley_threshold', '--start_valley',        type=float, help="Song range refinement valley detection threshold, range start will walk towards start of song until this sample value is reached", default=0.03)
    parser.add_argument('-ev', '--end_valley_threshold', '--end_valley',            type=float, help="Song range refinement valley detection threshold, range end will walk towards start of song until this sample value is reached", default=0.005)
    parser.add_argument('-spr', '--song_start_prerun', '--prerun',                  type=float, help="Time value in seconds added to the final detected start position after song range detection finished", default=-1.0)
    parser.add_argument('-spo', '--song_end_postrun', '--postrun',                  type=float, help="Time value in seconds added to the final detected end position after song range detection finished", default=0)
    parser.add_argument('-plots', '--plot_song_activity_detection', '--plot_activity',    action="store_true", help="Show plots of the song/activity detection responses")
    parser.add_argument('-nr', '--no_on_range_detection', '--no_on_range',          action="store_true", help="Do not automatically detect where the song/audio action start, ends and cut the output video to that range when muxing")

    parser.add_argument('-tn', '--extract_thumbnails_count', '--thumbnails_n',      type=int, help="Amount of thumbnails to pull from the video into a new sibling directory ./thumbnails ", default=0)
    parser.add_argument('-tk', '--keep_thumbnails', '--keep_thumbs',                action="store_true", help="Keep old thumbnails dir before extracting new thumbnails")

    parser.add_argument('-fxc', '--audio_effects_chain', '--fx_chain',              type=str, help="Reaper audio effects chain to be applied to the 'external audio' before muxing into the video", default='male-voice-mix')
    parser.add_argument('-np', '--no_chain_processing', '--no_processing',          action="store_true", help="Do not apply reaper FX chain onto the 'audio_src_path' before muxing/mixing")
    parser.add_argument('-dfx', '--disable_effects_chain', '--disable_fx',          action="store_true", help="Disable audio processing (applying effects chain on audio data)")

    parser.add_argument('-ta', '--create_audio_transcript', '--transcribe',         action="store_true", help="Create a transcript of the 'audio_src_path' file and store it next to that file as a .txt")

    parser.add_argument('-drm', '--disable_video_remuxing', '--disable_remux',      action="store_true", help="Disable remuxing stage where the audio stream of the passed video is replaced by the processed audio")

    parser.add_argument('-p', '--play_on_finish', '--play',                         action="store_true", help="Play video when finished with processing")
    parser.add_argument('-k', '--keep_transient_files', '--keep',                   action="store_true", help="Keep files that were the results of processing or extraction, but can be recalculated from the source files if needed")


def main():

    parser = argparse.ArgumentParser(
        description="Synchronize external audio with camera video, select only song range and apply sound processing effects in one go. Automatized postprocessing"
    )

    parser.add_argument('video_src_path', help="The video file that should be muxed with the audio_src_file, which audio stream is used for syncing and then discarded in favor if 'audio_src_path' file")
    parser.add_argument('audio_src_path', help="Audio file to be muxed/mixed and synced with the video data of the 'video_src_path' file")
    add_generic_optional_parser_arguments(parser)
    parser.add_argument('-range', '--on_range_timestamps', '--on_range', nargs='+', type=str, help="Manually specifies the range of interested which should be selected from the files (relative to the video file), 2 timestamps in the format HH:MM:SS.MSEC3DIGITS  ")
    parser.add_argument('-offset', '--remux_sync_offset_msec', '--sync_offset_msec', '--sync_offset', type=int, help="Manually set the offset between the 'video audio' and the 'external audio' for the purpose of syncing the 2")

    args: argparse.Namespace = parser.parse_args()

    if (args.on_range_timestamps and len(args.on_range_timestamps) > 2):
        raise Exception("--on_range can not have more than 2 timestamps: one for start and one for end")

    video_file_path = args.video_src_path
    audio_file_path = args.audio_src_path
    # video_file_path = "20231204_204607.mp4"
    # audio_file_path = "230412_0072S12.wav"

    if (not args.on_range_timestamps):
        args.on_range_timestamps = detect_song_time_range(video_file_path, args)

    if (not args.remux_sync_offset_msec):
        args.remux_sync_offset_msec = find_remux_sync_offset_msec(audio_file_path, video_file_path, args)

    processed_audio_file_path = audio_file_path

    if (not args.disable_effects_chain):
        # male-voice-mix, warmness-booster-effects
        processed_audio_file_path = apply_audio_fx_chains([audio_file_path], [args.audio_effects_chain])

    if (not args.disable_video_remuxing):

        # TODO song range timestamps have to consider the offset as we are truncating the output of the synced merge
        # args.on_range_timestamps = ('00:03:33.000', '00:09:24.000')
        # args.remux_sync_offset_msec = -3652
        remuxed_video_file_path = remux_video_audio_with_offset(video_file_path, processed_audio_file_path, args.remux_sync_offset_msec, args.on_range_timestamps)

        if (not args.keep_transient_files and processed_audio_file_path != audio_file_path):
            os.unlink(processed_audio_file_path)

    if (args.extract_thumbnails_count > 0):
        # remuxed_video_file_path = "20231204_204607_remuxed.mkv"
        thumbnail_paths = pull_video_thumbnail_samples(remuxed_video_file_path, args.extract_thumbnails_count, clean_dir=not args.keep_thumbnails)
        print("Extracted thumbnails: " + "\n".join(thumbnail_paths))

    if (args.create_audio_transcript):
        transcription = transcribe_audio_file(audio_file_path)
        print(transcription)

    if (args.play_on_finish):
        os.system(f'{video_player_binary} {remuxed_video_file_path}')


if __name__ == '__main__':
    sys.exit(main())
