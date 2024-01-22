
import argparse
from enum import Enum
import os
import sys
from apply_reaper_fx_chain import add_generic_optional_parser_arguments as add_apply_fx_optional_parser_arguments, apply_video_audio_fx_chains
from ffmpeg_processing import pull_video_thumbnail_samples, remux_video_audio_with_offset
from path_util import add_name_suffix_path
from play_on_finish import add_play_on_finish_arguments, play_if_enabled
from transcribe_audio import transcribe_audio_file
from audio_sync_offset_calc import add_generic_optional_parser_arguments as add_impulse_detection_optional_parser_arguments
from video_sync_audio_remux import add_generic_optional_parser_arguments as add_sync_remux_optional_parser_arguments, video_sync_audio_remux
from detect_audio_activity_trim import add_generic_optional_parser_arguments as add_range_detection_optional_parser_arguments, detect_audio_activity_trim

# video_player_binary = "mpv"


class Stages(Enum):
    START = 0
    SYNC = 1
    TRIM = 2
    FX = 3
    THUMBNAILS = 4
    TRANSCRIBE = 5


stage_processing_results = {}

# def detect_stage_processing_results(source_video_file: str):


def add_generic_optional_parser_arguments(parser: argparse.ArgumentParser):

    # Impulse/clap detection options
    # parser.add_argument('-ct', '--clap_loudness_threshold', '--clap_threshold',     type=float, help="Threshold of the loudness change that must have been at least occured for a sound/ramp to be marked as clap/impulse [0.0, 1.0]", default=0.4)
    # parser.add_argument('-cr', '--clap_release_time', '--clap_release',             type=float, help="Time in seconds from which another impulse can be detected after detecting an impulse in the audi", default=0.1)
    # parser.add_argument('-ploti', '--plot_impulse_detection', '--plot_impulses',    action="store_true", help="Show plots of the impulse/clap detection responses")
    # parser.add_argument('-ns', '--no_clap_sync', '--no_sync',                       action="store_true", help="Do not detect the positions of claps in the video and audio file for the purpose of synchronizing those 2 recordings/files")

    # Song range/Audio activity range detection options
    # parser.add_argument('-st', '--rough_active_threshold', '--song_threshold',           type=float, help="Threshold of 'rough_activity_block_size' window where activity is detected [0.0, 1.0]", default=0.1)
    # parser.add_argument('-sb', '--rough_activity_block_size', '--song_block_size',  type=float, help="Averaging window size for activity detection in seconds - all samples in each window are averaged to get an activity value over longer timescale", default=4.0)
    # parser.add_argument('-rb', '--fine_activity_block_size', '--refine_block_size',  type=float, help="Averaging window size for refining the the ranges, detected with the 'song_block_size' window (accuracy of final range result)", default=1.0)
    # parser.add_argument('-sv', '--start_valley_threshold', '--start_valley',        type=float, help="Song range refinement valley detection threshold, range start will walk towards start of song until this sample value is reached", default=0.03)
    # parser.add_argument('-ev', '--end_valley_threshold', '--end_valley',            type=float, help="Song range refinement valley detection threshold, range end will walk towards start of song until this sample value is reached", default=0.005)
    # parser.add_argument('-spr', '--activity_start_prerun', '--prerun',                  type=float, help="Time value in seconds added to the final detected start position after song range detection finished", default=-1.0)
    # parser.add_argument('-spo', '--activity_end_postrun', '--postrun',                  type=float, help="Time value in seconds added to the final detected end position after song range detection finished", default=0)
    # parser.add_argument('-plots', '--plot_song_activity_detection', '--plot_activity',    action="store_true", help="Show plots of the song/activity detection responses")
    # parser.add_argument('-nr', '--no_on_range_detection', '--no_on_range',          action="store_true", help="Do not automatically detect where the song/audio action start, ends and cut the output video to that range when muxing")

    parser.add_argument('-tn', '--extract_thumbnails_count', '--thumbnails_n',      type=int, help="Amount of thumbnails to pull from the video into a new sibling directory ./thumbnails ", default=0)
    parser.add_argument('-tk', '--keep_thumbnails', '--keep_thumbs',                action="store_true", help="Keep old thumbnails dir before extracting new thumbnails")

    # parser.add_argument('-fxc', '--audio_effects_chain', '--fx_chain',              type=str, help="Reaper audio effects chain to be applied to the 'external audio' before muxing into the video", default='male-voice-mix')
    # parser.add_argument('-np', '--no_chain_processing', '--no_processing',          action="store_true", help="Do not apply reaper FX chain onto the 'audio_src_path' before muxing/mixing")
    # parser.add_argument('-dfx', '--disable_effects_chain', '--disable_fx',          action="store_true", help="Disable audio processing (applying effects chain on audio data)")

    # parser.add_argument('-ta', '--create_audio_transcript', '--transcribe',         action="store_true", help="Create a transcript of the 'audio_src_path' file and store it next to that file as a .txt")

    # parser.add_argument('-drm', '--disable_video_remuxing', '--disable_remux',      action="store_true", help="Disable remuxing stage where the audio stream of the passed video is replaced by the processed audio")

    parser.add_argument('-k', '--keep_transient_files', '--keep',                   action="store_true", help="Keep files that were the results of processing or extraction, but can be recalculated from the source files if needed")
    parser.add_argument('-ram', '--store_transient_ram', '--store_ram',             action="store_true", help="Attempt to store transient files, the output of the processing stages, which are not needed anymore after finishing processing to RAM (/dev/shm). (only on linux)")
    parser.add_argument('-plot', '--show_plot', '--plot_activity',
                        action="store_true", help="Show plots of the song/activity detection responses")

    # parser.add_argument('-ns', '--no_video_audio_sync', '--no_sync',                action="store_true", help="Do not detect the positions of impulses in the video and audio file and sync remux video and audio synchronizing those 2 recordings/files")
    parser.add_argument('-nt', '--no_activity_trim', '--no_trim',                   action="store_true", help="Do not automatically detect where the audio action start, ends and cut the output video to that range when muxing")


def main():

    parser = argparse.ArgumentParser(
        description="Synchronize external audio with camera video, select only song range and apply sound processing effects in one go. Automatized postprocessing"
    )

    parser.add_argument('video_src_path', help="The video file that should be muxed with the audio_src_file, which audio stream is used for syncing and then discarded in favor if 'audio_src_path' file")
    parser.add_argument('audio_src_path', help="Audio file to be muxed/mixed and synced with the video data of the 'video_src_path' file")
    add_generic_optional_parser_arguments(parser)
    # parser.add_argument('-range', '--on_range_timestamps', '--on_range', nargs='+', type=str, help="Manually specifies the range of interested which should be selected from the files (relative to the video file), 2 timestamps in the format HH:MM:SS.MSEC3DIGITS  ")
    # parser.add_argument('-offset', '--remux_sync_offset_sec', '--sync_offset_msec', '--sync_offset', type=int, help="Manually set the offset between the 'video audio' and the 'external audio' for the purpose of syncing the 2")

    # add_impulse_detection_optional_parser_arguments(parser)
    add_sync_remux_optional_parser_arguments(parser)
    add_range_detection_optional_parser_arguments(parser)
    add_apply_fx_optional_parser_arguments(parser)
    add_play_on_finish_arguments(parser)

    parser.add_argument('-y', '--overwrite_output', '--overwrite',                  action="store_true", help="Overwrite output media if it exists")
    parser.add_argument('-stage', '--continue_at_stage', '--continue_at',           type=int, help="Detect existing transient files for previous stages and continue processing at the specified stage: continue after 0=start, 1=syncing, 2=trimming, 3=audio_fx, 4=thumnails, 5=transcribe", default=Stages.START.value)

    args: argparse.Namespace = parser.parse_args()

    if (args.active_range_timestamps and len(args.active_range_timestamps) > 2):
        raise Exception("--on_range can not have more than 2 timestamps: one for start and one for end")

    # Requires a bit more time -> 2x remux, but makes the stages stand on their own (manual intervention in single steps becomes possible)
    # Here: 'Syncing' is mostly solid, 'Detect Activity Range' might need manual parameter tuning, 'Apply fx' does not require tunable parameters

    # 1. Detect Impulses, Sync, remux

    if (args.no_impulse_sync):
        synced_video_path = args.video_src_path
    else:
        synced_video_path: str = video_sync_audio_remux(args.video_src_path, args.audio_src_path, args)

    # 2. Detect Activity Range (on remuxed video), truncate/trim

    if (args.no_activity_trim):
        activity_trimmed_video_path = synced_video_path
    else:
        activity_trimmed_video_path: str = detect_audio_activity_trim(synced_video_path, args)

    # 3. Apply audio fx on the audio stream (of the video) and then remux

    # final_remuxed_video = activity_trimmed_video_path

    if (args.audio_effect_chains):
        final_remuxed_video, processed_audio_file_paths = apply_video_audio_fx_chains(
            activity_trimmed_video_path,
            args.audio_effect_chains,
            options=args)
    else:
        final_remuxed_video = activity_trimmed_video_path

    # activity_trimmed_processed_audio_path = add_name_suffix_path(activity_trimmed_video_path, '_fx_chains_processed', )
    # processed_audio_file_paths = apply_audio_fx_chains([activity_trimmed_video_path], args.audio_effect_chains, options=args.__dict__)
    # final_remuxed_video = remux_video_audio_with_offset(activity_trimmed_video_path, processed_audio_file_paths[0], 0)

    if (not args.keep_transient_files):
        if (os.path.exists(synced_video_path)):
            os.unlink(synced_video_path)

        if (os.path.exists(synced_video_path)):
            os.unlink(synced_video_path)

        for processed_audio in processed_audio_file_paths:
            if (os.path.exists(processed_audio)):
                os.unlink(processed_audio)

        # os.unlink(final_remuxed_video)

    # 4. Pull thumbnails

    if (args.extract_thumbnails_count > 0):
        # final_remuxed_video = "20231204_204607_remuxed.mkv"
        thumbnail_paths = pull_video_thumbnail_samples(final_remuxed_video, args.extract_thumbnails_count, clean_dir=not args.keep_thumbnails)
        print("Extracted thumbnails: " + "\n".join(thumbnail_paths))

    # 5. Transcribe audio

    # if (args.create_audio_transcript):
    #    transcription = transcribe_audio_file(activity_trimmed_video_path)
    #    print(transcription)

    # 6. Pull images (use some of the transcription for stable diffusion, or use thumbnails for sd, do an image search with keywords)
    # 7. Upload to youtube
    # 8. Open upload page
    # 9. Play final video

    play_if_enabled(final_remuxed_video, args)

    print("Finished processing -> final video:")
    print(final_remuxed_video)

    # TODO Ram mode -> put temporary files on /dev/shm

    """video_file_path = args.video_src_path
    audio_file_path = args.audio_src_path
    # video_file_path = "20231204_204607.mp4"
    # audio_file_path = "230412_0072S12.wav"

    if (not args.on_range_timestamps):
        args.on_range_timestamps = detect_song_time_range(video_file_path, args)

    if (not args.remux_sync_offset_sec):
        args.remux_sync_offset_sec = remux_video_audio_with_offset(video_file_path, audio_file_path, args.remux_sync_offset_sec, args)

    processed_audio_file_path = audio_file_path

    if (not args.disable_effects_chain):
        # male-voice-mix, warmness-booster-effects
        processed_audio_file_path = apply_audio_fx_chains([audio_file_path], [args.audio_effects_chain])

    if (not args.disable_video_remuxing):

        # TODO song range timestamps have to consider the offset as we are truncating the output of the synced merge
        # args.on_range_timestamps = ('00:03:33.000', '00:09:24.000')
        # args.remux_sync_offset_sec = -3652
        remuxed_video_file_path = remux_video_audio_with_offset(video_file_path, processed_audio_file_path, args.remux_sync_offset_sec, args.on_range_timestamps)

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
        os.system(f'{video_player_binary} {remuxed_video_file_path}')"""


if __name__ == '__main__':
    sys.exit(main())
