
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

import scipy.io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from apply_reaper_fx_chain import apply_audio_processing_chain
from audio_activity_range_detection import detect_song_time_range
from audio_impulse_detection import find_remux_sync_offset_msec
from ffmpeg_processing import extract_audio_from_video, remux_video_audio_with_offset
from path_util import add_name_suffix_path, get_extracted_audio_path
from time_conversion_util import msec_to_samples

from transcribe_audio import transcribe_audio_file

# sample_rate, original_data = wavfile.read("230412_0072S12.wav")

# print(f"number of channels = {original_data.shape[1]}")
# length = original_data.shape[0] / sample_rate
# print(f"samplerate = {sample_rate}")
# print(f"length = {length}s")
# print(original_data.dtype)

# pip3 install python-reapy

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
