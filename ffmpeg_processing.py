
import os
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from path_util import add_name_suffix_path, get_extracted_audio_path
from time_conversion_util import msec_to_timestamp, sec_to_timestamp

# import ffmpegio_plugin_numpy
# pip3 install ffmpegio
import ffmpegio

ffmpeg_binary_location = "ffmpeg"
ffprobe_binary_location = "ffprobe"
thumbnails_subdir_name = "thumbnails"
video_extracted_audio_codec = 'pcm_s24le'
remuxed_video_container = ".mkv"
thumbnail_format = '.jpg'


def extract_audio_from_video(video_path):

    extracted_audio_path = get_extracted_audio_path(video_path)

    if (not os.path.exists(extracted_audio_path)):
        os.system(f"{ffmpeg_binary_location} -i {video_path} -n -vn -acodec {video_extracted_audio_codec} {extracted_audio_path}")

    return extracted_audio_path


def read_audio_ffmpeg(video_path) -> (float, np.ndarray):
    sample_rate, audio_data = ffmpegio.audio.read(video_path)
    return sample_rate, audio_data


def add_cmd_option(cmd_args_array: list[str], option_key: str, option_val: Any, skip_value=None, condition: bool = True):
    if (option_val == skip_value or not condition):
        return

    if (option_key):
        cmd_args_array.append(option_key)

    if (option_val):
        cmd_args_array.append(option_val)


def add_cmd_flag(cmd_args_array: list[str], flag_key: str, condition: bool = True):
    if (condition):
        cmd_args_array.append(flag_key)


def add_cmd_args(cmd_args_array: list[str], args_to_append: list[str]):
    cmd_args_array += args_to_append


def copy_streams_options():
    codec_args = [
        '-c:v',
        'copy',
        '-c:a',
        'copy',
    ]
    return codec_args


def trim_media_output(src_file_path: str, target_file_path: str, output_range_timestamps: tuple[str] = None, overwrite_output=False):
    if (not output_range_timestamps):
        output_range_timestamps = (None, None)

    cmd_args = [ffmpeg_binary_location]

    add_cmd_option(cmd_args, '-i', src_file_path)
    add_cmd_args(cmd_args, copy_streams_options())

    add_cmd_option(cmd_args, '-ss', output_range_timestamps[0])
    add_cmd_option(cmd_args, '-to', output_range_timestamps[1])
    add_cmd_flag(cmd_args, '-y', condition=overwrite_output)
    add_cmd_option(cmd_args, None, target_file_path)
    full_command_string = " ".join(cmd_args)
    print(full_command_string)
    os.system(full_command_string)
    return target_file_path


def remux_video_audio_with_offset(video_file_path: str, audio_file_path: str, target_video_path: str = None, audio_offset_sec: float = 0.0, output_range_timestamps: tuple = None, overwrite_output=False):
    # remuxed_video_path = add_name_suffix_path(video_file_path, "_remuxed", ".mp4")

    if (not output_range_timestamps):
        output_range_timestamps = (None, None)

    # MKV container can hold both, normal pcm audio and hevc, h264, etc.
    # Which is great because we are not forced to reencode video audio, to mix our streams together into a container, like we would need to do with an mp4 container

    remuxed_video_path = target_video_path
    if (not target_video_path):
        remuxed_video_path = add_name_suffix_path(video_file_path, "_remuxed", remuxed_video_container)

    audio_offset_string = sec_to_timestamp(abs(audio_offset_sec))

    cmd_args = [ffmpeg_binary_location]

    add_cmd_option(cmd_args, '-ss', audio_offset_string, condition=bool(audio_offset_sec < 0.0))

    add_cmd_option(cmd_args, '-i', video_file_path)

    add_cmd_option(cmd_args, '-ss', audio_offset_string, condition=bool(audio_offset_sec > 0.0))

    add_cmd_option(cmd_args, '-i', audio_file_path)

    add_cmd_args(cmd_args, copy_streams_options())

    mixing_muxing_args = [
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        '-shortest',
        # '-map_metadata',
        # '0'
    ]

    add_cmd_args(cmd_args, mixing_muxing_args)

    add_cmd_flag(cmd_args, '-y', condition=overwrite_output)

    add_cmd_option(cmd_args, '-ss', output_range_timestamps[0])
    add_cmd_option(cmd_args, '-to', output_range_timestamps[1])

    add_cmd_option(cmd_args, None, remuxed_video_path)

    """if (audio_offset_msec < 0):

        cmd_args.insert(1, '-ss')
        cmd_args.insert(2, audio_offset_string)

        # remux_command = f"{ffmpeg_binary_location} -ss {audio_offset_string} -i {video_file_path} -i {audio_file_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -ss {output_range_timestamps[0]} -to {output_range_timestamps[1]} {remuxed_video_path}"
    else:

        cmd_args.insert(3, '-ss')
        cmd_args.insert(4, audio_offset_string)

        # remux_command = f"{ffmpeg_binary_location} -i {video_file_path}  -ss {audio_offset_string} -i {audio_file_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -ss {output_range_timestamps[0]} -to {output_range_timestamps[1]} {remuxed_video_path}"

    # Example (video was started before audio): {ffmpeg_binary_location}  -ss 00:00:03.625 -i 20231204_204607.mp4 -i 230412_0072S12.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest remuxed_synced.mp4 -threads 6"""

    remux_command = " ".join(cmd_args)

    print(remux_command)

    os.system(remux_command)

    return remuxed_video_path

    # Audio starts before video -> sync clap is later in audio then in video ( -> audio needs to be trimmed at start, before syncing)

# '-vf',
# f"trim={output_range_timestamps[0]}:{output_range_timestamps[1]}",


match_duration = re.compile('(?<=duration=).+')


def get_ffprobe_meta_field(media_file_path: str, field_label: str):
    field_matcher = re.compile(f'(?<={field_label}=).+')

    cmd = f"{ffprobe_binary_location} -i {media_file_path} -show_format"

    # | grep -Po \"(?<=duration=).+\""

    print(cmd)

    # Note: there are cases where this will fail
    args = cmd.split(' ')

    video_format_meta = subprocess.check_output(args).decode(sys.stdout.encoding)
    field_match: re.Match = re.search(field_matcher, video_format_meta)
    matched_field_string = field_match.group(0)

    # duration_string = subprocess.check_output(args).decode(sys.stdout.encoding)
    return matched_field_string


def get_ffprobe_meta_tag(media_file_path: str, field_label: str):
    return get_ffprobe_meta_field(media_file_path, 'TAG:' + field_label)


def get_media_duration(media_file_path: str):
    # duration_string = subprocess.check_output(args).decode(sys.stdout.encoding)
    return float(get_ffprobe_meta_field(media_file_path, 'duration'))


def pull_thumbnail(video_file_path: str, target_thumbnail_path: str, at_timestamp: str):
    # thumbnail_path = add_name_suffix_path(video_file_path, "_thumbnail", ".jpg")

    os.system(f"{ffmpeg_binary_location} -ss {at_timestamp} -i {video_file_path} -y -vframes 1 {target_thumbnail_path}")

    return target_thumbnail_path


def pull_thumbnail_at_sec(video_file_path: str, target_thumbnail_path: str, seconds_time: float):
    return pull_thumbnail(video_file_path, target_thumbnail_path, sec_to_timestamp(seconds_time))


def pull_video_thumbnail_samples(video_file_path: str, thumbnail_cnt: int, clean_dir=False):

    if (thumbnail_cnt <= 0):
        return None

    if (not os.path.exists(video_file_path)):
        raise Exception(f"{video_file_path} does not exist -> can not create thumbnails of video")

    video_duration = get_media_duration(video_file_path)

    thumbnail_sec_timepoints = np.linspace(0.5, video_duration - 0.5, thumbnail_cnt).tolist()

    video_dir = os.path.dirname(video_file_path)
    thumbnails_dir = os.path.join(video_dir, thumbnails_subdir_name)

    if (os.path.exists(thumbnails_dir) and clean_dir and thumbnails_subdir_name in thumbnails_dir):
        shutil.rmtree(thumbnails_dir)

    os.makedirs(thumbnails_dir, exist_ok=True)

    # thumbnails_per_second = float(thumbnail_cnt / video_duration)
    # os.system(f"{ffmpeg_binary_location} -i {video_file_path} -ss 00.500 -threads 6 -vf fps={thumbnails_per_second} {join(thumbnails_dir, 'thumbnail_at_%03d.jpg')}")

    thumbnail_paths = []
    for index, time_point_seconds in enumerate(thumbnail_sec_timepoints):
        time_label = str(int(time_point_seconds))
        thumbnail_path = os.path.join(thumbnails_dir, 'thumbnail_at_' + time_label + thumbnail_format)
        thumbnail_path = pull_thumbnail_at_sec(video_file_path, thumbnail_path, time_point_seconds)

        thumbnail_paths.append(thumbnail_path)

    return thumbnail_paths


def apply_audio_effects(video_file_path):

    os.system(f"{ffmpeg_binary_location} -i {video_file_path} -af acompression=threshold=-21dB:ratio=4:attack=200:release=1000 {video_file_path}")


def normalize_audio():

    # Truepeak at -1.5dB, loudness normalization
    cmd = f"{ffmpeg_binary_location} -i input.mp3 -af loudnorm=TP=-1.5 output.mp3"
