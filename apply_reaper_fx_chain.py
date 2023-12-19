from enum import Enum
from os.path import join
import pathlib
import sys
import argparse
from pathlib import PurePath
import os
from os.path import join
from typing import Any

from path_util import add_name_suffix_path
from platform_bind_util import get_platform_store_value


class NormalizeModes(Enum):
    PEAK = 1
    TRUE_PEAK = 2
    LUFS_I = 3
    LUFS_S = 4
    LUFS_M = 5


class NormalizeWhen(Enum):
    ALWAYS = 0
    TOO_LOUD = 1


current_user_home = os.path.expanduser('~')

# Basic constants
fx_chain_extension = '.RfxChain'
reaper_processed_default_suffix = '_applied_fx'
reaper_batch_file_suffix = "_reaper_batch_processing_def.txt"


# Default variables
platform_values_store: dict[str, dict[str, str]] = {
    "default_fx_chains_dir": {
        'linux': join(current_user_home, ".config/REAPER/FXChains"),
        'osx': join(current_user_home, ".config/REAPER/FXChains"),
        'windows': join(current_user_home, 'AppData\Roaming\REAPER\FXChains')
    },
    "default_reaper_batch_list_target_dir": {
        'linux': "/dev/shm",
        'osx': "/tmp",
        'windows': join(current_user_home, 'AppData\Local\Temp')
    },
    "default_reaper_binary_location": {
        'linux': "/usr/local/bin/reaper",
        'osx': "/usr/local/bin/reaper",
        'windows': join(pathlib.Path.home().drive + ":", 'Program Files', 'REAPER (x64)', 'Reaper.exe')
    }
}

default_reaper_binary_location = get_platform_store_value('default_reaper_binary_location', platform_values_store)
default_fx_chains_dir = get_platform_store_value('default_fx_chains_dir', platform_values_store)
default_reaper_batch_list_target_dir = get_platform_store_value('default_reaper_batch_list_target_dir', platform_values_store)
default_fx_chain = "guitar-mix"
default_fx_chain_path = join(default_fx_chains_dir, default_fx_chain + fx_chain_extension)
default_output_norm_mode = NormalizeModes.TRUE_PEAK
default_output_norm_level = -3.0

wavpack_24_normal_render_string = "a3B2dwAAAAABAAAAAAAAAAAAAAA="
wav_32_pcm_render_string = 'ZXZhdyEAAQ=='
wav_24_pcm_render_string = 'ZXZhdxgAAQ=='
wav_16_pcm_render_string = 'ZXZhdxAAAQ=='
# flac at c 6 "ffmpeg -i 20231204_222341.mp4 -vn -compression_level 6 test_ffmpeg_enc_flac_6.flac" encodes at about 200x speed --> almost free compared to the effects
flac_24_c6_render_string = 'Y2FsZhgAAAAGAAAA'
flac_16_c6_render_string = 'Y2FsZhAAAAAGAAAA'

render_codec_mapping = {
    "flac_16": flac_16_c6_render_string,
    "flac_24": flac_24_c6_render_string,
    "wav_16_pcm": wav_16_pcm_render_string,
    "wav_24_pcm": wav_24_pcm_render_string,
    "wav_32_pcm": wav_32_pcm_render_string,
    "wavpack_32": wavpack_24_normal_render_string,
}


def get_container_extension_for_codec(codec_string):

    if (not codec_string):
        return '.wav'

    if ('flac' in codec_string):
        return '.flac'
    if ('wavpack' in codec_string):
        return '.wv'
    if ('wav' in codec_string):
        return '.wav'

    return '.wav'


def get_output_format_block(render_codec: str):
    if (not render_codec):
        return ""

    render_settings_string_id = render_codec_mapping[render_codec]

    if (not render_settings_string_id):
        raise Exception(f"No render string id found for codec key: {render_codec}")

    # To get this encoded string (denotes the rendering settings and options encoded in one string)
    # open a project, go to render, change render settings, click 'save settings', close render window, save project, open reaper project and find the 'RENDER_CFG' block where the encoded string is located
    output_format_render_block = f"""
        <OUTFMT
            {render_settings_string_id}
        >
    """

    return output_format_render_block


def get_batch_in_out_mapping_block(source_files: list[str], target_files: list[str]):

    batch_in_out_mapping_block = ""
    for source_file, target_file in zip(source_files, target_files):
        abs_source_file = os.path.realpath(source_file)
        abs_target_file = os.path.realpath(target_file)
        batch_in_out_mapping_block += abs_source_file + "\t" + abs_target_file + "\n"

    return batch_in_out_mapping_block


def get_batch_processing_file_content(source_files: list[str], target_files: list[str], fx_chain_path: str, norm_mode: int, norm_level: float, render_codec: str) -> str:

    # Call 'reaper -h' to find more possible options for this
    batch_processing_file_content = f"""
        {get_batch_in_out_mapping_block(source_files, target_files)}
        <CONFIG
        NORMALIZE	{norm_mode} {norm_level} {NormalizeWhen.ALWAYS.value}
        FXCHAIN	{fx_chain_path}
        USESRCMETADATA 1
        {get_output_format_block(render_codec)}
        >
    """

    return batch_processing_file_content


def get_option_or_default(key: str, default_value: Any, options: dict[str, Any]):
    if (key in options):
        return options[key]

    return default_value


def get_fx_chain_path_from_identifier(chain_identifier: str):

    fx_chain_path = chain_identifier
    if (not fx_chain_path.endswith(fx_chain_extension)):
        fx_chain_path += fx_chain_extension

    if (not os.path.isabs(fx_chain_path)):
        fx_chain_path = join(default_fx_chains_dir, fx_chain_path)

    return fx_chain_path


def perform_batch_processing(batch_file_target_path: str, fx_chain: str, source_audio_paths: str, target_audio_paths: str, normalization_mode: NormalizeModes, normalization_level: float, render_codec: str, reaper_binary_location: str):
    fx_chain_path = get_fx_chain_path_from_identifier(fx_chain)

    if (os.path.exists(batch_file_target_path)):
        os.unlink(batch_file_target_path)

    with open(batch_file_target_path, 'w+') as batch_file:

        batch_processing_file_content = get_batch_processing_file_content(source_audio_paths, target_audio_paths, fx_chain_path, normalization_mode.value, normalization_level, render_codec)

        batch_file.write(batch_processing_file_content)
        batch_file.flush()

    os.system(f"{reaper_binary_location} -batchconvert {batch_file_target_path}")

    if (os.path.exists(batch_file_target_path)):
        os.unlink(batch_file_target_path)
    if (os.path.exists(batch_file_target_path + ".log")):
        os.unlink(batch_file_target_path + ".log")

    return target_audio_paths


# Process multiple files multiple times, applying the fx chains in series (one after the other on the output of the last stage)
def apply_audio_fx_chains(source_audio_paths: list[str], fx_chain_paths: list[str], target_audio_paths: list[str] = [], options: dict[str, Any] = None):

    if (not target_audio_paths):
        target_audio_paths = []

    if (not fx_chain_paths):
        fx_chain_paths = [default_fx_chain_path]

    normalization_mode = get_option_or_default('normalize_mode', default_output_norm_mode, options.__dict__)
    normalization_level = get_option_or_default('normalize_level', default_output_norm_level, options.__dict__)
    reaper_binary_location = get_option_or_default('reaper_binary_location', default_reaper_binary_location, options.__dict__)
    batch_list_target_dir = get_option_or_default('batch_list_target_dir', default_reaper_batch_list_target_dir, options.__dict__)

    if (len(target_audio_paths) > 0 and len(source_audio_paths) != len(target_audio_paths)):
        raise Exception("Invalid input dimensions, lengths of audio_file_paths and target_audio_file_paths do not match, either omit the target_audio_file_paths parameter or make sure they have the same dimensions")

    if (len(target_audio_paths) <= 0):
        for audio_path in source_audio_paths:
            suffixed_path = add_name_suffix_path(audio_path, reaper_processed_default_suffix, new_extension=get_container_extension_for_codec(options.render_codec))
            target_audio_paths.append(suffixed_path)

    first_source_name = PurePath(source_audio_paths[0]).stem
    batch_processing_definition_path = join(batch_list_target_dir, first_source_name + reaper_batch_file_suffix)

    for fx_chain_path in fx_chain_paths:
        processed_audio_paths: list[str] = perform_batch_processing(batch_processing_definition_path, fx_chain_path, source_audio_paths, target_audio_paths, normalization_mode, normalization_level, options.render_codec, reaper_binary_location)
        source_audio_paths = processed_audio_paths

    return target_audio_paths


def apply_audio_fx_chains_single(source_audio_path: str, fx_chain_paths: list[str], target_audio_path: str = None, options: dict[str, Any] = None):
    target_paths = []
    if (target_audio_path):
        target_paths = [target_audio_path]
    return apply_audio_fx_chains([source_audio_path], fx_chain_paths, target_paths, options)


def apply_fx_chain_to_single(source_audio_path: str, fx_chain_path: str, target_audio_path: str = None, options: dict[str, Any] = None):
    apply_audio_fx_chains_single(source_audio_path, [fx_chain_path], target_audio_path, options)


# Process passed files multiple times, each time with a different fx_chain
def render_fx_candidates(source_audio_paths: list[str], fx_chain_paths: list[str], options: dict[str, Any] = None):

    fx_chain_to_output_paths: dict[str, list[str]] = {}

    for fx_chain_path in fx_chain_paths:

        fx_chain_path_obj = PurePath(fx_chain_path)
        fx_chain_id = fx_chain_path_obj.stem

        target_audio_paths: list[str] = list(map(lambda source_audio_path: add_name_suffix_path(source_audio_path, '_' + fx_chain_id), source_audio_paths))

        processed_audio_paths = apply_audio_fx_chains(source_audio_paths, [fx_chain_path], target_audio_paths, options)

        fx_chain_to_output_paths[fx_chain_id] = processed_audio_paths

    return fx_chain_to_output_paths


def main():

    parser = argparse.ArgumentParser(
        description="Apply reaper fx chain .RfxChain on the selected audio and write result to disk"
    )

    parser.add_argument('source_audio_paths', nargs='+', help="Audio file to be processed by fx chain (can be a video file with an audio stream as well)")
    parser.add_argument('-out', '-targets', '--target_audio_paths', '--processed_paths', nargs='+', help="Path of the output file to be created, (adds a suffix to input file by default if not specified)")
    parser.add_argument('-fx', '-fxc', '--audio_effect_chains', '--fx_chains', nargs='+', type=str, help="Reaper audio effects chains to be applied to the 'external audio' before muxing into the video, are applied one after the other", default=default_fx_chain)
    parser.add_argument('-r', '-rc', '-codec', '--render_codec', '--render_encoder', type=str, help="Specify the lossless codec and bit depth of the output, when the file will stay around for a time -> use flac: flac_16, flac_24, wav_16_pcm, wav_24_pcm, wav_32_pcm, wavpack_24 ", default=None)

    parser.add_argument('-nm', '--normalize_mode', type=float, help="Audio output normalization type: 1=peak, 2=true peak, 3=lufs-i, 4=lufs-s, 5=lufs-m", default=default_output_norm_mode)
    parser.add_argument('-nl', '--normalize_level', type=float, help="Audio output normalization level in dB [-inf, -0.0] dB", default=default_output_norm_level)

    parser.add_argument('-p', '--play_on_finish_binary', '--play_bin', help="Use this application to play back the processed audio, when the fx chains have been applied")
    parser.add_argument('-mult', '--multi_fx_candidates', '--mult_candidates', action="store_true", help="Render one output candidate for each specified fx_chain per input file, instead of applying all fx_chains on the same input file")

    # Compability patching options (adapting to uncommon configurations)
    parser.add_argument('-reaper', '--reaper_binary_location', help="Location of the reaper binary to process the audio batch with", default=default_reaper_binary_location)
    parser.add_argument('-batch_dir', '--batch_list_target_dir', help="Location of the directory to store the reaper batch file in, which is needed to run the processing when calling reaper binary", default=default_reaper_batch_list_target_dir)

    args: argparse.Namespace = parser.parse_args()

    if (args.multi_fx_candidates):
        if (args.target_audio_paths and len(args.target_audio_paths) > 0):
            raise Exception("'--target_audio_paths' option is not allowed when '--multi_fx_candidates' is enabled")

        fx_chain_to_output_paths = render_fx_candidates(args.source_audio_paths, args.audio_effect_chains, options=args.__dict__)
        print(fx_chain_to_output_paths)
    else:

        processed_audio_file_paths = apply_audio_fx_chains(args.source_audio_paths, args.audio_effect_chains, args.target_audio_paths, options=args.__dict__)

        if (args.play_on_finish_binary):
            processed_audio_files_line_list = " ".join(processed_audio_file_paths)
            os.system(f"{args.play_on_finish_binary} {processed_audio_files_line_list}")

        processed_audio_files_list = "\n".join(processed_audio_file_paths)
        print(processed_audio_files_list)


if __name__ == '__main__':
    sys.exit(main())
