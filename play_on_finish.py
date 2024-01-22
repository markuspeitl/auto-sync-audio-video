
import argparse
import os


def add_play_on_finish_arguments(parser: argparse.ArgumentParser, namespace: str = ""):

    parser.add_argument(f'-{namespace}p', f'--{namespace}play_on_finish', f'--{namespace}play',                         action="store_true",  help="Play media when finished with processing")
    parser.add_argument(f'-{namespace}pbin', f'--{namespace}play_on_finish_binary', f'--{namespace}play_bin',           help="Use this application to play back the media file", default='/usr/bin/mpv')


def get_play_options_state(args: argparse.Namespace, namespace: str = ""):
    args_dict: dict = vars(args)
    return args_dict.get(f"{namespace}play_on_finish"), args_dict.get(f"{namespace}play_on_finish_binary")


def is_playable(path: str):
    return os.path.exists(path)


def play_if_enabled(media_file_path: str, args: argparse.Namespace, namespace: str = ""):
    do_play, binary = get_play_options_state(args, namespace)

    if (not do_play or not binary or not is_playable(media_file_path)):
        return

    os.system(f'{binary} {media_file_path}')


def play_series_if_enabled(media_file_paths: list[str], args: argparse.Namespace, namespace: str = ""):
    for media_file_path in media_file_paths:
        play_if_enabled(media_file_path, args, namespace)


def play_all_if_enabled(media_file_paths: list[str], args: argparse.Namespace, namespace: str = ""):
    do_play, binary = get_play_options_state(args, namespace)
    if (not do_play or not binary):
        return

    playable_paths = filter(is_playable, media_file_paths)

    if (playable_paths and len(playable_paths) > 0):

        all_media_files_string = " ".join(media_file_paths)
        os.system(f'{binary} {all_media_files_string}')
