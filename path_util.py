from os.path import join
from pathlib import PurePath


def add_name_suffix_path(path, suffix, new_extension=None):
    path_obj = PurePath(path)

    extension = path_obj.suffix
    if (new_extension):
        extension = new_extension

    return join(path_obj.parent, path_obj.stem + suffix + extension)


def get_extracted_audio_path(video_path):
    return add_name_suffix_path(video_path, "_extracted_audio", ".wav")
