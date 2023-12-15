
from pathlib import PurePath
import os
from os.path import join

from path_util import add_name_suffix_path


def apply_reaper_effects_and_render():
    import reapy
    reapy.print("Hello world!")


def apply_audio_processing_chain(audio_file_path: str, processing_chain_path: str):

    audio_path_obj = PurePath(audio_file_path)
    batch_processing_chain_file = join('/dev/shm', audio_path_obj.stem + "_reaper_process_list.txt")

    abs_processed_audio_file_path = os.path.realpath(add_name_suffix_path(audio_file_path, '_chain_processed'))
    abs_audio_file_path = os.path.realpath(audio_file_path)

    true_peak_normalize_db = -3.0

    if (not processing_chain_path):
        processing_chain_path = "/home/pmarkus/.config/REAPER/FXChains/guitar-mix.RfxChain"

    if (not processing_chain_path.endswith('.RfxChain')):
        processing_chain_path += '.RfxChain'

    with open(batch_processing_chain_file, 'w+') as batch_file:
        batch_file.writelines([
            f"{abs_audio_file_path}\t{abs_processed_audio_file_path}\n",
            f"<CONFIG\n",
            f"NORMALIZE	2 {true_peak_normalize_db}\n",
            f"FXCHAIN	{processing_chain_path}\n",
            f">\n"])

        batch_file.flush()

    os.system(f"reaper -batchconvert {batch_processing_chain_file}")

    return abs_processed_audio_file_path
