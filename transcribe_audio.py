# Takes a couple of GB to install

# pip3 install SpeechRecognition
# https://github.com/Uberi/speech_recognition/blob/master/examples/audio_transcribe.py
# ((pip3 install pocketsphinx))

# pip3 install -U openai-whisper
# pip3 install soundfile

import sys
from pathlib import PurePath
import speech_recognition as sr

from path_util import add_name_suffix_path


def transcribe_audio_file(audio_file_path):
    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = r.record(source)  # read the entire audio file

    try:
        print("openai-whisper thinks you said ")
        transcription = r.recognize_whisper(audio, model="small", language="en")

        transcription_path = add_name_suffix_path(audio_file_path, '_transcription', '.txt')
        with open(transcription_path, 'w+') as transcription_file:
            transcription_file.write(transcription)

        print(transcription)
        return transcription

    except sr.UnknownValueError:
        print("openai-whisper could not understand audio")
    except sr.RequestError as e:
        print("openai-whisper error; {0}".format(e))

    return None


if __name__ == '__main__':
    sys.exit(transcribe_audio_file(sys.argv[1]))
