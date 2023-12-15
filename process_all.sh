#! /usr/bin/env sh

DIR=/mnt/CRU1TB/recordingprojects


python3 detect_audio_claps.py $DIR/122-yemanja/20231204_214535.mp4 $DIR/122-yemanja/230412_0076S12.wav
python3 detect_audio_claps.py $DIR/floresta-calma-meu-pensamento/20231204_215946.mp4 $DIR/floresta-calma-meu-pensamento/20231204_215946.mp4
python3 detect_audio_claps.py $DIR/minha-estrella-guia/20231204_222341.mp4 $DIR/minha-estrella-guia/230412_0079S12.wav

#python3 detect_audio_claps.py $DIR/sound-of-light/20231204_212619.mp4 $DIR/sound-of-light/230412_0075S12.wav