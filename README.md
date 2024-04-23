# Auto audio video sync

Audio quality is important, in many cases of videos even more important than video quality.
The sound quality of a condenser microphone compared to small electret capsules like in many cameras or
MEMS sensors as in mobile phones is not quite comparable and it does not have the issue of picking up
sounds made by the camera (image stabilization motor, etc.) or phone (rumbling, handling noise).

The solution for me is frequently to use one device for recording video (dslr, phone, videocamera)
and another one for recording audio (statinary or mobile audio interface with a small or large condenser mic).

The problem is that syncing those 2 file and mixing them into a single file usually takes the usage of a
video cutting program and is just a simple step that is easy to perform, but annoying to do each time.
Also these prgrams take a lot of resources and might not properly perform when playing the video in full sizes on most computers.

Therefore i created these scripts that can automatize syncing a video (with embedded audio) together with an external audio
track and discarding the embedded audio track while cutting both to the size of the video (or the selected range of the video).
They even go 1 step further and can use REAPER to automatically apply an audio effects chain on the audio track,
therefore making it possible to generate multiple options for the videos and then just pick one to publish.

## How to use:
``python3 sync_remux_process_video_audio.py /home/my_video.mp4 /home/external_audio.wav``

## Partial usage:
### Full processing chain  
``python3 sync_remux_process_video_audio.py --help``  
### Sync audio and video  
``python3 video_sync_audio_remux.py --help``  
### Detecting active audio range  
 and trim file to range    
``python3 detect_audio_activity_trim.py --help``  
### Calculate the offset  
in msec between 2 audio files of the same sound source    
``python3 audio_sync_offset_calc.py --help``  
### Apply an audio effects  
chain using reaper and render to output file    
``python3 apply_reaper_fx_chain.py --help``  

## How does it work?
1. **Detect Impulses, Sync, remux:**  
Detects impulses in embedded audio and on the external track (using algorithmic deviation) and matches those impulses to find out the difference offset.
Then mixes the external audio with the video, discarding the embedded track in the process.
2. **Detect Activity Range (on remuxed video), truncate/trim:**  
Uses an averaging window and threshold to detect zones with high audio activity, then descends the start and end ramp to optimize
where the audio activity starts and where it end.
Through this the parts before and after we start talking can be discarded.
3. **Apply audio fx on the audio stream (of the video) and then remux:**  
Use reaper and apply the selected FX chain to the audio, after normalizing it.
Useful when not wanting to spend a lot of time on audio postprocessing and optimization.
4. **Pull thumbnails:**  
Use ffmpeg to pull n thumbnail images at even intervals from the video.
5. **(Transcribe audio) was more of an experiment:** 
does work somewhat even when utilizing machine learning transcription options.  
Probably works fine for just talking, look at `transcribe_audio.py` if you want to do this.
6. **Play final video using selected program:**  
Uses a system call to play the rendered video with a binary/program of choice

### Platforms:
Tested on Linux.
Should theoretically work on Windows as well. (if it does not this can likely be solved with configuring the used binaries manually)

Depending on used features depends on:
ffmpeg, mpv, reaper
numpy, matplotlib, .etc
