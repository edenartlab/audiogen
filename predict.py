# never push DEBUG_MODE = True to Replicate!
DEBUG_MODE = False
#DEBUG_MODE = True

import os
import time
import random
import sys
import json
import tempfile
import requests
import subprocess
import signal
from typing import Iterator, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from cog import BasePredictor, BaseModel, File, Input, Path as cogPath

import torch
import torchaudio

from audiocraft.data.audio import audio_write
from audiocraft.models import AudioGen, MusicGen

load_dotenv()

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"


if DEBUG_MODE:
    debug_output_dir = "/src/tests/server/debug_output"
    if os.path.exists(debug_output_dir):
        shutil.rmtree(debug_output_dir)
    os.makedirs(debug_output_dir, exist_ok=True)

class CogOutput(BaseModel):
    files: Optional[list[cogPath]] = []
    name: Optional[str] = None
    thumbnails: Optional[list[cogPath]] = []
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False


import subprocess

def convert_wav_to_mp3(wav_file_path, mp3_file_path, bitrate='192k'):
    """
    Converts a WAV file to an MP3 file using FFmpeg.

    Args:
    wav_file_path (str): The path to the input WAV file.
    mp3_file_path (str): The path to the output MP3 file.
    bitrate (str): The bitrate of the output MP3 file. Default is 192k.
    """
    try:
        subprocess.run(['ffmpeg', '-i', wav_file_path, '-ab', bitrate, mp3_file_path], check=True)
        print(f"Conversion complete: {mp3_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


#https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md
MODEL_INFO = {
    'facebook/audiogen-medium':  '1.5B model, text to sound - 🤗 Hub',
    'facebook/musicgen-medium':  '1.5B model, text to music - 🤗 Hub',
    'facebook/musicgen-large':   '3.3B model, text to music - 🤗 Hub',
    #'facebook/musicgen-melody':  '1.5B model, text to music and text+melody to music - 🤗 Hub',
    #'facebook/musicgen-melody-large':   '3.3B model, text to music and text+melody to music - 🤗 Hub',
}

class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = cogPath if DEBUG_MODE else CogOutput

    def setup(self):
        print("cog:setup")

    def generate(self, model_name, text_input, desired_duration):

        if "musicgen" in model_name:
            model = MusicGen.get_pretrained(model_name)
        elif "audiogen" in model_name:
            model = AudioGen.get_pretrained(model_name)
        else:
            raise ValueError("model_name must contain 'musicgen' or 'audiogen'")
        
        desired_duration = min(desired_duration, model.max_duration)
        model.set_generation_params(duration=int(desired_duration))

        # Generate the audio:
        descriptions = [text_input] # just generate a single sample for now
        wav = model.generate(descriptions)
        wav = wav[0].cpu()

        return wav, model.sample_rate

    def predict(
        self,
        
        # Universal args
        model_name: str = Input(
            description="Model name", default="facebook/audiogen-medium",
            choices=MODEL_INFO.keys()
        ),
        # Create mode
        text_input: str = Input(
            description="Text description of the sound / music", default=None
        ),
        duration_seconds: float = Input(
            description="Duration of the audio in seconds",
            ge=1.0, le=120.0, default=10.0
        ),

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:

        t_start = time.time()
        for i in range(3):
            print("-------------------------------------------------------")

        if not text_input:
            raise ValueError("text_input is required")

        if model_name not in MODEL_INFO.keys():
            print(f"Invalid model_name: {model_name}")
            print(f"Valid options are:")
            print(MODEL_INFO)
            raise ValueError(f"Invalid audio model_name: {model_name}")

        print(f"cog:predict: {model_name}")

        wav, sample_rate = self.generate(model_name, text_input, duration_seconds)

        audio_write('/src/tmp_wav', wav, sample_rate, strategy="loudness", loudness_compressor=True)
        out_path = f"{str(int(time.time()))}_{model_name}_output.mp3"
        out_path = out_path.replace('/', "_")
        convert_wav_to_mp3('/src/tmp_wav.wav', out_path)
        print(f"Final audio saved to {out_path}")

        attributes = {
            "model_name": model_name,
            "model_info": MODEL_INFO[model_name],
            "duration_seconds": duration_seconds,
            "text_input": text_input,
            "job_time_seconds": time.time() - t_start,
        }

        if DEBUG_MODE:
            print(attributes)
            #shutil.copyfile(out_path, os.path.join(debug_output_dir, prediction_name + ".mp4"))
            yield out_path
        else:
            yield CogOutput(files=[cogPath(out_path)], name=text_input, thumbnails=[cogPath('/src/sound.png')], attributes=attributes, isFinal=True, progress=1.0)

        if DEBUG_MODE:
            print("--------------------------------")
            print("--- cog was in DEBUG mode!!! ---")
            print("--------------------------------")

        t_end = time.time()
        print(f"predict.py: done in {t_end - t_start:.2f} seconds")
