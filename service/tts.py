from TTS.api import TTS
import frontend
import re


def run(text_input):
    tts=TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)
    #tts = TTS(model_path="mlflow_artifacts/vits_ljspeech_v2/optimized_model.pth",config_path="mlflow_artifacts/vits_ljspeech_v2/config.json", progress_bar=False)
    text=text_input
    tts.tts_to_file(text=text, file_path="output.wav")

'''
if error is thrown-> AttributeError: 'TTS' object has no attribute 'is_multi_lingual'
replace-> self.config = load_config(config_path) if config_path else None
with-> self.config = None
in environment's-> lib/site-packages/TTS/api.py at line 65'''