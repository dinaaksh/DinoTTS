from TTS.api import TTS
from core.stages.stage_02 import llm
from core.utils.common import read_yaml
from core.utils.constants import *
from pathlib import Path

class tts:
    def clone():
        # config=read_yaml(CONFIG_FILE_PATH)
        # model_path=Path(config["MODEL_PATH"]).as_posix()
        # config_path=Path(config["CONFIG_PATH"]).as_posix()
        # generated_speech = TTS(model_path=model_path,config_path=config_path, progress_bar=False)
        generated_speech=TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)
        text=llm.message()
        generated_speech.tts_to_file(text=text, file_path="output.wav")

'''
if error is thrown-> AttributeError: 'TTS' object has no attribute 'is_multi_lingual'
replace-> self.config = load_config(config_path) if config_path else None
with-> self.config = None
in environment's-> lib/site-packages/TTS/api.py at line 65'''