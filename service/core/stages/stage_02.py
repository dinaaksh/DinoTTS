import os
import re
from core.utils.common import read_yaml
from core.utils.constants import *

config=read_yaml(CONFIG_FILE_PATH)
os.environ["TOGETHER_API_KEY"]=config["TOGETHER_API_KEY"]
from together import Together

client = Together()

class llm:
    def message():
        response=client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
            {
                "role": "user",
                "content": "what is an apple in two small sentences."
            }
            ]
        )
        raw_text=response.choices[0].message.content
        return llm.sanitize_text(raw_text)

    def sanitize_text(text):
        text=text.strip()
        if text in{'"', "''", '""'}:
            return ""
        text=re.sub(r'["]{2,}', '"', text)
        text=text.replace('"', '')
        return text.strip()