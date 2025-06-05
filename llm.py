import os
import re
os.environ["TOGETHER_API_KEY"] = "383ee051b2cbb563ac497a4dab58b3031607fbef7dc334a07e50fa12585231c1"
from together import Together

client = Together()

def message():
    response=client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
        {
            "role": "user",
            "content": "recite a 5 line song for me"
        }
        ]
    )
    raw_text=response.choices[0].message.content
    return sanitize_text(raw_text)

def sanitize_text(text):
    text=text.strip()
    if text in{'"', "''", '""'}:
        return ""
    text=re.sub(r'["]{2,}', '"', text)
    text=text.replace('"', '')
    return text.strip()

