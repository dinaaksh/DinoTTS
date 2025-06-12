import os
import re
from together import Together

TOGETHER_API_KEY=os.getenv('TOGETHER_API_KEY')

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