import streamlit as st
import pandas as pd
import numpy as np
import llm
import requests

TTS_API_URL = "http://localhost:5005/tts"

st.title('Voice Cloning Model')

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
if "disabled" not in st.session_state:
    st.session_state.disabled = False
if "placeholder" not in st.session_state:
    st.session_state.placeholder = "Type something..."

text_input = st.text_input(
    "Enter prompt for LLM 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder=st.session_state.placeholder,
)

if text_input:
    st.write("Input accepted: ", text_input)
    st.write("Sending input to LLM...")

    output = llm.message(text_input)
    st.write("Answer generated: ", output)

    st.write("Sending to TTS API...")

    try:
        response = requests.post(TTS_API_URL, json={"text": output})
        if response.status_code == 200:
            st.success("Your audio has been cloned.")
            st.audio("output.wav") 
        else:
            st.error(f"TTS Error: {response.json().get('error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
