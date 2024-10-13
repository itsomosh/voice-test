import streamlit as st
import websocket
import json
import base64
import threading
import os
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import io
import tempfile

# Audio settings
SAMPLE_RATE = 24000
CHANNELS = 1

# Streamlit app state
if 'ws' not in st.session_state:
    st.session_state.ws = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

def on_message(ws, message):
    event = json.loads(message)
    if event['type'] == 'response.output_item.added':
        if event['item']['type'] == 'message':
            for content in event['item']['content']:
                if content['type'] == 'text':
                    st.session_state.conversation.append(('assistant', content['text']))
                elif content['type'] == 'audio':
                    audio_data = base64.b64decode(content['audio'])
                    play_audio(audio_data)

def on_error(ws, error):
    st.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket connection closed")

def on_open(ws):
    st.success("Connected to OpenAI Realtime API")
    ws.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": "You are a helpful AI assistant. Respond concisely.",
        }
    }))

def connect_websocket():
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
        header={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1",
        },
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    return ws

def play_audio(audio_data):
    # For simplicity, we'll just save the audio to a file and use st.audio to play it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio.write(audio_data)
        temp_audio.flush()
        st.audio(temp_audio.name, format='audio/wav')

def record_audio(duration=5):
    st.write(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    st.write("Recording finished.")
    return audio_data

def send_audio(audio_data):
    # Convert float32 array to int16
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file in memory
    with io.BytesIO() as wav_buffer:
        wavfile.write(wav_buffer, SAMPLE_RATE, audio_data_int16)
        wav_buffer.seek(0)
        wav_data = wav_buffer.read()
    
    base64_audio = base64.b64encode(wav_data).decode('utf-8')
    st.session_state.ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_audio",
                "audio": base64_audio
            }]
        }
    }))

def main():
    st.title("OpenAI Realtime Voice Interaction")

    if st.session_state.ws is None:
        if st.button("Connect to OpenAI"):
            st.session_state.ws = connect_websocket()

    if st.session_state.ws:
        if st.button("Record Audio (5 seconds)"):
            audio_data = record_audio()
            send_audio(audio_data)

        for role, message in st.session_state.conversation:
            st.write(f"{role.capitalize()}: {message}")

if __name__ == "__main__":
    main()