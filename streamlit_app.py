import streamlit as st
import websocket
import json
import base64
import pyaudio
import threading
import os
from pydub import AudioSegment
from pydub.playback import play
import io
import tempfile

# Initialize PyAudio
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000

p = pyaudio.PyAudio()

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
    audio = AudioSegment.from_raw(io.BytesIO(audio_data), sample_width=2, frame_rate=24000, channels=1)
    play(audio)

def record_audio():
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    st.write("Recording... Press 'Stop Recording' when finished.")
    frames = []
    
    stop_recording = st.button("Stop Recording")
    
    while not stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)
        stop_recording = st.button("Stop Recording")
    
    st.write("Recording stopped.")
    stream.stop_stream()
    stream.close()
    
    audio_data = b''.join(frames)
    return audio_data

def send_audio(audio_data):
    base64_audio = base64.b64encode(audio_data).decode('utf-8')
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
        if st.button("Start Recording"):
            audio_data = record_audio()
            send_audio(audio_data)

        for role, message in st.session_state.conversation:
            st.write(f"{role.capitalize()}: {message}")

if __name__ == "__main__":
    main()