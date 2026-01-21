import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/generate"

def generate_voice(text):
    response = requests.post(
        API_URL,
        params={"text": text}
    )

    if response.status_code != 200:
        return None

    data = response.json()
    return data["audio_path"]

gr.Interface(
    fn=generate_voice,
    inputs=gr.Textbox(lines=2, placeholder="Enter dialogue..."),
    outputs=gr.Audio(type="filepath"),
    title="Sage Voice Generator"
).launch()
