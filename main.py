from collections.abc import AsyncIterator  # from typing import asynccontextmanager
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import StreamingResponse

from models import generate_audio, generate_text, load_audio_model, load_text_model
from schemas import VoicePresets
from utils import audio_array_to_buffer

models = {} # global variable

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    models["text"] = load_text_model()

    yield

    models.clear()

app = FastAPI(lifespan=lifespan)

# Serving functions

@app.get("/generate/text")
def serve_language_model_controller(prompt: str) -> str:
    #pipe = load_text_model()
    output = generate_text(models["text"], prompt)
    return output

@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_audio_model_controller(
    prompt: str,
    preset: VoicePresets = "v2/en_speaker_1",
):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate), media_type="audio/wav"
    )