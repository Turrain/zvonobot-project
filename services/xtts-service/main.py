import os
import torch
import numpy as np
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
from scipy.signal import resample
# Import the xTTS model and config
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download

# Global variables for the model and conditioning parameters
model = None
gpt_cond_latent = None
speaker_embedding = None


def convert_float32_to_pcm(input_audio, input_sample_rate=24000, target_sample_rate=16000):
    """
    Converts a numpy array of audio data from float32 format to int16 PCM format, with resampling.

    Parameters:
    - input_audio: numpy array with dtype=float32. Shape can be (n_samples,) for mono or (n_samples, 2) for stereo.
    - input_sample_rate: Sample rate of the input audio (e.g., 24000)
    - target_sample_rate: Desired sample rate for output audio (e.g., 8000 or 16000)

    Returns:
    - PCM encoded byte data of the resampled audio
    """
    # Ensure audio data is a numpy array
    if not isinstance(input_audio, np.ndarray):
        raise ValueError("Input audio must be a numpy array.")

    # Normalize the data to be in the range [-1.0, 1.0]
    input_audio = np.clip(input_audio, -1.0, 1.0)

    # Determine if the input is stereo or mono
    if len(input_audio.shape) == 1:
        # Mono input
        is_stereo = False
        input_audio = input_audio[:, np.newaxis]  # Convert to (n_samples, 1)
    elif len(input_audio.shape) == 2 and input_audio.shape[1] == 2:
        # Stereo input
        is_stereo = True
    else:
        raise ValueError("Input audio must be mono or stereo with shape (n_samples,) or (n_samples, 2).")

    # Resample the audio
    num_samples = int(len(input_audio) * target_sample_rate / input_sample_rate)
    if is_stereo:
        resampled_audio = np.zeros((num_samples, 2), dtype=np.float32)
        resampled_audio[:, 0] = resample(input_audio[:, 0], num_samples)
        resampled_audio[:, 1] = resample(input_audio[:, 1], num_samples)
    else:
        resampled_audio = resample(input_audio[:, 0], num_samples)
        resampled_audio = resampled_audio[:, np.newaxis]  # Convert back to 2D for mono

    # Convert to int16 PCM format
    pcm_audio = (resampled_audio * 32767).astype(np.int16)

    # Return as bytes
    return pcm_audio.tobytes()

##
def float_to_byte(sig):
    return float2pcm(sig, dtype='int16').tobytes()

def byte_to_float(byte):
    return pcm2float(np.frombuffer(byte, dtype=np.int16), dtype='float32')

def pcm2float(sig, dtype='float32'):
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)
##

# Sample rate of the audio output (adjust as per your model)
SAMPLE_RATE = 24000  # Hz

# Initialize FastAPI app
app = FastAPI(
    title="Streaming TTS Server",
    description="A server that generates speech from text inputs using xTTSv2 in streaming mode.",
    version="1.0.0"
)

# Allow CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Async function to download the model from Hugging Face Hub
async def download_model():
    print("Downloading model...")
    return snapshot_download("coqui/XTTS-v2")

# Async function to load the model
async def load_model(checkpoint_path):
    print("Loading model...")
    config_path = os.path.join(checkpoint_path, "config.json")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_path)
    model.cuda()
    return model

# Event handler for server startup
@app.on_event("startup")
async def startup_event():
    global model, gpt_cond_latent, speaker_embedding
    checkpoint_path = await download_model()
    model = await load_model(checkpoint_path)
    # Load conditioning latents (replace 'ref4.wav' with your reference audio file)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["ref4.wav"])
    print("Model is ready.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.
    Receives text input and sends back audio chunks as they are generated.
    """
    await websocket.accept()
    processing_task = None  # To keep track of the current processing task

    try:
        while True:
            # Receive data from the WebSocket client
            data = await websocket.receive_json()
            message = data.get('message', '')
            language = data.get('language', 'ru')  # Default to Russian
            speed = data.get('speed', 1.0)
            # Add other TTS settings as needed

            if not message:
                await websocket.send_json({"error": "No message provided."})
                continue

            # If a previous processing task is running, cancel it
            if processing_task and not processing_task.done():
                processing_task.cancel()

            # Start a new processing task
            processing_task = asyncio.create_task(
                process_tts_stream(message, language, speed, websocket)
            )

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({"error": f"Internal server error: {str(e)}"})
    finally:
        await websocket.close()
async def process_tts_stream(text, language, speed, websocket):
    """
    Asynchronously process TTS inference in streaming mode and send audio chunks via WebSocket.

    Args:
        text (str): Text input to be synthesized.
        language (str): Language code (e.g., 'ru' for Russian).
        speed (float): Speed of the speech.
        websocket (WebSocket): WebSocket connection to the client.
    """
    try:
        # Begin streaming inference
        chunks = model.inference_stream(
            text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=speed
        )
        print(f"Streaming TTS started for: {text}")
        for chunk in chunks:
            # Convert the chunk tensor to numpy array
            chunk_np = chunk.cpu().numpy().squeeze().astype('float32')
            
            # Convert numpy array to bytes
            audio_bytes = convert_float32_to_pcm(chunk_np, 24000, 8000)
            print(f"Generated audio chunk of size: {len(audio_bytes)} bytes")

            # Split and send the audio chunk in parts if it exceeds the size limit
            size_limit = 320
            for i in range(0, len(audio_bytes), size_limit):
                part = audio_bytes[i:i + size_limit]
                print(f"Sending audio part of size: {len(part)} bytes")
                await websocket.send_bytes(part)

        await websocket.send_json({"type": "end_of_audio"})
        print("End of audio chunks sent.")
        # Optional: Send progress updates or metadata
        # await websocket.send_json({"progress": "chunk_sent"})

    except asyncio.CancelledError:
        print("TTS streaming task was cancelled.")
    except Exception as e:
        await websocket.send_json({"error": f"Error during TTS processing: {str(e)}"})
@app.get("/languages")
async def get_languages():
    """
    HTTP endpoint to get the supported languages for TTS.
    """
    return {"languages": ["en", "ru", "es"]}

@app.post("/synthesize")
async def synthesize(request: Request):
    """
    HTTP endpoint for full speech synthesis.
    Receives text input and returns the full audio as a streaming response.

    Request body (JSON):
        { 
            "text": "Your text here",
            "language": "ru",
            "speed": 1.0
            // Add other TTS settings as needed
        }
    """ 
    data = await request.json()
    text = data.get('text', '')
    language = data.get('language', 'ru')  # Default to Russian
    speed = data.get('speed', 1.0)
    # Add other TTS settings as needed

    if not text:
        return {"error": "No text provided."}

    # Start the TTS inference in non-streaming mode
    try:
        # Generate the full speech audio
        with torch.no_grad():
            audio = model.inference(
                text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                speed=speed
            )

        # Convert the audio tensor to numpy array
        audio_np = audio.cpu().numpy().squeeze().astype('float32')

        # Convert numpy array to bytes 
        audio_bytes = audio_np.tobytes()

        # Return the audio as a streaming response
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav"  # Adjust media type as needed
        )

    except Exception as e:
        return {"error": f"Error during TTS processing: {str(e)}"}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 