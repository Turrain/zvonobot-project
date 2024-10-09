from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import asyncio
import logging
import numpy as np
import webrtcvad
from fastapi.middleware.cors import CORSMiddleware
import torch
from scipy.signal import resample


def convert_pcm8_to_float32(pcm_data, input_sample_rate=8000, target_sample_rate=24000, is_stereo=False):
    # Convert PCM bytes to int8 numpy array
    pcm_audio = np.frombuffer(pcm_data, dtype=np.int8)

    # Reshape if the input is stereo
    if is_stereo:
        pcm_audio = pcm_audio.reshape((-1, 2))

    # Normalize the PCM data to be in the range [-1.0, 1.0]
    float_audio = pcm_audio.astype(np.float32) / 127.0  # Max value for int8 is 127

    # Resample the audio if necessary
    if input_sample_rate != target_sample_rate:
        num_samples = int(len(float_audio) * target_sample_rate / input_sample_rate)
        if is_stereo:
            resampled_audio = np.zeros((num_samples, 2), dtype=np.float32)
            resampled_audio[:, 0] = resample(float_audio[:, 0], num_samples)
            resampled_audio[:, 1] = resample(float_audio[:, 1], num_samples)
        else:
            resampled_audio = resample(float_audio, num_samples)
    else:
        resampled_audio = float_audio

    return resampled_audio

def convert_pcm_to_float32(pcm_data, input_sample_rate=8000, target_sample_rate=24000, is_stereo=False):
    """
    Converts PCM (int16) byte data to a numpy array of audio data in float32 format, with resampling.

    Parameters:
    - pcm_data: PCM encoded byte data (int16).
    - input_sample_rate: Sample rate of the PCM data (e.g., 16000)
    - target_sample_rate: Desired sample rate for output audio (e.g., 24000)
    - is_stereo: Boolean indicating if the PCM data is stereo (True) or mono (False).

    Returns:
    - A numpy array of audio data in float32 format, resampled to the target sample rate.
    """
    # Convert PCM bytes to int16 numpy array
    pcm_audio = np.frombuffer(pcm_data, dtype=np.int16)

    # Reshape if the input is stereo
    if is_stereo:
        pcm_audio = pcm_audio.reshape((-1, 2))

    # Normalize the PCM data to be in the range [-1.0, 1.0]
    float_audio = pcm_audio.astype(np.float32) / 32767.0

    # Resample the audio if necessary
    if input_sample_rate != target_sample_rate:
        num_samples = int(len(float_audio) * target_sample_rate / input_sample_rate)
        if is_stereo:
            resampled_audio = np.zeros((num_samples, 2), dtype=np.float32)
            resampled_audio[:, 0] = resample(float_audio[:, 0], num_samples)
            resampled_audio[:, 1] = resample(float_audio[:, 1], num_samples)
        else:
            resampled_audio = resample(float_audio, num_samples)
    else:
        resampled_audio = float_audio

    return resampled_audio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

# Initialize the Whisper model for transcription
print("Model initializing")
# Check if CUDA is available and print the result
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
model = WhisperModel("large-v3", device="cuda", compute_type='float32')  # Assuming GPU usage; "cpu" can be used otherwise
print("Model initialized")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # Aggressive mode

# Constants
SAMPLE_RATE = 8000
FRAME_DURATION_MS = 20
FRAME_SIZE = 640  # 2 bytes per sample for 16-bit audio
MIN_CHUNK_FRAMES = SAMPLE_RATE // (FRAME_SIZE // 2)  # Number of frames for 1 second of audio

@app.get("/")
async def read_root():
    return {"message": "Welcome to Whisper Service"}

@app.post("/complete_transcribe_r")
async def transcribe_audio(request: Request):
    try:
        audio_data = await request.body()
        logger.info(f"Received audio data of size: {len(audio_data)} bytes")
        audio_array_np = np.frombuffer(audio_data, dtype=np.float32)
        
        segments, _ = await asyncio.to_thread(model.transcribe, audio_array_np, language="ru", temperature=0.2)
        arr = [segment for segment in segments]
      #  print(arr)
        transcription = " ".join([segment.text.strip() for segment in arr])
        logger.info(f"Transcription: {transcription}")
        return {"transcription": transcription}
    
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during transcription")
    
@app.websocket("/complete_transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_sentence = await websocket.receive_bytes()
            print("Buffer size: ", len(audio_sentence))
            audio_array_np = np.frombuffer(audio_sentence, dtype=np.float32)
            segments, _ = await asyncio.to_thread(model.transcribe, audio_array_np, language="ru")
            transcription = " ".join([segment.text.strip() for segment in segments])
            logger.info(f"Transcription: {transcription}")
            await websocket.send_text(transcription)
            logger.info("Transcription sent")
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Ensure to close websocket in finally only if it's open, avoid redundant closing
        if not websocket.application_state.closed:
            await websocket.close()

@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray() 
    frame_count = 0
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            buffer.extend(audio_chunk)
            logger.info("Received audio chunk")

            # Process audio when buffer is sufficiently filled
            if len(buffer) >= 64000: 
                print("Buffer size: ", len(buffer))
                # Convert buffer to NumPy ndarray
                audio_array_np = np.frombuffer(buffer, dtype=np.float32)

             #   audio_array_np = convert_pcm8_to_float32(buffer, input_sample_rate=16000)
                # Clear the buffer after processing
                buffer = bytearray()
                # Transcribe in a non-blocking manner
                segments, _ = await asyncio.to_thread(model.transcribe, audio_array_np, language="ru")

                transcription = " ".join([segment.text.strip() for segment in segments])
                logger.info(f"Transcription: {transcription}")
                await websocket.send_text(transcription)
                logger.info("Transcription sent")

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Ensure to close websocket in finally only if it's open, avoid redundant closing
        if not websocket.application_state.closed:
            await websocket.close()

# @app.websocket("/transcribe")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     buffer = bytearray() 
#     silence_buffer = bytearray()
#     frame_count = 0
#     try:
#         while True:
#             audio_chunk = await websocket.receive_bytes()
#             buffer.extend(audio_chunk) 
#             logger.info("Received audio chunk")
#             # Process audio in chunks of FRAME_SIZE
#             while len(buffer) >= FRAME_SIZE: 
#                 frame = buffer[:FRAME_SIZE]
#                 buffer = buffer[FRAME_SIZE:]

#                 is_speech = vad.is_speech(frame, 16000)

#                 if is_speech:
#                     silence_buffer.extend(frame)
#                     frame_count += 1
#                 else:
#                     if len(silence_buffer) > 0 and frame_count >= MIN_CHUNK_FRAMES:
#                         # Convert buffer to NumPy ndarray
#                         audio_array_np = float2pcm(silence_buffer, dtype='int8')
#                         # Transcribe in a non-blocking manner
#                         segments, _ = await asyncio.to_thread(model.transcribe, audio_array_np, language="ru", chunk_length=640, vad_filter=True)

#                         transcription = " ".join([segment.text.strip() for segment in segments])
#                         logger.info(f"Transcription: {transcription}")
#                         await websocket.send_text(transcription)
#                         logger.info("Transcription sent")
#                         # Clear the silence buffer and reset frame count after processing
#                         silence_buffer = bytearray()
#                         frame_count = 0

#     except WebSocketDisconnect:
#         logger.info("Client disconnected")
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
#     finally:
#         # Ensure to close websocket in finally only if it's open, avoid redundant closing
#         if not websocket.application_state.closed:
#             await websocket.close()

# Run the application

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)