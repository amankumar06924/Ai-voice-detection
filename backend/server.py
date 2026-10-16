from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from converting_audio_file import convert_to_standard_audio
from checking_audio_file import preprocessing_audio_file
from collections import deque
import numpy as np
import json
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin = ["*"],
    allow_credentials = True,
    allow_method = ["*"],
    allow_headers = ["*"]
)

@app.post("/detect-audio-file")
async def check_audio_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("/audio"):
        raise HTTPException(
            status_code= 400,
            detail= "File must be an audio file"
        )
    try:
        total_read_bytes = 0
        max_file_size = 50*1024*1024 # maximum audio file size: 50Mb
        chunk_size = 64*1024 # maximum chunk size 64kb
        in_memory_buffer = bytearray()
        while chunk:= file.read(chunk_size):
            total_read_bytes = total_read_bytes + len(chunk)
            if total_read_bytes > max_file_size:
                raise HTTPException(
                    status_code= 413,
                    detail= "maximum file size supported is 50mb"
                )
            in_memory_buffer.append(chunk)
            audio_1d = convert_to_standard_audio(in_memory_buffer)
            answer = preprocessing_audio_file(audio_1d, hpp_seconds = 2.5)
            if answer == 1:
                print("most probable voice ai ")
                return {
                    "success": True,
                    "AI_Voice": 1,
                    "Human_Voice": 0
                }
            return {
                "success": True,
                "AI_Voice": 0,
                "Human_Voice": 1
            }
    except:
        raise HTTPException(
            status_code= 500,
            detail= "something went wrong in the server"
        )

@app.websocket("/ws/detect-real-audio")
async def check_real_audio(websocket: WebSocket):
    await websocket.accept()
    print("Websocket connection accpted with frontend")
    #rolling buffer to hold 5sec chunk data
    sample_rate = 16000
    chunk_duration = 5
    queue_size = sample_rate * chunk_duration
    audio_buffer = deque(maxlen = queue_size)
    try:
        while True:
            # receiving chunk of 1sec from the frontend
            raw_bytes = await websocket.receive()
            if "text" in raw_bytes:
                data = json.load(raw_bytes["text"])
                if data.get("action") == "STOP_RECORDING":
                    print("User finished his speech")
                    await websocket.send_json({
                        "success": True,
                        "AI_Voice": 0,
                        "Human_Voice": 1
                    })
            audio_chunk = np.frombuffer(raw_bytes, dtype= np.float32)
            audio_buffer.extend(audio_chunk)
            if len(audio_buffer) == queue_size:
                audio_bytes_array = np.array(audio_buffer, dtype= np.float32)
                answer = preprocessing_audio_file(audio_bytes_array,hop_seconds=2.5)
                if answer == 1:
                    await websocket.send_json({
                        "success": True,
                        "AI_Voice": 1,
                        "Human_Voice": 0
                    })
    except Exception as e:
        if e == WebSocketDisconnect:
            print("client discoonected")
        else:
            print(f"some error occured: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host = "0.0.0.0", port = 8000, reload = True)