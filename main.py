import cv2
import numpy as np
import tensorflow as tf
import io
import threading
import json
import asyncio
import os

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)

# ─────────────────────────────────────────
# MODEL (TFLite)
# ─────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path="iris_pure_float32.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter_lock = threading.Lock()

_dummy = np.zeros((1,384,384,3), dtype=np.float32)
with interpreter_lock:
    interpreter.set_tensor(input_details[0]['index'], _dummy)
    interpreter.invoke()

print("[model] Warmed up")

# ─────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────
face_mesh_live = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

face_mesh_photo = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
)

# ─────────────────────────────────────────
# LENS CACHE (PRELOAD)
# ─────────────────────────────────────────
_lens_cache = {}
_resized_lens_cache = {}

for file in os.listdir("images"):
    if file.endswith(".png"):
        lens_id = file.split(".")[0]
        _lens_cache[lens_id] = cv2.imread(f"images/{file}", cv2.IMREAD_UNCHANGED)

print(f"[cache] Loaded {len(_lens_cache)} lenses")


def get_resized_lens(lens_id, w, h):
    key = f"{lens_id}_{w}_{h}"
    if key not in _resized_lens_cache:
        t = _lens_cache.get(lens_id)
        if t is None:
            return None
        _resized_lens_cache[key] = cv2.resize(t, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return _resized_lens_cache[key]

# ─────────────────────────────────────────
# EAR (EYE OPENNESS)
# ─────────────────────────────────────────
EAR_OPEN_THRESH  = 0.18
EAR_CLOSE_THRESH = 0.10

def get_ear(lm, t, b, l, r, w, h):
    top   = np.array([lm[t].x*w, lm[t].y*h])
    bot   = np.array([lm[b].x*w, lm[b].y*h])
    left  = np.array([lm[l].x*w, lm[l].y*h])
    right = np.array([lm[r].x*w, lm[r].y*h])

    vertical   = np.linalg.norm(top - bot)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal else 0.0


def get_eye_openness(lm, w, h):
    return (
        get_ear(lm,159,145,33,133,w,h),
        get_ear(lm,386,374,362,263,w,h)
    )


def ear_to_alpha(ear):
    if ear <= EAR_CLOSE_THRESH: return 0.0
    if ear >= EAR_OPEN_THRESH:  return 1.0
    return (ear - EAR_CLOSE_THRESH) / (EAR_OPEN_THRESH - EAR_CLOSE_THRESH)

# ─────────────────────────────────────────
# LENS APPLICATION
# ─────────────────────────────────────────
def apply_lens(frame, lm, lens_id):
    h, w = frame.shape[:2]

    L_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
    R_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

    left_ear, right_ear = get_eye_openness(lm, w, h)

    for (iris, edge, pts, ear) in [
        (468,471,L_EYE,left_ear),
        (473,476,R_EYE,right_ear)
    ]:
        alpha_scale = ear_to_alpha(ear)
        if alpha_scale <= 0: continue

        try:
            cx = int(lm[iris].x*w)
            cy = int(lm[iris].y*h)
            ex = int(lm[edge].x*w)
            ey = int(lm[edge].y*h)

            r = int(np.hypot(cx-ex, cy-ey))
            if r < 4: continue

            pad = int(r*1.35)
            y1,y2 = max(0,cy-pad), min(h,cy+pad)
            x1,x2 = max(0,cx-pad), min(w,cx+pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            ch, cw = crop.shape[:2]

            iris_mask = np.zeros((ch,cw), np.uint8)
            cv2.circle(iris_mask,(cx-x1,cy-y1),int(r*0.92),255,-1)

            occ_mask = np.zeros((ch,cw), np.uint8)
            poly = np.array([
                [(int(lm[p].x*w)-x1),(int(lm[p].y*h)-y1)]
                for p in pts
            ], np.int32)
            cv2.fillPoly(occ_mask,[poly],255)

            mask = cv2.GaussianBlur(cv2.bitwise_and(iris_mask, occ_mask),(5,5),0)

            lens = get_resized_lens(lens_id, cw, ch)
            if lens is None or lens.shape[2] != 4: continue

            alpha = (lens[:,:,3]/255.0) * (mask/255.0) * 0.75 * alpha_scale
            a3 = np.stack([alpha]*3, axis=2)

            out = lens[:,:,:3]*a3 + crop*(1-a3)
            frame[y1:y2,x1:x2] = out.astype(np.uint8)

        except:
            continue

    return frame

# ─────────────────────────────────────────
# FRAME PROCESSOR
# ─────────────────────────────────────────
def process_frame(frame_bytes, lens_id):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    frame = cv2.resize(frame,(640,480))
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh_live.process(rgb)

    if results.multi_face_landmarks:
        frame = apply_lens(frame, results.multi_face_landmarks[0].landmark, lens_id)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()

# ─────────────────────────────────────────
# WEBSOCKET (BINARY, LOW LATENCY)
# ─────────────────────────────────────────
@app.websocket("/ws/live-ar")
async def live_ar(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Connected")

    loop = asyncio.get_event_loop()
    pending = None

    try:
        while True:
            data = await websocket.receive_bytes()

            if pending and not pending.done():
                continue

            def job():
                return process_frame(data, "1")

            pending = loop.run_in_executor(executor, job)
            result = await pending

            if result:
                await websocket.send_bytes(result)

    except WebSocketDisconnect:
        print("[WS] Disconnected")

# ─────────────────────────────────────────
# PHOTO MODE
# ─────────────────────────────────────────
@app.post("/apply-lens")
async def photo(image: UploadFile = File(...), lens_id: str = Form(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "decode fail"}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_photo.process(rgb)

    if results.multi_face_landmarks:
        frame = apply_lens(frame, results.multi_face_landmarks[0].landmark, lens_id)

    _, buf = cv2.imencode(".png", frame)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
