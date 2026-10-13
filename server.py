import os
import io
import base64
import threading
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# Initialize FastAPI App
app = FastAPI(
    title="Voice2Sign API",
    description="Real-Time Indian Sign Language AI Translation & Learning Backend",
    version="1.0.0"
)

# Enable CORS for maximum flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
SIGNS_DIR = os.path.join(BASE_DIR, "signs", "signs")

# Mount static sign images
if os.path.exists(SIGNS_DIR):
    app.mount("/signs", StaticFiles(directory=SIGNS_DIR), name="signs")

# Load backend AI model and live prediction helper
try:
    from live_prediction import live_predict, model, class_names, preprocess_for_model, process_single_frame
    HAS_MODEL = True
except Exception as e:
    HAS_MODEL = False
    model_load_error = str(e)


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class RegisterSchema(BaseModel):
    fullName: str
    email: str
    password: str
    role: Optional[str] = "Learner / Ally"
    language: Optional[str] = "English"

class SignInSchema(BaseModel):
    email: str
    password: str

class TextToISLSchema(BaseModel):
    text: str

class ImagePredictSchema(BaseModel):
    image: str  # Base64 string

class LivePredictFrameSchema(BaseModel):
    image: str  # Base64 string or data URL
    show_skeleton: bool = True
    mode: Optional[str] = "letters"
    threshold: Optional[float] = 85.0

class QuizSubmitSchema(BaseModel):
    selected_option: str
    correct_option: str


# ==========================================
# HTML PAGE ROUTES
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def get_home_page():
    file_path = os.path.join(FRONTEND_DIR, "home_page.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Home page not found</h1>", status_code=404)

@app.get("/home", response_class=HTMLResponse)
async def get_home_alias():
    return await get_home_page()

@app.get("/register", response_class=HTMLResponse)
async def get_register_page():
    file_path = os.path.join(FRONTEND_DIR, "register_page.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Register page not found</h1>", status_code=404)

@app.get("/signin", response_class=HTMLResponse)
async def get_signin_page():
    file_path = os.path.join(FRONTEND_DIR, "sign_in_page.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Sign in page not found</h1>", status_code=404)

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard_page():
    file_path = os.path.join(FRONTEND_DIR, "main_page.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Dashboard page not found</h1>", status_code=404)


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.post("/api/register")
async def api_register(data: RegisterSchema):
    if not data.email or not data.password or not data.fullName:
        raise HTTPException(status_code=400, detail="Missing required registration fields.")
    
    user_data = {
        "name": data.fullName,
        "email": data.email,
        "role": data.role or "Learner / Ally",
        "lang": data.language or "English",
        "xp": 450,
        "streak": 7
    }
    return {"status": "success", "message": "Account registered successfully", "user": user_data}


@app.post("/api/signin")
async def api_signin(data: SignInSchema):
    if not data.email or not data.password:
        raise HTTPException(status_code=400, detail="Email and password required.")
    
    # Extract name from email as placeholder if needed
    name = data.email.split("@")[0].capitalize() if "@" in data.email else "Learner"
    user_data = {
        "name": name if name != "Aarav" else "Aarav Sharma",
        "email": data.email,
        "role": "Learner / Ally",
        "lang": "English",
        "xp": 450,
        "streak": 7
    }
    return {"status": "success", "message": "Signed in successfully", "user": user_data}


@app.post("/api/text-to-isl")
async def api_text_to_isl(data: TextToISLSchema):
    clean_text = data.text.strip().upper()
    results = []
    
    for char in clean_text:
        if char.isalnum():
            file_name = f"{char}.jpg"
            full_path = os.path.join(SIGNS_DIR, file_name)
            exists = os.path.exists(full_path)
            results.append({
                "char": char,
                "url": f"/signs/{file_name}" if exists else None,
                "found": exists
            })
    
    return {"status": "success", "text": clean_text, "signs": results}


@app.post("/api/live-predict")
async def api_live_predict():
    if not HAS_MODEL:
        raise HTTPException(status_code=500, detail=f"AI Model not available: {model_load_error}")
    
    # Run OpenCV window in a separate thread so it doesn't block the server loop
    t = threading.Thread(target=live_predict, daemon=True)
    t.start()
    
    return {"status": "success", "message": "Live OpenCV prediction camera window started. Press 'q' in camera window to stop."}


@app.post("/api/live-predict-frame")
async def api_live_predict_frame(data: LivePredictFrameSchema):
    if not HAS_MODEL:
        raise HTTPException(status_code=500, detail=f"AI Model not available: {model_load_error}")
    
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image frame.")

        threshold_val = float(data.threshold) if data.threshold is not None else 85.0
        annotated_frame, hand_detected, predicted_class, confidence, meets_threshold, layout_data = process_single_frame(
            img, show_skeleton=bool(data.show_skeleton), threshold=threshold_val
        )

        # Encode annotated frame back to JPEG base64 for real-time web display
        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        encoded_frame = base64.b64encode(buffer).decode("utf-8")
        annotated_data_url = f"data:image/jpeg;base64,{encoded_frame}"

        return {
            "status": "success",
            "hand_detected": hand_detected,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "meets_threshold": meets_threshold,
            "annotated_image": annotated_data_url,
            "layout_data": layout_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Live frame error: {str(e)}")


@app.websocket("/ws/live-predict")
async def websocket_live_predict(websocket: WebSocket):
    await websocket.accept()
    if not HAS_MODEL:
        await websocket.close(code=1011, reason=f"AI Model not available: {model_load_error}")
        return
        
    try:
        while True:
            data = await websocket.receive_json()
            
            image_data = data.get("image", "")
            show_skeleton = data.get("show_skeleton", True)
            threshold_val = float(data.get("threshold", 85.0))
            
            header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
            if not encoded:
                continue
            
            img_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            annotated_frame, hand_detected, predicted_class, confidence, meets_threshold, layout_data = process_single_frame(
                img, show_skeleton=bool(show_skeleton), threshold=threshold_val
            )

            await websocket.send_json({
                "status": "success",
                "hand_detected": hand_detected,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "meets_threshold": meets_threshold,
                "layout_data": layout_data
            })
            
    except WebSocketDisconnect:
        print("Client disconnected from live prediction WebSocket")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/predict-image")
async def api_predict_image(data: ImagePredictSchema):
    if not HAS_MODEL:
        raise HTTPException(status_code=500, detail=f"AI Model not available: {model_load_error}")
    
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        processed = preprocess_for_model(img)
        preds = model.predict(processed)
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        predicted_sign = class_names[class_idx]

        return {
            "status": "success",
            "predicted_class": predicted_sign,
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


@app.get("/api/dictionary")
async def api_dictionary(q: Optional[str] = "", category: Optional[str] = "all"):
    signs = []
    if os.path.exists(SIGNS_DIR):
        files = sorted([f for f in os.listdir(SIGNS_DIR) if f.endswith(".jpg")])
        for f in files:
            symbol = f.replace(".jpg", "")
            if q and q.upper() not in symbol.upper():
                continue
            
            # Simple category categorizer
            cat = "Alphabets" if symbol.isalpha() else "Numbers"
            if category and category != "all" and category.lower() != cat.lower():
                continue

            signs.append({
                "symbol": symbol,
                "url": f"/signs/{f}",
                "category": cat
            })

    return {"status": "success", "total": len(signs), "signs": signs}


@app.get("/api/quiz")
async def api_quiz():
    # Pick a sample quiz entry (e.g., 'A')
    quiz_symbol = "A"
    return {
        "status": "success",
        "question": "Identify the Indian Sign Language gesture below:",
        "sign_symbol": quiz_symbol,
        "image_url": f"/signs/{quiz_symbol}.jpg",
        "options": ["Sign 'B'", "Sign 'A'", "Sign 'C'", "Sign 'O'"],
        "correct_option": "Sign 'A'",
        "reward_xp": 50
    }


@app.post("/api/quiz/submit")
async def api_quiz_submit(data: QuizSubmitSchema):
    is_correct = data.selected_option == data.correct_option
    return {
        "status": "success",
        "correct": is_correct,
        "message": "🎉 Correct! +50 XP Earned!" if is_correct else "❌ Incorrect. Try again!",
        "earned_xp": 50 if is_correct else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
