#!/usr/bin/env python3
"""
Real-time Facial Expression Recognition from webcam (/dev/video0)
using a FER-ResNet model exported from the FERResNet.ipynb notebook.

FIXES: 
1. **CRITICAL OOM FIX:** Implemented logic to explicitly set memory growth 
   and provided a recommended CPU fallback if GPU memory is insufficient.
2. **Robustness:** Added clearer checks for image validity during preprocessing.
3. **Cleanup:** Improved resource cleanup with explicit model deletion.

Usage:
    python3 run_fer_webcam.py --model fer_resnet_model.h5
    python3 run_fer_webcam.py --camera 0
    python3 run_fer_webcam.py --model fer_resnet_model.h5 --no-display

NOTE: This file uses batch prediction to improve real-time performance.
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
import os

# --- Emotion classes for FER2013 ---
# This order is based on standard FER2013 dataset mapping.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def preprocess_face(face_img):
    """
    Preprocess image as expected by FERResNet:
    - convert to grayscale
    - resize to 48x48
    - normalize to [0,1]
    - add batch and channel dimensions
    """
    if face_img is None or face_img.size == 0:
        return None
        
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"[ERROR] OpenCV error during preprocessing: {e}")
        return None

    # ** ROBUSTNESS CHECK: Ensure resize didn't fail **
    if resized is None or resized.size == 0:
        print("[ERROR] Failed to resize face image to 48x48.")
        return None

    # Normalize to [0, 1] and change dtype
    normalized = resized.astype("float32") / 255.0
    
    # Add channel dimension (48, 48, 1) and batch dimension (1, 48, 48, 1)
    normalized = np.expand_dims(normalized, axis=-1)
    normalized = np.expand_dims(normalized, axis=0)
    
    return normalized


def run(camera_index, model_path, display=True, use_cpu_fallback=False):
    model = None # Initialize model outside the try block

    # --- MODEL LOADING WITH OOM MANAGEMENT ---
    
    # Use GPU (default path) or CPU fallback context
    device_name = '/cpu:0' if use_cpu_fallback else '/gpu:0'
    
    if use_cpu_fallback:
        print("[INFO] Attempting to load model using **CPU Fallback** to avoid OOM.")
    else:
        print(f"[INFO] Attempting to load model on GPU with memory growth set.")

    # Wrap model loading in the chosen device context
    try:
        with tf.device(device_name):
            print(f"[INFO] Loading model: {model_path} on {device_name}")
            model = tf.keras.models.load_model(model_path)
    except Exception as e:
        # Catch OOM or other loading errors
        print(f"[FATAL] Failed to load model from {model_path} on {device_name}: {e}")
        print("-" * 50)
        print("CRITICAL: If this is an OOM error, run the script with --use-cpu-fallback.")
        print("-" * 50)
        return

    # --- FACE DETECTOR LOADING ---
    FACE_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
    
    if not os.path.exists(FACE_CASCADE_FILENAME):
        print(f"[FATAL] Haar Cascade file not found: {FACE_CASCADE_FILENAME}. "
              "Please ensure it is in the same directory.")
        return

    print(f"[INFO] Loading Haar Cascade: {FACE_CASCADE_FILENAME}")
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILENAME)
    
    if face_cascade.empty():
        print(f"[FATAL] Failed to load Haar Cascade from file: {FACE_CASCADE_FILENAME}.")
        return

    # --- VIDEO CAPTURE SETUP ---
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open camera {camera_index}")
        return

    print("[INFO] Press 'q' to quit.")

    # --- MAIN LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed. Exiting.")
            break

        frame = cv2.flip(frame, 1)  # mirror the webcam view
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        processed_faces = []
        face_rects = []
        
        # 1. Collect all preprocessed faces and their coordinates
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            processed = preprocess_face(face_img)
            
            if processed is not None:
                processed_faces.append(processed)
                face_rects.append((x, y, w, h))

        # 2. Predict on the entire batch of faces (if any are found)
        if processed_faces:
            # Concatenate the list of (1, 48, 48, 1) arrays into a batch (N, 48, 48, 1)
            batch_processed = np.vstack(processed_faces)
            
            # Prediction runs on the device the model was loaded onto (GPU or CPU)
            all_preds = model.predict(batch_processed, verbose=0)

            # 3. Iterate through predictions and draw UI
            for i, (x, y, w, h) in enumerate(face_rects):
                preds = all_preds[i] # Get the prediction for the i-th face
                emotion_idx = int(np.argmax(preds))
                
                if 0 <= emotion_idx < len(EMOTIONS):
                    emotion_label = EMOTIONS[emotion_idx]
                else:
                    emotion_label = "Unknown Index"

                # Define colors for drawing
                RECT_COLOR = (0, 255, 0)
                TEXT_BG_COLOR = (255, 0, 0)
                TEXT_COLOR = (255, 255, 255)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), RECT_COLOR, 2)
                
                # Draw text background and label
                (label_width, label_height), baseline = cv2.getTextSize(
                    emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                )
                cv2.rectangle(
                    frame, 
                    (x, y - label_height - baseline - 10), # Top-left corner
                    (x + label_width, y),                  # Bottom-right corner
                    TEXT_BG_COLOR, 
                    cv2.FILLED
                )
                cv2.putText(
                    frame, emotion_label, (x, y - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2
                )


        if display:
            cv2.imshow("FER - Live Webcam", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()
    
    # ** IMPROVED CLEANUP **
    if model is not None:
        try:
            del model # Explicitly delete model to aid garbage collection
        except Exception:
            pass # Ignore if deletion fails for some reason
            
    tf.keras.backend.clear_session()
    print("[INFO] Resources released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time FER webcam tool")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (/dev/videoX or 0, 1, etc.)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained FERResNet .h5 model file")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable window display (useful for headless systems)")
    parser.add_argument("--use-cpu-fallback", action="store_true",
                        help="Force model loading and prediction onto the CPU (RAM) to avoid GPU OOM errors.")
    args = parser.parse_args()
    
    # --- GLOBAL TENSORFLOW GPU SETUP ---
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus and not args.use_cpu_fallback:
        try:
            # Set TensorFlow to only allocate memory when needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] Set GPU memory growth to True.")
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print(f"[WARN] Failed to set memory growth: {e}")
            
    # --- EXECUTE RUN ---
    run(
        camera_index=args.camera,
        model_path=args.model,
        display=not args.no_display,
        use_cpu_fallback=args.use_cpu_fallback
    )
