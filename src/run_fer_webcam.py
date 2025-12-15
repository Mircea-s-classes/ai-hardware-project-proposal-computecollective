#!/usr/bin/env python3
"""
Real-time Facial Expression Recognition from webcam (/dev/video0)
using a FER-ResNet model exported from the FERResNet.ipynb notebook.

Usage:
    python3 run_fer_webcam.py --model fer_resnet_model.h5
    python3 run_fer_webcam.py --camera 0
    python3 run_fer_webcam.py --model fer_resnet_model.h5 --no-display

NOTE: This file has been fixed to use batch prediction to improve
      real-time performance and prevent prediction lag.
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf


# Emotion classes — modify if your model uses a different order
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def preprocess_face(face_img):
    """
    Preprocess image as expected by FERResNet:
    - convert to grayscale
    - resize to 48x48
    - normalize to [0,1]
    - add batch and channel dimensions
    
    NOTE: For batch processing, we ensure the output has shape (1, 48, 48, 1)
          so it can be easily stacked later.
    """
    # Crop to focus on the face before preprocessing
    # This might help with consistency, though the original code used the full face_img
    
    # The original implementation was:
    # gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(gray, (48, 48))
    
    # A safer approach for FER models is often to convert, resize, and normalize:
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
    except cv2.error as e:
        print(f"[ERROR] OpenCV error during preprocessing: {e}")
        # Return an empty array or handle error gracefully if face_img is invalid
        return None

    normalized = resized.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=-1)   # (48,48,1)
    normalized = np.expand_dims(normalized, axis=0)    # (1,48,48,1)
    return normalized


def run(camera_index, model_path, display=True):
    # Load model
    print(f"[INFO] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load face detector
    FACE_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
    print(f"[INFO] Loading Haar Cascade: {FACE_CASCADE_FILENAME}")
    face_cascade = cv2.CascadeClassifier(
        FACE_CASCADE_FILENAME
    )
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from file: {FACE_CASCADE_FILENAME}. Ensure the file is in the same directory as the script.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            break

        frame = cv2.flip(frame, 1)  # mirror the webcam view
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # --- FIX: Batch Prediction for Real-time Stability ---
        
        processed_faces = []
        face_rects = []
        
        # 1. Collect all preprocessed faces and their coordinates
        for (x, y, w, h) in faces:
            # Note: face_img is BGR (color frame slice)
            face_img = frame[y:y+h, x:x+w]
            
            processed = preprocess_face(face_img)
            
            # Only add validly processed faces
            if processed is not None:
                processed_faces.append(processed)
                face_rects.append((x, y, w, h))

        # 2. Predict on the entire batch of faces (if any are found)
        if processed_faces:
            # Concatenate the list of (1, 48, 48, 1) arrays into a batch (N, 48, 48, 1)
            batch_processed = np.vstack(processed_faces) 
            
            # Predict once for all faces in the current frame
            # This is the key fix for performance and update lag
            all_preds = model.predict(batch_processed, verbose=0) 

            # 3. Iterate through predictions and draw UI
            for i, (x, y, w, h) in enumerate(face_rects):
                preds = all_preds[i] # Get the prediction for the i-th face
                emotion_idx = int(np.argmax(preds))
                emotion_label = EMOTIONS[emotion_idx]
                
                # Draw UI
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw text background
                (label_width, label_height), baseline = cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), (255, 0, 0), cv2.FILLED)

                # Draw text label
                cv2.putText(frame, emotion_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


        if display:
            cv2.imshow("FER - Live Webcam", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tf.keras.backend.clear_session()
    print("[INFO] GPU memory released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time FER webcam tool")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (/dev/videoX)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained FERResNet .h5 model file")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable window display (useful for headless systems)")
    args = parser.parse_args()
    
    # TensorFlow setup block is kept as is
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Set TensorFlow to only allocate memory when needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] Set GPU memory growth to True.")
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print(e)
            
    run(
        camera_index=args.camera,
        model_path=args.model,
        display=not args.no_display
    )
