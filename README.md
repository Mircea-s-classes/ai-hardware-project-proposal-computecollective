[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/v3c0XywZ)
# AI Hardware Project : Emotion Detector
ECE 4332 / ECE 6332 — AI Hardware  
Fall 2025
Team : Compute Collective 
Members: Sansshita Baskaran, Garrett Delaney, Evan Sage

## Project Overview:
Emotion recognition is an increasingly important area in computer vision and human-computer interaction (HCI), with applications ranging from mental health monitoring to driver alertness systems and in customer service. Many existing emotion recognition systems rely on cloud-based computation, where video data is streamed to remote servers for processing. While this is effective, this approach could also introduce issues such as high latency and bandwidth usage, privacy concerns, and reduced reliability in environments with unstable or unavailable internet connectivity. This project, Emotion Detector, allows us to investigate whether real-time facial emotion recognition can be performed entirely on an edge device. By running the full computation locally, the system aims to improve privacy, reduce latency, and enable deployment in real-world settings without cloud dependence. This project focuses on building and evaluating an on-device emotion recognition system using the NVIDIA Jetson platform.

## Hardware 
-NVIDIA Jetson
-Zoom Q4 USB Webcam
-USB Keyboard
-USB Mouse
-HDMI Monitor

## Software
-OpenCV
-TensorFlow
-Jetson Software Development Kit 
-V4L2- Video for Linux

## Dataset
- We used FER-2013 dataset 
- 48x48 pixel grayscale images of faces
- 7 Emotions: Angry, Disgusted, Fear, Happy, Neutral, Sad, Surprised 
- Training set includes 28,709 examples 
- Public test set includes 3,589 examples


## How To:
In order to use the emotion tracker, once the Jetson has finished logging in and a usb webcam is plugged in,
1. Change Directories to /face-detection/
2. Run python3 third_fer_webcam.py, arguements include --model MODEL.h5 for which trained model to use, --camera # for which V4L2 number the webcam ends up using, and --use-cpu-fallback if OOM error occurs
3. Once the model is running, simply use Cntr C to quit


## References 
[1]“Jetson Nano,” NVIDIA Developer. Accessed: Dec. 15, 2025. [Online]. Available: https://developer.nvidia.com/embedded/jetson-nano 
[2]	Boesch, “Explore Computer Vision with NVIDIA Jetson Modules,” viso.ai. Accessed: Dec. 15, 2025. [Online]. Available: https://viso.ai/edge-ai/nvidia-jetson/ 
[3]	“OpenCV,” OpenCV. Accessed: Dec. 15, 2025. [Online]. Available: https://opencv.org/ 
[4]	Google, “TensorFlow,” TensorFlow. Accessed: Dec. 15, 2025. [Online]. Available: https://www.tensorflow.org/ 
[5]	M. Sambare, “FER-2013,” Kaggle. Accessed: Dec. 15, 2025. [Online]. Available: https://www.kaggle.com/datasets/msambare/fer2013 
[6]	“NVIDIA JetPack Software Stack,” NVIDIA Developer. Accessed: Dec. 15, 2025. [Online]. Available: https://developer.nvidia.com/embedded/jetpack 


## 🧭 Overview
This repository provides a structured template for your team project in the AI Hardware class.  
Each team will **clone this template** to start their own project repository.


## 🗂 Folder Structure
- `docs/` – project proposal and documentation  
- `presentations/` – midterm and final presentation slides  
- `report/` – final written report (PDF version included)  
- `src/` – source code for software, hardware, and experiments  
- `data/` – datasets or pointers to data used

## 🧑‍🤝‍🧑 Team Setup
Each team should have **2–4 members (3 preferred)**.  
List all team members in `docs/Project_Proposal.md`.


## 🧾 Submissions
- Commit and push all deliverables before each deadline.
- Tag final submissions with:
   ```bash
   git tag v1.0-final
   git push origin v1.0-final
   ```

## 📜 License
This project is released under the MIT License.
