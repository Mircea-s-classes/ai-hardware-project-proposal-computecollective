# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal - Emotion Detector

## 1. Project Title: Emotion Detector
Name of the Team: Compute Collective

List of students in the team: Garrett Delaney, Evan Sage, Sansshita Baskaran

## 2. Platform Selection

We will use the NVIDIA Jetson, which is in the Edge AI category. We chose this because one team member has prior experience with it at his job. He also said they have good tutorials and documentation. 

## 3. Problem Definition

Most AI based computer vision and recognition is currently performed in the cloud and requires a continuous internet connection. This reliance on the cloud has challenges such as latency, high data and bandwidth usage, security and privacy concerns, plus it stops working whenever you are in an environment without internet. This limits the deployment of real world applications of AI, especially in environments with poor or slow internet.

In this project, we aim to utilize the NVIDIA Jetson development kit to design and implement a real-time emotion recognition system (for example to detect if someone if happy or sad). The computations will run completely locally on the device, requiring no internet, delivering speedy results while protecting user privacy.

## 4. Technical Objectives
List 3–5 measurable objectives with quantitative targets when possible.

1. Train an AI model on the dataset (FER-2013 or NVIDIA's DetectNet) and get it working well with still test images on the Jetson.
2. Set up the Jetson to accept realtime camera footage.
3. Connect the video feed to the emotion detection model to run it in real time.
4. Optimize the model for edge and resource constrained use cases through methods like quantization.

## 5. Methodology
Describe your planned approach: hardware setup, software tools, model design, performance metrics, and validation strategy.

**Hardware Setup**: 

Nvidia Jetson dev kit connected to a USB webcam. Possibly also a monitor and keyboard for interfacing with the Jetson, otherwise using SSH/VNC.

**Software Tools**:

NVIDIA JetPack (includes CUDA, cuDNN, TensorRT) for OS/SDK, PyTorch for the AI framework, TensorRT for inference acceleration, ONNX for model export, OpenCV (computer vision library) for realtime facial detection.

**Model Design**: 

A lightweight CNN-based facial emotion model suitable for edge deployment, such as MobileNetV2 which we used in a previous homework. It will be trained on a dataset such as FER-2013, which contains sample images in seven emotional categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). The training may be performed on our laptops or in the cloud with Google Colab. Once complete, it can be transferred deployed locally on the Jetson via the ONNX export format. We will convert to TensorRT to reduce latency and memory footprint and we will evaluate INT8 or FP16 quantization to minimize resource usage.

**Performance Metrics**: 

1. Classification accuracy of human emotion
2. Latency between an image being captured and an emotion being predicted
3. Memory usage (both footprint of the model and when running on the Jetson)

**Validation Strategy**: 

We will test the system under various conditions such as different lighting, backgrounds, or end users.

## 6. Expected Deliverables
List tangible outputs: working demo, GitHub repository, documentation, presentation slides, and final report.

For our working demo, we would like for someone to be able to smile or frown into the webcam and have the Jetson detect the person's emotion (either output to a monitor or LEDs).

The GitHub repo will house all our code for the project as well as your documentations, slides, report.

## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| Garrett Delaney | Team Lead | Coordination, documentation |
| Evan Sage | Hardware | Setup, integration |
| Sansshita Baskaran | Software | Model training, inference |
| Everyone | Evaluation | Testing, benchmarking |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 6 | Integration & testing | Working prototype |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
List special hardware, datasets, or compute access needed.

**Hardware**
1. Nvidia Jetson, probaly the Nano model, but the Jetson Xavier NX or Jetson AGX Xavier should work too
2. USB webcam
3. A monitor and keyboard (optional)

**Datasets**
1. FER-2013, should have free access through kaggle.com

**Compute Access**
1. Nothing specifically

## 10. References
Include relevant papers, repositories, and documentation.

[FER-2013 Face Emotion Data Set](https://www.kaggle.com/datasets/msambare/fer2013)

[Jetson Interface Deploy Deep Learning](https://github.com/dusty-nv/jetson-inference/blob/master/README.md)
