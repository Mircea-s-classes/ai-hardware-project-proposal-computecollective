[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/v3c0XywZ)
# AI Hardware Project Template
ECE 4332 / ECE 6332 — AI Hardware  
Fall 2025
## How To:
In order to use the emotion tracker, once the Jetson has finished logging in and a usb webcam is plugged in,
1. Change Directories to /face-detection/
2. Run python3 third_fer_webcam.py, arguements include --model MODEL.h5 for which trained model to use, --camera # for which V4L2 number the webcam ends up using, and --use-cpu-fallback if OOM error occurs
3. Once the model is running, simply use Cntr C to quit
## 🧭 Overview
This repository provides a structured template for your team project in the AI Hardware class.  
Each team will **clone this template** to start their own project repository.

## 🗂 Folder Structure
- `docs/` – project proposal and documentation  
- `presentations/` – midterm and final presentation slides  
- `report/` – final written report (IEEE LaTeX and DOCX versions included)  
- `src/` – source code for software, hardware, and experiments  
- `data/` – datasets or pointers to data used

## 🧑‍🤝‍🧑 Team Setup
Each team should have **2–4 members (3 preferred)**.  
List all team members in `docs/Project_Proposal.md`.

## 📋 Required Deliverables
1. **Project Proposal** — due Nov. 5, 2025, 11:59 PM  
2. **Midterm Presentation** — Nov. 19,2025, 11:59 PM  
3. **Final Presentation and Report** — Dec. 17, 11:59 PM

## 🚀 How to Use This Template
1. Click **“Use this template”** on GitHub.  
2. Name your repo `ai-hardware-teamXX` (replace XX with your team name or number).  
3. Clone it locally:
   ```bash
   git clone https://github.com/YOUR-ORG/ai-hardware-teamXX.git
   ```
4. Add your work in the appropriate folders.

## 🧾 Submissions
- Commit and push all deliverables before each deadline.
- Tag final submissions with:
   ```bash
   git tag v1.0-final
   git push origin v1.0-final
   ```

## 📜 License
This project is released under the MIT License.
