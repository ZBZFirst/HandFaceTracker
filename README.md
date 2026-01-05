# Gesture-Controlled Face & Hand Tracking (Python)

This project uses **MediaPipe** and **OpenCV** to perform real-time face and hand tracking from a webcam.  
Detected gestures can optionally control **OBS Studio** via WebSocket.

---

## Features

- Real-time **face landmark** detection
- Real-time **hand gesture** recognition
- Visual overlay of landmarks
- Optional OBS Studio control (filters, scene effects)
- Graceful handling when optional components are missing

---

## System Requirements

### Operating System
- **Windows 10/11** (recommended)
- macOS / Linux may work with camera backend changes

### Python
- **Python 3.9 – 3.11**
- Python 3.10 is strongly recommended

Check version:
```bash
python --version


opencv-python>=4.8,<5
numpy>=1.23,<2.0
mediapipe>=0.10.9
obs-websocket-py>=1.0
#i think?
