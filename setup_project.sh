#!/bin/bash
echo "=========================================================="
echo " Installing dependencies for AI Dashboard Project"
echo "=========================================================="

echo "[1/4] Updating package lists..."
sudo apt update -y

echo "[2/4] Installing system dependencies, Hailo stack, OpenCV and PyQt6..."
# hailo-all installs Hailo RT and driver for the RPi AI Kit
# python3-opencv and python3-pyqt6 are best installed via apt on Raspberry Pi
sudo apt install -y hailo-all python3-opencv python3-numpy python3-pyqt6 python3-pip

echo "[3/4] Installing Python libraries (FastAPI, Uvicorn, Streamlit, etc.)..."
# On Raspberry Pi OS Bookworm, we might need --break-system-packages if not using venv
pip3 install fastapi uvicorn websockets pyyaml streamlit --break-system-packages

echo "[4/4] Setting up HailoRT multi-process service..."
sudo systemctl enable hailort.service
sudo systemctl start hailort.service

echo "=========================================================="
echo " Setup Complete! 🎉"
echo " To run the AI Dashboard API / WebSocket Server:"
echo "   cd ~/wf/ai_dashboard"
echo "   python3 main.py"
echo ""
echo " To run the Receiver GUI:"
echo "   cd ~/wf/ai_receiver_deploy"
echo "   ./run_receiver.sh"
echo "=========================================================="
