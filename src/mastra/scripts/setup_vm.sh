#!/bin/bash
# Setup script for Vast.ai VM — installs XTTS-v2 and starts the inference server.
# Called by the Mastra workflow after provisioning the VM.
set -e

echo "=== XTTS-v2 VM Setup ==="

# System deps
apt-get update -qq && apt-get install -y -qq git wget sox ffmpeg unzip curl > /dev/null 2>&1
echo "[1/5] System deps installed"

# Python deps (exact versions from proven playbook)
pip install --upgrade pip -q
pip install --force-reinstall certifi -q
pip install TTS==0.22.0 -q
pip install faster-whisper -q
pip install fastapi uvicorn python-multipart -q
echo "[2/5] Python deps installed"

# Download the XTTS server script
mkdir -p /root/xtts_server
cat > /root/xtts_server/server.py << 'SERVEREOF'
SERVEREOF

echo "[3/5] Server script ready"

# Generate a reference audio from a Polish sentence using base XTTS
# (This gives the model a voice to clone from — any Polish audio works)
python3 -c "
from TTS.api import TTS
import torch
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda')
# Use the built-in speaker for initial setup
tts.tts_to_file(
    text='Witaj, jestem polskim lektorem. Czytam dla Ciebie najpiękniejsze książki z biblioteki Wolnych Lektur.',
    language='pl',
    file_path='/root/ref_audio.wav',
    speaker='Claribel Dervla'
)
print('Reference audio generated')
" 2>/dev/null
echo "[4/5] Reference audio generated"

# Start the server in background
cd /root/xtts_server
nohup python3 -c "
import sys
sys.path.insert(0, '.')
exec(open('/root/xtts_server/server.py').read() if __import__('os').path.exists('/root/xtts_server/server.py') else '')
" > /root/xtts_server/server.log 2>&1 &

# Alternative: start with uvicorn directly
nohup uvicorn server:app --host 0.0.0.0 --port 5002 > /root/xtts_server/server.log 2>&1 &
echo "[5/5] XTTS server starting on port 5002"

# Wait for it to be healthy
for i in $(seq 1 60); do
    if curl -s http://localhost:5002/health | grep -q '"status":"ok"'; then
        echo "=== XTTS Server READY ==="
        exit 0
    fi
    sleep 5
done

echo "WARNING: Server not ready after 5 minutes, check logs"
cat /root/xtts_server/server.log | tail -30
exit 1
