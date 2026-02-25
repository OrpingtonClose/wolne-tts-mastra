#!/bin/bash
set -euo pipefail

# Deploy Mastra + XTTS-v2 to a Vast.ai GPU VM
# Usage: VASTAI_API_KEY=xxx ./scripts/deploy-to-vast.sh [instance_id]
#
# If instance_id is provided, deploys to that existing instance.
# Otherwise, provisions a new one.

VASTAI_API_KEY="${VASTAI_API_KEY:?Set VASTAI_API_KEY}"
VAST_API="https://cloud.vast.ai/api/v0"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID="${1:-}"

auth_header="Authorization: Bearer $VASTAI_API_KEY"

provision_vm() {
  echo ">> Searching for GPU offers..."
  local offers
  offers=$(curl -s -X POST "$VAST_API/bundles/" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "$auth_header" \
    -d '{
      "q": {
        "gpu_ram": {"gte": 24576},
        "dph_total": {"lte": 0.50},
        "reliability2": {"gte": 0.95},
        "num_gpus": {"eq": 1},
        "disk_space": {"gte": 80},
        "cuda_max_good": {"gte": 12.0},
        "rented": {"eq": false},
        "type": "on-demand"
      },
      "sort": [["dph_total", "asc"]],
      "limit": 3
    }')

  local offer_id
  offer_id=$(echo "$offers" | python3 -c "import sys,json; o=json.load(sys.stdin)['offers']; print(o[0]['id']) if o else exit(1)")
  echo ">> Best offer: $offer_id"

  echo ">> Creating instance..."
  local result
  result=$(curl -s -X PUT "$VAST_API/asks/$offer_id/" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "$auth_header" \
    -d '{
      "client_id": "me",
      "image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
      "disk": 80,
      "label": "wolne-tts-mastra",
      "onstart": "apt-get update -qq && apt-get install -y -qq curl wget git ffmpeg sox openssh-client > /dev/null 2>&1",
      "runtype": "ssh ssh_proxy",
      "ports": {"3000/http": {}, "5002/http": {}}
    }')

  INSTANCE_ID=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
  echo ">> Instance created: $INSTANCE_ID"

  echo ">> Waiting for instance to start..."
  for i in $(seq 1 60); do
    sleep 5
    local info
    info=$(curl -s "$VAST_API/instances/$INSTANCE_ID/" -H "Accept: application/json" -H "$auth_header")
    local status
    status=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status',''))" 2>/dev/null || echo "")
    if [ "$status" = "running" ]; then
      local ssh_host ssh_port
      ssh_host=$(echo "$info" | python3 -c "import sys,json; print(json.load(sys.stdin)['ssh_host'])")
      ssh_port=$(echo "$info" | python3 -c "import sys,json; print(json.load(sys.stdin)['ssh_port'])")
      echo ">> Instance running! SSH: $ssh_host:$ssh_port"
      echo "$ssh_host" > /tmp/vast_ssh_host
      echo "$ssh_port" > /tmp/vast_ssh_port
      echo "$INSTANCE_ID" > /tmp/vast_instance_id
      return 0
    fi
    echo "   poll $i/60 — status: $status"
  done
  echo ">> FAILED: Instance did not start in 5 minutes"
  exit 1
}

deploy_to_vm() {
  local ssh_host ssh_port
  if [ -f /tmp/vast_ssh_host ]; then
    ssh_host=$(cat /tmp/vast_ssh_host)
    ssh_port=$(cat /tmp/vast_ssh_port)
  else
    local info
    info=$(curl -s "$VAST_API/instances/$INSTANCE_ID/" -H "Accept: application/json" -H "$auth_header")
    ssh_host=$(echo "$info" | python3 -c "import sys,json; print(json.load(sys.stdin)['ssh_host'])")
    ssh_port=$(echo "$info" | python3 -c "import sys,json; print(json.load(sys.stdin)['ssh_port'])")
  fi

  local SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $ssh_port root@$ssh_host"
  local SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $ssh_port"

  echo ">> Installing Node.js 22 on VM..."
  $SSH "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs > /dev/null 2>&1 && node --version"

  echo ">> Installing Python TTS deps..."
  $SSH "pip install -q TTS==0.22.0 fastapi uvicorn python-multipart 2>/dev/null"

  echo ">> Copying project files..."
  $SSH "mkdir -p /root/wolne-tts"
  $SCP -r "$PROJECT_DIR/src" "root@$ssh_host:/root/wolne-tts/"
  $SCP "$PROJECT_DIR/package.json" "$PROJECT_DIR/tsconfig.json" "root@$ssh_host:/root/wolne-tts/"

  echo ">> Installing npm dependencies on VM..."
  $SSH "cd /root/wolne-tts && npm install --omit=dev 2>&1 | tail -5"

  echo ">> Generating reference audio..."
  $SSH 'python3 -c "
from TTS.api import TTS
tts = TTS(\"tts_models/multilingual/multi-dataset/xtts_v2\").to(\"cuda\")
tts.tts_to_file(text=\"Witaj, jestem polskim lektorem.\", language=\"pl\", file_path=\"/root/ref_audio.wav\", speaker=\"Claribel Dervla\")
print(\"Reference audio generated\")
"'

  echo ">> Starting XTTS server (background)..."
  $SSH "cd /root/wolne-tts && nohup python3 src/mastra/scripts/xtts_server.py > /root/xtts.log 2>&1 &"

  echo ">> Waiting for XTTS to load model (~2-3 min)..."
  for i in $(seq 1 60); do
    sleep 5
    if $SSH "curl -sf http://localhost:5002/health" 2>/dev/null | grep -q '"model_loaded": true'; then
      echo ">> XTTS server healthy!"
      break
    fi
    echo "   waiting $i/60..."
  done

  echo ">> Building Mastra..."
  $SSH "cd /root/wolne-tts && npx mastra build 2>&1 | tail -5"

  echo ">> Starting Mastra server (background, port 3000)..."
  $SSH "cd /root/wolne-tts && XTTS_URL=http://localhost:5002 nohup npx mastra start --port 3000 > /root/mastra.log 2>&1 &"

  sleep 3
  echo ">> Checking Mastra health..."
  if $SSH "curl -sf http://localhost:3000/api/health" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "  DEPLOYED SUCCESSFULLY"
    echo "  Instance: $INSTANCE_ID"
    echo "  SSH: ssh -p $ssh_port root@$ssh_host"
    echo "=========================================="

    local info
    info=$(curl -s "$VAST_API/instances/$INSTANCE_ID/" -H "Accept: application/json" -H "$auth_header")
    local public_ip
    public_ip=$(echo "$info" | python3 -c "import sys,json; print(json.load(sys.stdin).get('public_ipaddr',''))" 2>/dev/null || echo "")
    local port_3000
    port_3000=$(echo "$info" | python3 -c "import sys,json; p=json.load(sys.stdin).get('ports',{}); e=p.get('3000/tcp',[{}]); print(e[0].get('HostPort','3000'))" 2>/dev/null || echo "3000")

    if [ -n "$public_ip" ]; then
      echo "  Mastra API: http://$public_ip:$port_3000"
      echo "  XTTS direct: (internal only, via Mastra)"
    fi
    echo "=========================================="
  else
    echo ">> WARNING: Mastra not responding yet. Check logs: ssh -p $ssh_port root@$ssh_host cat /root/mastra.log"
  fi
}

if [ -z "$INSTANCE_ID" ]; then
  provision_vm
fi
deploy_to_vm
