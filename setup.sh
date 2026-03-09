#!/usr/bin/env bash
set -euo pipefail

echo "=== Updating package index ==="
sudo apt-get update

echo "=== Installing Python and pip ==="
sudo apt-get install -y python3 python3-pip python3-venv curl

echo "=== Setting up swap space to prevent OOM kills ==="
if [ ! -f /swapfile ]; then
    FREE_SPACE=$(df / | awk 'NR==2 {print int($4)}')
    SWAP_SIZE=$(awk "BEGIN {printf \"%.0f\", $FREE_SPACE * 0.9 / 1024}")
    echo "Free storage: ${FREE_SPACE}KB, allocating ${SWAP_SIZE}GB for swap (90%)"
    sudo fallocate -l ${SWAP_SIZE}G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1G count=${SWAP_SIZE}
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    echo "Swap space created and enabled"
else
    echo "Swap space already exists"
fi

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Running init.py ==="
nohup python init.py > init.log 2>&1 &
INIT_PID=$!
wait $INIT_PID

echo "=== Starting server in background ==="
nohup python server.py > server.log 2>&1 &
SERVER_PID=$!

echo "=== Waiting for server to start ==="
sleep 5

echo "=== Querying /health ==="
curl -s http://localhost:5000/health
echo ""

echo "=== Querying /test ==="
curl -s http://localhost:5000/test
echo ""

echo "=== Server is running (PID: $SERVER_PID) ==="
wait $SERVER_PID
