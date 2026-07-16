#!/bin/bash
# Principal Architect's Production Bootstrap

# Define absolute paths for stability
PROJECT_ROOT=$(pwd)
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=$PROJECT_ROOT/.ollama_models
# Force LD_LIBRARY_PATH to see Lightning AI's CUDA drivers
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

echo "📡 Starting Ollama Production Server..."
pkill -9 ollama || true
sleep 2

# Start server using the local binary
nohup ./ollama serve > ollama.log 2>&1 &

echo "⏳ Liveness Probe: Waiting for API binding..."
for i in {1..20}; do
    if curl -s http://127.0.0.1:11434 > /dev/null; then
        echo "✅ Heartbeat Detected. System is ALIVE."
        break
    fi
    if [ $i -eq 20 ]; then
        echo "❌ ERROR: Log Dump:"
        tail -n 10 ollama.log
        exit 1
    fi
    echo "   Checking... ($i/20)"
    sleep 3
done

echo "🧠 Provisioning Qwen2.5-3B-Instruct..."
./ollama pull qwen2.5:3b-instruct

echo "🎯 MNC-READY: Your infrastructure is now stable."