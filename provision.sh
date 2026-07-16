#!/bin/bash
echo "🚀 MNC-Grade System Provisioning..."

# 1. Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 2. Cleanup and Start
pkill -9 ollama || true
export OLLAMA_HOST=127.0.0.1:11434
# Use GPU (Automatic detection)
nohup ollama serve > ollama.log 2>&1 &

echo "⏳ Waiting for Heartbeat..."
until curl -s http://127.0.0.1:11434 > /dev/null; do
  sleep 2
done

echo "🧠 Pulling Qwen 2.5 (3B Instruct)..."
ollama pull qwen2.5:3b-instruct

echo "✅ INFRASTRUCTURE READY."