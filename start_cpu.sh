#!/bin/bash
echo "🖥️  Configuring LegalSathi for CPU Mode..."

# 1. Start the server if the system service didn't start it properly
# We use 127.0.0.1 to stay local
export OLLAMA_HOST=127.0.0.1:11434

# 2. Check if server is already running, if not, start it
if ! curl -s http://127.0.0.1:11434 > /dev/null; then
    echo "📡 Starting Ollama background process..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 5
fi

# 3. Pull the model (3B runs great on CPU)
echo "📥 Pulling Qwen 2.5 (3B)..."
ollama pull qwen2.5:3b-instruct

echo "✅ System Ready in CPU Mode."