#!/bin/bash
echo "🚀 Starting LegalSathi AI Engine..."
pkill ollama
sleep 2
export OLLAMA_HOST=127.0.0.1:11434
nohup ollama serve > ollama.log 2>&1 &
echo "⏳ Waiting for server to wake up..."
sleep 8
ollama run qwen2.5:3b-instruct "hi" # This forces the model into VRAM
echo "✅ Engine is ready!"