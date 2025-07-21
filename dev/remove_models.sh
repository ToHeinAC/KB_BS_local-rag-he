#!/bin/bash

# Script to remove Ollama models listed in ollama_remove.md
# Created on 2025-07-21

# Models to remove
models=(
  "llama4:latest"
  "llama3.3:70b-instruct-q4_K_M"
  "mistral-nemo:latest"
  "mistrallite:latest"
  "mistral:instruct"
  "gemma3:4b"
  "phi4-mini:latest"
  "mistral-small:latest"
  "deepseek-r1:1.5b"
  "qwen2.5:14b-instruct"
  "gemma3:1b"
  "phi4:latest"
  "gemma3:27b"
  "gemma3:latest"
  "llama3.3:latest"
  "deepseek-r1:14b"
  "nemotron:latest"
  "llama3.1:8b-instruct-q8_0"
  "EvilAtomicLobbyist:latest"
  "SuperMarioExample:latest"
  "stable-code:latest"
  "codegemma:latest"
  "llama3.1:latest"
  "llama3.1:70b"
  "llama3.1:8b-instruct-q4_0"
  "llama3-groq-tool-use:latest"
  "llava:13b"
  "mistral:latest"
  "llava:latest"
  "phi3:14b"
  "phi3:latest"
  "llama3:70b-instruct-q8_0"
  "llama3:8b-instruct-q8_0"
  "all-minilm:latest"
  "mxbai-embed-large:latest"
  "quentinz/bge-large-zh-v1.5:latest"
  "llama3:instruct"
  "llama3:70b-instruct"
  "mistral:7b-instruct"
  "llama3:70b"
)

# Remove each model
for model in "${models[@]}"; do
  echo "Removing $model..."
  ollama rm "$model"
done

echo "All specified models have been removed."
echo "Listing remaining models:"
ollama list
