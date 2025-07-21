#!/bin/bash

# Script to update remaining Ollama models to latest official releases
# Created on 2025-07-21

# Function to extract base model name (remove version/tag)
get_base_model() {
  local model=$1
  
  # Handle special cases with namespaces
  if [[ $model == *"/"* ]]; then
    echo "$model"  # Return as is for namespaced models
    return
  fi
  
  # Extract base model name (before the colon)
  echo "${model%%:*}"
}

echo "Updating remaining Ollama models to their latest official releases..."

# Get list of current models
models=$(ollama list | tail -n +2 | awk '{print $1}')

# Update each model
for model in $models; do
  base_model=$(get_base_model "$model")
  
  # Skip updating models that are already tagged as latest
  if [[ "$model" == "$base_model:latest" ]]; then
    echo "Updating $model..."
    ollama pull "$model"
  # Skip updating models with custom namespaces
  elif [[ "$model" == *"/"* ]]; then
    echo "Skipping custom namespaced model: $model"
  # Update other models to their latest version
  else
    echo "Updating $base_model to latest version..."
    ollama pull "$base_model:latest"
  fi
done

echo "All models updated. Current models:"
ollama list
