#!/bin/bash

# Run the filename extractor app on port 8508 using uv

uv run streamlit run dev/filename_extractor_app.py --server.port 8508
