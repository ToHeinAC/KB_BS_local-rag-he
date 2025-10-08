# Deploying Launcher to HuggingFace Spaces

This guide explains how to deploy the launcher app to HuggingFace Spaces.

## Prerequisites

- HuggingFace account
- Git installed locally

## Deployment Steps

### 1. Create a New Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Name**: `rag-researcher-launcher` (or your preferred name)
   - **License**: Your choice
   - **Space SDK**: Streamlit
   - **Visibility**: Private (recommended for launcher apps)

### 2. Prepare Files

The following files need to be in your Space repository:

```
launcher_app.py       # Main application
requirements.txt      # Python dependencies
.env                 # Environment variables (create this)
README.md            # Documentation
```

### 3. Set Up Environment Variables

In your HuggingFace Space settings:

1. Go to **Settings** > **Variables and secrets**
2. Add the following secrets:
   - `LAUNCHER_PASSWORD`: Your secure password for the launcher

### 4. Clone and Deploy

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/rag-researcher-launcher
cd rag-researcher-launcher

# Copy launcher files
cp /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login/launcher_app.py app.py
cp /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login/requirements.txt .

# Create README
cat > README.md << 'EOF'
---
title: RAG Researcher Launcher
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# RAG Researcher Launcher

Remote launcher for the RAG Researcher application.
EOF

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### 5. Access Your Launcher

Your launcher will be available at:
`https://huggingface.co/spaces/YOUR_USERNAME/rag-researcher-launcher`

## Important Notes

⚠️ **Security Considerations:**

1. **Always use Private Spaces** for launcher apps
2. **Use strong passwords** for LAUNCHER_PASSWORD
3. **Don't expose sensitive ports** publicly
4. Consider using HuggingFace's **authentication** features

## Alternative: Local Deployment

If you prefer not to use HuggingFace, you can run the launcher locally:

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Set environment variable
export LAUNCHER_PASSWORD="your_secure_password"

# Run the launcher
streamlit run launcher_app.py --server.port 8502
```

Then access it via your Cloudflare Tunnel.

## Troubleshooting

### App won't start from launcher

- Check that the path in `launcher_app.py` matches your setup
- Verify `uv` is installed and accessible
- Check logs in `/tmp/rag_app.log`

### Password not working

- Ensure `LAUNCHER_PASSWORD` is set in environment or HuggingFace secrets
- Check for typos in password

### Port already in use

- Use the "Stop App" button to kill existing processes
- Or manually: `lsof -ti:8501 | xargs -r kill -9`
