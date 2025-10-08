# Quick Start Guide

## 🎯 What This Does

**Access your RAG Researcher from anywhere in the world via web browser:**

```
┌─────────────────────────────────────────────────────────┐
│  You (anywhere) → Browser                               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────┐
│  Cloudflare Tunnel (encrypted)                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────┐
│  Your Remote Machine                                    │
│  ┌─────────────────────────────────────────────┐        │
│  │ Port 8502: Launcher App                     │        │
│  │ - Login with password                       │        │
│  │ - Click "Start App" button                  │ ───┐   │
│  │ - Click "Open App" button                   │    │   │
│  └─────────────────────────────────────────────┘    │   │
│                                                      │   │
│  ┌─────────────────────────────────────────────┐    │   │
│  │ Port 8501: RAG App (app_v2_0g.py)           │ ←──┘   │
│  │ - Started/stopped by launcher               │        │
│  │ - Full RAG researcher interface             │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

**Result:** Use your RAG app in browser without VPN or SSH!

---

## Option A: Automated Cloudflare Tunnel Setup (Recommended)

This is the easiest way to get remote access to your RAG application.

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Run the automated setup script
./cloudflared-setup.sh
```

Follow the prompts. The script will:
1. Install cloudflared (if needed)
2. Authenticate with Cloudflare
3. Create a tunnel
4. Configure DNS routes
5. Set up systemd service to auto-start

After setup completes, your tunnel will be running automatically!

## Option B: Manual Quick Test

Just want to test locally first?

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Set your password
export LAUNCHER_PASSWORD="your_secure_password"

# Start the launcher
./start-launcher.sh
```

Then open: http://localhost:8502

## Accessing Your Apps

Once Cloudflare Tunnel is set up:

1. **Launcher Interface**: `https://launcher-YOUR_TUNNEL.your-domain.com`
   - Login with your LAUNCHER_PASSWORD
   - Start/stop/restart the main RAG app
   - View logs and status

2. **Main RAG App**: `https://rag-app-YOUR_TUNNEL.your-domain.com`
   - Your app_v2_0g.py running on port 8501
   - Access from anywhere in the world

## Common Issues

### "cloudflared: command not found"

```bash
# Install manually
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

### "Port 8501 already in use"

From the launcher app, click "Stop App" or run:
```bash
lsof -ti:8501 | xargs -r kill -9
```

### Tunnel not working

Check the service status:
```bash
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -f
```

## Security Recommendations

1. **Use a strong password** for LAUNCHER_PASSWORD
2. **Keep the tunnel private** - don't share URLs publicly
3. **Enable Cloudflare Access** for additional authentication
4. **Monitor logs** regularly for suspicious activity

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [README_HUGGINGFACE.md](README_HUGGINGFACE.md) for HuggingFace deployment
- Configure your domain in `~/.cloudflared/config.yml`
