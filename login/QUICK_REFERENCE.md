# 🚀 Quick Reference Card

## 🤖 Option A: Fully Automated (Recommended)

### First Time Setup (10 minutes)

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# 1. Setup tunnel
./setup-persistent-tunnel.sh

# 2. Set password
export LAUNCHER_PASSWORD="BrAIn-#1"
echo 'export LAUNCHER_PASSWORD="BrAIn-#1"' >> ~/.bashrc

# 3. Setup launcher auto-start
./setup-launcher-autostart.sh
```

### Daily Usage (0 seconds!)

**Nothing!** Just open: `https://brain-nw1-launcher.trycloudflare.com`

Everything auto-starts on boot! 🎉

---

## 👤 Option B: Manual Launcher Start

### First Time Setup (5 minutes)

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# 1. Run setup (authenticate with Cloudflare once)
./setup-persistent-tunnel.sh

# 2. Set your password
export LAUNCHER_PASSWORD="BrAIn-#1"
echo 'export LAUNCHER_PASSWORD="BrAIn-#1"' >> ~/.bashrc

# 3. Start launcher manually
./start-persistent-launcher.sh
```

### Daily Usage (30 seconds)

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./start-persistent-launcher.sh
```

Then open: `https://brain-nw1-launcher.trycloudflare.com`

---

## 🌐 Your URLs (Always the Same)

- 🔐 Launcher: `https://brain-nw1-launcher.trycloudflare.com`
- 🚀 Main App: `https://brain-nw1-app.trycloudflare.com`

---

## Common Commands

### Service Management (If Using Auto-Start)

```bash
# Check tunnel service
sudo systemctl status cloudflared-brain

# Check launcher service
sudo systemctl status brain-launcher

# Restart launcher
sudo systemctl restart brain-launcher

# Stop launcher
sudo systemctl stop brain-launcher

# Disable auto-start
sudo systemctl disable brain-launcher
```

### Manual Process Management

```bash
# Check if launcher is running
lsof -i :8502

# Check if main app is running
lsof -i :8501

# Stop launcher (manual mode)
lsof -ti:8502 | xargs -r kill -9

# Stop main app
lsof -ti:8501 | xargs -r kill -9
```

### Logs

```bash
# Tunnel logs
sudo journalctl -u cloudflared-brain -f

# Launcher service logs (if auto-start)
sudo journalctl -u brain-launcher -f

# Launcher app logs
tail -f /tmp/launcher.log

# Main app logs
tail -f /tmp/rag_app.log
```

---

## Workflow

1. **Access Launcher**: `https://brain-nw1-launcher.trycloudflare.com`
2. **Login** with your password
3. **Click "Start App"** → starts app_v2_0g.py on port 8501
4. **Click "Open RAG Researcher App"** → opens main app
5. **Use your RAG app** from anywhere!
6. **Click "Stop App"** when finished

---

## Troubleshooting

**Tunnel not working?**
```bash
sudo systemctl restart cloudflared-brain
sudo journalctl -u cloudflared-brain -f
```

**Launcher not starting?**
```bash
tail -f /tmp/launcher.log
```

**Port already in use?**
```bash
lsof -ti:8502 | xargs -r kill -9  # Launcher
lsof -ti:8501 | xargs -r kill -9  # Main app
```

---

## Why This Setup?

✅ **No HuggingFace/Streamlit Cloud needed**  
✅ **Everything runs on your machine**  
✅ **Access from anywhere via HTTPS**  
✅ **Persistent URLs**  
✅ **Free, fast, secure**  

See `WHY_CLOUDFLARE_NOT_CLOUD.md` for detailed explanation.

---

## File Locations

- **Setup scripts**: `/home/he/ai/dev/langgraph/KB_BS_local-rag-he/login/`
- **Launcher app**: `login/launcher_app.py`
- **Main app**: `apps/app_v2_0g.py`
- **Launcher logs**: `/tmp/launcher.log`
- **Main app logs**: `/tmp/rag_app.log`
- **Tunnel config**: `~/.cloudflared/config.yml`
- **Tunnel credentials**: `~/.cloudflared/<tunnel-id>.json`

---

## Security

- 🔐 Password-protected launcher
- 🔒 HTTPS encrypted (Cloudflare)
- 🚫 No open ports on your machine
- 🌐 Accessible from anywhere with internet

**Optional enhancements:**
- Add custom domain for prettier URLs
- Set up Cloudflare Access for 2FA
- Add IP restrictions

---

## Support

Read detailed guides:
- `SETUP_PERSISTENT_NOW.md` - Step-by-step setup guide
- `WHY_CLOUDFLARE_NOT_CLOUD.md` - Why this approach is best
- `PERSISTENT_SETUP.md` - Comprehensive documentation
- `WORKFLOW.md` - How the system works
