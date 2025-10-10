# 🚀 Quick Start: Persistent Tunnel Setup

## You Already Have Everything! Just Follow These Steps:

### 1️⃣ One-Time Setup (5 minutes)

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Run the setup script
./setup-persistent-tunnel.sh
```

**What happens:**
- Opens browser for Cloudflare authentication (do it once)
- Creates named tunnel "brain-nw1" 
- Sets up systemd service for auto-start
- Your URLs will be:
  - **Launcher**: `https://brain-nw1-launcher.trycloudflare.com`
  - **Main App**: `https://brain-nw1-app.trycloudflare.com`

### 2️⃣ Set Your Password

```bash
# Set password for launcher authentication
export LAUNCHER_PASSWORD="BrAIn-#1"

# Make it permanent (optional)
echo 'export LAUNCHER_PASSWORD="BrAIn-#1"' >> ~/.bashrc
```

### 3️⃣ Start the Launcher

```bash
./start-persistent-launcher.sh
```

### 4️⃣ Access From Anywhere

1. **Open**: `https://brain-nw1-launcher.trycloudflare.com`
2. **Login** with your password
3. **Click "Start App"** → launches `app_v2_0g.py` on port 8501
4. **Click "Open RAG Researcher App"** → opens `brain-nw1-app.trycloudflare.com`
5. **Use your app** from anywhere in the world!
6. **Click "Stop App"** when finished

---

## 🔄 Daily Usage (After Setup)

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./start-persistent-launcher.sh
```

Then open: `https://brain-nw1-launcher.trycloudflare.com`

---

## ⚙️ How It Works

```
Internet → Cloudflare Tunnel → Your Machine
  ↓
brain-nw1-launcher.trycloudflare.com → localhost:8502 (launcher)
brain-nw1-app.trycloudflare.com → localhost:8501 (app_v2_0g.py)
```

**Architecture:**
- **Single Cloudflare Tunnel** routes both URLs to your machine
- **Launcher app** (port 8502) controls the main app
- **Main app** (port 8501) runs app_v2_0g.py when you click "Start"
- **Systemd service** keeps tunnel running automatically

---

## 🛠️ Management Commands

```bash
# Check if tunnel is running
sudo systemctl status cloudflared-brain

# View tunnel logs
sudo journalctl -u cloudflared-brain -f

# Restart tunnel
sudo systemctl restart cloudflared-brain

# Check what's running on ports
lsof -i :8502  # Launcher
lsof -i :8501  # Main app

# Stop launcher
lsof -ti:8502 | xargs -r kill -9

# Stop main app
lsof -ti:8501 | xargs -r kill -9
```

---

## ❓ Why Not HuggingFace/Streamlit Cloud?

You **don't need** external hosting because:

| Cloudflare Tunnel (Your Setup) | HuggingFace/Streamlit Cloud |
|-------------------------------|----------------------------|
| ✅ Everything runs on your machine | ❌ Need to expose machine to internet |
| ✅ Access to all local data/models | ❌ Can't access remote files |
| ✅ No upload/sync needed | ❌ Must sync code changes |
| ✅ Full control | ❌ Limited resources |
| ✅ Free, fast, secure | ❌ More complex setup |

**Cloudflare Tunnel = Your machine becomes the server, accessible from anywhere**

---

## 🔒 Security Notes

✅ **Password-protected launcher** - Only you can start/stop apps  
✅ **HTTPS encrypted** - All traffic encrypted via Cloudflare  
✅ **No open ports** - Tunnel is outbound connection only  
✅ **No external hosting** - Data never leaves your machine  

**Optional Enhancements:**
- Add custom domain for prettier URLs
- Set up Cloudflare Access for 2FA
- Add IP restrictions via Cloudflare firewall

---

## 🎯 Summary

**You already have the perfect setup!** Just run:

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./setup-persistent-tunnel.sh  # One-time
export LAUNCHER_PASSWORD="your_password"
./start-persistent-launcher.sh  # Daily
```

Then access from anywhere: `https://brain-nw1-launcher.trycloudflare.com`

**No HuggingFace/Streamlit Cloud needed!** 🎉
