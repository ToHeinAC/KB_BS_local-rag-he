# Complete Workflow: Remote Access to RAG Researcher

## Architecture Overview

```
Internet
   ↓
Cloudflare Tunnel (encrypted)
   ↓
Your Remote Machine
   ├─→ Port 8502: Launcher App (launcher_app.py)
   └─→ Port 8501: Main RAG App (apps/app_v2_0g.py)
```

## Step-by-Step Workflow

### 1. Initial Setup (One-time)

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Run automated setup
./cloudflared-setup.sh
```

This creates:
- Cloudflare Tunnel
- DNS routes
- Systemd service (auto-starts on boot)

### 2. Access from Anywhere

**Open your browser and go to:**
```
https://launcher.your-domain.com
```

### 3. Login to Launcher

- Enter your `LAUNCHER_PASSWORD`
- You'll see the control panel

### 4. Start the RAG App

Click the **"🚀 Start App"** button

- The launcher executes: `uv run streamlit run apps/app_v2_0g.py --server.port 8501`
- App starts in the background on your remote machine
- Status changes to "✅ App is RUNNING"

### 5. Access the RAG App

Click the **"🚀 Open RAG Researcher App"** button

This opens: `https://rag-app.your-domain.com` in a new tab

**Now you're using app_v2_0g.py in your browser!**

### 6. Stop the App (When Done)

Go back to the launcher tab and click **"🛑 Stop App"**

- Kills the process on port 8501
- App shuts down cleanly
- Frees up resources

## URLs Summary

| Service | URL | Purpose |
|---------|-----|---------|
| **Launcher** | `https://launcher.your-domain.com` | Control panel - start/stop apps |
| **Main App** | `https://rag-app.your-domain.com` | The actual RAG researcher (app_v2_0g.py) |

## Environment Variables

Set these for proper URL display in the launcher:

```bash
export LAUNCHER_PASSWORD="your_secure_password"
export LAUNCHER_URL="https://launcher.your-domain.com"
export MAIN_APP_URL="https://rag-app.your-domain.com"
```

Add to your `~/.bashrc` or `/etc/systemd/system/launcher.service`

## Security Notes

✅ **What's Secure:**
- All traffic encrypted via Cloudflare Tunnel
- Password authentication on launcher
- No direct SSH or VPN needed
- Apps only accessible via your domain

⚠️ **Recommendations:**
- Use a strong `LAUNCHER_PASSWORD` (32+ characters)
- Enable Cloudflare Access for additional authentication
- Monitor logs regularly: `sudo journalctl -u cloudflared -f`
- Keep launcher private (don't share the URL)

## Troubleshooting

### Launcher won't start

```bash
# Check tunnel status
sudo systemctl status cloudflared

# Check launcher logs
tail -f /tmp/rag_app.log
```

### Can't access via browser

1. Verify tunnel is running: `sudo systemctl status cloudflared`
2. Check DNS: `nslookup launcher.your-domain.com`
3. Test locally: `curl http://localhost:8502`
4. Check firewall: `sudo ufw status`

### App won't start from launcher

1. Check `uv` is installed: `which uv`
2. Verify app path: `ls -la /home/he/ai/dev/langgraph/KB_BS_local-rag-he/apps/app_v2_0g.py`
3. Check logs: `cat /tmp/rag_app.log`

## Example Session

```
# You (from anywhere in the world):
1. Open browser → https://launcher.your-domain.com
2. Login with password
3. Click "Start App" button
4. Wait 3 seconds for app to start
5. Click "Open RAG Researcher App" button
6. New tab opens → https://rag-app.your-domain.com
7. Use the RAG researcher normally
8. When done, go back to launcher tab
9. Click "Stop App" button
10. App shuts down
```

## Benefits of This Approach

✅ **No VPN Required** - Just open a URL  
✅ **Secure** - Encrypted Cloudflare Tunnel  
✅ **Convenient** - Start/stop apps remotely  
✅ **Clean** - Proper shutdown when done  
✅ **Monitored** - See CPU, memory, logs  
✅ **Simple** - Just click buttons in a web UI  

## Advanced: Auto-Start Launcher

If you want the launcher to start automatically:

```bash
# Create systemd service for launcher
sudo tee /etc/systemd/system/launcher.service > /dev/null << 'EOF'
[Unit]
Description=RAG Researcher Launcher
After=network.target cloudflared.service

[Service]
Type=simple
User=he
WorkingDirectory=/home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
Environment="LAUNCHER_PASSWORD=YOUR_PASSWORD_HERE"
Environment="LAUNCHER_URL=https://launcher.your-domain.com"
Environment="MAIN_APP_URL=https://rag-app.your-domain.com"
ExecStart=/usr/bin/streamlit run launcher_app.py --server.port 8502 --server.headless true
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable launcher
sudo systemctl start launcher
```

Now both the tunnel AND launcher start on boot!
