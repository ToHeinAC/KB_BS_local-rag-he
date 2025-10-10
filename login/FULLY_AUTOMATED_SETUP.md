# 🤖 Fully Automated Setup - Zero Daily Maintenance

## 🎯 Goal: Set It Up Once, Never Touch It Again

After this setup, **everything auto-starts on boot**. Just open the URL and go!

---

## 📋 Complete One-Time Setup (10 minutes)

### Step 1: Set Up Cloudflare Tunnel (Persistent URLs)

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Install and configure Cloudflare Tunnel
./setup-persistent-tunnel.sh
```

**What this does:**
- Authenticates with Cloudflare (browser opens once)
- Creates named tunnel "brain-nw1"
- Sets up systemd service `cloudflared-brain`
- Tunnel auto-starts on boot ✅

### Step 2: Set Your Password

```bash
# Set password
export LAUNCHER_PASSWORD="BrAIn-#1"

# Make it permanent
echo 'export LAUNCHER_PASSWORD="BrAIn-#1"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Set Up Launcher Auto-Start

```bash
# Install launcher as systemd service
./setup-launcher-autostart.sh
```

**What this does:**
- Creates systemd service `brain-launcher`
- Launcher auto-starts on boot ✅
- Automatically restarts if it crashes ✅

---

## ✅ What Happens After Setup?

### On Every Boot (Automatic):

1. **Cloudflared Tunnel** starts → URLs are live
2. **Launcher App** starts → `https://brain-nw1-launcher.trycloudflare.com` is ready
3. **You do nothing!** Just open the URL whenever you want

### When You Want to Use It:

1. Open `https://brain-nw1-launcher.trycloudflare.com`
2. Login with password: `BrAIn-#1`
3. Click "Start App" → launches app_v2_0g.py
4. Click "Open RAG Researcher App"
5. Done! 🎉

---

## 🔄 What Runs Automatically?

| Component | Auto-Start? | Auto-Restart on Crash? | Command |
|-----------|-------------|------------------------|---------|
| **Cloudflare Tunnel** | ✅ YES | ✅ YES | `sudo systemctl status cloudflared-brain` |
| **Launcher App** | ✅ YES | ✅ YES | `sudo systemctl status brain-launcher` |
| **Main App (app_v2_0g.py)** | ❌ NO | - | Started via launcher UI |

**Main app** is controlled by you through the launcher interface - start/stop as needed!

---

## 📊 Service Dependencies

```
Boot Sequence:
  1. Network comes up
  2. cloudflared-brain starts → Tunnel is live
  3. brain-launcher starts → Launcher is accessible
  4. You start main app via launcher → app_v2_0g.py runs
```

**If tunnel fails**, launcher waits for it (systemd `Wants` directive).

---

## 🛠️ Management Commands

### Check Status

```bash
# Check tunnel
sudo systemctl status cloudflared-brain

# Check launcher
sudo systemctl status brain-launcher

# Check if launcher is accessible
curl -I http://localhost:8502

# Check if main app is running
lsof -i :8501
```

### View Logs

```bash
# Tunnel logs
sudo journalctl -u cloudflared-brain -f

# Launcher service logs
sudo journalctl -u brain-launcher -f

# Launcher app logs
tail -f /tmp/launcher.log

# Main app logs
tail -f /tmp/rag_app.log
```

### Manual Control

```bash
# Restart launcher
sudo systemctl restart brain-launcher

# Stop launcher
sudo systemctl stop brain-launcher

# Start launcher
sudo systemctl start brain-launcher

# Disable auto-start (stops on next boot)
sudo systemctl disable brain-launcher

# Re-enable auto-start
sudo systemctl enable brain-launcher
```

---

## 🔧 Troubleshooting

### Launcher Not Accessible After Boot

```bash
# Check service status
sudo systemctl status brain-launcher

# View recent logs
sudo journalctl -u brain-launcher -n 50

# Check if port is listening
lsof -i :8502

# Restart manually
sudo systemctl restart brain-launcher
```

### Tunnel Not Working

```bash
# Check tunnel status
sudo systemctl status cloudflared-brain

# View tunnel logs
sudo journalctl -u cloudflared-brain -n 50

# Restart tunnel
sudo systemctl restart cloudflared-brain
```

### Password Not Working

The password is embedded in the systemd service file. To change it:

```bash
# 1. Update environment variable
export LAUNCHER_PASSWORD="NewPassword"
echo 'export LAUNCHER_PASSWORD="NewPassword"' >> ~/.bashrc

# 2. Re-run setup
./setup-launcher-autostart.sh
```

### Completely Reset Everything

```bash
# Stop and disable services
sudo systemctl stop brain-launcher
sudo systemctl disable brain-launcher
sudo systemctl stop cloudflared-brain
sudo systemctl disable cloudflared-brain

# Remove service files
sudo rm /etc/systemd/system/brain-launcher.service
sudo rm /etc/systemd/system/cloudflared-brain.service

# Reload systemd
sudo systemctl daemon-reload

# Delete tunnel (optional - creates new URLs)
cloudflared tunnel delete brain-nw1

# Start fresh
./setup-persistent-tunnel.sh
./setup-launcher-autostart.sh
```

---

## 🔒 Security Considerations

### Current Setup

✅ **Password-protected launcher** - Only you can start apps  
✅ **HTTPS encrypted** - All traffic via Cloudflare  
✅ **No open ports** - Outbound tunnel only  
✅ **Auto-restart** - Service recovers from crashes  
✅ **User-level service** - Runs as your user (not root)

### Password Storage

⚠️ **Note**: Password is stored in plaintext in:
- `/etc/systemd/system/brain-launcher.service`
- `~/.bashrc` (if you added it there)

**More secure alternatives:**

1. **Use systemd credentials** (recommended):
   ```bash
   # Create encrypted credential
   systemd-ask-password "Launcher password" | \
     sudo systemd-creds encrypt --name=launcher_password - \
     /etc/credstore/launcher_password.cred
   
   # Update service file to use LoadCredential
   ```

2. **Use environment file**:
   ```bash
   # Create /etc/brain-launcher.env (readable by service only)
   sudo nano /etc/brain-launcher.env
   # Add: LAUNCHER_PASSWORD=BrAIn-#1
   sudo chmod 600 /etc/brain-launcher.env
   
   # Update service to use EnvironmentFile=/etc/brain-launcher.env
   ```

For a personal server, plaintext in systemd is usually acceptable.

---

## 📈 Resource Usage

The launcher app is very lightweight:

- **CPU**: ~0.5% idle, ~5% during operations
- **Memory**: ~150-200 MB
- **Disk**: Logs in `/tmp/launcher.log` (auto-rotated)

**Main app** (app_v2_0g.py) uses more resources when running - you control when it runs!

---

## ✅ Benefits of Full Auto-Start

1. **Zero Maintenance** - Set up once, runs forever
2. **Survives Reboots** - Everything comes back up automatically
3. **Self-Healing** - Crashes trigger automatic restart
4. **Always Accessible** - URLs work 24/7
5. **Professional Setup** - Production-grade systemd services
6. **Easy Monitoring** - Standard systemctl commands

---

## 🎉 You're All Set!

After running the setup scripts:

```bash
./setup-persistent-tunnel.sh
./setup-launcher-autostart.sh
```

**You never need to run anything again!**

Just bookmark: `https://brain-nw1-launcher.trycloudflare.com`

The tunnel and launcher are **always running**, waiting for you to login and use the app! 🚀

---

## 📚 Related Documentation

- `SETUP_PERSISTENT_NOW.md` - Basic persistent setup (manual launcher start)
- `QUICK_REFERENCE.md` - Quick command reference
- `WHY_CLOUDFLARE_NOT_CLOUD.md` - Why this approach is best
- `PERSISTENT_SETUP.md` - Detailed tunnel documentation
