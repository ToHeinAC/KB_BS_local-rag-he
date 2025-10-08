# RAG Researcher Remote Access Setup

Access your RAG Researcher application from anywhere in the world via web browser!

## 🎯 How It Works

**The Complete Workflow:**

1. **Cloudflare Tunnel** → Points to your remote machine (encrypted, no port forwarding needed)
2. **Launcher App** (Port 8502) → Password-protected control panel
3. **Login** → Enter your password in the launcher
4. **Click "Start App"** → Launcher starts `apps/app_v2_0g.py` on port 8501
5. **Click "Open App"** → Browser opens the RAG researcher
6. **Use your RAG app** → Full functionality in browser
7. **Click "Stop App"** → Cleanly shuts down when you're done

**Result:** No VPN, no SSH tunnels, just open a URL and click buttons!

---

This directory contains everything needed to set up this remote access system.

## 📁 Files Overview

- **`launcher_app.py`** - Streamlit app for remotely starting/stopping the main RAG app
- **`cloudflared-setup.sh`** - Automated script to set up Cloudflare Tunnel
- **`cloudflared-config.yml`** - Example configuration for Cloudflare Tunnel
- **`start-launcher.sh`** - Quick start script for the launcher
- **`test-setup.sh`** - Verify your setup is correct
- **`requirements.txt`** - Python dependencies for the launcher
- **`.env.example`** - Example environment variables file
- **`WORKFLOW.md`** - Detailed workflow documentation
- **`QUICK_START.md`** - Quick start guide

## 🚀 Quick Start Guide

### Option 1: Full Remote Access (Recommended)

This setup allows you to access your app from anywhere via a public URL.

#### Step 1: Set Up Cloudflare Tunnel

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Run the automated setup script
./cloudflared-setup.sh
```

The script will:
1. Install cloudflared (if needed)
2. Authenticate with Cloudflare
3. Create a tunnel
4. Configure DNS routes
5. Set up a systemd service

#### Step 2: Start the Launcher

```bash
# Set your password
export LAUNCHER_PASSWORD="your_secure_password_here"

# Start the launcher on port 8502
streamlit run launcher_app.py --server.port 8502 --server.headless true
```

#### Step 3: Access Your Apps

- **Launcher**: `https://launcher-YOUR_TUNNEL.your-domain.com`
- **Main App**: `https://rag-app-YOUR_TUNNEL.your-domain.com`

### Option 2: Local Testing

For testing without Cloudflare Tunnel:

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Set password
export LAUNCHER_PASSWORD="test123"

# Run launcher
streamlit run launcher_app.py --server.port 8502
```

Access at: `http://localhost:8502`

## 🔧 Manual Cloudflare Tunnel Setup

If the automated script doesn't work, follow these manual steps:

### 1. Install cloudflared

```bash
# Download and install
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

### 2. Authenticate

```bash
cloudflared tunnel login
```

### 3. Create Tunnel

```bash
cloudflared tunnel create rag-researcher
```

Note the tunnel ID from the output.

### 4. Create Configuration

```bash
mkdir -p ~/.cloudflared

cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: YOUR_TUNNEL_ID_HERE
credentials-file: /home/YOUR_USER/.cloudflared/YOUR_TUNNEL_ID_HERE.json

ingress:
  - hostname: rag-app.your-domain.com
    service: http://localhost:8501
  
  - hostname: launcher.your-domain.com
    service: http://localhost:8502
  
  - service: http_status:404
EOF
```

Replace:
- `YOUR_TUNNEL_ID_HERE` with your actual tunnel ID
- `YOUR_USER` with your username
- `your-domain.com` with your actual domain

### 5. Configure DNS

```bash
cloudflared tunnel route dns rag-researcher rag-app.your-domain.com
cloudflared tunnel route dns rag-researcher launcher.your-domain.com
```

### 6. Create Systemd Service

```bash
sudo tee /etc/systemd/system/cloudflared.service > /dev/null << 'EOF'
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=YOUR_USER
ExecStart=/usr/bin/cloudflared tunnel --config /home/YOUR_USER/.cloudflared/config.yml run
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

# Replace YOUR_USER with your actual username
sudo sed -i "s/YOUR_USER/$USER/g" /etc/systemd/system/cloudflared.service
```

### 7. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

## 🔐 Security Best Practices

1. **Use Strong Passwords**
   ```bash
   # Generate a secure password
   openssl rand -base64 32
   ```

2. **Restrict Access**
   - Use Cloudflare Access policies
   - Enable Cloudflare WAF rules
   - Consider IP whitelisting

3. **Environment Variables**
   - Never commit `.env` files with real passwords
   - Use environment variables or secrets management
   - Rotate passwords regularly

4. **Monitor Access**
   ```bash
   # View tunnel logs
   sudo journalctl -u cloudflared -f
   
   # View app logs
   tail -f /tmp/rag_app.log
   ```

## 🛠️ Troubleshooting

### Cloudflared Service Issues

**Problem**: `Unit cloudflared.service not found`

**Solution**: 
1. Check if config exists: `cat ~/.cloudflared/config.yml`
2. Verify service file: `cat /etc/systemd/system/cloudflared.service`
3. Run setup script again

**Problem**: `Cannot determine default configuration path`

**Solution**: Create config file at `~/.cloudflared/config.yml`

### Launcher Issues

**Problem**: Can't start app from launcher

**Solution**:
1. Check that `uv` is installed: `which uv`
2. Verify app path is correct in `launcher_app.py`
3. Check logs: `cat /tmp/rag_app.log`

**Problem**: Port already in use

**Solution**:
```bash
# Kill process on port
lsof -ti:8501 | xargs -r kill -9
```

### Connection Issues

**Problem**: Can't access via tunnel URL

**Solution**:
1. Check tunnel status: `sudo systemctl status cloudflared`
2. Verify DNS: `nslookup rag-app.your-domain.com`
3. Check local app is running: `curl http://localhost:8501`

## 📊 Monitoring

### Check Tunnel Status

```bash
# Service status
sudo systemctl status cloudflared

# View logs
sudo journalctl -u cloudflared -f

# List tunnels
cloudflared tunnel list

# Get tunnel info
cloudflared tunnel info rag-researcher
```

### Check App Status

```bash
# Check if app is running
lsof -i :8501

# View app logs
tail -f /tmp/rag_app.log

# Check resource usage
ps aux | grep streamlit
```

## 🌐 Alternative: HuggingFace Deployment

See [README_HUGGINGFACE.md](README_HUGGINGFACE.md) for deploying the launcher to HuggingFace Spaces.

## 📝 Useful Commands

```bash
# Restart tunnel
sudo systemctl restart cloudflared

# Stop tunnel
sudo systemctl stop cloudflared

# Disable tunnel autostart
sudo systemctl disable cloudflared

# Test tunnel configuration
cloudflared tunnel --config ~/.cloudflared/config.yml ingress validate

# Run tunnel manually (for testing)
cloudflared tunnel --config ~/.cloudflared/config.yml run
```

## 🆘 Getting Help

If you encounter issues:

1. Check the logs: `sudo journalctl -u cloudflared -xe`
2. Verify configuration: `cat ~/.cloudflared/config.yml`
3. Test local connectivity: `curl http://localhost:8501`
4. Consult [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)

## 📄 License

This launcher is part of the RAG Researcher project.
