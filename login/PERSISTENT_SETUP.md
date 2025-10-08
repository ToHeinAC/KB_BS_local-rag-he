# Persistent Tunnel Setup - BrAIn-nw1

## 🎯 What This Provides

✅ **Persistent URLs** - Same URLs every time you restart  
✅ **Single Tunnel** - Both apps accessible through one tunnel  
✅ **Named Tunnel** - "brain-nw1" identifier  
✅ **Auto-start** - Tunnel runs as systemd service  
✅ **No Manual Steps** - Start and forget  

## 🚀 Quick Setup (One-Time)

### Step 1: Run the Setup Script

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./setup-persistent-tunnel.sh
```

This will:
1. ✅ Install cloudflared (if needed)
2. ✅ Authenticate with Cloudflare (browser opens once)
3. ✅ Create named tunnel "brain-nw1"
4. ✅ Configure routing for both services
5. ✅ Set up systemd service for auto-start
6. ✅ Start the tunnel

### Step 2: Set Your Password

```bash
export LAUNCHER_PASSWORD="your_secure_password"
```

Add to `~/.bashrc` to make it permanent:
```bash
echo 'export LAUNCHER_PASSWORD="your_secure_password"' >> ~/.bashrc
```

### Step 3: Start the Launcher

```bash
./start-persistent-launcher.sh
```

## 🌐 Your URLs

Once setup is complete, these URLs will always work:

- **Launcher**: `https://brain-nw1-launcher.trycloudflare.com`
- **Main App**: `https://brain-nw1-app.trycloudflare.com`

Both URLs are routed through a **single tunnel** - no need to manage multiple tunnels!

## 📋 How It Works

### Single Tunnel Architecture

```
┌──────────────────────────────────────────────────┐
│  Cloudflare Tunnel: brain-nw1                    │
│  (One tunnel, multiple routes)                   │
└──────────────┬───────────────────────────────────┘
               │
               ├──→ brain-nw1-launcher.trycloudflare.com
               │    └──→ http://localhost:8502 (Launcher)
               │
               └──→ brain-nw1-app.trycloudflare.com
                    └──→ http://localhost:8501 (Main App)
```

### Configuration File

The tunnel uses `~/.cloudflared/config.yml`:

```yaml
tunnel: <tunnel-id>
credentials-file: /home/he/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: brain-nw1-launcher.trycloudflare.com
    service: http://localhost:8502
  
  - hostname: brain-nw1-app.trycloudflare.com
    service: http://localhost:8501
  
  - service: http_status:404
```

This means:
- **One tunnel daemon** handles both services
- **Automatic routing** based on hostname
- **No manual tunnel management** needed

## 🔄 Daily Usage

### Start Everything

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./start-persistent-launcher.sh
```

This:
1. Checks tunnel is running (starts if needed)
2. Starts the launcher app
3. Shows you the URLs

### Access Your Apps

1. **Open launcher**: `https://brain-nw1-launcher.trycloudflare.com`
2. **Login** with your password
3. **Click "Start App"** to launch the main RAG app
4. **Click "Open RAG Researcher App"** 
   - Opens: `https://brain-nw1-app.trycloudflare.com`
5. **Use your RAG researcher** - full functionality
6. **Click "Stop App"** when done

### Stop Everything

```bash
# Stop launcher
lsof -ti:8502 | xargs -r kill -9

# Stop main app (if running)
lsof -ti:8501 | xargs -r kill -9

# Stop tunnel (optional - it can stay running)
sudo systemctl stop cloudflared-brain
```

## 🛠️ Management Commands

### Tunnel Management

```bash
# Check tunnel status
sudo systemctl status cloudflared-brain

# View tunnel logs
sudo journalctl -u cloudflared-brain -f

# Restart tunnel
sudo systemctl restart cloudflared-brain

# Stop tunnel
sudo systemctl stop cloudflared-brain

# Start tunnel
sudo systemctl start cloudflared-brain

# Disable auto-start
sudo systemctl disable cloudflared-brain

# Enable auto-start
sudo systemctl enable cloudflared-brain
```

### Cloudflared Commands

```bash
# List all tunnels
cloudflared tunnel list

# Get tunnel info
cloudflared tunnel info brain-nw1

# Delete tunnel (careful!)
cloudflared tunnel delete brain-nw1

# Test configuration
cloudflared tunnel --config ~/.cloudflared/config.yml run brain-nw1
```

### Process Management

```bash
# Check what's running on ports
lsof -i :8502  # Launcher
lsof -i :8501  # Main app

# View launcher logs
tail -f /tmp/launcher.log

# View main app logs
tail -f /tmp/rag_app.log
```

## 🔧 Troubleshooting

### Tunnel Not Starting

```bash
# Check service status
sudo systemctl status cloudflared-brain

# View detailed logs
sudo journalctl -u cloudflared-brain -n 50

# Test tunnel manually
cloudflared tunnel --config ~/.cloudflared/config.yml run brain-nw1
```

### URLs Not Working

1. **Check tunnel is running**:
   ```bash
   sudo systemctl status cloudflared-brain
   ```

2. **Check DNS routes**:
   ```bash
   cloudflared tunnel route dns brain-nw1 brain-nw1-launcher.trycloudflare.com
   cloudflared tunnel route dns brain-nw1 brain-nw1-app.trycloudflare.com
   ```

3. **Test locally first**:
   ```bash
   curl http://localhost:8502  # Should return Streamlit HTML
   ```

### Authentication Issues

If you need to re-authenticate:

```bash
# Remove old cert
rm ~/.cloudflared/cert.pem

# Re-authenticate
cloudflared tunnel login
```

### Recreate Tunnel

If something is broken, start fresh:

```bash
# Stop and disable service
sudo systemctl stop cloudflared-brain
sudo systemctl disable cloudflared-brain
sudo rm /etc/systemd/system/cloudflared-brain.service

# Delete tunnel
cloudflared tunnel delete brain-nw1

# Remove config
rm ~/.cloudflared/config.yml

# Run setup again
./setup-persistent-tunnel.sh
```

## 🔒 Security Notes

### Current Setup

- ✅ Password-protected launcher
- ✅ Encrypted Cloudflare Tunnel (HTTPS)
- ✅ No open ports on your machine
- ⚠️ Using trycloudflare.com subdomain (temporary certificate)

### Enhanced Security (Optional)

1. **Add Custom Domain**:
   - Add your domain to Cloudflare
   - Update config to use: `brain-nw1-launcher.yourdomain.com`
   - Get proper SSL certificates

2. **Add Cloudflare Access**:
   - Set up Cloudflare Access in dashboard
   - Add authentication layer (Google, GitHub, etc.)
   - Require 2FA

3. **IP Restrictions**:
   - Configure Cloudflare firewall rules
   - Whitelist specific IPs
   - Block countries

4. **Rate Limiting**:
   - Set up rate limits in Cloudflare
   - Prevent brute force attacks

## 🎓 Advanced: Custom Domain

### If You Have a Domain

1. **Add domain to Cloudflare**:
   - Go to: https://dash.cloudflare.com
   - Click "+ Add site"
   - Follow setup instructions

2. **Update tunnel config**:
   ```yaml
   ingress:
     - hostname: launcher.yourdomain.com
       service: http://localhost:8502
     
     - hostname: app.yourdomain.com
       service: http://localhost:8501
   ```

3. **Update DNS routes**:
   ```bash
   cloudflared tunnel route dns brain-nw1 launcher.yourdomain.com
   cloudflared tunnel route dns brain-nw1 app.yourdomain.com
   ```

4. **Update environment variables**:
   ```bash
   export LAUNCHER_URL="https://launcher.yourdomain.com"
   export MAIN_APP_URL="https://app.yourdomain.com"
   ```

## 📊 Comparison: Quick vs Persistent

| Feature | Quick Tunnel | Persistent Tunnel |
|---------|-------------|-------------------|
| **Setup Time** | 30 seconds | 2 minutes |
| **URL Changes** | Every restart | Never (with domain) |
| **Custom Naming** | Random | brain-nw1 |
| **Auto-start** | No | Yes (systemd) |
| **Multiple Services** | Need 2 tunnels | 1 tunnel |
| **Production Ready** | Testing only | Yes |
| **Domain Required** | No | Optional (better) |

## ✅ Benefits Summary

### What You Get with Persistent Setup

1. **Single URL for Launcher**: Always `brain-nw1-launcher.trycloudflare.com`
2. **Single URL for App**: Always `brain-nw1-app.trycloudflare.com`
3. **One Tunnel Process**: Manages both services automatically
4. **Auto-start on Boot**: Tunnel starts automatically with systemd
5. **Easy Management**: Standard systemctl commands
6. **Better Performance**: Single tunnel = lower overhead
7. **Cleaner Architecture**: No manual tunnel juggling

## 🎉 You're All Set!

With this setup:
- ✅ Tunnel runs automatically on boot
- ✅ URLs never change
- ✅ Both services accessible through one tunnel
- ✅ Professional, production-ready setup
- ✅ Easy to manage and monitor

Just run `./start-persistent-launcher.sh` and open your browser to `https://brain-nw1-launcher.trycloudflare.com`!
