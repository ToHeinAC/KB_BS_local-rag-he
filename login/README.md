# Quick Cloudflare Tunnels Setup

## 🚀 What This Does

Provides instant remote access to your RAG Deep Researcher application via temporary Cloudflare tunnels.

**Key Features:**
- ✅ Works immediately (no configuration needed)
- ✅ No domain name required
- ✅ Encrypted HTTPS tunnels
- ✅ Access from anywhere
- ⚠️ URLs change each restart (temporary)

---

## 📋 Prerequisites

1. **Install cloudflared:**
   ```bash
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. **Ensure your apps are ready:**
   - Launcher app should be on port 8502
   - Main RAG app should be on port 8501

---

## 🎯 Quick Start

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./start-quick-tunnels.sh
```

**Output:**
```
✅ Quick Tunnels Running!
======================================

🔐 Launcher URL:  https://quiet-mountain-3d2f.trycloudflare.com
🚀 Main App URL:  https://proud-river-8a1b.trycloudflare.com

⚠️  IMPORTANT: These URLs are temporary!
   They will change each time you restart the tunnels.
```

**Usage:**
1. Copy the Launcher URL and open in browser
2. Login with your password
3. Click "Start App" to launch the RAG researcher
4. Click "Open App" or use the Main App URL

---

## 🛑 Stopping Tunnels

```bash
pkill -f "cloudflared tunnel"
```

---

## 📝 How It Works

```
Your Browser → HTTPS → Cloudflare → Tunnel → localhost:8502 (Launcher)
                                           → localhost:8501 (RAG App)
```

1. Script starts two Cloudflare quick tunnels
2. Each tunnel gets a random `*.trycloudflare.com` URL
3. URLs are saved to `/tmp/launcher-url.txt` and `/tmp/app-url.txt`
4. Tunnels route external traffic to your local ports

---

## ⚠️ Important Notes

- **Temporary URLs:** Change every time you restart
- **No Authentication:** Quick tunnels are publicly accessible (use strong launcher password!)
- **Development Use:** Best for testing and development
- **Production:** Consider setting up persistent tunnels with custom domains
  - See `cloudflared-config.yml` template for persistent tunnel configuration

---

## 🔒 Security

Always set a strong launcher password:
```bash
export LAUNCHER_PASSWORD="your_secure_password_here"
```

---

## 🐛 Troubleshooting

### URLs not showing
```bash
# Check logs
tail -f /tmp/launcher-tunnel.log
tail -f /tmp/app-tunnel.log
```

### Port already in use
```bash
lsof -ti:8501 | xargs -r kill -9
lsof -ti:8502 | xargs -r kill -9
```

### Tunnels not accessible
```bash
# Verify cloudflared is running
ps aux | grep cloudflared

# Restart tunnels
pkill -f "cloudflared tunnel"
./start-quick-tunnels.sh
```
