# Choose Your Setup

Two options for remote access to your RAG Researcher:

## 🚀 Option 1: Quick Tunnel (Testing)

**Best for**: Quick testing, temporary access

### Features
- ✅ Setup in 30 seconds
- ✅ No authentication needed
- ✅ Works immediately
- ⚠️ URLs change on every restart
- ⚠️ Need 2 separate tunnels (launcher + app)

### How to Start
```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./start-quick-tunnel.sh
```

**URLs**: Random (e.g., `https://random-words-xyz.trycloudflare.com`)

---

## 🧠 Option 2: Persistent Tunnel (Production)

**Best for**: Daily use, production, stable URLs

### Features
- ✅ Persistent URLs with "BrAIn-nw1" name
- ✅ Single tunnel for both services
- ✅ Auto-starts on boot (systemd)
- ✅ Professional setup
- ✅ URLs never change

### How to Start

**One-time setup:**
```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./setup-persistent-tunnel.sh
```

**Daily use:**
```bash
./start-persistent-launcher.sh
```

**URLs**: 
- Launcher: `https://brain-nw1-launcher.trycloudflare.com`
- Main App: `https://brain-nw1-app.trycloudflare.com`

---

## 📊 Comparison

| Feature | Quick Tunnel | Persistent Tunnel |
|---------|-------------|-------------------|
| **Setup** | 30 seconds | 2 minutes |
| **URLs** | Random | brain-nw1-* |
| **Stability** | Changes | Permanent |
| **Tunnels** | 2 separate | 1 combined |
| **Auto-start** | No | Yes |
| **Production** | ❌ | ✅ |

---

## 🎯 Recommendation

**Use Persistent Tunnel** for your use case because:
1. You want "BrAIn-nw1" in the URL ✅
2. You want the main app accessible via launcher ✅
3. You want consistent URLs ✅
4. You'll use it regularly ✅

---

## 📚 Documentation

- **Quick Tunnel**: See [QUICK_TUNNEL.md](QUICK_TUNNEL.md)
- **Persistent Tunnel**: See [PERSISTENT_SETUP.md](PERSISTENT_SETUP.md)
- **Detailed Workflow**: See [WORKFLOW.md](WORKFLOW.md)
- **Main README**: See [README.md](README.md)

---

## 🚀 Get Started Now

For your requirements (persistent URLs + single tunnel for both apps):

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./setup-persistent-tunnel.sh
```

Follow the prompts, then start the launcher:

```bash
export LAUNCHER_PASSWORD="your_password"
./start-persistent-launcher.sh
```

Access via: `https://brain-nw1-launcher.trycloudflare.com`

Done! 🎉
