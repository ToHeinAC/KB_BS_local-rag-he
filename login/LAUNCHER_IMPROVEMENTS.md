# 🎨 Launcher App Improvements - Tunnel URL Display

## ✅ Changes Made to launcher_app.py

### 1. Dynamic URL Detection

Added `get_tunnel_urls()` function that automatically detects tunnel URLs from:
- Environment variables (`LAUNCHER_URL`, `MAIN_APP_URL`) - for persistent tunnels
- Saved files (`/tmp/launcher-url.txt`, `/tmp/app-url.txt`) - for quick tunnels
- Falls back to localhost if neither is available

### 2. Tunnel Type Detection

Added intelligent detection:
```python
IS_PERSISTENT = "brain-nw1" in LAUNCHER_URL  # Named tunnel with custom domain
IS_QUICK_TUNNEL = "trycloudflare.com" in LAUNCHER_URL and not IS_PERSISTENT  # Quick tunnels
```

### 3. Prominent URL Display Section

Added expandable **"🌐 Current Tunnel URLs"** section that:
- **Auto-expands** for quick tunnels (URLs are temporary)
- Shows both launcher and main app URLs in copyable code blocks
- Provides clickable "Open" buttons next to each URL
- Main app button is:
  - **Primary** (highlighted) when app is running
  - **Disabled "Start First"** when app is not running

### 4. Visual Indicators

**Header alerts:**
- ✅ Green success for persistent tunnels
- ⚠️ Orange warning for quick tunnels (temporary)

**Info boxes:**
- Quick tunnels: Shows reminder that URLs change on restart
- Persistent tunnels: Confirms URLs are permanent

---

## 🎯 User Experience Improvements

### Before:
- URLs were hidden in a footer expander
- Hard to find and copy
- No clear indication if URLs were temporary or permanent
- No direct "Open" buttons

### After:
- URLs displayed prominently near the top
- Auto-expanded for quick tunnels (most important case)
- Clear visual indicators for tunnel type
- Clickable "Open" buttons for easy access
- Copyable code blocks for sharing
- Context-aware button states (disabled until app runs)

---

## 📋 How It Works

### Quick Tunnels Flow:

1. User runs `./start-quick-tunnels.sh`
2. Script saves URLs to `/tmp/launcher-url.txt` and `/tmp/app-url.txt`
3. Launcher app reads these files automatically
4. URLs displayed in auto-expanded section with warning
5. User can directly click "Open" buttons to access

### Persistent Tunnels Flow:

1. User sets environment variables in systemd service or script
2. Launcher app reads from environment
3. URLs displayed in collapsible section with success indicator
4. URLs stay the same across restarts

---

## 🚀 Current Setup

**Active Tunnel URLs:**
- 🔐 Launcher: `https://grades-functional-frequencies-full.trycloudflare.com`
- 🚀 Main App: `https://static-retrieve-fonts-papua.trycloudflare.com`

**Status:**
- ✅ Launcher is running (PID varies)
- ✅ Quick tunnels are active
- ✅ URLs are auto-detected from saved files
- ✅ Both URLs accessible via HTTPS

---

## 💡 Usage Examples

### Opening the Launcher:

**From browser:** Open `https://grades-functional-frequencies-full.trycloudflare.com`

**You'll see:**
1. ⚠️ "Using Quick Tunnels (URLs are temporary)" warning
2. Auto-expanded "🌐 Current Tunnel URLs" section showing both URLs
3. Clickable "Open" buttons next to each URL
4. Info message about URLs being temporary

### Starting the Main App:

1. Login with password: `BrAIn-#1`
2. See tunnel URLs section (auto-expanded)
3. Click "🚀 Start App" button
4. Wait for app to start
5. Click "Open" button next to Main App URL
6. Or manually open: `https://static-retrieve-fonts-papua.trycloudflare.com`

---

## 🔄 Restart Workflow

When you restart tunnels to get new URLs:

```bash
# Stop everything
pkill -f "cloudflared tunnel"
lsof -ti:8502 | xargs -r kill -9

# Start new tunnels (generates NEW URLs)
./start-quick-tunnels.sh

# Restart launcher (picks up new URLs automatically)
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
LAUNCHER_PASSWORD="BrAIn-#1" nohup uv run --with streamlit --with psutil \
  streamlit run launcher_app.py --server.port 8502 --server.headless true \
  > /tmp/launcher.log 2>&1 &

# New URLs shown in output and in launcher UI
```

---

## 📊 Code Structure

```python
# URL detection
def get_tunnel_urls():
    """Reads from env vars or saved files"""
    # Priority: env vars > saved files > localhost
    
# Tunnel type detection
IS_PERSISTENT = "brain-nw1" in LAUNCHER_URL
IS_QUICK_TUNNEL = "trycloudflare.com" in LAUNCHER_URL

# UI display (auto-expands for quick tunnels)
with st.expander("🌐 Current Tunnel URLs", expanded=IS_QUICK_TUNNEL):
    # Launcher URL with Open button
    # Main App URL with conditional button (running/disabled)
    # Context-specific info message
```

---

## ✅ Benefits

1. **No more searching** - URLs front and center
2. **Copy-paste ready** - Code blocks for easy copying
3. **One-click access** - Direct "Open" buttons
4. **Clear context** - Visual indicators for tunnel type
5. **Smart defaults** - Auto-expands when URLs are temporary
6. **Automated detection** - Works with both quick and persistent tunnels
7. **User-friendly** - No need to check terminal output

---

## 🎉 Result

Users can now **directly see and use** their Cloudflare tunnel URLs from within the launcher interface, with clear visual cues about whether the URLs are temporary or permanent, and one-click buttons to open them!
