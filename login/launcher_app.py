import streamlit as st
import subprocess
import os
import signal
import time
import psutil
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RAG Researcher Launcher",
    page_icon="🚀",
    layout="centered"
)

# Simple authentication - use environment variables for security
ADMIN_PASSWORD = os.getenv("LAUNCHER_PASSWORD", "changeme123")
APP_PORT = 8501
APP_COMMAND = "uv run streamlit run apps/app_v2_0g.py --server.port 8501 --server.headless true"

# Cloudflare Tunnel URLs (set these in environment or they'll default to localhost)
# For persistent tunnel setup, these will be set to brain-nw1 URLs
LAUNCHER_URL = os.getenv("LAUNCHER_URL", f"http://localhost:8502")
MAIN_APP_URL = os.getenv("MAIN_APP_URL", f"http://localhost:{APP_PORT}")

# Check if we're using the persistent tunnel setup
IS_PERSISTENT = "brain-nw1" in LAUNCHER_URL

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == ADMIN_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "🔐 Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "🔐 Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

def get_process_on_port(port):
    """Check if there's a process running on the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                connections = proc.connections()
                for conn in connections:
                    if conn.laddr.port == port:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        st.error(f"Error checking port: {e}")
    return None

def kill_process_on_port(port):
    """Kill process running on specified port."""
    try:
        # Try using lsof command
        result = subprocess.run(
            f"lsof -ti:{port}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(f"kill -9 {pid}", shell=True, check=True)
                    st.success(f"✅ Killed process {pid} on port {port}")
                except Exception as e:
                    st.error(f"Error killing process {pid}: {e}")
            return True
        else:
            st.warning(f"No process found on port {port}")
            return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def start_app():
    """Start the Streamlit app."""
    try:
        # Check if something is already running
        proc = get_process_on_port(APP_PORT)
        if proc:
            st.warning(f"⚠️ Process already running on port {APP_PORT} (PID: {proc.pid})")
            return False
        
        # Change to the project directory
        project_dir = "/home/he/ai/dev/langgraph/KB_BS_local-rag-he"
        
        # Start the app in background
        st.info(f"🚀 Starting app with command: {APP_COMMAND}")
        
        # Use nohup to keep process running
        subprocess.Popen(
            f"cd {project_dir} && nohup {APP_COMMAND} > /tmp/rag_app.log 2>&1 &",
            shell=True,
            start_new_session=True
        )
        
        time.sleep(3)  # Wait for app to start
        
        # Verify it started
        proc = get_process_on_port(APP_PORT)
        if proc:
            st.success(f"✅ App started successfully! PID: {proc.pid}")
            st.info(f"📍 Access the app at: http://localhost:{APP_PORT}")
            return True
        else:
            st.error("❌ Failed to start app - no process detected on port")
            return False
            
    except Exception as e:
        st.error(f"❌ Error starting app: {e}")
        return False

def get_app_status():
    """Get current status of the app."""
    proc = get_process_on_port(APP_PORT)
    if proc:
        try:
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            return {
                "running": True,
                "pid": proc.pid,
                "cpu": cpu_percent,
                "memory_mb": memory_mb,
                "cmdline": " ".join(proc.cmdline()) if proc.cmdline() else "N/A"
            }
        except Exception as e:
            return {"running": True, "pid": proc.pid, "error": str(e)}
    return {"running": False}

# Main app
st.title("🧠 BrAIn-nw1 RAG Researcher Launcher")
st.markdown("### Remote Control Panel for RAG Deep Researcher")
if IS_PERSISTENT:
    st.success("✅ Using persistent tunnel URLs")
st.caption("Start, stop, and monitor your RAG application from anywhere")
st.markdown("---")

# Authentication
if not check_password():
    st.stop()

# Logged in - show controls
st.success("✅ Authenticated")
st.markdown("---")

# Status section
st.subheader("📊 Application Status")

status = get_app_status()

if status["running"]:
    st.success(f"✅ **App is RUNNING** (PID: {status['pid']})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", f"{status.get('cpu', 'N/A')}%")
    with col2:
        st.metric("Memory Usage", f"{status.get('memory_mb', 'N/A'):.1f} MB")
    
    if "cmdline" in status:
        with st.expander("📝 Process Details"):
            st.code(status["cmdline"], language="bash")
    
    st.success(f"🌐 **App is ready!**")
    
    # Prominent button to open the app
    col_open1, col_open2 = st.columns([2, 1])
    with col_open1:
        st.link_button(
            "🚀 Open RAG Researcher App",
            MAIN_APP_URL,
            use_container_width=True,
            type="primary"
        )
    with col_open2:
        if st.button("📋 Copy URL", use_container_width=True):
            st.code(MAIN_APP_URL, language="text")
    
    st.caption(f"📍 Direct URL: {MAIN_APP_URL}")
    
else:
    st.warning("⚠️ **App is NOT running**")

st.markdown("---")

# Control buttons
st.subheader("🎮 Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Start App", type="primary", disabled=status["running"], use_container_width=True):
        with st.spinner("Starting application..."):
            if start_app():
                time.sleep(2)
                st.rerun()

with col2:
    if st.button("🔄 Restart App", type="secondary", disabled=not status["running"], use_container_width=True):
        with st.spinner("Restarting application..."):
            kill_process_on_port(APP_PORT)
            time.sleep(2)
            if start_app():
                time.sleep(2)
                st.rerun()

with col3:
    if st.button("🛑 Stop App", type="secondary", disabled=not status["running"], use_container_width=True):
        with st.spinner("Stopping application..."):
            if kill_process_on_port(APP_PORT):
                time.sleep(2)
                st.rerun()

st.markdown("---")

# Logs section
st.subheader("📋 Application Logs")

if st.button("🔄 Refresh Logs"):
    st.rerun()

try:
    if os.path.exists("/tmp/rag_app.log"):
        with open("/tmp/rag_app.log", "r") as f:
            logs = f.read()
            # Show last 50 lines
            log_lines = logs.split('\n')
            recent_logs = '\n'.join(log_lines[-50:])
            st.text_area("Recent logs (last 50 lines)", recent_logs, height=300)
    else:
        st.info("No log file found yet")
except Exception as e:
    st.error(f"Error reading logs: {e}")

# Footer
st.markdown("---")

# Info box
with st.expander("ℹ️ How This Works"):
    tunnel_info = "**persistent** BrAIn-nw1 tunnel" if IS_PERSISTENT else "Cloudflare Tunnel"
    st.markdown("""
    **Workflow:**
    1. 🔐 You access this launcher via {tunnel_type}
    2. ✅ After login, you can start the main RAG app
    3. 🚀 Click "Open RAG Researcher App" to access it in your browser
    4. 🛑 Use "Stop App" to shut it down when finished
    
    **URLs (both accessible via the same tunnel):**
    - **Launcher**: `{launcher_url}` (this page)
    - **Main App**: `{main_app_url}` (the RAG researcher)
    
    Both URLs are routed through a single Cloudflare Tunnel - no need for separate tunnels!
    {persistence_note}
    """.format(
        tunnel_type=tunnel_info,
        launcher_url=LAUNCHER_URL,
        main_app_url=MAIN_APP_URL,
        persistence_note="\n\n**Note:** Using persistent named tunnel - URLs stay the same on restart!" if IS_PERSISTENT else ""
    ))

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
