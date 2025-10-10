# Cloudflare Tunnel URL Options

## ❌ Current Problem

Your config uses:
- `brain-nw1-launcher.trycloudflare.com` 
- `brain-nw1-app.trycloudflare.com`

**These don't work!** You cannot choose custom subdomains with `trycloudflare.com`.

---

## ✅ Solution Options

### Option 1: Quick Tunnel (Recommended for Testing)

**Pros:**
- ✅ Works immediately
- ✅ Completely free
- ✅ No domain needed
- ✅ No DNS configuration

**Cons:**
- ❌ URLs are random (e.g., `https://quiet-mountain-3d2f.trycloudflare.com`)
- ❌ URLs change each time you restart
- ❌ Must share new URL each time

**How to use:**

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login

# Start both tunnels (generates new URLs)
./start-quick-tunnels.sh

# Output shows URLs like:
# 🔐 Launcher: https://quiet-mountain-3d2f.trycloudflare.com
# 🚀 Main App: https://red-forest-8a1b.trycloudflare.com
```

**When to use:**
- Testing the setup
- Personal use where changing URLs is acceptable
- No domain available

---

### Option 2: Named Tunnel with Custom Domain (Persistent URLs)

**Pros:**
- ✅ Custom URLs (e.g., `launcher.yourdomain.com`)
- ✅ URLs never change
- ✅ Professional appearance
- ✅ Can use own domain

**Cons:**
- ❌ Requires owning a domain (~$10-15/year)
- ❌ Domain must be added to Cloudflare
- ❌ Requires DNS configuration

**Requirements:**
1. Own a domain (e.g., `example.com`)
2. Domain nameservers pointing to Cloudflare
3. Domain added to your Cloudflare account

**Setup steps:**

#### Step 1: Add Domain to Cloudflare

1. Go to: https://dash.cloudflare.com
2. Click "Add site"
3. Enter your domain
4. Follow setup wizard
5. Update nameservers at your registrar

#### Step 2: Create DNS Routes

```bash
# Route launcher subdomain
cloudflared tunnel route dns brain-nw1 launcher.yourdomain.com

# Route main app subdomain
cloudflared tunnel route dns brain-nw1 app.yourdomain.com
```

#### Step 3: Update Config File

Edit `~/.cloudflared/config.yml`:

```yaml
tunnel: 5bc36850-dc39-48be-81ab-fd15f8071bd0
credentials-file: /home/he/.cloudflared/5bc36850-dc39-48be-81ab-fd15f8071bd0.json

ingress:
  # Launcher app
  - hostname: launcher.yourdomain.com
    service: http://localhost:8502
  
  # Main RAG app
  - hostname: app.yourdomain.com
    service: http://localhost:8501
  
  # Catch-all
  - service: http_status:404
```

#### Step 4: Restart Tunnel

```bash
sudo systemctl restart cloudflared-brain
```

**Your URLs:**
- `https://launcher.yourdomain.com`
- `https://app.yourdomain.com`

**When to use:**
- Production deployment
- Want professional URLs
- Share with team/clients
- URLs must never change

---

### Option 3: Hybrid Approach

Use **Quick Tunnel** for development, then migrate to **Custom Domain** for production.

1. **Development:** Use `./start-quick-tunnels.sh`
2. **Production:** Buy domain, set up DNS routes

---

## 🎯 Recommendation

### For You Right Now:

**Use Option 1 (Quick Tunnel)**

Why:
- Works immediately
- No cost
- No domain setup needed
- Perfect for personal use

Just run:
```bash
./start-quick-tunnels.sh
```

Then bookmark the URLs it generates. When you restart your server, just run the script again and bookmark the new URLs.

---

### If You Want Persistent URLs:

**Use Option 2 (Custom Domain)**

1. Buy domain from: Namecheap, GoDaddy, Cloudflare Registrar, etc.
2. Add to Cloudflare (free)
3. Set up DNS routes (5 minutes)
4. Professional persistent URLs forever

---

## 🔍 Why trycloudflare.com URLs Are Random

From Cloudflare's documentation:

> "Quick Tunnels use random subdomains under trycloudflare.com. These URLs are ephemeral and will change on each run. For persistent URLs, use a Named Tunnel with your own domain."

**trycloudflare.com = Random temporary URLs**
**yourdomain.com = Persistent custom URLs**

---

## 📋 Quick Commands

### Check what's currently running:

```bash
# Check tunnel service
sudo systemctl status cloudflared-brain

# Check launcher
lsof -i :8502

# Check main app
lsof -i :8501
```

### Stop everything:

```bash
# Stop named tunnel service
sudo systemctl stop cloudflared-brain

# Stop quick tunnels
pkill -f "cloudflared tunnel"

# Stop launcher
lsof -ti:8502 | xargs -r kill -9

# Stop main app
lsof -ti:8501 | xargs -r kill -9
```

### Start fresh with Quick Tunnels:

```bash
# Stop all existing tunnels
sudo systemctl stop cloudflared-brain
pkill -f "cloudflared tunnel"

# Start quick tunnels
./start-quick-tunnels.sh
```

---

## ❓ FAQ

**Q: Can I make trycloudflare.com URLs persistent?**
A: No, they are always random and temporary.

**Q: How much does a domain cost?**
A: $10-15/year from most registrars. Cloudflare Registrar offers domains at cost (no markup).

**Q: Do I need to pay for Cloudflare?**
A: No! Cloudflare Tunnel is free. You only pay for the domain itself.

**Q: Can I use a subdomain of a free service?**
A: Not with Cloudflare Tunnel. You need your own domain.

**Q: What if my URLs keep changing?**
A: Use Quick Tunnels and just bookmark the new URL each time, or buy a domain for persistent URLs.

---

## 🚀 Next Steps

**Right now, run:**
```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he/login
./start-quick-tunnels.sh
```

This will give you working URLs immediately!
