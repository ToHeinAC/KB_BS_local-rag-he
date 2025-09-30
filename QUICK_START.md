# Quick Start Guide

## Starting the Application

### Method 1: Using Make (Recommended)
```bash
make start       # Start English version (default)
make start-de    # Start German version  
make start-v1    # Start V1.1 version
make test        # Run source linking tests
make help        # Show all commands
```

### Method 2: Using the Run Script
```bash
./run.sh         # Start English version (default)
./run.sh v2_0    # Start English version
./run.sh v2_0g   # Start German version
./run.sh v1_1    # Start V1.1 version
```

### Method 3: Manual Start
```bash
uv run streamlit run apps/app_v2_0.py --server.port 8501 --server.address localhost
```

## What's Been Fixed

### Source Linking Fix (2025-09-30)

**Problem:** Source references like `[filename.pdf]` were being converted to massive base64-encoded data URLs, making the output unreadable.

**Solution:** Changed the `linkify_sources` function to use clean `file://` URLs instead of base64 encoding.

**Result:** 
- ✅ Clean, clickable links
- ✅ No massive HTML strings
- ✅ Files open in browser with file:// URLs
- ✅ Graceful handling of missing files

**Example output:**
```html
<a href="file:///path/to/file.pdf" target="_blank">📄 filename.pdf</a>
```

## Testing

### Automated Tests
Run the automated tests:
```bash
make test
# or
uv run test_source_linking.py
uv run test_empty_response.py
```

### Interactive Source Link Testing
Test source linking in your browser:
```bash
make test-sources
# or
uv run streamlit run dev/basic_report-source-tester_app.py --server.port 8502
```

This will open a test app at http://localhost:8502 where you can:
- Select a database
- See sample agent answers with source references
- Click on source links to verify they open PDFs correctly
- Test custom source references

## App Versions

- **app_v2_0.py** - English version (default)
- **app_v2_0g.py** - German version
- **app_v1_1.py** - Legacy V1.1 version

## Accessing the App

Once started, the app will be available at:
- **URL:** http://localhost:8501
- The terminal will show the URL and automatically open it in your browser
