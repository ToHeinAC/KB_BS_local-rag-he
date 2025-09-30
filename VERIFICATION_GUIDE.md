# Source Linking Verification Guide

## Quick Verification Commands

### 1. Run Automated Tests
```bash
make test
```
This runs:
- `test_source_linking.py` - Tests source reference conversion
- `test_empty_response.py` - Tests LLM empty response handling

### 2. Run Diagnostic Check
```bash
make diagnose
```
This comprehensive diagnostic checks:
- ✅ file:// URLs are being used (not base64)
- ✅ PDF files can be found
- ✅ Links are properly formatted
- ✅ KB directory structure is correct
- ✅ Database-specific resolution works
- ✅ No base64 encoding in implementation

### 3. Interactive Browser Test
```bash
make test-sources
```
Opens a test app at http://localhost:8502 where you can:
- Select different databases
- See sample agent answers with source references
- Click on source links to verify PDFs open correctly
- Test custom source references

## What to Check

### Expected Behavior ✅
1. **Source links look like this:**
   ```html
   <a href="file:///absolute/path/to/file.pdf" target="_blank" style="color: #1f77b4; text-decoration: underline;">📄 filename.pdf</a>
   ```

2. **Clicking a source link:**
   - Opens PDF in a new browser tab/window
   - Uses browser's built-in PDF viewer
   - No massive data loading

3. **Missing files:**
   ```html
   <span style="color: orange;">📄 filename.pdf (Not found in directory)</span>
   ```

### What Should NOT Happen ❌
1. **No base64 data URLs:**
   ```html
   <!-- This is BAD and should NOT appear: -->
   <a href="data:application/pdf;base64,JVBERi0xLjYNJeLjz9...">
   ```

2. **No massive HTML strings** (megabytes of text)

3. **No empty LLM responses** without error messages

## Verifying the Empty Response Fix

The empty LLM response issue has been fixed. To verify:

1. **Check error handling is in place:**
   ```bash
   make test
   ```
   The `test_empty_response.py` should pass all tests.

2. **Error message format:**
   When an LLM returns empty response, you should see:
   ```
   Error: The LLM model gpt-oss:20b returned an empty response. 
   This may indicate the model is not working properly or the prompt is too complex. 
   Please try a different model or simplify the query.
   ```

## Full Test Workflow

Run these commands in order to fully verify the fixes:

```bash
# 1. Run automated tests
make test

# 2. Run diagnostic
make diagnose

# 3. Test in browser (interactive)
make test-sources
```

## Troubleshooting

### If source links don't work:
1. Check KB directory exists: `ls -la ./kb`
2. Check PDF files exist: `ls -la ./kb/StrlSch__db_inserted/`
3. Run diagnostic: `make diagnose`

### If you see base64 in links:
1. This should NOT happen with the fix
2. Run diagnostic to confirm: `make diagnose`
3. Check the test output for "❌ FAIL: Found base64.b64encode"

### If LLM returns empty response:
1. The error should be caught and displayed clearly
2. Try a different LLM model
3. Check if the model is available: `ollama list`

## Integration with Main App

The fixes are automatically integrated into:
- `apps/app_v2_0.py` (English version)
- `apps/app_v2_0g.py` (German version)  
- `apps/app_v1_1.py` (Legacy version)

All apps use the same fixed `linkify_sources` function from `src/rag_helpers_v1_1.py`.

## Files Modified (2025-09-30)

1. **Source Linking Fix:**
   - `src/rag_helpers_v1_1.py` (lines 911-926)
   - `src/rag_helpers_v1_1.py` (lines 765-793)

2. **Empty Response Fix:**
   - `src/utils_v1_1.py` (lines 163-208)
   - `src/graph_v2_0.py` (lines 1063-1082)

3. **Test/Diagnostic Tools:**
   - `test_source_linking.py` (created)
   - `test_empty_response.py` (created)
   - `diagnose_source_linking.py` (created)
   - `dev/basic_report-source-tester_app.py` (updated)

## Next Steps After Verification

Once you've verified everything works:

1. **Run the main app:**
   ```bash
   make start
   ```

2. **Test with a real query** to see source linking in action

3. **Monitor for any issues** with the LLM models

If you encounter the empty response issue again, it may indicate:
- The LLM model is not working properly
- The prompt is too complex
- The model needs to be restarted: `ollama restart gpt-oss:20b`
