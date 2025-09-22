# Report Source Tester App

## Overview

The `basic_report-source-tester_app.py` is a minimal Streamlit application that demonstrates clickable sources in LLM agent answers. It allows users to click on source references like `[StrlSchG--250508.pdf]` to view the actual PDF files inline.

## Features

- **Dynamic Database Selection**: Choose from available vector databases to determine source document location
- **Flexible Source Resolution**: Automatically map database names to corresponding source directories
- **Clickable Source References**: Convert text references like `[filename.pdf]` into clickable links
- **PDF New Window Opening**: Open PDF files in new browser tabs/windows using base64 data URLs
- **Smart File Resolution**: Map source references with timestamps to actual files
- **Visual Status Indicators**: Show different colors for found/missing/error states
- **Context-Aware Samples**: Sample answers adapt based on selected database
- **Custom Input**: Allow users to test their own text with source references

## Usage

### Running the App

```bash
cd /home/he/ai/dev/langgraph/KB_BS_local-rag-he
uv run streamlit run dev/basic_report-source-tester_app.py --server.port 8502
```

### Source Reference Format

The app recognizes source references in the following formats:

- `[StrlSchG--250508.pdf]` → Links to `StrlSchG.pdf`
- `[filename--timestamp.pdf]` → Links to `filename.pdf`
- `[AtG.pdf]` → Links to `AtG.pdf`

### File Resolution Logic

1. **Direct Match**: First tries to find the exact filename
2. **Timestamp Removal**: Removes `--timestamp` patterns (e.g., `--250508`)
3. **Case-Insensitive Search**: Searches for files with matching base names
4. **Error Handling**: Shows available files if the target is not found

## Configuration

### Database Selection

The app dynamically discovers available databases from:
```
./kb/database/
```

**Database to Source Directory Mapping:**
- `StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600` → `./kb/StrlSch__db_inserted/`
- `NORM__Qwen--Qwen3-Embedding-0.6B--3000--600` → `./kb/NORM__db_inserted/`

**Flexible Pattern Matching:**
The app tries multiple patterns to find source directories:
1. `{prefix}__db_inserted` (standard)
2. `{prefix}_db_inserted` (alternative)
3. `{prefix}__inserted` (short form)
4. `{prefix}_inserted` (alternative short)
5. `{prefix}` (just prefix)

### PDF Directory

The app is configured to look for PDF files in:
```
./kb/StrlSch__db_inserted/
```

### Available Test Files

**StrlSch Database (Radiation Protection):**
- `StrlSchG.pdf` - German Radiation Protection Act
- `StrlSchV.pdf` - German Radiation Protection Ordinance  
- `AtG.pdf` - Atomic Energy Act
- `AtEV.pdf` - Atomic Energy Ordinance
- `AtDeckV.pdf` - Atomic Energy Cover Ordinance
- `KTA 1401_2017-11.pdf` - Technical Standard
- `KTA 1402_2017-11.pdf` - Technical Standard

**NORM Database (Naturally Occurring Radioactive Materials):**
- 54+ PDF files related to NORM regulations and guidelines

## Technical Implementation

### Core Functions

1. **`get_available_databases()`**: Discovers available vector databases
2. **`extract_database_prefix(database_name)`**: Extracts prefix from database names
3. **`resolve_source_directory(database_name)`**: Maps database to source directory
4. **`resolve_pdf_path(source_name, selected_database)`**: Maps source names to actual file paths
5. **`linkify_sources(markdown_text, selected_database)`**: Converts source references to clickable HTML links

### PDF Opening Method

- Reads PDF files as binary data
- Encodes using base64 for data URL creation
- Creates HTML links with `target="_blank"` to open in new windows
- Uses `data:application/pdf;base64,` URL scheme for direct PDF access
- Provides visual indicators for file status (found/missing/error)

### Status Indicators

- **Blue clickable link**: PDF found and ready to open
- **Orange text**: PDF file not found in directory
- **Red text**: Error occurred while reading PDF file

## Example Usage in RAG Applications

This pattern can be integrated into RAG applications to make source citations clickable:

```python
# In your RAG application
agent_answer = "According to the regulations in [StrlSchG--250508.pdf], the requirements are..."

# Get selected database from session state
selected_database = st.session_state.get('selected_database')

# Convert to clickable links that open in new windows
clickable_answer = linkify_sources(agent_answer, selected_database)

# Display in Streamlit with HTML enabled
st.markdown(clickable_answer, unsafe_allow_html=True)

# PDFs will automatically open in new windows when clicked - no additional handling needed!
```

## Limitations

- **Browser Compatibility**: PDF inline viewing depends on browser PDF support
- **File Size**: Large PDF files may take time to load
- **Security**: Uses base64 encoding which increases memory usage
- **Local Files Only**: Currently only supports local file access

## Future Enhancements

- Support for remote PDF URLs
- Integration with vector databases for source metadata
- Annotation support for highlighting relevant sections
- Caching for frequently accessed PDFs
- Support for other document formats (Word, PowerPoint, etc.)
