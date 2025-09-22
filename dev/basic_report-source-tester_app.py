#!/usr/bin/env python3
"""
Basic Report Source Tester App

This minimal Streamlit app demonstrates clickable sources in LLM agent answers.
It allows users to click on source references like [StrlSchG--250508.pdf] to view the actual PDF files.

Usage:
    streamlit run dev/basic_report-source-tester_app.py
"""

import urllib.parse
from pathlib import Path
import streamlit as st

# Import source handling functions from rag_helpers_v1_1
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.rag_helpers_v1_1 import (
    get_available_databases,
    extract_database_prefix,
    resolve_source_directory,
    resolve_pdf_path,
    linkify_sources
)

# Configuration
DATABASE_PATH = Path("./kb/database")  # Directory containing vector databases
KB_PATH = Path("./kb")  # Root KB directory
SUPPORTED_EXTENSIONS = [".pdf"]


def main():
    """Main Streamlit app function."""
    
    # Page configuration
    st.set_page_config(
        page_title="Report Source Tester",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("📄 Report Source Tester")
    st.markdown("""
    This app demonstrates clickable sources in LLM agent answers. 
    Click on any source reference to view the corresponding PDF file.
    """)
    
    # Initialize session state for database selection
    if "selected_database" not in st.session_state:
        st.session_state.selected_database = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database selection section
        with st.expander("🗄️ Database Selection", expanded=True):
            # Get available databases
            database_options = get_available_databases()
            
            if database_options:
                # Select database
                selected_db = st.selectbox(
                    "Select Database",
                    options=["None"] + database_options,
                    index=database_options.index(st.session_state.selected_database) + 1 if st.session_state.selected_database in database_options else 0,
                    help="Choose a database to determine source document location"
                )
                
                # Update session state
                st.session_state.selected_database = selected_db if selected_db != "None" else None
                
                # Show database info
                if st.session_state.selected_database:
                    prefix = extract_database_prefix(st.session_state.selected_database)
                    source_dir = resolve_source_directory(st.session_state.selected_database)
                    
                    st.info(f"**Selected Database:** `{st.session_state.selected_database}`")
                    st.info(f"**Database Prefix:** `{prefix}`")
                    st.info(f"**Source Directory:** `{source_dir}`")
                    
                    # Show available PDF files in selected source directory
                    if source_dir.exists():
                        pdf_files = list(source_dir.glob("*.pdf"))
                        st.success(f"**Available PDFs:** {len(pdf_files)} files")
                        
                        with st.expander("📂 View Available Files", expanded=False):
                            for pdf_file in sorted(pdf_files):
                                st.write(f"- `{pdf_file.name}`")
                    else:
                        st.warning(f"Source directory not found: `{source_dir}`")
                else:
                    st.info("No database selected - using default directory")
            else:
                st.error(f"No databases found in: `{DATABASE_PATH}`")
                st.info("Please ensure the database directory exists and contains database folders.")
    
    # Sample agent answers for testing
    st.header("🤖 Sample Agent Answers")
    
    # Predefined sample answers (dynamically updated based on selected database)
    base_sample_answers = {
        "StrlSch": {
            "Legal Framework": """
            According to the German radiation protection law, as mentioned in [StrlSchG--250508.pdf], 
            the regulatory requirements are clearly defined. Additional details can be found in 
            [StrlSchV--250508.pdf] regarding implementation guidelines.
            """,
            
            "Technical Standards": """
            The technical requirements for nuclear facilities are specified in [KTA 1401_2017-11.pdf] 
            and [KTA 1402_2017-11.pdf]. These documents provide comprehensive guidance for 
            safety systems and operational procedures.
            """,
            
            "Atomic Energy Law": """
            The Atomic Energy Act, as detailed in [AtG--250508.pdf], establishes the fundamental 
            legal framework. Related provisions can be found in [AtEV--250508.pdf] and 
            [AtDeckV--250508.pdf] for specific operational aspects.
            """
        },
        "NORM": {
            "NORM Regulations": """
            The naturally occurring radioactive materials (NORM) guidelines, as specified in 
            [NORM_Guidelines--123456.pdf], provide comprehensive requirements for handling 
            radioactive materials in industrial processes.
            """,
            
            "Industrial Applications": """
            NORM applications in various industries are documented in [NORM_Industrial--789012.pdf] 
            and related technical specifications can be found in [NORM_Technical--345678.pdf].
            """
        },
        "Default": {
            "Sample Document": """
            This is a sample answer referencing [SampleDoc--123456.pdf] to demonstrate 
            the clickable source functionality with any database selection.
            """
        }
    }
    
    # Select appropriate sample answers based on database
    if st.session_state.selected_database:
        prefix = extract_database_prefix(st.session_state.selected_database)
        sample_answers = base_sample_answers.get(prefix, base_sample_answers["Default"])
    else:
        sample_answers = base_sample_answers["StrlSch"]  # Default to StrlSch samples
    
    # Sample answer selector
    selected_sample = st.selectbox(
        "Choose a sample answer:",
        options=list(sample_answers.keys()),
        help="Select a predefined sample answer to test clickable sources"
    )
    
    # Display selected sample answer
    if selected_sample:
        st.subheader(f"Sample: {selected_sample}")
        sample_text = sample_answers[selected_sample]
        
        # Convert sources to clickable links and display
        linked_text = linkify_sources(sample_text, st.session_state.selected_database)
        st.markdown(linked_text, unsafe_allow_html=True)
    
    # Custom input section
    st.header("✏️ Custom Input")
    
    # Text area for custom agent answer
    custom_answer = st.text_area(
        "Enter your own agent answer with source references:",
        value="This is a custom answer referencing [StrlSchG--250508.pdf] and other sources.",
        height=100,
        help="Use format [filename.pdf] or [filename--timestamp.pdf] for source references"
    )
    
    if custom_answer.strip():
        st.subheader("Custom Answer with Clickable Sources")
        linked_custom = linkify_sources(custom_answer, st.session_state.selected_database)
        st.markdown(linked_custom, unsafe_allow_html=True)
    
    # Note: PDFs now open in new windows automatically when clicked
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Click on any blue source reference (📄 filename.pdf) to open the PDF in a new window/tab
    2. The PDF will open directly in your browser's PDF viewer
    3. You can close the PDF tab to return to this app
    
    **Source Reference Format:**
    - `[StrlSchG--250508.pdf]` → Links to `StrlSchG.pdf`
    - `[filename--timestamp.pdf]` → Links to `filename.pdf`
    
    **Status Indicators:**
    - 📄 Blue link: PDF found and ready to open
    - 📄 Orange text: PDF file not found
    - 📄 Red text: Error reading PDF file
    """)

if __name__ == "__main__":
    main()
