- [ ] static cloudflare link
    - priority: high (current version)
    - cloudflare handling for the start of the app
    - additional cloudflare handling for the restart of the app
    - additional (not this version): cloudflare handling for the dynamic document links
- [ ] clear button resolved
    - priority: high (current version)
    - clear button shall be changed to start new button which starts a complete new session
- [x] simplified GUI ✅ COMPLETED
    - priority: high (current version)
    - only single main window, no 3 phases (i.e. phases 2 and 3 are merged into phase 1)
    - GUI shall reflect the current status of the merged phases to give the user a clear overview of the current state (st.progress and st.spinner or st.status)
    - Implementation: When /end is typed, automatically execute Phase 2 and Phase 3 in sequence with st.status progress indicators
    - Status: Fully implemented in app_v2_1.py - unified workflow with automatic phase execution
- [ ] advanced vectordatabase handling
    - priority: high (current version)
    - 1 stage: multiselect of vectordatabases via GUI possible, for the search process all selected databases shall be used in the search such that in the retrieval phase the documents are retrieved from all selected databases, e.g. k=5 then in total the 5 most similar documents are retrieved from all selected databases and then the final k=5 most similar documents are selected for the next phase 
    - 2 stage: (later version) The metadata of each database containing the document names in the database as well as some information on the documents and their relation to the other documents in the database are dynamically added to the state from the multiselect and used in the langgraph workflow as additional context called "database context"
- [ ] langchain open deep agent
    - priority: medium (future version)
    - A LangChain deep agent is a modular, multi-tool agent designed to handle truly complex workflows that require memory, delegation, and adaptive planning—representing the current frontier in agent architectures
- [ ] multimodality
    - priority: medium (future version)
    - 1 stage: other text based data as .doc, .docx, .ppt, .pptx, .txt, .md, .csv, .xls, .xlsx
    - 2 stage: images as .png, .jpg, .jpeg, .gif, .bmp, .webp, .svg, .tiff, .heic, .heif




Remember to restore the port settings when done testing:

Change 
/home/he/.streamlit/config.toml
 line 5 back to port = 8502
Change 
/home/he/ai/dev/langgraph/KB_BS_local-rag-he/.streamlit/config.toml
 line 16 back to serverPort = 8501
This keeps ports 8501 and 8502 available for deployment as required.

    