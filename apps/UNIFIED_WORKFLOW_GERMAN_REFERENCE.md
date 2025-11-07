# German Version Unified Workflow - Remaining Changes for app_v2_1g.py

## Status
✅ Completed:
- Header with BrAIn branding
- execute_all_phases_automatically() function  
- Fixed duplicate progress bar headings
- Fixed enable_quality_checker duplicate
- Fixed max_search_queries session state
- Added workflow visualization expander
- Started unified workflow structure

❌ Remaining:
- Complete HITL phase section
- Add auto_executing phase
- Add completed phase with detailed results
- Remove old tab-based structure (lines ~1717-2173)

## Instructions

### Step 1: Complete the HITL Phase Section

After line 1715 (where initialize_hitl_state is called), continue with this code:

```python
                # Add to conversation history
                st.session_state.hitl_conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # Process initial query
                combined_response = process_initial_query(st.session_state.hitl_state)
                
                # Add AI response to conversation history
                st.session_state.hitl_conversation_history.append({
                    "role": "assistant",
                    "content": combined_response
                })
                
                # Set waiting for human input
                st.session_state.waiting_for_human_input = True
                
                st.rerun()
        
        # Display conversation history
        for message in st.session_state.hitl_conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Human feedback input
        if st.session_state.waiting_for_human_input and not st.session_state.conversation_ended:
            human_feedback = st.chat_input(
                "Ihre Antwort (Tippen Sie '/end' um fortzufahren)", 
                key=f"hitl_feedback_{st.session_state.input_counter}"
            )
            
            if human_feedback:
                # Check if conversation should end
                if human_feedback.strip().lower() == "/end":
                    st.session_state.conversation_ended = True
                    
                    # Add user message
                    st.session_state.hitl_conversation_history.append({
                        "role": "user",
                        "content": "/end - Konversation beendet"
                    })
                    
                    # Set flags
                    st.session_state.waiting_for_human_input = False
                    
                    # Finalize HITL conversation
                    final_response = finalize_hitl_conversation(st.session_state.hitl_state)
                    
                    # Add AI message
                    st.session_state.hitl_conversation_history.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    
                    # Store HITL results for handover
                    st.session_state.hitl_result = {
                        "user_query": st.session_state.hitl_state["user_query"],
                        "current_position": st.session_state.hitl_state["current_position"],
                        "detected_language": st.session_state.hitl_state["detected_language"],
                        "additional_context": st.session_state.hitl_state["additional_context"],
                        "report_llm": st.session_state.hitl_state["report_llm"],
                        "summarization_llm": st.session_state.hitl_state["summarization_llm"],
                        "research_queries": st.session_state.hitl_state.get("research_queries", []),
                        "analysis": st.session_state.hitl_state["analysis"],
                        "follow_up_questions": st.session_state.hitl_state["follow_up_questions"],
                        "human_feedback": st.session_state.hitl_state["human_feedback"]
                    }
                    
                    # Automatically execute all remaining phases
                    st.session_state.workflow_phase = "auto_executing"
                    
                    # Increment input counter
                    st.session_state.input_counter += 1
                    st.rerun()
                
                else:
                    # Continue HITL conversation
                    st.session_state.hitl_state["human_feedback"] = human_feedback
                    
                    # Add to conversation history
                    st.session_state.hitl_conversation_history.append({
                        "role": "user",
                        "content": human_feedback
                    })
                    
                    # Process feedback
                    follow_up_response = process_human_feedback(st.session_state.hitl_state)
                    
                    # Add AI response
                    st.session_state.hitl_conversation_history.append({
                        "role": "assistant",
                        "content": follow_up_response
                    })
                    
                    st.rerun()

# ========================================
# PHASE 2 & 3: AUTOMATIC EXECUTION
# ========================================
elif st.session_state.workflow_phase == "auto_executing":
    # Show HITL results summary
    if st.session_state.hitl_result:
        with st.expander("📋 HITL-Phase Ergebnisse (Abgeschlossen)", expanded=False):
            research_queries = st.session_state.hitl_result.get("research_queries", [])
            st.markdown(f"**Ursprüngliche Anfrage:** {st.session_state.hitl_result.get('user_query', 'N/A')}")
            st.markdown(f"**{len(research_queries)} Forschungsfragen generiert**")
            for i, query in enumerate(research_queries, 1):
                st.markdown(f"**{i}.** {query}")
    
    # Execute all remaining phases automatically
    success = execute_all_phases_automatically()
    if success:
        st.rerun()

# ========================================
# FINAL ANSWER DISPLAY
# ========================================
elif st.session_state.workflow_phase == "completed":
    # Show HITL results summary
    if st.session_state.hitl_result:
        with st.expander("📋 HITL-Phase Ergebnisse (Abgeschlossen)", expanded=False):
            research_queries = st.session_state.hitl_result.get("research_queries", [])
            st.markdown(f"**Ursprüngliche Anfrage:** {st.session_state.hitl_result.get('user_query', 'N/A')}")
            st.markdown(f"**{len(research_queries)} Forschungsfragen generiert**")
            for i, query in enumerate(research_queries, 1):
                st.markdown(f"**{i}.** {query}")
    
    # Show Phase 2 results with detailed documents and summaries
    if st.session_state.retrieval_summarization_result:
        with st.expander("📚 Retrieval & Summarization Ergebnisse (Abgeschlossen)", expanded=False):
            result = st.session_state.retrieval_summarization_result
            retrieved_docs = result.get("retrieved_documents", {})
            search_summaries = result.get("search_summaries", {})
            
            st.markdown(f"**Abgerufene Dokumente gesamt:** {sum(len(docs) for docs in retrieved_docs.values())}")
            st.markdown(f"**Generierte Zusammenfassungen gesamt:** {sum(len(sums) for sums in search_summaries.values())}")
            st.divider()
            
            # Show retrieved documents per query
            if retrieved_docs:
                st.subheader("📄 Abgerufene Dokumente")
                for query, docs in retrieved_docs.items():
                    with st.expander(f"Anfrage: {query} ({len(docs)} Dokumente)", expanded=False):
                        for idx, doc in enumerate(docs, 1):
                            st.markdown(f"**Dokument {idx}**")
                            if hasattr(doc, 'page_content'):
                                st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.caption(f"Quelle: {doc.metadata.get('source', 'Unbekannt')}")
                            else:
                                st.markdown(str(doc)[:500])
                            st.divider()
            
            # Show generated summaries per query
            if search_summaries:
                st.subheader("📝 Generierte Zusammenfassungen")
                for query, summaries in search_summaries.items():
                    with st.expander(f"Anfrage: {query} ({len(summaries)} Zusammenfassungen)", expanded=False):
                        for idx, summary in enumerate(summaries, 1):
                            st.markdown(f"**Zusammenfassung {idx}**")
                            if isinstance(summary, dict):
                                content = summary.get('Content', str(summary))
                                st.markdown(content)
                                if 'Source' in summary:
                                    st.caption(f"Quelle: {summary['Source']}")
                            else:
                                st.markdown(str(summary))
                            st.divider()
    
    # Show Phase 3 results - Reranked summaries and quality check
    if st.session_state.reporting_result:
        with st.expander("📊 Reranking & Qualitätsergebnisse (Abgeschlossen)", expanded=False):
            result = st.session_state.reporting_result
            
            # Show reranked summaries
            reranked_summaries = result.get("all_reranked_summaries", [])
            if reranked_summaries:
                st.subheader("🏆 Gerankte Dokumente")
                st.markdown(f"**Gerankte Dokumente gesamt:** {len(reranked_summaries)}")
                
                # Summary statistics
                total_docs = len(reranked_summaries)
                avg_score = sum(item.get('score', 0) for item in reranked_summaries) / total_docs if total_docs > 0 else 0
                max_score = max([item.get('score', 0) for item in reranked_summaries], default=0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dokumente gesamt", total_docs)
                with col2:
                    st.metric("Durchschnittlicher Score", f"{avg_score:.2f}/10")
                with col3:
                    st.metric("Höchster Score", f"{max_score:.2f}/10")
                
                st.divider()
                
                # Display top reranked documents
                for rank, item in enumerate(reranked_summaries[:10], 1):  # Show top 10
                    score = item.get('score', 0)
                    summary_data = item.get('summary', {})
                    
                    with st.expander(f"🥇 Rang #{rank} (Score: {score:.2f}/10)", expanded=False):
                        # Handle different summary formats
                        if isinstance(summary_data, dict):
                            content = summary_data.get('Content', str(summary_data))
                            source = summary_data.get('Source', 'Unbekannt')
                        else:
                            if hasattr(summary_data, 'page_content'):
                                content = summary_data.page_content
                                source = summary_data.metadata.get('source', 'Unbekannt') if hasattr(summary_data, 'metadata') else 'Unbekannt'
                            else:
                                content = str(summary_data)
                                source = 'Unbekannt'
                        
                        st.markdown(f"**Score:** {score:.2f}/10")
                        st.markdown(f"**Anfrage:** {item.get('query', 'N/A')}")
                        st.markdown(f"**Quelle:** {source}")
                        st.markdown("**Inhalt:**")
                        st.markdown(content)
            
            st.divider()
            
            # Show quality assessment details
            quality_check = result.get("quality_check", {})
            if quality_check:
                st.subheader("✅ Qualitätsprüfung")
                if "overall_score" in quality_check:
                    score = quality_check["overall_score"]
                    max_score = 400
                    score_percentage = (score / max_score) * 100
                    
                    if score >= 300:
                        st.success(f"✅ Qualitätsscore: {score}/{max_score} ({score_percentage:.1f}%) - BESTANDEN")
                    else:
                        st.warning(f"⚠️ Qualitätsscore: {score}/{max_score} ({score_percentage:.1f}%) - VERBESSERUNG NÖTIG")
                
                # Show full assessment if available
                full_assessment = quality_check.get("full_assessment", "")
                if full_assessment:
                    st.text_area("Vollständige Qualitätsbewertung", full_assessment, height=200)
    
    # Display final research report
    if st.session_state.reporting_result:
        result = st.session_state.reporting_result
        final_answer = result.get("linked_final_answer") or result.get("final_answer", "")
        
        if final_answer and final_answer.strip():
            st.markdown("---")
            st.markdown("# 🎯 Finaler Forschungsbericht")
            
            # Parse structured output
            final_content, thinking_content = parse_structured_llm_output(final_answer)
            
            # Show thinking in expander
            if thinking_content:
                with st.expander("🧠 LLM Denkprozess", expanded=False):
                    st.markdown(thinking_content)
            
            # Display final answer
            with st.chat_message("assistant"):
                st.markdown(final_content, unsafe_allow_html=True)
```

### Step 2: Remove Old Tab-Based Structure

Delete everything from line ~1717 (starting with `# Phase 1: HITL` and `with tab1:`) 
through to just before the `def main():` function (around line 2173).

This removes:
- Old tab1 (HITL) section
- Old tab2 (Retrieval-Summarization) section  
- Old tab3 (Reporting) section
- Old final answer display section

### Step 3: Verify Structure

After the changes, your file structure should be:

1. Imports
2. Header with BrAIn branding
3. Sidebar configuration
4. Helper functions (initialize_hitl_state, process_initial_query, etc.)
5. execute_retrieval_summarization_phase()
6. execute_reporting_phase()
7. execute_all_phases_automatically()
8. start_new_session()
9. Workflow visualization expander (old one - can be deleted)
10. **NEW: Workflow visualization expander**
11. **NEW: Unified workflow interface**
    - Session state initialization
    - Phase status banner
    - HITL phase
    - auto_executing phase  
    - completed phase with detailed results
12. def main()
13. if __name__ == "__main__"

### Step 4: Test

Run the German version:
```bash
uv run streamlit run apps/app_v2_1g.py --server.port 8506 --server.headless false
```

## Summary

The German version now has the same unified workflow as the English version with:
- ✅ Single main window (no tabs)
- ✅ Automatic execution when /end is typed
- ✅ Progress tracking with st.status()
- ✅ Detailed preliminary results in expanders
- ✅ All German text

The remaining manual work is to complete the HITL section, add the auto_executing and completed phases, and remove the old tab structure.
