#!/usr/bin/env python3
"""
Demo script showing the enhanced structured output functionality in apps/app_v2_0.py and apps/app_v2_0g.py
"""

import streamlit as st
import sys
import os

# Add the apps directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'apps'))

from app_v2_0 import parse_structured_llm_output

st.set_page_config(
    page_title="Structured Output Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Enhanced Structured Output Parsing Demo")
st.markdown("This demo shows how the improved `apps/app_v2_0.py` and `apps/app_v2_0g.py` handle structured LLM output.")

# Example data like the user provided
example_data = {
    "thinking": "The user requests a comprehensive, scientifically grounded explanation of the Kd‑Wert, its mathematical definition, units, relevance for Ra‑226, typical value ranges, and key influencing factors. The highest‑ranked primary summary contains the most relevant data, so I will base the core answer on that, citing its specific statements. I will supplement with information from the secondary and tertiary summaries to provide context on value ranges and additional correlations.",
    "final": """# Kd‑Wert (Verteilungskoeffizient) – Begriff, mathematische Definition und Bedeutung für ^226Ra

## Einleitung
Der **Kd‑Wert** (Distribution‑Coefficient) quantifiziert die Adsorption eines Stoffes an feste Partikel im Verhältnis zur Konzentration in der umgebenden Flüssigkeit. In der Radioökologie dient er als Maß für die Mobilität radioaktiver Isotope wie Radium‑226 (Ra‑226).

## Definition und mathematische Formel
Der Kd‑Wert wird definiert als
\\[
K_d = \\frac{C_{\\text{fest}}}{C_{\\text{Wasser}}}
\\]
wobei \\(C_{\\text{fest}}\\) die Konzentration des Sorptionsobjekts an der festen Phase und \\(C_{\\text{Wasser}}\\) die Konzentration in der Lösung beschreibt.

## Typische Kd‑Werte für ^226Ra
Die Werte für Ra‑226 variieren stark (über zwei Größenordnungen), was auf die hohe Sensitivität der Adsorption gegenüber Bodencharakteristika zurückzuführen ist.

**Information Fidelity Score:** 10/10"""
}

st.markdown("## 📝 Input Example")
st.markdown("Here's an example of structured LLM output with thinking and final parts:")
st.code(str(example_data), language="python")

st.markdown("## 🔄 Processing")
st.markdown("The `parse_structured_llm_output()` function processes this data...")

# Process the data
final_content, thinking_content = parse_structured_llm_output(example_data)

st.markdown("## 📊 Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Processing Stats")
    st.metric("Thinking Content Found", "Yes" if thinking_content else "No")
    st.metric("Final Content Length", f"{len(final_content)} chars")
    st.metric("Thinking Content Length", f"{len(thinking_content) if thinking_content else 0} chars")

with col2:
    st.markdown("### 🎯 Key Features")
    st.markdown("""
    - ✅ Automatic format detection
    - ✅ Flexible key matching
    - ✅ Content cleanup
    - ✅ Error handling
    - ✅ Markdown rendering
    """)

st.markdown("## 🎨 UI Display (as it appears in the apps)")

st.markdown("### Final Research Report")

# Simulate the chat message display
with st.chat_message("assistant"):
    # Show thinking process in expander if available
    if thinking_content:
        with st.expander("🧠 LLM Thinking Process", expanded=False):
            st.markdown(thinking_content)
    
    # Display the main content
    st.markdown(final_content)

st.markdown("---")
st.markdown("### 🚀 Ready to Use!")
st.success("Both `apps/app_v2_0.py` and `apps/app_v2_0g.py` are now enhanced with this structured output parsing capability!")

st.markdown("### 🔧 Supported Formats")
st.markdown("""
The enhanced parsing supports:
- **JSON strings**: `'{"thinking": "...", "final": "..."}'`
- **Python dict strings**: `"{'reasoning': '...', 'response': '...'}"`
- **Direct dictionaries**: `{'thought': '...', 'answer': '...'}`
- **Regular strings**: Pass-through with cleanup
- **Various key patterns**: thinking/final, thought/answer, reasoning/response, etc.
""")
