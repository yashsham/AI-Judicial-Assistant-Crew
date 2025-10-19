# app.py

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables at the very top
load_dotenv()

# CrewAI and LangChain Imports
from crewai import Crew, Process, Task
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# NEW: Import for universal document loading
from langchain_community.document_loaders import UnstructuredFileLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from fpdf import FPDF

# Import all agent classes
from agents.case_intake_agent import CaseIntakeAgents
from agents.constitutional_query_agent import ConstitutionalQueryAgents
from agents.legal_precedent_agent import LegalPrecedentAgents
from agents.synthesis_agent import SynthesisAgents

# --- Page Configuration ---
st.set_page_config(page_title="AI Judicial Assistant", layout="wide")
st.title("⚖️ AI Indian Judicial Assistant")
st.write("This tool uses AI agents to analyze legal documents against the Constitution of India.")

# --- UPGRADED Helper Functions using 'unstructured' ---

# FIX: Add an underscore to the 'files' argument to make it unhashable for caching
@st.cache_data
def get_text_from_files(_files):
    """Extracts raw text content from various file formats using 'unstructured'."""
    full_text = ""
    # FIX: Use the new argument name '_files' in the loop
    if not _files:
        return ""
        
    for file_obj in _files:
        st.write(f"Processing '{file_obj.name}'...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
            tmp_file.write(file_obj.getvalue())
            tmp_file_path = tmp_file.name

        try:
            loader = UnstructuredFileLoader(tmp_file_path)
            pages = loader.load()
            for page in pages:
                full_text += page.page_content + "\n\n"
            st.success(f"Successfully extracted text from '{file_obj.name}'.")
        except Exception as e:
            st.error(f"Error processing '{file_obj.name}': {e}")
        finally:
            os.remove(tmp_file_path)
            
    if not full_text.strip():
        st.error("Could not extract any text from the uploaded files. Please check the files.")

    return full_text

@st.cache_resource
def create_rag_retriever_from_text(_text, collection_name_suffix):
    """Creates a searchable retriever from a block of text."""
    if not _text: return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.create_documents([_text])
    
    if not docs: return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=f"rag_{collection_name_suffix}")
    return vectorstore.as_retriever()

# --- File Uploaders ---
st.header("Step 1: Upload Documents")
col1, col2 = st.columns(2)
with col1:
    uploaded_case_files = st.file_uploader(
        "Upload Case Files (PDF, DOCX, TXT, etc.)",
        accept_multiple_files=True, 
        type=['pdf', 'docx', 'txt']
    )
with col2:
    constitution_path = "knowledge_base/Constitution_of_India.pdf"
    if not os.path.exists(constitution_path):
        st.error("`Constitution_of_India.pdf` not found in 'knowledge_base' folder.")
        st.stop()
    st.success("✅ Constitution of India loaded.")

# --- Main Logic ---
if uploaded_case_files:
    if st.button("Begin Full Analysis"):
        with st.spinner("The AI crew is analyzing your case... 🤖"):
            from crewai import LLM

            llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)
            
            st.info("Extracting text from your case files...")
            case_text = get_text_from_files(uploaded_case_files)

            if not case_text.strip():
                st.error("Could not extract any text from the uploaded files. Aborting analysis.")
                st.stop()

            st.info("Processing the Constitution of India for RAG...")
            with open(constitution_path, "rb") as f:
                class InMemoryFile:
                    def __init__(self, content, name):
                        self._content = content
                        self.name = name
                    def getvalue(self):
                        return self._content
                
                constitution_text = get_text_from_files([InMemoryFile(f.read(), "Constitution.pdf")])
            
            if not constitution_text:
                st.error("Could not extract text from the Constitution PDF. Aborting analysis.")
                st.stop()
                
            constitution_retriever = create_rag_retriever_from_text(constitution_text, "constitution")

            @tool("Constitution RAG Tool")
            def constitution_rag_tool(query: str) -> str:
                """Searches the Constitution of India for relevant articles and clauses."""
                docs = constitution_retriever.invoke(query)
                return "\n".join([doc.page_content for doc in docs])

            intake_agents = CaseIntakeAgents()
            constitutional_agents = ConstitutionalQueryAgents()
            precedent_agents = LegalPrecedentAgents()
            synthesis_agents = SynthesisAgents()

            intake_agent = intake_agents.make_intake_agent(llm)
            constitutional_agent = constitutional_agents.make_query_agent(llm, tools=[constitution_rag_tool])
            precedent_agent = precedent_agents.make_precedent_agent(llm)
            synthesis_agent = synthesis_agents.make_synthesis_agent(llm)

            summarization_task = intake_agents.make_summarization_task(intake_agent, case_text)
            
            constitutional_analysis_task = constitutional_agents.make_constitutional_analysis_task(constitutional_agent, context=[summarization_task])
            precedent_research_task = precedent_agents.make_precedent_research_task(precedent_agent, context=[constitutional_analysis_task])
            synthesis_task = synthesis_agents.make_synthesis_task(synthesis_agent, context=[summarization_task, constitutional_analysis_task, precedent_research_task])

            crew = Crew(
                agents=[intake_agent, constitutional_agent, precedent_agent, synthesis_agent],
                tasks=[summarization_task, constitutional_analysis_task, precedent_research_task, synthesis_task],
                process=Process.sequential
            )
            
            result = crew.kickoff()

            st.success("Analysis complete! Converting to PDF...")
            
            markdown_text = result.raw
            pdf_filename = "preliminary_judicial_note.pdf"
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 5, text=markdown_text.encode('latin-1', 'replace').decode('latin-1'))
            pdf.output(pdf_filename)

            st.header("Preliminary Judicial Note")
            st.markdown(markdown_text)
            
            with open(pdf_filename, "rb") as f:
                st.download_button(
                    label="⬇️ Download Full Note as PDF",
                    data=f.read(),
                    file_name=pdf_filename,
                    mime="application/pdf"
                )