# AI Judicial Assistant ⚖️

This project implements an advanced, multi-agent AI system designed to assist the Indian judiciary by automating the preliminary analysis of legal cases. The system uses a crew of specialized AI agents to process uploaded case documents, analyze them against the Constitution of India using a RAG pipeline, research legal precedents, and synthesize a comprehensive "Preliminary Judicial Note."

## Features ✨

*   **🤖 Autonomous Four-Agent Crew:** A team of specialized AI agents collaborate to perform a full legal analysis workflow:
    1.  **`CaseIntakeAgent`**: Summarizes the core facts from uploaded case files.
    2.  **`ConstitutionalQueryAgent`**: Identifies constitutional issues and retrieves relevant articles using a RAG pipeline built on the Constitution of India.
    3.  **`LegalPrecedentAgent`**: Researches and identifies relevant historical case law.
    4.  **`SynthesisAgent`**: Compiles all information into a final, structured judicial note.
*   **🧠 Smart Provider Selection:** Automatically selects the best available AI provider (LLM & Embeddings) based on your API keys.
    *   **LLMs:** OpenRouter (DeepSeek), OpenAI, Google Gemini, Groq.
    *   **Embeddings:** OpenAI, Google Gemini, HuggingFace (Local).
*   **🇮🇳 Constitutional RAG:** A powerful Retrieval-Augmented Generation system built on the full text of the Constitution of India.
*   **📄 Universal Document Support:** Processes PDF, DOCX, and TXT files using `unstructured`.
*   **🌐 Clean UI:** A streamlined Streamlit interface with no manual configuration needed.

## Tech Stack 🛠️

*   **Agent Framework:** CrewAI
*   **Orchestration:** LangChain
*   **LLM Providers:** Dynamic (OpenAI, Google, Groq, OpenRouter)
*   **Embeddings:** Dynamic (OpenAI, Google, HuggingFace)
*   **Vector Store:** ChromaDB
*   **UI Framework:** Streamlit
*   **Document Processing:** unstructured, PyPDF

## Setup & Installation ⚙️

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/AI-Judicial-Assistant-Crew.git
cd AI-Judicial-Assistant-Crew
```

**2. Create and activate a virtual environment:**
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up your API Keys:**
Create a file named `.env` in the root directory and add your keys. The app will automatically detect which ones are available and pick the best provider.

```env
# Add as many as you have. The app will prioritize them automatically.
OPENROUTER_API_KEY="sk-..."
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
GROQ_API_KEY="gsk_..."
HUGGINGFACE_API_KEY="hf_..."
```

**5. Prepare the Knowledge Base:**
Place a PDF version of the Constitution of India into the `knowledge_base/` folder. The file must be named `Constitution_of_India.pdf`.

## How to Run 🚀

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The app will launch in your browser. It will seamlessly switch to the best available AI models based on your credentials.