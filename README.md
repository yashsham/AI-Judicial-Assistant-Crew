# AI Judicial Assistant ⚖️

This project implements an advanced, multi-agent AI system designed to assist the Indian judiciary by automating the preliminary analysis of legal cases. The system uses a crew of specialized AI agents to process uploaded case documents, analyze them against the Constitution of India using a RAG pipeline, research legal precedents, and synthesize a comprehensive "Preliminary Judicial Note."

## Features ✨

* **🤖 Autonomous Four-Agent Crew:** A team of specialized AI agents collaborate to perform a full legal analysis workflow:
    1.  **`CaseIntakeAgent`**: Summarizes the core facts from uploaded case files.
    2.  **`ConstitutionalQueryAgent`**: Identifies constitutional issues and retrieves relevant articles using a RAG pipeline built on the Constitution of India.
    3.  **`LegalPrecedentAgent`**: Researches and identifies relevant historical case law.
    4.  **`SynthesisAgent`**: Compiles all information into a final, structured judicial note.
* **🇮🇳 Constitutional RAG:** A powerful Retrieval-Augmented Generation system built on the full text of the Constitution of India allows for precise, context-aware legal queries.
* **📄 PDF Processing:** Capable of ingesting and analyzing multiple PDF case documents.
* **🌐 Interactive UI:** A clean and user-friendly web interface built with Streamlit for uploading files and viewing the final analysis.

## Tech Stack 🛠️

* **Agent Framework:** CrewAI
* **LLM:** OpenRouter (DeepSeek R1)
* **Embeddings:** Google Gemini Pro
* **RAG & Vector Store:** LangChain, ChromaDB
* **UI Framework:** Streamlit
* **PDF Processing:** PyPDF
* **Core Language:** Python

## Setup & Installation ⚙️

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/AI-Judicial-Assistant-Crew.git](https://github.com/YOUR_USERNAME/AI-Judicial-Assistant-Crew.git)
cd AI-Judicial-Assistant-Crew
```

**2. Create and activate a virtual environment:**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up your API Key:**
Create a file named `.env` in the root directory and add your Google API key:
```
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY_HERE"
```

**5. Prepare the Knowledge Base:**
Place a PDF version of the Constitution of India into the `knowledge_base/` folder. The file must be named `Constitution_of_India.pdf`.

## How to Run 🚀

Launch the Streamlit web application with the following command:

```bash
streamlit run app.py
```
Navigate to the local URL provided by your terminal to interact with the application.