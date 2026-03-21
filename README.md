🤖 AI Documentation Assistant (RAG-based)
A professional-grade Retrieval-Augmented Generation (RAG) application that allows users to chat with technical PDF manuals in real-time. This project bridges the gap between structured documentation and high-speed AI inference.

🌟 Key Features
High-Speed Inference: Powered by Groq LPUs for near-instant responses.

Local Vector Intelligence: Uses HuggingFace (all-MiniLM-L6-v2) to process document mathematics locally.

Interactive UI: Built with Streamlit for a seamless browser-based user experience.

Context-Aware: Not just a chatbot; it uses specific document context to provide 100% accurate technical answers.

🛠️ Tech Stack
Language: Python 3.12

Framework: LangChain (v1.0 Architecture)

LLM: Llama 3.1 (via Groq)

Vector Database: ChromaDB

UI: Streamlit

🚀 Getting Started
1. Prerequisites
Python 3.12 (Stable)

Groq API Key (Get it at console.groq.com)

2. Installation
Bash
# Clone the repository
git clone https://github.com/yourusername/MyChatbot.git
cd MyChatbot

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Environment Setup
Create a .env file in the root directory:

Plaintext
GROQ_API_KEY=your_gsk_key_here
4. Running the App
Bash
streamlit run ui_app.py
🏗️ Architecture Design
The system uses a modular "Chains" approach. When a PDF is uploaded, it is split into 1,000-character chunks with a 200-character overlap to preserve context. These are then embedded into a 384-dimensional vector space for semantic retrieval.
