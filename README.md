# Chatbot RAG 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-rag-hiyubudyfurchjq9nn5pt3.streamlit.app/)

**Link do Projeto:** [Acessar Demo Online](https://chatbot-rag-hiyubudyfurchjq9nn5pt3.streamlit.app/)

Implementação de um chatbot utilizando **RAG (Retrieval-Augmented Generation)** para consulta contextual em documentos PDF. 

Diferente de implementações tradicionais caras, este projeto utiliza a **Groq API** (Llama-3) para inferência ultrarrápida e **HuggingFace** para embeddings locais, tornando a solução extremamente performática e gratuita.

##  Tech Stack

* **Python 3.9+**
* **Streamlit:** Interface de usuário interativa.
* **LangChain:** Framework de orquestração para LLMs.
* **Groq API:** Processamento de linguagem natural usando o modelo **Llama-3-70b** (Baixa latência).
* **HuggingFace Embeddings:** Vetorização de texto rodando na CPU (`sentence-transformers/all-MiniLM-L6-v2`).
* **FAISS:** Banco de dados vetorial para busca semântica eficiente.
* **PyPDF2:** Extração de texto dos arquivos.

##  Arquitetura

Fluxo de dados do upload até a resposta:

    ```mermaid
    graph TD
        subgraph Setup
            A[PDF Upload] --> B[Text Extraction]
            B --> C[Chunking]
            C --> D[HuggingFace Embeddings]
            D --> E[(FAISS Vector Store)]
        end
        
        subgraph Runtime
            U[User Query] --> F[Query Embedding]
            F --> G{Semantic Search}
            E --> G
            G -->|Top k Chunks| H[Context + Query]
            H --> I[Groq API / Llama-3]
            I --> J[Response]
        end


Como Rodar Localmente

Clone o repositório:

Bash

git clone [https://github.com/SEU_USUARIO/Chatbot-RAG-Final.git](https://github.com/SEU_USUARIO/Chatbot-RAG.git)
cd Chatbot-RAG
Crie um ambiente virtual (Opcional, mas recomendado):

Bash

python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
Instale as dependências:

Bash

pip install -r requirements.txt
Configure a API Key:

Crie uma conta na Groq Cloud.

Crie um arquivo .env na raiz do projeto (ou insira diretamente na interface do Streamlit quando rodar).

Conteúdo do .env:

GROQ_API_KEY=gsk_sua_chave_aqui...
Execute a aplicação:

Bash

streamlit run app.py
