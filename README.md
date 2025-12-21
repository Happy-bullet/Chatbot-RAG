Chatbot RAG

Implementação de um chatbot utilizando RAG (Retrieval-Augmented Generation) para consulta contextual em documentos PDF. O sistema vetoriza o conteúdo carregado e utiliza a OpenAI API para gerar respostas fundamentadas no texto fornecido.

Stack

Python 3.9+

Streamlit (Interface)

LangChain (Orquestração LLM)

OpenAI API (GPT-3.5/4)

FAISS (Vector Store)

PyPDF2 (Parser)

Arquitetura

Fluxo de dados do upload até a resposta:

    graph TD
        subgraph Setup
            A[PDF Upload] --> B[Text Extraction]
            B --> C[Chunking]
            C --> D[OpenAI Embeddings]
            D --> E[(FAISS Vector Store)]
        end
    
        subgraph Runtime
            U[User Query] --> F[Query Embedding]
            F --> G{Semantic Search}
            E --> G
            G -->|Top k Chunks| H[Context + Query]
            H --> I[LLM Generation]
            I --> J[Response]
        end


Setup

Clone o repo.

Instale as libs:

    pip install streamlit langchain langchain-openai faiss-cpu pypdf tiktoken


Rode a aplicação:

    streamlit run app.py
