# Aaraki 
Retriever based QA using Llama-2 (13b)

- ⚠️ Under development ⚠️

## Setup

1. Create and Activate Python Virtual Environment
```bash
python3 -m venv aaraki-env
source aaraki-env/bin/activate
```

2. Set Environment Variables
Create a .env file based on the provided dummy.env. Set the values for Huggingface and Pinecone.
```env
HUGGINGFACE_ENV=your_huggingface_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENV=your_pinecone_env_here
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Run
- Keep all your PDF files (which is to be searched) in datasets folder.
> Note : Keep all the PDF files in ./datasets/ folder itself unnested.

- Also make necessary changes in src/app.py, config

```python
config = {
    "model_name": "meta-llama/Llama-2-13b-chat-hf",
    "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "pdf_directory": "./datasets/",
    "text_split_chunk_size": 800,
    "chunk_overlap": 40,
    "pinecone_index": "power-index",
    "huggingface_key": os.environ.get("HUGGINGFACE_ENV") or "HUGGINGFACE_ENV",
    "pinecone_key": os.environ.get("PINECONE_API_KEY") or "PINECONE_API_KEY",
    "pinecone_env": os.environ.get("PINECONE_ENV") or "PINECONE_ENV",
    "device": "cuda",
    "max_opt_token": 512,
    "log": False,
}
```

- Run from Terminal

```bash
python3 src/app.py 
```

> Note : Initially, it will take a considerable amount of time before prompting you. This is due to the installation of the Llama-2 13b model, which is 24GB in size. You can install other LLM models by adding the model's Hugging Face path/id to the config.

4. To get the desired results, the prompt template can be improved by updating `template` in `src/app.py`

## TODO 
- Impliment a chat streaming
- Improve condense question prompt
- Improve character text splitter for better text embeddings
- Impliment a GUI/CLI application for chat.