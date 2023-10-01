# Aaraki 

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
> Note : Keep all the PDF files in ./datasets folder itself unnested.

- Also make necessary changes in src/app.py, config

```python
config = {
    'model_name':'meta-llama/Llama-2-13b-chat-hf',
    'embed_model_name':'sentence-transformers/all-MiniLM-L6-v2',
    'pdf_directory': "./datasets/",
    'text_split_chunk_size':1000,
    'chunk_overlap':20,
    'pinecone_index': "ix-history-2",
    'huggingface_key': os.environ.get('HUGGINGFACE_ENV') or 'HUGGINGFACE_ENV',
    'pinecone_key':os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY',
    'pinecone_env': os.environ.get('PINECONE_ENV') or 'PINECONE_ENV',
    'device':'cuda',
    'max_opt_token': 512,
    'vectorstore_similarity_query': 4
}
```

- Run from Terminal

```bash
python3 src/app.py --topic "Russian Revolution"
```

> Note : Initially, it will take a considerable amount of time before prompting you. This is due to the installation of the Llama-2 13b model, which is 24GB in size. You can install other LLM models by adding the model's Hugging Face path/id to the config.