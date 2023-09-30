import os
import logging as log
import argparse

config = {
    'llm_model':'meta-llama/Llama-2-13b-chat-hf',
    'embedding_model':'sentence-transformers/all-MiniLM-L6-v2',
    'pdf_directory': "",
    'text_split_chunk_size':1000,
    'chunk_overlap':20,
    'pinecone_index': "",
    'huggingface_key': os.environ.get('HUGGINGFACE_ENV') or 'HUGGINGFACE_ENV',
    'pinecone_key':os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY',
    'pinecone_env': os.environ.get('PINECONE_ENV') or 'PINECONE_ENV',
    'device':'cuda',
    'max_opt_token': 512,
    'vectorstore_similarity_query': 4
}

log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
log.info(f"Loaded config: {config}\n")

def main():
    parser = argparse.ArgumentParser(description='Process prompts.')
    parser.add_argument('--prompt', type=str, help='Input prompt')
    args = parser.parse_args()

    if args.prompt:
        log.info(f'Prompt received: {args.prompt}')
    else:
        log.error('No prompt provided.')

    


if __name__ == '__main__':
    main()