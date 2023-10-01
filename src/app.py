import os
import argparse
import logging as log
from dotenv import load_dotenv
from aaraki_model.model import AskAaraki 

load_dotenv()

config = {
    'model_name':'meta-llama/Llama-2-13b-chat-hf',
    'embed_model_name':'sentence-transformers/all-MiniLM-L6-v2',
    'pdf_directory': "./datasets/Social/",
    'text_split_chunk_size':1000,
    'chunk_overlap':20,
    'pinecone_index': "ix-history-2",
    'huggingface_key': os.environ.get('HUGGINGFACE_ENV') or 'HUGGINGFACE_ENV',
    'pinecone_key':os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY',
    'pinecone_env': os.environ.get('PINECONE_ENV') or 'PINECONE_ENV',
    'device':'cuda',
    'max_opt_token': 512,
    'vectorstore_similarity_query': 10,
    'log': False
}

if config['log']:
    log.basicConfig(level=log.INFO, format=' %(levelname)s %(message)s')
log.info(f"Loaded config: {config}\n")

def main():
    parser = argparse.ArgumentParser(description='Process prompts.')
    # parser.add_argument('--prompt', type=str, help='Input prompt')
    parser.add_argument('--topic', type=str, help='Input topic')
    args = parser.parse_args()

    # if args.prompt:
    #     log.info(f'Prompt received: {args.prompt}')
    if args.topic:
        log.info(f'Topic received: {args.topic}')
    else:
        log.error('No Topic provided provided.')
        exit()

    aaraki = AskAaraki(
        model_name=config['model_name'],
        embed_model_name=config['embed_model_name'],
        tokenizer=None,
        device=config['device'],
        hf_key=config['huggingface_key'],
        pinecone_key=config['pinecone_key'],
        pinecone_env=config['pinecone_env'],
        pdf_directory=config['pdf_directory'],
        text_split_chunk_size=config['text_split_chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        pinecone_index=config['pinecone_index'],
        max_opt_token=config['max_opt_token'],
        vectorstore_similarity_query=config['vectorstore_similarity_query'] 
    )
   
    aaraki.llm_rag_pipeline(topic=args.topic)

    while True:
        prompt = input("\nEnter your prompt: ")

        if prompt == 'quit':
            break
        else:
            log.info(f'Prompt received: {prompt}')

        answer = aaraki.ask(prompt=prompt)
        print(f"Answer: {answer['result']}")
        
        print('\n')
        end = input("Do you want to end (y/n)? ")
        if end.lower() == 'y':
            break

    


if __name__ == '__main__':
    main()