import os
import argparse
import logging as log
from dotenv import load_dotenv
from aaraki_model.askAaraki import AskAaraki

load_dotenv()

config = {
    "model_name": "meta-llama/Llama-2-13b-chat-hf",
    "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "pdf_directory": "./datasets/pow_books/",
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

template = """
You are AI, and your job is to answer questions to Humans in most helpful and polite way possible.
You are provided with your previous answers inside <history> tag, and you can use them to answer the question. 
Also use the following pieces of information inside <context> tag to answer the question at the end.
Don't try to make up an answer, if you don't know or it dosen't exists in context just say that you don't know.
Answer in a way that is easy to understand. Answer in the the way it is asked for like, in steps.., in short, in points.... 
Do not say "Based on the information you provided, ..." or "I think the answer is...". Just answer the question directly in detail.

<history>
 {chat_history}
</history>
------------------
<context>
 {context}
</context>
------------------
Question: {question}
Helpful Answer:"""

if config["log"]:
    log.basicConfig(level=log.INFO, format=" %(levelname)s %(message)s")
log.info(f"Loaded config: {config}\n")


def main():
    parser = argparse.ArgumentParser(description="Process prompts.")
    parser.add_argument("--prompt", type=str, help="Input Prompt")
    args = parser.parse_args()

    if args.prompt:
        log.info(f"Prompt received: {args.prompt}")
    else:
        log.info("No prompt provided provided.")

    print(f'Preparing the model...')
    aaraki = AskAaraki(
        model_name=config["model_name"],
        embed_model_name=config["embed_model_name"],
        tokenizer=None,
        device=config["device"],
        hf_key=config["huggingface_key"],
        pinecone_key=config["pinecone_key"],
        pinecone_env=config["pinecone_env"],
        pdf_directory=config["pdf_directory"],
        text_split_chunk_size=config["text_split_chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        pinecone_index=config["pinecone_index"],
        max_opt_token=config["max_opt_token"],
        log=config["log"],
        template=template,
    )
    print(f'Model ready!\n')
    
    while True:
        prompt = input("\nEnter your prompt: ")

        if prompt == "quit":
            break
        else:
            log.info(f"Prompt received: {prompt}")

        print(f"Generating answer...")
        answer = aaraki.ask(prompt=prompt)
        # print(f"Answer: {aaraki.process_response(answer)}")
        print(f"Answer: {answer}")

        print("\n")
        end = input("Do you want to end (y/n)? ")
        if end.lower() == "y":
            break


if __name__ == "__main__":
    main()
