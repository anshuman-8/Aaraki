import os
import argparse
import logging as log
from dotenv import load_dotenv
from aaraki_model.askAaraki import AskAaraki

load_dotenv()

config = {
    "model_name": "meta-llama/Llama-2-13b-chat-hf",
    "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "pdf_directory": "./datasets/explain_me/",
    "text_split_chunk_size": 750,
    "chunk_overlap": 40,
    "pinecone_index": "anshuman-info",
    "huggingface_key": os.environ.get("HUGGINGFACE_ENV") or "HUGGINGFACE_ENV",
    "pinecone_key": os.environ.get("PINECONE_API_KEY") or "PINECONE_API_KEY",
    "pinecone_env": os.environ.get("PINECONE_ENV") or "PINECONE_ENV",
    "device": "cuda",
    "max_opt_token": 512,
    "vectorstore_similarity_query": 5,
    "log": True,
}

template = """
        You help everyone by answering questions, and improve your answers from previous answer in History.
        Don't try to make up an answer, if you don't know just say that you don't know.
        Answer in the same language the question was asked.
        Answer in a way that is easy to understand.
        Do not say "Based on the information you provided, ..." or "I think the answer is...". Just answer the question directly in detail.
        Use only the following pieces of context to answer the question at the end.

        History: {chat_history}

        Context: {context}

        Question: {question}
        Answer:"""

if config["log"]:
    log.basicConfig(level=log.INFO, format=" %(levelname)s %(message)s")
log.info(f"Loaded config: {config}\n")


def main():
    parser = argparse.ArgumentParser(description="Process prompts.")
    parser.add_argument("--topic", type=str, help="Input topic")
    args = parser.parse_args()

    if args.topic:
        log.info(f"Topic received: {args.topic}")
    else:
        log.error("No Topic provided provided.")
        exit()

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
        template=template,
        vectorstore_similarity_query=config["vectorstore_similarity_query"],
    )

    aaraki.llm_rag_pipeline(topic=args.topic)

    while True:
        prompt = input("\nEnter your prompt: ")

        if prompt == "quit":
            break
        else:
            log.info(f"Prompt received: {prompt}")

        answer = aaraki.ask(prompt=prompt)
        # print(f"Answer: {aaraki.process_response(answer)}")
        print(f"Answer: {answer}")

        print("\n")
        end = input("Do you want to end (y/n)? ")
        if end.lower() == "y":
            break


if __name__ == "__main__":
    main()
