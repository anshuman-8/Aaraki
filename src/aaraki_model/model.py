import os
import time
import pinecone
import transformers
import logging as log
from torch import bfloat16
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class AskAaraki():
    def __init__(self, model_name, embed_model_name, tokenizer, device, hf_key, pinecone_key, pinecone_env, pdf_directory, text_split_chunk_size, chunk_overlap, pinecone_index, max_opt_token=512, vectorstore_similarity_query=2):
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.embed_model = None
        self.model = None
        self.index_name = pinecone_index
        self.index = None
        self.pdf_directory = pdf_directory
        self.hf_key = hf_key
        self.pinecone_key = pinecone_key
        self.pinecone_env = pinecone_env
        self.tokenizer = tokenizer
        self.device = device
        self.vectorstore = None
        self.max_opt_token = 512
        self.vectorstore_similarity_query = vectorstore_similarity_query
        self.text_split_chunk_size = text_split_chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_dim = 384

        self.rag_pipeline = None

        self.load_model()
        self._embed()
        self._upsert_vectorstore()

    def load_model(self):
        bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
        )
        log.debug(f"4 bit quantization enabled")
        hf_auth = self.hf_key
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_name,
            use_auth_token=hf_auth
        )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth,
            resume_download = True

        )
        self.model.eval()
        log.info(f"Loaded model: {self.model_name} on {self.device}")

    def _embed(self):
        embed_model_id = self.embed_model_name

        self.embed_model = HuggingFaceEmbeddings(
            model_name = embed_model_id,
            model_kwargs = {'device': self.device},
        )
        log.info(f"Loaded embed model: {embed_model_id}")

    def _get_texts(self):
        pdf_files = [os.path.join(self.pdf_directory, f) for f in os.listdir(self.pdf_directory) if os.path.isfile(os.path.join(self.pdf_directory, f)) and f.endswith('.pdf')]
        log.debug(f"Found {len(pdf_files)} pdf files in {self.pdf_directory}")
        if len(pdf_files) == 0:
            raise ValueError(f"No pdf files found in {self.pdf_directory}")
        loader = PyPDFDirectoryLoader(self.pdf_directory)
        data = loader.load()
        log.debug(f"Loaded {len(data)} documents from {self.pdf_directory}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.text_split_chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(data)
        log.debug(f"Split {len(texts)} documents into texts chunks")
        return texts

    def _upsert_vectorstore(self):
        log.debug(f"Initializing pinecone")
        log.debug(f"pinecone key: {self.pinecone_key}")

        pinecone.init(
            api_key = self.pinecone_key,
            environment = self.pinecone_env,
            )
        log.debug(f"pinecone initialized")

        if self.index_name == None or self.index_name == "":
            raise ValueError("Please provide a pinecone index name")


        if self.index_name not in pinecone.list_indexes():
            log.debug(f"Creating index: {self.index_name}")
            pinecone.create_index(
                self.index_name,
                dimension=self.embed_dim,
                metric='cosine'
            )

        while not pinecone.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        self.index = pinecone.Index(self.index_name)
        log.info(f"Index created: {self.index.describe_index_stats()}")

        texts = self._get_texts()

        ids = [f"{i}-{texts[i].metadata['page']}" for i in range(len(texts))]
        textList = [text.page_content for text in texts]
        embeds = self.embed_model.embed_documents(textList) 
        metadata = [{
            'text': text.page_content,
                'source':text.metadata['source'],
                'page': text.metadata['page']
        } for text in texts]
        log.debug(f"Upserting {len(ids)} vectors into {self.index_name}")

        self.index.upsert(vectors=zip(ids, embeds, metadata), show_progress=False)
        log.info(f"Upsert Done!! {len(ids)} vectors into {self.index_name}")

    def llm_rag_pipeline(self, topic):
        log.debug(f'Loading llm_rag_pipeline')
        log.debug(f"Topic: {topic}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_key
        )
        log.debug(f"Loaded tokenizer: {self.model_name}")

        generate_text = transformers.pipeline(
            model=self.model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1,  # without this output begins repeating
            framework="pt",
        )

        llm = HuggingFacePipeline(pipeline=generate_text)
        log.debug(f"Loaded pipeline: {self.model_name}")
        self.vectorstore = Pinecone(
            self.index, self.embed_model.embed_query, "text"
        )
        self.vectorstore.similarity_search(topic, k=6)
        log.debug(f"Loaded vectorstore: {self.vectorstore}")

        self.rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm, chain_type='stuff',
            retriever=self.vectorstore.as_retriever()
        )
        log.info(f'Topic received: {topic}')
        log.info(f"Retrival QA Pipeline Ready!")

    def ask(self, prompt):
        log.debug(f'Prompt received: {prompt}')
        if self.rag_pipeline is None:
            raise ValueError("Please run llm_rag_pipeline(topic) first")
        answer = self.rag_pipeline(prompt)
        log.debug(f"Answer to {answer['query']} ready!")
        return answer


    