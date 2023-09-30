import os
from torch import bfloat16
import transformers
import logging as log

class AskAaraki():
    def __inti__(self, model_name, tokenizer, device, hf_key, pinecone_key, pinecone_env, pdf_directory, text_split_chunk_size, chunk_overlap, pinecone_index, max_opt_token=512, vectorstore_similarity_query=2):
        self.model_name = model_name
        self.pdf_directory = pdf_directory
        self.model = None
        self.index = pinecone_index
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

    def load_model(self):
        bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
        )

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

        # print(f"Model loaded on {self.device}")
        log.info(f"Loaded model: {self.model_name}")

    