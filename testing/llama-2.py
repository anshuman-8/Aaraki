import os
import torch
from transformers import LlamaModel, LlamaConfig, pipeline
import logging as log

log.basicConfig(level=log.DEBUG)

configuration = LlamaConfig()
log.debug(f'LLaMA config: {configuration}')

model = LlamaModel(configuration)
log.info(f'model prep')
