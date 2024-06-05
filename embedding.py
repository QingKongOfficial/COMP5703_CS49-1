import json,faiss,random,re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
import torch
from transformers import HfArgumentParser, GenerationConfig, BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
)


def read_text_file_to_df(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        df = pd.DataFrame(lines, columns=['sentence'])
        df['sentence'] = df['sentence'].str.strip()
    return df

def save_sentence_embeddings(text_file_path, model_path, index_path, texts_path):
    try:
        df = read_text_file_to_df(text_file_path)
        sentences = df['sentence'].tolist()
        model = SentenceTransformer(model_path)
        embeddings = model.encode(sentences, show_progress_bar=True)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(model.get_sentence_embedding_dimension()))
        ids = np.array(range(len(sentences)), dtype=np.int64)
        index.add_with_ids(embeddings, ids)
        faiss.write_index(index, index_path)
        print(f"Index saved to {index_path}")
        with open(texts_path, 'w', encoding='utf-8') as file:
            for sentence in sentences:
                file.write(sentence + '\n')
        print(f"Texts saved to {texts_path}")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    save_sentence_embeddings("data.txt", "/root/autodl-tmp/tao-8k", "embeddings.index", "output.txt")
    
