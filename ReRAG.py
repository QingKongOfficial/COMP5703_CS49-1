import os
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import os
import torch
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
import json
from tqdm import tqdm
import argparse
import json,faiss,random,re
import numpy as np
import pandas as pd
import nltk
from transformers import HfArgumentParser, GenerationConfig, BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
)

from keybert import KeyBERT



os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

## retrieval grader
class CheckRetrievalNeeded(BaseModel):
    """Binary score to check if a question needs retrieval."""

    binary_score: str = Field(
        description="Question needs retrieval, 'yes' or 'no'"
    )
system_message = """You are an assessor determining whether a user question needs document retrieval. 
    If the question is specific or factual and can be answered with a direct response, it may not need retrieval. 
    If the question is broad, open-ended, or requires detailed information, it likely needs retrieval.
    Give a binary score 'yes' or 'no' to indicate whether the question needs retrieval."""
### prompt
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "User question: {question}")
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_checker = llm.with_structured_output(CheckRetrievalNeeded)
retrieval_checker = grade_prompt | structured_llm_checker


## set up retriever

def load_index_and_search(input_string, index_path="embeddings.index", model_path="Amu/tao-8k", texts_path="output.txt", k=2):
    index = faiss.read_index(index_path)
    with open(texts_path, 'r', encoding='utf-8') as file:
        original_inputs = [line.strip() for line in file]   
    embed_model = SentenceTransformer(model_path)
    input_embedding = embed_model.encode([input_string], show_progress_bar=False)
    D, I = index.search(input_embedding, k)
    retrieved_inputs = [original_inputs[id_] for id_ in I[0]]
    return retrieved_inputs

### Relevant Grader

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

## question rewrite component

class rewrite():

    def __init__(self):
        self.model = KeyBERT(model='all-MiniLM-L6-v2')

    def text_keywords(self, text, text_range, top_n):
        output = self.model.extract_keywords(text, keyphrase_ngram_range=text_range, stop_words=None, use_maxsum=True, nr_candidates=20, top_n=top_n)

        return output

    def file_keywords(self, input_file, output_file, text_range, top_n):
        with open(input_file, 'rb') as input:
            df_input = json.load(input)

        df_output = df_input
        size = len(df_input)

        for i in tqdm(range(0, size), total=size, desc=""):

            text = df_input[i]['input']
            split = self.model.extract_keywords(text, keyphrase_ngram_range=text_range, stop_words=None, use_maxsum=True, nr_candidates=20, top_n=top_n)

            if split == []:
                df_output[i]['split'] = ''
                continue

            new = split
            for j in range(top_n):
                new[j] = split[j][0]

            df_output[i]['split'] = new

        out_file = open(output_file, "w")
        json.dump(df_output, out_file, indent=6)
        out_file.close()


rewrite = rewrite()

def process_data(input_json_path, output_json_path):
    # read file
    with open(input_json_path, 'r') as f:
        data_list = json.load(f)
    
    #data_list = data_list[:30]

    processed_data = []

    for data in tqdm(data_list, desc="Processing data"):
        try:
            question = data["input"]
            expected_output = data["output"]
        # retrieval node
            result = retrieval_checker.invoke({"question": question})
        
            if result.binary_score == 'no':
            # if no relevant retrieval text needed, use original question
                output_data = {
                    "input": question,
                    "output": expected_output
                }
            else:
                ## retrieval node
                retrieve_doc = load_index_and_search(question)

            # check if the retrieved texts are relevant
                relevant_docs = []
                for doc in retrieve_doc:
                    grade_result = retrieval_grader.invoke({"question": question, "document": doc})
                    if grade_result.binary_score == 'yes':
                        relevant_docs.append(doc)
            
                if not relevant_docs:
                    # if there is no relevant text after the first retrieval
                    # question rewrite node
                    key_phrase = rewrite.text_keywords(text= question,         
						text_range=(3, 4),
                        top_n=1)
                    ## retrieve the new phrase
                    retrieve_doc_new = load_index_and_search(key_phrase[0][0])
                    for doc in retrieve_doc_new:
                        grade_result = retrieval_grader.invoke({"question": question, "document": doc})
                        if grade_result.binary_score == 'yes':
                            relevant_docs.append(doc)
                

                if len(relevant_docs) >= 2:
                # if there are relevant texts found, build new prompt
                    new_question = f"""Documents: 1.{relevant_docs[0]}
                    2.{relevant_docs[1]}
                    Based on the above documents, please answer the following question:
                    {question}"""
                    output_data = {
                        "input": new_question,
                        "output": expected_output
                    }
                elif len(relevant_docs) == 0:
                    output_data = {
                    "input": question,
                    "output": expected_output
                    }

                else:
                    new_question = f"""Document: {relevant_docs[0]}
                    Based on the above document, please answer the following question:
                    {question}"""
                    output_data = {
                        "input": new_question,
                        "output": expected_output
                    }
        
            processed_data.append(output_data)
        except Exception as e:
            print(f"Error processing:{e}")

    with open(output_json_path, 'w') as f:
        json.dump(processed_data, f, indent=4)



parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()

input_json_path = args.input_file
output_json_path = args.output_file

process_data(input_json_path, output_json_path)
