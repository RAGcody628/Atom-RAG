import os
import json
import time

from PathCoRAG import PathCoRAG, QueryParam
from PathCoRAG.llm.openai import gpt_4o_mini_complete, openai_embed

def insert_text(rag, file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.txt':
        pass
    else:
        with open(file_path, mode="r") as f:
            unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            if file_extension == '.txt': 
                with open(file_path, "r", encoding="utf-8") as f:
                    rag.insert(f.read(), param = construction_param)
            else:
                rag.insert(unique_contexts, param = construction_param)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


cls = "test"
mode = "ours_experiment2" # experiment1: doc->chunk->atomic / experiment2: doc->chunk->atomic->triple
construction_param = QueryParam(Mode = mode)
WORKING_DIR = f"../{cls}"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathCoRAG(
    working_dir=WORKING_DIR,
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete,
)

if cls == "hotpot" or cls == "multihoprag" or cls == "musique" or cls == "2wikimultihopqa" or cls == "test" or cls == "triviaqa" or cls == "nq":
    insert_text(rag, f"../datasets/unique_contexts/{cls}_unique_contexts.txt")
else:
    insert_text(rag, f"../datasets/unique_contexts/{cls}_unique_contexts.json")