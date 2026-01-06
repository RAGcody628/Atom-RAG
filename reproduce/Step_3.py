import re
import json
import asyncio
from PathCoRAG import PathCoRAG, QueryParam
from PathCoRAG.llm.openai import gpt_4o_mini_complete, openai_embed
from PathCoRAG.llm.groqapi import groq_llama_4_scout_complete
from PathCoRAG.llm.ollama import ollama_model_complete
from PathCoRAG.llm.ollama import ollama_embed
from PathCoRAG.llm.gemini import gemini_complete
from PathCoRAG.utils import EmbeddingFunc
from tqdm import tqdm


# =========================================================
# Query Extraction Functions
# =========================================================
def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    data = data.replace("**", "")
    queries = re.findall(r" - Question \d+: (.+)", data)
    return queries


def extract_queries_novelqa(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [v["Question"] for v in data.values() if v.get("Question")]


def extract_queries_infinite(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if question := item.get("input"):
                queries.append(question)
    return queries


def extract_queries_hotpot(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if question := item.get("question"):
                queries.append(question)
    return queries


def extract_queries_musique(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if question := item.get("question"):
                queries.append(question.strip())
    return queries


def extract_queries_2wikimultihopqa(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if question := item.get("question"):
                queries.append(question.strip())
    return queries

def extract_queries_triviaqa(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if question := item.get("question"):
                queries.append(question.strip())
    return queries

def extract_queries_nq(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if question := item.get("question"):
                queries.append(question)
    return queries


def extract_queries_multihoprag(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if question := item.get("query"):
                queries.append(question)
    return queries


# =========================================================
# Async processing logic
# =========================================================
async def process_query(query_text, rag_instance, query_param):
    """단일 쿼리 실행"""
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


async def process_queries_concurrently(queries, rag_instance, query_param, batch_size=10):
    """
    여러 쿼리를 비동기 병렬로 처리 (batch 단위)
    """
    sem = asyncio.Semaphore(batch_size)
    results, errors = [], []

    async def sem_task(query_text):
        async with sem:  # 동시 실행 제한
            return await process_query(query_text, rag_instance, query_param)

    tasks = [sem_task(q) for q in queries]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing queries", unit="query"):
        result, error = await coro
        if result:
            results.append(result)
        elif error:
            errors.append(error)

    return results, errors


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_queries_and_save_to_json_parallel(
    start_idx, end_idx, queries, rag_instance, query_param, output_file, error_file, batch_size=10
):
    """비동기 병렬 처리 + 결과 저장"""
    queries = queries[start_idx:end_idx]
    loop = always_get_an_event_loop()

    results, errors = loop.run_until_complete(
        process_queries_concurrently(queries, rag_instance, query_param, batch_size=batch_size)
    )

    # 결과 저장
    with open(output_file, "w", encoding="utf-8") as rf:
        json.dump(results, rf, ensure_ascii=False, indent=4)

    with open(error_file, "w", encoding="utf-8") as ef:
        json.dump(errors, ef, ensure_ascii=False, indent=4)

    print(f"✅ Saved {len(results)} results, {len(errors)} errors to disk.")


# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    cls = "multihoprag"
    query_mode = "experiment3"  # query mode) #0 : naive query #1 : decomposition #2 : keyword extraction #3 : decomposition(based atomic fact) #4 : origin ours method #5 : decomposition(seq/parall -> based atomic fact)
    mode = "ours_experiment1"   # retrieve mode) #0 : chunk, #1 : atomic, #2 : triple #3 : 1-hop
    top = 7
    start_idx = 0
    end_idx = 1000
    WORKING_DIR = f"../{cls}"

    rag = PathCoRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    query_param = QueryParam(
        mode=mode,
        query_mode=query_mode,
        top_mode=top,
        addon_params={"embedding_func": rag.embedding_func},
    )

    # dataset별 query 로드
    if cls == "novelqa":
        queries = extract_queries_novelqa(f"../datasets/{cls}.json")
    elif (cls == "infiniteqa") or (cls == "infinitechoice"):
        queries = extract_queries_infinite(f"../datasets/{cls}.jsonl")
    elif cls == "hotpot":
        queries = extract_queries_hotpot(f"../datasets/{cls}.jsonl")
    elif cls == "multihoprag":
        queries = extract_queries_multihoprag(f"../datasets/{cls}.jsonl")
    elif cls == "musique":
        queries = extract_queries_musique(f"../datasets/{cls}.jsonl")
    elif cls == "2wikimultihopqa":
        queries = extract_queries_2wikimultihopqa(f"../datasets/{cls}.jsonl")
    elif cls == "triviaqa":
        queries = extract_queries_triviaqa(f"../datasets/{cls}.jsonl")
    elif cls == "nq":
        queries = extract_queries_triviaqa(f"../datasets/{cls}.jsonl")
    else:
        queries = extract_queries(f"../datasets/questions/{cls}_questions.txt")

    # 병렬 실행 (batch_size 조정 가능)
    run_queries_and_save_to_json_parallel(
        start_idx, end_idx, queries, rag, query_param,
        f"/workspace/PathCoRAG/PathCoRAG/pred/atomic_20_top{top}_{mode}_{cls}_result.json", f"/workspace/PathCoRAG/PathCoRAG/pred/atomic_20_top{top}_{mode}_{cls}_errors.json",
        batch_size=10  # 동시에 10개 쿼리 실행
    )


# import re
# import json
# import asyncio
# from PathCoRAG import PathCoRAG, QueryParam
# from PathCoRAG.llm.openai import gpt_4o_mini_complete, openai_embed
# from PathCoRAG.llm.groqapi import groq_llama_4_scout_complete
# from PathCoRAG.llm.ollama import ollama_model_complete
# from PathCoRAG.llm.ollama import ollama_embed
# from PathCoRAG.llm.gemini import gemini_complete
# from PathCoRAG.utils import EmbeddingFunc
# from tqdm import tqdm


# # =========================================================
# # Query Extraction Functions
# # =========================================================
# def extract_queries(file_path):
#     with open(file_path, "r") as f:
#         data = f.read()
#     data = data.replace("**", "")
#     queries = re.findall(r" - Question \d+: (.+)", data)
#     return queries


# def extract_queries_novelqa(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return [v["Question"] for v in data.values() if v.get("Question")]


# def extract_queries_infinite(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("input"):
#                 queries.append(question)
#     return queries


# def extract_queries_hotpot(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question)
#     return queries


# def extract_queries_musique(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question.strip())
#     return queries


# def extract_queries_2wikimultihopqa(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question.strip())
#     return queries


# def extract_queries_triviaqa(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question.strip())
#     return queries


# def extract_queries_multihoprag(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("query"):
#                 queries.append(question)
#     return queries


# # =========================================================
# # Async sequential processing logic
# # =========================================================
# async def process_query(query_text, rag_instance, query_param):
#     """단일 쿼리 직렬 실행"""
#     try:
#         result = await rag_instance.aquery(query_text, param=query_param)
#         return {"query": query_text, "result": result}, None
#     except Exception as e:
#         return None, {"query": query_text, "error": str(e)}


# async def process_queries_sequentially(queries, rag_instance, query_param):
#     """❗ 완전 직렬 실행 (한 query씩 순서대로 처리)"""
#     results = []
#     errors = []

#     for q in tqdm(queries, desc="Processing queries", unit="query"):
#         result, error = await process_query(q, rag_instance, query_param)
#         if result:
#             results.append(result)
#         else:
#             errors.append(error)

#     return results, errors


# def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
#     try:
#         return asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop


# def run_queries_and_save_to_json_sequential(
#     start_idx, end_idx, queries, rag_instance, query_param, output_file, error_file
# ):
#     """직렬 처리 + 결과 저장"""
#     queries = queries[start_idx:end_idx]
#     loop = always_get_an_event_loop()

#     results, errors = loop.run_until_complete(
#         process_queries_sequentially(queries, rag_instance, query_param)
#     )

#     # 결과 저장
#     with open(output_file, "w", encoding="utf-8") as rf:
#         json.dump(results, rf, ensure_ascii=False, indent=4)

#     with open(error_file, "w", encoding="utf-8") as ef:
#         json.dump(errors, ef, ensure_ascii=False, indent=4)

#     print(f"✅ Saved {len(results)} results, {len(errors)} errors to disk.")


# # =========================================================
# # Main Execution
# # =========================================================
# if __name__ == "__main__":
#     cls = "multihoprag"
#     query_mode = "experiment3"
#     mode = "ours_experiment1"
#     start_idx = 0
#     end_idx = 1000
#     WORKING_DIR = f"../{cls}"

#     rag = PathCoRAG(
#         working_dir=WORKING_DIR,
#         embedding_func=openai_embed,
#         llm_model_func=gpt_4o_mini_complete,
#     )

#     query_param = QueryParam(
#         mode=mode,
#         query_mode=query_mode,
#         addon_params={"embedding_func": rag.embedding_func},
#     )

#     # dataset별 query 로드
#     if cls == "novelqa":
#         queries = extract_queries_novelqa(f"../datasets/{cls}.json")
#     elif cls in ("infiniteqa", "infinitechoice"):
#         queries = extract_queries_infinite(f"../datasets/{cls}.jsonl")
#     elif cls == "hotpot":
#         queries = extract_queries_hotpot(f"../datasets/{cls}.jsonl")
#     elif cls == "multihoprag":
#         queries = extract_queries_multihoprag(f"../datasets/{cls}.jsonl")
#     elif cls == "musique":
#         queries = extract_queries_musique(f"../datasets/{cls}.jsonl")
#     elif cls == "2wikimultihopqa":
#         queries = extract_queries_2wikimultihopqa(f"../datasets/{cls}.jsonl")
#     elif cls == "triviaqa":
#         queries = extract_queries_triviaqa(f"../datasets/{cls}.jsonl")
#     else:
#         queries = extract_queries(f"../datasets/questions/{cls}_questions.txt")

#     # 직렬 실행
#     run_queries_and_save_to_json_sequential(
#         start_idx, end_idx, queries, rag, query_param,
#         f"{mode}_{cls}_result.json", f"{mode}_{cls}_errors.json"
#     )
