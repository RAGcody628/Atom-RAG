import json
import os
from openai import OpenAI
from transformers import GPT2Tokenizer


def openai_complete_if_cache(
    model="gpt-4o-mini", prompt=None, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_client = OpenAI()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = openai_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return response.choices[0].message.content


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def get_summary(context, tot_tokens=2000):
    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000 : 1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens) : 1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary


clses = ["test"]
for cls in clses:
    with open(f"../datasets/unique_contexts/{cls}_unique_contexts.json", mode="r") as f:
        unique_contexts = json.load(f)

    summaries = [get_summary(context) for context in unique_contexts]

    total_description = "\n\n".join(summaries)

    prompt = f"""
    Given the following description of a dataset:

    {total_description}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """

    result = openai_complete_if_cache(model="gpt-4o-mini", prompt=prompt)

    file_path = f"../datasets/questions/{cls}_questions.txt"
    
    if not os.path.exists("../datasets/questions"):
        os.makedirs("../datasets/questions")
    
    with open(file_path, "w") as file:
        file.write(result)

    print(f"{cls}_questions written to {file_path}")

# import json
# from transformers import AutoTokenizer
# from transformers import GPT2Tokenizer
# import google.generativeai as genai
# import os

# # Gemini 클라이언트 구성
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# client = genai.GenerativeModel("gemini-2.0-flash")


# def gemini_complete(prompt: str, system_prompt: str = None, history_messages: list[dict] = None) -> str:
#     if history_messages is None:
#         history_messages = []

#     messages = []
#     if system_prompt:
#         messages.append(f"System Instruction: {system_prompt}")
#     for msg in history_messages:
#         role = msg.get("role", "user")
#         content = msg.get("content", "")
#         if role == "user":
#             messages.append(f"User: {content}")
#         elif role == "assistant":
#             messages.append(f"Assistant: {content}")
#         else:
#             messages.append(f"{role.capitalize()}: {content}")

#     messages.append(f"User: {prompt}")
#     full_prompt = "\n\n".join(messages)

#     try:
#         response = client.generate_content(full_prompt)
#         return response.candidates[0].content.parts[0].text
#     except Exception as e:
#         print(f"[ERROR] Gemini API call failed: {e}")
#         return ""


# # 기존 요약 로직 유지
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# def get_summary(context, tot_tokens=2000):
#     tokens = tokenizer.tokenize(context)
#     half_tokens = tot_tokens // 2

#     start_tokens = tokens[1000 : 1000 + half_tokens]
#     end_tokens = tokens[-(1000 + half_tokens) : 1000]

#     summary_tokens = start_tokens + end_tokens
#     summary = tokenizer.convert_tokens_to_string(summary_tokens)

#     return summary


# # 실제 실행
# clses = ["test"]
# for cls in clses:
#     with open(f"../datasets/unique_contexts/{cls}_unique_contexts.json", mode="r") as f:
#         unique_contexts = json.load(f)

#     summaries = [get_summary(context) for context in unique_contexts]
#     total_description = "\n\n".join(summaries)

#     prompt = f"""
#     Given the following description of a dataset:

#     {total_description}

#     Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

#     Output the results in the following structure:
#     - User 1: [user description]
#         - Task 1: [task description]
#             - Question 1:
#             - Question 2:
#             - Question 3:
#             - Question 4:
#             - Question 5:
#         - Task 2: [task description]
#             ...
#         - Task 5: [task description]
#     - User 2: [user description]
#         ...
#     - User 5: [user description]
#         ...
#     """

#     result = gemini_complete(prompt=prompt)

#     file_path = f"../datasets/questions/{cls}_questions.txt"
    
#     if not os.path.exists("../datasets/questions"):
#         os.makedirs("../datasets/questions")

#     with open(file_path, "w") as file:
#         file.write(result)

#     print(f"{cls}_questions written to {file_path}")

# import json
# import os
# from google.generativeai import GenerativeModel
# from transformers import GPT2Tokenizer

# # Gemini API 호출 함수
# def gemini_complete_if_cache(
#     model="gemini-2.0-flash", prompt=None, system_prompt=None, history_messages=[], **kwargs
# ) -> str:
#     gemini_client = GenerativeModel(model)

#     # Gemini API의 메시지 포맷 (OpenAI 'messages' → Gemini 'contents')
#     contents = []

#     if system_prompt:
#         contents.append({
#             "role": "user",
#             "parts": [{"text": f"System Instruction: {system_prompt}"}]
#         })

#     for msg in history_messages:
#         role = msg.get("role", "user")
#         content = msg.get("content", "")
#         if role == "user":
#             contents.append({"role": "user", "parts": [{"text": content}]})
#         elif role == "assistant":
#             contents.append({"role": "model", "parts": [{"text": content}]})
#         else:
#             contents.append({"role": role, "parts": [{"text": content}]})

#     contents.append({"role": "user", "parts": [{"text": prompt}]})

#     try:
#         response = gemini_client.generate_content(contents=contents, **kwargs)
#         return response.text
#     except Exception as e:
#         print(f"[ERROR] Gemini API call failed: {e}")
#         return ""

# # 토크나이저
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# def get_summary(context, tot_tokens=2000):
#     tokens = tokenizer.tokenize(context)
#     half_tokens = tot_tokens // 2

#     start_tokens = tokens[1000:1000 + half_tokens]
#     end_tokens = tokens[-(1000 + half_tokens):1000]

#     summary_tokens = start_tokens + end_tokens
#     summary = tokenizer.convert_tokens_to_string(summary_tokens)

#     return summary

# # 전체 실행 코드
# clses = ["mix"]
# for cls in clses:
#     with open(f"../datasets/unique_contexts/{cls}_unique_contexts.json", mode="r") as f:
#         unique_contexts = json.load(f)

#     summaries = [get_summary(context) for context in unique_contexts]

#     total_description = "\n\n".join(summaries)

#     prompt = f"""
#     Given the following description of a dataset:

#     {total_description}

#     Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

#     Output the results in the following structure:
#     - User 1: [user description]
#         - Task 1: [task description]
#             - Question 1:
#             - Question 2:
#             - Question 3:
#             - Question 4:
#             - Question 5:
#         - Task 2: [task description]
#             ...
#         - Task 5: [task description]
#     - User 2: [user description]
#         ...
#     - User 5: [user description]
#         ...
#     """

#     result = gemini_complete_if_cache(model="gemini-2.0-flash", prompt=prompt)

#     file_path = f"../datasets/questions/{cls}_questions.txt"

#     if not os.path.exists("../datasets/questions"):
#         os.makedirs("../datasets/questions")

#     with open(file_path, "w") as file:
#         file.write(result)

#     print(f"{cls}_questions written to {file_path}")
