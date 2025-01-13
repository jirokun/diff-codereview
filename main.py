import argparse
import os
import sys

from google import genai
from google.genai import types
from openai import OpenAI

prompt = """以下のGit diffの内容をコードレビューしてください。
変更内容について、以下の点に着目してコメントをお願いします。

- コードの可読性、保守性
- バグやセキュリティ上のリスク
- パフォーマンスへの影響
- 設計上の問題点
- その他改善点"""

MAX_DIFF_SIZE = 100000

def deepseek_chat(diff: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": diff},
        ],
        stream=False
    )

    print(response.choices[0].message.content)

def gemini_2_0_flash_exp(diff: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp', 
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature= 0.3,
        ),
        contents=diff,
    )
    print(response.text)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-chat")
args = parser.parse_args()

model: str = args.model

if model not in ["deepseek-chat", "gemini-2.0-flash-exp"]:
    raise ValueError("Invalid model")

diff = sys.stdin.read()
if len(diff) > MAX_DIFF_SIZE:
    raise ValueError("Diff is too large")

if model == "deepseek-chat":
    deepseek_chat(diff)
elif model == "gemini-2.0-flash-exp":
    gemini_2_0_flash_exp(diff)