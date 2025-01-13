from google import genai
from google.genai import types
import os
import sys
from openai import OpenAI
import argparse

prompt = """以下のGit diffの内容をコードレビューしてください。
変更内容について、以下の点に着目してコメントをお願いします。

- コードの可読性、保守性
- バグやセキュリティ上のリスク
- パフォーマンスへの影響
- 設計上の問題点
- その他改善点"""

def deepseek_chat(diff: str) -> str:
    # 環境変数からAPIキーを取得してクライアントを初期化
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
    # 環境変数からAPIキーを取得してクライアントを初期化

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
# コマンドライン引数を解析してモデルを選択
# 選択できるモデルは以下の通り
# - deepseek-chat
# - chatgpt-4o
# - gemini-2.0-flash-exp

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-chat")
args = parser.parse_args()

model: str = args.model

if model not in ["deepseek-chat", "chatgpt-4o", "gemini-2.0-flash-exp"]:
    raise ValueError("Invalid model")

diff = sys.stdin.read()
if len(diff) > 100000:
    raise ValueError("Diff is too large")

if model == "deepseek-chat":
    deepseek_chat(diff)
elif model == "gemini-2.0-flash-exp":
    gemini_2_0_flash_exp(diff)