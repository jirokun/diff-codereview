#!/usr/bin/env python3

import argparse
import os
import sys

from anthropic import Anthropic, HUMAN_PROMPT
from google import genai
from google.genai import types
from openai import OpenAI

prompt = """以下のGit diffの内容をコードレビューしてください。
変更内容について、以下の点に着目してコメントをお願いします。
問題となり得るものだけを挙げてください。

- コードの可読性、保守性の問題点
- バグやセキュリティの問題点
- パフォーマンスに問題がある点
- 設計上の問題点
- その他改善点"""

def get_api_key(env_var: str) -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"{env_var} environment variable is not set")
    return api_key

def deepseek_chat(diff: str) -> str:
    api_key = get_api_key("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": diff},
        ],
        stream=False,
    )

    return response.choices[0].message.content

def gemini_2_0_flash_exp(diff: str) -> str:
    api_key = get_api_key("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.3,
        ),
        contents=diff,
    )
    return response.text

def gpt_4o(diff: str) -> str:
    api_key = get_api_key("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "user", "content": diff},
        ],
        response_format={"type": "text"},
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content

def claude_sonnet(diff: str) -> str:
    api_key = get_api_key("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        system=prompt,
        messages=[{"role": "user", "content": diff}],
    )
    return response.content[0].text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--max-diff-size", type=int, default=10000, 
                       help="maximum allowed diff size in characters")
    args = parser.parse_args()

    model: str = args.model

    if model not in [
        "deepseek-chat",
        "gemini-2.0-flash-exp",
        "gpt-4o",
        "claude-sonnet",
    ]:
        raise ValueError("Invalid model")

    diff = sys.stdin.read()
    if len(diff) > args.max_diff_size:
        raise ValueError(f"Diff is too large (max {args.max_diff_size} characters)")

    response = ""
    if model == "deepseek-chat":
        response = deepseek_chat(diff)
    elif model == "gemini-2.0-flash-exp":
        response = gemini_2_0_flash_exp(diff)
    elif model == "gpt-4o":
        response = gpt_4o(diff)
    elif model == "claude-sonnet":
        response = claude_sonnet(diff)

    print(response)

if __name__ == "__main__":
    main()
