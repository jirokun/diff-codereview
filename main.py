# Please install OpenAI SDK first: `pip3 install openai`

import os
import sys
from openai import OpenAI


api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 標準入力からdiffをすべて受け取る
diff = sys.stdin.read()

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": """以下のGit diffの内容をコードレビューしてください。
変更内容について、以下の点に着目してコメントをお願いします。

- コードの可読性、保守性
- バグやセキュリティ上のリスク
- パフォーマンスへの影響
- 設計上の問題点
- その他改善点"""},
        {"role": "user", "content": diff},
    ],
    stream=False
)

print(response.choices[0].message.content)