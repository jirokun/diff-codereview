# コードレビュー自動化ツール

このツールは、Git の diff 内容を解析し、コードレビューを自動で行います。

## 初期設定

```sh
uv sync
```

## 使い方

以下のコマンドを使用して、`main.py`を実行します。

```sh
git diff | uv run main.py --model <model_name>
```

`model_name`のデフォルト値は`deepseek-chat`です。

### 引数

- `--model`: 使用するモデルを指定します。以下のいずれかを指定できます。
  - `deepseek-chat`
  - `gemini-2.0-flash-exp`
  - `gpt-4o`

### 例

```sh
git diff | uv run main.py --model deepseek-chat
```

このコマンドは、diff の内容を読み込み、`deepseek-chat`モデルを使用してコードレビューを行います。

## 環境変数

以下の環境変数を設定する必要があります。

- `DEEPSEEK_API_KEY`: DeepSeek API のキー
- `GEMINI_API_KEY`: Gemini API のキー
- `OPENAI_API_KEY`: OpenAI API のキー
