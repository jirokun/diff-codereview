import pytest
from unittest.mock import patch, MagicMock
import os
import sys
from io import StringIO
from main import (
    get_api_key,
    deepseek_chat,
    gemini_2_0_flash_exp,
    gpt_4o,
    claude_sonnet,
    main,
)

# Test get_api_key
def test_get_api_key_success(monkeypatch):
    """環境変数からAPIキーを正常に取得できることをテスト"""
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    assert get_api_key("TEST_API_KEY") == "test-key"

def test_get_api_key_failure():
    """環境変数が設定されていない場合に適切なエラーが発生することをテスト"""
    with pytest.raises(ValueError):
        get_api_key("NON_EXISTENT_KEY")

# Test deepseek_chat
@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY environment variable not set"
)
@patch("main.OpenAI")
def test_deepseek_chat(mock_openai):
    """Deepseek Chat APIのレスポンスが正しく返されることをテスト"""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "test response"
    mock_client.chat.completions.create.return_value = mock_response
    
    result = deepseek_chat("test diff")
    assert result == "test response"
    mock_client.chat.completions.create.assert_called_once()

# Test gemini_2_0_flash_exp
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set"
)
@patch("main.genai.Client")
def test_gemini_2_0_flash_exp(mock_genai):
    """Gemini 2.0 Flash APIのレスポンスが正しく返されることをテスト"""
    mock_client = MagicMock()
    mock_genai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = "test response"
    mock_client.models.generate_content.return_value = mock_response
    
    result = gemini_2_0_flash_exp("test diff")
    assert result == "test response"
    mock_client.models.generate_content.assert_called_once()

# Test gpt_4o
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set"
)
@patch("main.OpenAI")
def test_gpt_4o(mock_openai):
    """GPT-4o APIのレスポンスが正しく返されることをテスト"""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "test response"
    mock_client.chat.completions.create.return_value = mock_response
    
    result = gpt_4o("test diff")
    assert result == "test response"
    mock_client.chat.completions.create.assert_called_once()

# Test claude_sonnet
@patch("main.Anthropic")
@patch("main.get_api_key", return_value="test-key")
def test_claude_sonnet(mock_get_api_key, mock_anthropic):
    """Claude Sonnet APIのレスポンスが正しく返されることをテスト"""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="test response")]
    mock_client.messages.create.return_value = mock_response
    
    result = claude_sonnet("test diff")
    assert result == "test response"
    mock_client.messages.create.assert_called_once()
    mock_get_api_key.assert_called_once_with("ANTHROPIC_API_KEY")

# Test main function
def test_main_valid_model(capsys):
    """有効なモデル指定時に正しく動作することをテスト"""
    test_args = ["--model", "deepseek-chat"]
    test_input = "test diff"
    
    with (
        patch.object(sys, "argv", ["main.py"] + test_args),
        patch("main.deepseek_chat", return_value="test response"),
        patch("sys.stdin", StringIO(test_input)),
    ):
        main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "test response"

def test_main_invalid_model():
    """無効なモデル指定時に適切なエラーが発生することをテスト"""
    test_args = ["--model", "invalid-model"]
    
    with patch.object(sys, "argv", ["main.py"] + test_args):
        with pytest.raises(ValueError):
            main()

def test_main_large_diff():
    """差分サイズが制限を超えた場合に適切なエラーが発生することをテスト"""
    test_args = ["--model", "deepseek-chat"]
    large_diff = "a" * 10001
    
    with (
        patch.object(sys, "argv", ["main.py"] + test_args),
        patch("sys.stdin", StringIO(large_diff)),
        pytest.raises(ValueError),
    ):
        main()
