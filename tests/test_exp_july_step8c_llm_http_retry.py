import json
import unittest
from unittest.mock import patch

from src.exp_july.perception.trajectory_pattern_closed_loop import http_llm


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()


class Step8CLLMHTTPRetryTests(unittest.TestCase):
    def test_timeout_is_retried_and_then_succeeds(self):
        env = {
            "OPENAI_API_KEY": "test-key",
            "CAUVID_STEP8C_LLM_TIMEOUT_SECONDS": "1",
            "CAUVID_STEP8C_LLM_MAX_ATTEMPTS": "3",
            "CAUVID_STEP8C_LLM_RETRY_BACKOFF_SECONDS": "0",
        }
        with patch.dict("os.environ", env, clear=False), patch(
            "src.exp_july.perception.trajectory_pattern_closed_loop.urllib.request.urlopen",
            side_effect=[TimeoutError("read timed out"), _Response()],
        ) as request:
            self.assertEqual(http_llm("test"), {})
        self.assertEqual(request.call_count, 2)

    def test_repeated_timeout_still_stops_pipeline(self):
        env = {
            "OPENAI_API_KEY": "test-key",
            "CAUVID_STEP8C_LLM_TIMEOUT_SECONDS": "1",
            "CAUVID_STEP8C_LLM_MAX_ATTEMPTS": "2",
            "CAUVID_STEP8C_LLM_RETRY_BACKOFF_SECONDS": "0",
        }
        with patch.dict("os.environ", env, clear=False), patch(
            "src.exp_july.perception.trajectory_pattern_closed_loop.urllib.request.urlopen",
            side_effect=TimeoutError("read timed out"),
        ) as request:
            with self.assertRaisesRegex(RuntimeError, "after 2 attempt"):
                http_llm("test")
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
