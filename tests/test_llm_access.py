"""Exercise standalone access with a fake provider boundary."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tyremind.agents.llm import ROLES, request_text


class AccessTests(unittest.TestCase):
    def test_roles_and_model_override(self):
        choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=" result "))
        sdk = Mock()
        sdk.completion.return_value = SimpleNamespace(choices=[choice])
        config = {"TYREMIND_LLM_MODEL": "default", "TYREMIND_LLM_MODEL_ROOT_CAUSE_ANALYSIS": "override"}
        with patch.dict(sys.modules, litellm=sdk), patch.dict(os.environ, config, clear=True):
            for role, instruction in ROLES.items():
                self.assertEqual(request_text(role, "sample"), "result")
                args = sdk.completion.call_args.kwargs
                self.assertEqual(args["model"], "override" if role == "root_cause_analysis" else "default")
                self.assertEqual(args["messages"][0]["content"], instruction)
            choice.finish_reason = "length"
            with self.assertRaises(RuntimeError):
                request_text("health_assessment", "sample")

    def test_invalid_requests_never_call_sdk(self):
        sdk = Mock()
        with patch.dict(sys.modules, litellm=sdk), patch.dict(os.environ, {}, clear=True):
            for role, prompt in [("unknown", "sample"), ("health_assessment", ""), ("maintenance_planning", "sample")]:
                with self.assertRaises(ValueError):
                    request_text(role, prompt)
        sdk.completion.assert_not_called()
