import json
import os
import unittest
from unittest.mock import patch, MagicMock

# ----------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------
def make_client():
    """
    Import the Flask app after patching, so every patch below affects the
    global symbols inside app.py.  Returns an in‑process test client.
    """
    # Import *inside* the function so each test gets a fresh app instance.
    from app import app as flask_app
    return flask_app.test_client()

def base_payload():
    """Minimal JSON body the /miloh route expects."""
    return {
        "assignment":  "HW 1",
        "question":    "1",
        "location":    "OH Queue",
        "description": "I am stuck on part (b).",
        "chat":        ["Here is a follow‑up clarification."]
    }

# ----------------------------------------------------------------------
#  fixture‑level patches applied to every test
# ----------------------------------------------------------------------
class MilohTests(unittest.TestCase):
    def setUp(self):
        # Ensure the API key check passes
        os.environ["API_KEY"]           = "secret"
        os.environ["QA_TOP_K"]          = "1"
        os.environ["ASSIGNMENT_CATEGORIES"] = "Homeworks"
        # everything else can be empty
        os.environ["CONTENT_CATEGORIES"]    = ""
        os.environ["LOGISTICS_CATEGORIES"]  = ""
        os.environ["WORKSHEET_CATEGORIES"]  = ""

    # ------------------------------------------------------------------
    # 1) Unauthorized header should yield 401 (proves the guard works)
    # ------------------------------------------------------------------
    def test_auth_failure(self):
        client = make_client()
        resp   = client.post("/miloh")          # no header
        self.assertEqual(resp.status_code, 401)

    # ------------------------------------------------------------------
    # 2) Happy path: returns 200 and the mocked LLM answer
    # ------------------------------------------------------------------
    @patch("app.load_course_config", lambda *_: None)
    @patch("app.retrieve_qa",          lambda *a, **k: "None")
    @patch("app.retrieve_docs_hybrid", lambda *a, **k: "none")
    @patch("app.retrieve_docs_manual", lambda *a, **k: ("none","none","none"))
    @patch("app.generate",             lambda *a, **k: "LLM‑answer")
    @patch("app.log_blob",             lambda *a, **k: None)
    @patch("app.log_local",            lambda *a, **k: None)
    def test_success_200(self):
        client = make_client()
        resp = client.post(
            "/miloh",
            data    = json.dumps(base_payload()),
            headers = {"Authorization": "secret"},
            content_type = "application/json"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["Miloh"], "LLM‑answer")

    # ------------------------------------------------------------------
    # 3) Missing optional flags (prod/log_blob/log_local) must NOT crash
    # ------------------------------------------------------------------
    @patch("app.load_course_config", lambda *_: None)
    @patch("app.retrieve_qa", lambda *a, **k: "None")
    @patch("app.generate",    lambda *a, **k: "dummy")
    def test_no_optional_flags(self):
        payload = base_payload()
        # deliberately *do not* supply 'prod', 'log_blob', 'log_local'
        client  = make_client()
        resp = client.post(
            "/miloh",
            data=json.dumps(payload),
            headers={"Authorization":"secret"},
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)

    # ------------------------------------------------------------------
    # 4) Blank chat strings should be filtered out (otherwise model sees
    #    meaningless empty turns, and you risk an index error later)
    # ------------------------------------------------------------------
    @patch("app.prompts", MagicMock())  # prompt helpers not used here
    @patch("app.process_conversation_search", lambda *a, **k: "")
    @patch("app.load_course_config", lambda *_: None)
    @patch("app.generate", lambda *a, **k: "dummy")
    def test_blank_chat_skipped(self):
        seen_turns = {}
        def fake_ocr(thread_title, conversation_history):
            seen_turns["n"] = len(conversation_history)
            return conversation_history

        with patch("app.ocr_process_input", fake_ocr):
            payload = base_payload()
            payload["chat"] = [""]      # <-- blank entry
            client = make_client()
            client.post(
                "/miloh",
                data=json.dumps(payload),
                headers={"Authorization":"secret"},
                content_type="application/json"
            )
        # Only the initial ticket should remain
        self.assertEqual(seen_turns["n"], 1)

    # ------------------------------------------------------------------
    # 5) Verify KeyError on 'prod' is gone once you switch to .get()
    #    (Run once; if it fails with 500, change your code accordingly)
    # ------------------------------------------------------------------
    @patch("app.load_course_config", lambda *_: None)
    @patch("app.retrieve_qa", lambda *a, **k: "None")
    @patch("app.generate",    lambda *a, **k: "dummy")
    def test_key_error_prod(self):
        payload = base_payload()
        # no 'prod' field!  Should still return 200.
        client = make_client()
        resp = client.post(
            "/miloh",
            data=json.dumps(payload),
            headers={"Authorization":"secret"},
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)