"""Public API rejects bad inputs and serves a usable local app without credentials."""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from smartphone_isp.app.server import app


class AppTests(unittest.TestCase):
    def test_home_and_health(self):
        with patch("smartphone_isp.app.server.warmup"), TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("PrismRAW", response.text)
            self.assertEqual(client.get("/api/health").json()["status"], "ok")

    def test_non_raw_upload_has_actionable_error(self):
        with patch("smartphone_isp.app.server.warmup"), TestClient(app) as client:
            response = client.post("/api/upload", files={"raw": ("photo.jpg", b"not RAW", "image/jpeg")})
            self.assertEqual(response.status_code, 400)
            self.assertIn("JPEG is not RAW", response.json()["detail"])

    def test_unknown_sample_is_not_a_file_path(self):
        with patch("smartphone_isp.app.server.warmup"), TestClient(app) as client:
            self.assertEqual(client.post("/api/sample/unknown").status_code, 404)
