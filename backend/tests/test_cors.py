"""Explicit cross-origin uploads and result access."""
import os
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from smartphone_isp.app.cors import add_frontend_cors


class CorsTests(unittest.TestCase):
    def test_frontend_upload_preflight_and_result_access(self):
        app = FastAPI()
        with patch.dict(os.environ, {'PRISMRAW_FRONTEND_ORIGINS': 'https://studio.example.test'}):
            add_frontend_cors(app)
        @app.get('/results/demo/report.json')
        def result():
            return {'ok': True}
        with TestClient(app) as client:
            headers = {'Origin': 'https://studio.example.test', 'Access-Control-Request-Method': 'POST',
                       'Access-Control-Request-Headers': 'Content-Type'}
            response = client.options('/api/upload', headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['access-control-allow-origin'], headers['Origin'])
            response = client.get('/results/demo/report.json', headers={'Origin': headers['Origin']})
            self.assertEqual(response.headers['access-control-allow-origin'], headers['Origin'])
            headers['Origin'] = 'https://other.example.test'
            response = client.options('/api/upload', headers=headers)
            self.assertEqual(response.status_code, 400)
            self.assertNotIn('access-control-allow-origin', response.headers)

    def test_invalid_origins(self):
        for value in ('*', 'https://studio.example.test/path'):
            with patch.dict(os.environ, {'PRISMRAW_FRONTEND_ORIGINS': value}), self.assertRaises(ValueError):
                add_frontend_cors(FastAPI())
