"""Explicit browser origins for a separately hosted static frontend."""
import os
from urllib.parse import urlsplit
from fastapi.middleware.cors import CORSMiddleware


def add_frontend_cors(application):
    origins = []
    for value in os.environ.get('PRISMRAW_FRONTEND_ORIGINS', '').split(','):
        value = value.strip().rstrip('/')
        if not value:
            continue
        url = urlsplit(value)
        if url.scheme not in {'http', 'https'} or not url.netloc or url.path or url.query or url.fragment or url.username or url.password:
            raise ValueError('PRISMRAW_FRONTEND_ORIGINS requires comma-separated HTTP(S) origins')
        origins.append(value)
    application.add_middleware(CORSMiddleware, allow_origins=origins,
                               allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
