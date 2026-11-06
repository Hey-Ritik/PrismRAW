FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src TORCH_HOME=/app/.cache/torch PYTHONUNBUFFERED=1
ENV PORT=7860 PRISMRAW_HOSTED=1
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[raw,eval,app]'
COPY app ./app
COPY scripts/serve.py ./scripts/serve.py
COPY reports ./reports
COPY checkpoints/final-hybrid.pt checkpoints/final-learned.pt checkpoints/final-hybrid-fp16.pt ./checkpoints/
COPY data/processed/demo ./data/processed/demo
RUN python -c "from smartphone_isp.evaluation.metrics import perceptual_model; perceptual_model()"
RUN useradd -m prism && mkdir -p /app/outputs && chown -R prism:prism /app/outputs /app/.cache
USER prism
EXPOSE 7860
HEALTHCHECK --interval=30s --start-period=45s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/api/health', timeout=5)" || exit 1
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0"]
