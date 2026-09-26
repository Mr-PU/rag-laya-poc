FROM python:3.11-slim

WORKDIR /app

# torch CPU wheel keeps the image reasonably small; laya[serve] adds fastapi+uvicorn
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir "laya[serve]"

# Preload the English checkpoint at container start so the first request isn't
# stuck downloading + building it (that's a one-time hit on first `docker compose up`
# while the model is pulled from Hugging Face and cached in the laya-cache volume).
ENV LAYA_HOST=0.0.0.0
ENV LAYA_PORT=8000
ENV LAYA_DEVICE=cpu
ENV LAYA_PRELOAD=1
ENV LAYA_MODELS=english
ENV LAYA_THREADS=4

EXPOSE 8000

CMD ["laya-serve"]
