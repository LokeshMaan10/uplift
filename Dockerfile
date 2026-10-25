# Hugging Face Spaces (Docker SDK) — alternative free host for the dashboard.
# Streamlit Community Cloud does NOT need this file.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir --user -r app/requirements.txt

COPY --chown=user src src
COPY --chown=user app app
COPY --chown=user .streamlit .streamlit

EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
