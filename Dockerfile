# Container for the reactivity-screen web app (works on Hugging Face Spaces, a lab server, or a laptop with Docker)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*
# xtb 6.7.1 (GFN2-xTB) and the g-xTB-enabled build, both from the Grimme group's GitHub
RUN mkdir -p /opt/xtb /opt/gxtb \
 && curl -fsSL https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz \
    | tar xJ -C /opt/xtb --strip-components=1 \
 && (curl -fsSL https://github.com/grimme-lab/g-xtb/raw/main/binaries/xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz \
    | tar xJ -C /opt/gxtb --strip-components=1 || echo "g-xTB download failed: app works without it")
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    XTB=/opt/xtb/bin/xtb GXTB=/opt/gxtb/bin/xtb \
    MAX_JOBS=1 JOB_CPUS=2 MAX_HEAVY=60 JOB_TIMEOUT=900
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
