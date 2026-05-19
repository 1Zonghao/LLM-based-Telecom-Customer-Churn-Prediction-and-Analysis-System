FROM python:3.11-slim

LABEL description="无头 Jupyter 环境 — 有状态数据科学代码执行沙箱"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m ipykernel install --name python3 --display-name "Python 3"

WORKDIR /workspace
RUN mkdir -p /workspace/data

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

CMD ["python", "-m", "ipykernel_launcher", "-f", "/tmp/kernel-connection.json"]
