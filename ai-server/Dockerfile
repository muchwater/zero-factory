# AI Model Server Dockerfile
# GPU 지원 + Python 3.10 + PyTorch

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 심볼릭 링크 설정
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /app

# pip 업그레이드
RUN python -m pip install --upgrade pip

# Python 패키지 설치 (레이어 분리 및 타임아웃 설정)
COPY requirements.txt .

# 무거운 패키지들 먼저 설치 (타임아웃 증가)
RUN pip install --default-timeout=1000 --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2

# 나머지 패키지 설치
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 모델 가중치 디렉토리 생성
RUN mkdir -p /app/models/weights

# 업로드 디렉토리 생성
RUN mkdir -p /app/uploads/reusables

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 개발 모드 실행 (자동 재로드)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# 프로덕션 모드는 docker-compose.yml에서 override
