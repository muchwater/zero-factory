# 시작하기

Zero Factory 프로젝트를 시작하는 방법을 안내합니다.

## 사전 요구사항

- Docker
- Docker Compose

## 빠른 시작

### 1. 환경 변수 설정

프로젝트 루트에서 `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다:

```bash
cp .env.example .env
```

필요한 경우 `.env` 파일을 열어 값들을 수정하세요:

```bash
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=zerowaste_dev
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/zerowaste_dev"

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:3000

# Kakao Map API Key (선택사항)
NEXT_PUBLIC_KAKAO_MAP_KEY=
```

> Kakao Map API Key 설정 방법은 [API Keys 가이드](./api-keys.md)를 참조하세요.

### 2. Docker로 실행

```bash
docker compose up -d --build
```

### 3. 서비스 접속

빌드가 완료되면 다음 주소에서 서비스에 접속할 수 있습니다:

- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:3000
- **Database**: localhost:5432

## 다음 단계

- [로컬 개발 환경 설정](./development.md)
- [Docker 명령어 가이드](./docker.md)
- [Kakao Map API 설정](./api-keys.md)
- [트러블슈팅](./troubleshooting.md)
