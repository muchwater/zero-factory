# Zero Factory

제로웨이스트 가게 위치 정보 제공 서비스

## 프로젝트 구조

```
zero-factory/
├── server/          # Backend API (NestJS)
├── web/             # Frontend (Next.js)
├── docker-compose.yml
├── .env             # Environment variables
└── README.md
```

## 기술 스택

### Backend (API)
- NestJS
- Prisma ORM
- PostgreSQL with PostGIS

### Frontend (Web)
- Next.js 15
- React

### Infrastructure
- Docker & Docker Compose
- PostgreSQL 15 with PostGIS 3.4

## 시작하기

### 사전 요구사항
- Docker
- Docker Compose

### 설치 및 실행

1. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 값들을 설정하세요
```

2. Docker 컨테이너 빌드 및 실행
```bash
docker compose up -d --build
```

3. 서비스 접속
- Frontend: http://localhost:3001
- Backend API: http://localhost:3000
- Database: localhost:5432

### 개발 모드

각 서비스를 개별적으로 실행할 수도 있습니다:

#### Backend
```bash
cd server
npm install
npm run dev
```

#### Frontend
```bash
cd web
npm install
npm run dev
```

## Docker 명령어

### 컨테이너 시작
```bash
docker compose up -d
```

### 컨테이너 중지
```bash
docker compose down
```

### 컨테이너 재빌드
```bash
docker compose up -d --build
```

### 로그 확인
```bash
# 전체 로그
docker compose logs -f

# 특정 서비스 로그
docker compose logs -f web
docker compose logs -f api
docker compose logs -f db
```

### 컨테이너 상태 확인
```bash
docker compose ps
```

## 데이터베이스

PostgreSQL with PostGIS 확장을 사용합니다.

### 마이그레이션

```bash
cd server
npm run prisma:migrate
```

### Prisma Studio
```bash
cd server
npm run prisma:studio
```

## 환경 변수

`.env` 파일에서 다음 변수들을 설정할 수 있습니다:

- `POSTGRES_USER`: PostgreSQL 사용자명
- `POSTGRES_PASSWORD`: PostgreSQL 비밀번호
- `POSTGRES_DB`: 데이터베이스 이름
- `DATABASE_URL`: 데이터베이스 연결 URL

## 라이선스

MIT
