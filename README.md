# Zero Factory

제로웨이스트 가게 위치 정보 제공 서비스

## 프로젝트 개요

Zero Factory는 제로웨이스트 라이프스타일을 실천하는 사람들을 위한 친환경 가게 및 시설 위치 정보 서비스입니다. 카카오맵을 기반으로 다회용컵 사용 가능한 카페, 반납함, 텀블러 포인트 적립 장소 등을 쉽게 찾을 수 있습니다.

## 주요 기능

- 🗺️ **지도 기반 검색**: 카카오맵을 활용한 제로웨이스트 장소 검색
- ♻️ **카테고리별 필터링**: 다회용컵 카페, 반납함, 텀블러 포인트 적립 등
- 📍 **근처 장소 추천**: 사용자 위치 기반 제로웨이스트 장소 추천
- 🔍 **검색 기능**: 상점명 또는 지역명으로 검색

## 빠른 시작

### 사전 요구사항

- Docker
- Docker Compose

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/muchwater/zero-factory.git
cd zero-factory

# 2. 환경 변수 설정
cp .env.example .env

# 3. Docker로 실행
docker compose up -d --build

# 4. 서비스 접속
# Frontend: http://localhost:3001
# Frontend-admin: http://localhost:3001/admin
# Backend API: http://localhost:3000
```

**더 자세한 설명은 [시작하기 가이드](./docs/getting-started.md)를 참조하세요.**

## 프로젝트 구조

```
zero-factory/
├── server/          # Backend API (NestJS)
├── web/             # Frontend (Next.js)
├── docs/            # 프로젝트 문서
├── docker-compose.yml
├── .env             # 환경 변수
└── README.md
```

## 기술 스택

### Backend (API)

- NestJS - Node.js 프레임워크
- Prisma ORM - 데이터베이스 ORM
- PostgreSQL with PostGIS - 공간 데이터베이스
- TypeScript - 타입 안전성

### Frontend (Web)

- Next.js 15 - React 프레임워크
- React 18 - UI 라이브러리
- Tailwind CSS - 스타일링
- TypeScript - 타입 안전성

### Infrastructure

- Docker & Docker Compose - 컨테이너화
- PostgreSQL 15 with PostGIS 3.4 - 데이터베이스
- GitHub Actions - CI/CD

## 문서

### 📚 시작하기

- **[시작하기 가이드](./docs/getting-started.md)** - 프로젝트 설치 및 실행 방법
- **[API Keys 설정](./docs/api-keys.md)** - Kakao Map API 키 발급 및 설정

### 💻 개발

- **[로컬 개발 환경](./docs/development.md)** - 로컬에서 개발하는 방법
- **[Docker 가이드](./docs/docker.md)** - Docker 명령어 및 사용법

### 🔧 운영 및 문제 해결

- **[배포 설정 가이드](./docs/deployment.md)** - GitHub Actions 자동 배포 설정
- **[HTTPS 설정 가이드](./docs/HTTPS_SETUP.md)** - Let's Encrypt SSL 인증서 설정
- **[트러블슈팅](./docs/troubleshooting.md)** - 자주 발생하는 문제 해결 방법

## 환경 변수

프로젝트는 다음 환경 변수를 사용합니다:

```bash
# Database (Server)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=zerowaste_dev
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/zerowaste_dev"

# Frontend (Web)
NEXT_PUBLIC_API_URL=http://localhost:3000
NEXT_PUBLIC_KAKAO_MAP_KEY=your_kakao_map_key_here
```

자세한 설정 방법은 [API Keys 가이드](./docs/api-keys.md)를 참조하세요.

## 배포

프로젝트는 GitHub Actions를 통해 자동으로 EC2에 배포됩니다.

`main` 브랜치에 push하면:

1. 자동으로 EC2 서버에 배포
2. Docker 이미지 빌드
3. 서비스 재시작
4. 헬스 체크 수행

자세한 내용은 [Docker 가이드 - 프로덕션 배포](./docs/docker.md#프로덕션-배포)를 참조하세요.

## 라이선스

MIT License

## 문의

프로젝트에 대한 문의사항이나 버그 리포트는 [GitHub Issues](https://github.com/muchwater/zero-factory/issues)에 등록해 주세요.
