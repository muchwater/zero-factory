# 시작하기

Zero Factory 프로젝트를 시작하는 방법을 안내합니다.

## 사전 요구사항

- Docker
- Docker Compose

## 빠른 시작

Zero Factory는 개발 환경과 배포 환경을 자동으로 구분하여 실행할 수 있습니다.

### 방법 1: 환경별 실행 스크립트 사용 (추천)

#### 개발 환경 (Development)

```bash
# 개발 환경으로 실행
./start-dev.sh

# 중지
./start-dev.sh down
```

개발 환경에서는:
- HTTP만 사용 (SSL 없음)
- 모든 포트가 직접 노출됨 (3000, 3001, 5432)
- Hot reload 활성화
- 개발용 nginx 설정 사용

**접속 URL:**
- Frontend: http://localhost 또는 http://localhost:3001
- Backend API: http://localhost:3000
- Database: localhost:5432

#### 배포 환경 (Production)

```bash
# 배포 환경으로 실행
./start-prod.sh

# 중지
./start-prod.sh down
```

배포 환경에서는:
- HTTPS 사용 (Let's Encrypt SSL)
- Nginx를 통한 프록시만 노출
- Production 최적화
- 자동 SSL 인증서 갱신

**접속 URL:**
- Frontend: https://zeromap.store
- Backend API: https://zeromap.store/api

### 방법 2: Docker Compose 직접 사용

```bash
# 개발 환경
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 배포 환경
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 환경 변수 설정

환경별로 다른 설정 파일을 사용합니다:

- `.env.dev` - 개발 환경 설정 (git에 포함)
- `.env.prod` - 배포 환경 설정 (git에 포함)
- `.env` - 자동 생성 파일 (gitignore, 직접 수정 금지)

환경 변수를 수정하려면 `.env.dev` 또는 `.env.prod` 파일을 수정하세요.

> Kakao Map API Key 설정 방법은 [API Keys 가이드](./api-keys.md)를 참조하세요.

## 상세 가이드

환경 설정에 대한 더 자세한 내용은 [환경 설정 가이드](./ENVIRONMENT_SETUP.md)를 참조하세요.

## 다음 단계

- [환경 설정 가이드](./ENVIRONMENT_SETUP.md) - 환경별 상세 설정
- [로컬 개발 환경 설정](./development.md)
- [Docker 명령어 가이드](./docker.md)
- [Kakao Map API 설정](./api-keys.md)
- [트러블슈팅](./troubleshooting.md)
