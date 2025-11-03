# Docker 가이드

Docker Compose를 사용한 Zero Factory 프로젝트 관리 방법을 안내합니다.

## 기본 명령어

### 컨테이너 시작

```bash
docker compose up -d
```

백그라운드에서 모든 서비스를 시작합니다.

### 컨테이너 중지

```bash
docker compose down
```

모든 서비스를 중지하고 컨테이너를 제거합니다.

> 주의: 데이터베이스 볼륨은 삭제되지 않습니다.

### 컨테이너 재빌드

코드나 Dockerfile을 수정한 경우:

```bash
docker compose up -d --build
```

또는 특정 서비스만 재빌드:

```bash
docker compose up -d --build web
docker compose up -d --build api
```

### 컨테이너 재시작

```bash
docker compose restart
```

특정 서비스만 재시작:

```bash
docker compose restart web
docker compose restart api
docker compose restart db
```

## 로그 확인

### 전체 로그 확인

```bash
# 실시간 로그 확인 (tail -f 와 유사)
docker compose logs -f

# 최근 로그만 확인
docker compose logs --tail=100
```

### 특정 서비스 로그 확인

```bash
# Web 서비스 로그
docker compose logs -f web

# API 서비스 로그
docker compose logs -f api

# Database 로그
docker compose logs -f db
```

## 컨테이너 상태 확인

### 실행 중인 컨테이너 확인

```bash
docker compose ps
```

출력 예시:
```
NAME                 IMAGE                    COMMAND                  SERVICE   CREATED          STATUS          PORTS
postgresdb           postgis/postgis:15-3.4   "docker-entrypoint.s…"   db        2 minutes ago    Up 2 minutes    0.0.0.0:5432->5432/tcp
zero-factory-api-1   zero-factory-api         "docker-entrypoint.s…"   api       2 minutes ago    Up 2 minutes    0.0.0.0:3000->3000/tcp
zero-factory-web-1   zero-factory-web         "docker-entrypoint.s…"   web       2 minutes ago    Up 2 minutes    0.0.0.0:3001->3001/tcp
```

### 리소스 사용량 확인

```bash
docker stats
```

## 데이터베이스 관리

### 데이터베이스 접속

```bash
docker compose exec db psql -U postgres -d zerowaste_dev
```

### 데이터베이스 백업

```bash
docker compose exec db pg_dump -U postgres zerowaste_dev > backup.sql
```

### 데이터베이스 복원

```bash
docker compose exec -T db psql -U postgres zerowaste_dev < backup.sql
```

### 데이터베이스 볼륨 삭제 (주의!)

모든 데이터가 삭제됩니다:

```bash
docker compose down -v
```

## 컨테이너 내부 접속

### API 컨테이너 접속

```bash
docker compose exec api sh
```

### Web 컨테이너 접속

```bash
docker compose exec web sh
```

### DB 컨테이너 접속

```bash
docker compose exec db bash
```

## 환경 변수 변경

### 중요: Next.js 환경 변수 변경 시

Next.js는 빌드 타임에 환경 변수를 코드에 포함시키므로, 환경 변수 변경 후 **반드시 재빌드**가 필요합니다:

```bash
# 1. .env 파일 수정
vim .env

# 2. 컨테이너 중지
docker compose down

# 3. 재빌드 및 시작
docker compose up -d --build
```

### Backend 환경 변수는 재시작만으로 충분

```bash
# 1. .env 파일 수정
vim .env

# 2. API 서비스만 재시작
docker compose restart api
```

## 디스크 공간 정리

### 사용하지 않는 이미지 삭제

```bash
docker image prune -a
```

### 모든 미사용 리소스 삭제

```bash
docker system prune -a
```

> 주의: 다른 프로젝트의 컨테이너와 이미지도 삭제될 수 있습니다.

## 프로덕션 배포

프로젝트는 GitHub Actions를 통해 자동으로 EC2에 배포됩니다.

### 자동 배포 워크플로우

`main` 브랜치에 push하면 자동으로:

1. EC2 서버에 SSH 접속
2. 최신 코드 pull
3. Docker 이미지 빌드
4. 서비스 재시작
5. 헬스 체크 수행

워크플로우 파일: [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)

## 트러블슈팅

Docker 관련 문제는 [트러블슈팅 가이드](./troubleshooting.md)를 참조하세요.

## 다음 단계

- [로컬 개발 환경 설정](./development.md)
- [트러블슈팅](./troubleshooting.md)
