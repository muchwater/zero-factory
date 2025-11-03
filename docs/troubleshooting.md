# 트러블슈팅

Zero Factory 프로젝트에서 자주 발생하는 문제들과 해결 방법을 안내합니다.

## Kakao Map API 관련

### 문제: 지도가 로드되지 않음

**증상**:
- 브라우저에서 지도가 표시되지 않음
- 콘솔에 Kakao API 관련 에러 발생

**해결 방법**:

#### 1. 플랫폼 등록 확인

현재 서버의 IP 또는 도메인이 Kakao Developers 콘솔에 등록되어 있는지 확인:

```bash
# 서버 IP 확인
curl ifconfig.me
```

[Kakao Developers 콘솔](https://developers.kakao.com/)에서:
- 애플리케이션 선택
- 플랫폼 > Web 플랫폼 등록
- 사이트 도메인 등록: `http://YOUR_IP:3001` 또는 `http://localhost:3001`

#### 2. 환경 변수 확인

```bash
# .env 파일에 API 키가 있는지 확인
cat .env | grep KAKAO

# Docker 컨테이너 로그에서 환경 변수 확인
docker compose logs web 2>&1 | head -20
```

#### 3. 빌드된 파일에 API 키 포함 확인

```bash
# Next.js 빌드 결과에 API 키가 포함되었는지 확인
curl -s http://localhost:3001 | grep -o "appkey=[^&\"]*"
```

API 키가 보이지 않으면 Docker 재빌드 필요:

```bash
docker compose down
docker compose up -d --build
```

#### 4. 브라우저 콘솔 확인

F12 키를 눌러 개발자 도구를 열고 Console 탭에서 에러 메시지 확인:

- `Failed to load resource`: 네트워크 문제 또는 플랫폼 미등록
- `Kakao Map API error`: API 키 오류 또는 플랫폼 미등록

자세한 설정 방법은 [API Keys 가이드](./api-keys.md)를 참조하세요.

## Docker 관련

### 문제: 컨테이너가 시작되지 않음

**증상**:
- `docker compose up` 실행 시 컨테이너가 계속 재시작됨
- 서비스에 접속할 수 없음

**해결 방법**:

#### 1. 로그 확인

```bash
# 전체 로그 확인
docker compose logs

# 특정 서비스 로그 확인
docker compose logs api
docker compose logs web
docker compose logs db
```

#### 2. 포트 충돌 확인

다른 프로세스가 같은 포트를 사용하고 있는지 확인:

```bash
# 포트 사용 확인
sudo lsof -i :3000  # API
sudo lsof -i :3001  # Web
sudo lsof -i :5432  # DB
```

포트를 사용 중인 프로세스가 있다면 종료하거나, `docker-compose.yml`에서 포트를 변경:

```yaml
services:
  web:
    ports:
      - "3002:3001"  # 3001 대신 3002 사용
```

#### 3. 볼륨 권한 문제

```bash
# 볼륨 삭제 후 재시작
docker compose down -v
docker compose up -d
```

### 문제: 이미지 빌드 실패

**증상**:
- `docker compose build` 실행 시 에러 발생

**해결 방법**:

#### 1. Docker 캐시 삭제

```bash
# 캐시 없이 빌드
docker compose build --no-cache

# 또는 전체 재빌드
docker compose down
docker system prune -a
docker compose up -d --build
```

#### 2. 디스크 공간 확인

```bash
# 디스크 사용량 확인
df -h

# Docker 디스크 사용량 확인
docker system df
```

디스크 공간이 부족하면:

```bash
# 사용하지 않는 이미지 삭제
docker image prune -a

# 모든 미사용 리소스 삭제
docker system prune -a
```

## 데이터베이스 관련

### 문제: 데이터베이스 연결 실패

**증상**:
- API 서버가 데이터베이스에 연결할 수 없음
- `Connection refused` 에러

**해결 방법**:

#### 1. 데이터베이스 컨테이너 상태 확인

```bash
# 컨테이너 상태 확인
docker compose ps

# DB 로그 확인
docker compose logs db
```

#### 2. DATABASE_URL 확인

`.env` 파일의 `DATABASE_URL`이 올바른지 확인:

```bash
# Docker 환경에서는 호스트를 'db'로 사용
DATABASE_URL="postgresql://postgres:postgres@db:5432/zerowaste_dev"

# 로컬 개발 환경에서는 'localhost' 사용
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/zerowaste_dev"
```

#### 3. 서비스 재시작

```bash
docker compose restart db
docker compose restart api
```

### 문제: 마이그레이션 실패

**증상**:
- `prisma migrate` 실행 시 에러 발생

**해결 방법**:

#### 1. 데이터베이스 연결 확인

```bash
# 데이터베이스 접속 테스트
docker compose exec db psql -U postgres -d zerowaste_dev
```

#### 2. 마이그레이션 상태 확인

```bash
cd server
pnpm prisma migrate status
```

#### 3. 마이그레이션 재설정 (주의: 데이터 손실)

```bash
# 개발 환경에서만 사용
cd server
pnpm prisma migrate reset
```

## 환경 변수 관련

### 문제: 환경 변수가 적용되지 않음

**증상**:
- `.env` 파일을 수정했지만 변경사항이 반영되지 않음

**해결 방법**:

#### Next.js (Web) 환경 변수

Next.js는 빌드 타임에 환경 변수를 포함시키므로 **반드시 재빌드** 필요:

```bash
docker compose down
docker compose up -d --build web
```

#### NestJS (API) 환경 변수

재시작만으로 충분:

```bash
docker compose restart api
```

## 네트워크 관련

### 문제: API 요청 실패

**증상**:
- 프론트엔드에서 API를 호출할 수 없음
- CORS 에러 발생

**해결 방법**:

#### 1. API 서버 상태 확인

```bash
# API 헬스 체크
curl http://localhost:3000/health

# Places 엔드포인트 확인
curl http://localhost:3000/places
```

#### 2. 네트워크 확인

```bash
# Docker 네트워크 확인
docker network ls
docker network inspect zero-factory_default
```

#### 3. NEXT_PUBLIC_API_URL 확인

`.env` 파일에서 올바른 API URL 설정:

```bash
# 로컬 개발
NEXT_PUBLIC_API_URL=http://localhost:3000

# Docker 환경
NEXT_PUBLIC_API_URL=http://localhost:3000

# 프로덕션 (EC2)
NEXT_PUBLIC_API_URL=http://YOUR_SERVER_IP:3000
```

## 권한 관련

### 문제: Permission denied

**증상**:
- 파일 생성/수정 시 권한 에러 발생

**해결 방법**:

#### 1. 파일 권한 확인

```bash
ls -la
```

#### 2. 소유권 변경

```bash
# 현재 사용자로 소유권 변경
sudo chown -R $USER:$USER .
```

#### 3. Docker 볼륨 권한

```bash
# 볼륨 재생성
docker compose down -v
docker compose up -d
```

## 성능 관련

### 문제: 서비스가 느림

**해결 방법**:

#### 1. 리소스 사용량 확인

```bash
# Docker 컨테이너 리소스 사용량
docker stats

# 시스템 리소스
top
htop
```

#### 2. 로그 크기 확인

```bash
# 로그 크기 확인
docker compose logs --tail=0 | wc -l

# 오래된 로그 삭제 (Docker 재시작)
docker compose restart
```

#### 3. 데이터베이스 인덱스 확인

```bash
# Prisma Studio에서 인덱스 확인
cd server
pnpm prisma:studio
```

## 추가 도움이 필요한 경우

위 방법으로 해결되지 않는 문제가 있다면:

1. [GitHub Issues](https://github.com/muchwater/zero-factory/issues)에 문제 등록
2. 로그 파일 첨부:
   ```bash
   docker compose logs > logs.txt
   ```
3. 환경 정보 제공:
   ```bash
   docker --version
   docker compose version
   node --version
   ```
