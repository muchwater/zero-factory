# Mixed Content 오류 방지 가이드

## 문제 개요

HTTPS 페이지에서 HTTP API를 호출하려고 하면 브라우저가 **Mixed Content** 오류를 발생시킵니다.

```
Mixed Content: The page at 'https://www.zeromap.store/' was loaded over HTTPS,
but requested an insecure resource 'http://43.201.190.116:3000/places?state=ACTIVE'.
This request has been blocked; the content must be served over HTTPS.
```

## 근본 원인

1. **하드코딩된 폴백 URL**: `web/src/services/api.ts`에 하드코딩된 HTTP URL
2. **환경 변수 누락**: 빌드 시 `NEXT_PUBLIC_API_URL`이 설정되지 않음
3. **빌드 캐시**: 환경 변수 변경 후 캐시된 빌드 사용

## 적용된 해결책

### 1. 하드코딩된 폴백 URL 제거

**변경 전** (`web/src/services/api.ts`):
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://43.201.190.116:3000'
```

**변경 후**:
```typescript
// Validate that NEXT_PUBLIC_API_URL is set at build time
if (!process.env.NEXT_PUBLIC_API_URL) {
  throw new Error(
    'NEXT_PUBLIC_API_URL environment variable is not set. ' +
    'Please set it in your .env file before building.'
  )
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL
```

**효과**:
- 환경 변수가 없으면 빌드가 실패하므로 잘못된 설정을 조기에 발견
- 하드코딩된 URL로 인한 Mixed Content 오류 원천 차단

### 2. Docker Compose 설정 개선

**변경 전** (`docker-compose.prod.yml`):
```yaml
web:
  build:
    args:
      NEXT_PUBLIC_API_URL: "https://zeromap.store/api"  # 하드코딩됨
```

**변경 후**:
```yaml
web:
  build:
    args:
      NEXT_PUBLIC_API_URL: "${NEXT_PUBLIC_API_URL}"  # .env에서 읽음
```

**효과**:
- `.env` 파일의 값을 사용하여 환경별 설정 유연성 확보
- 중앙 집중식 설정 관리

### 3. 환경 변수 설정 표준화

**프로덕션** (`.env`):
```bash
DATABASE_URL="postgresql://postgres:postgres@db:5432/zerowaste_dev"
NEXT_PUBLIC_API_URL=https://zeromap.store/api
```

**개발** (`.env.dev`):
```bash
DATABASE_URL="postgresql://postgres:postgres@db:5432/zerowaste_dev"
NEXT_PUBLIC_API_URL=http://localhost:3000
```

## 재발 방지 체크리스트

### 환경 변수 변경 시

- [ ] `.env` 파일에서 `NEXT_PUBLIC_API_URL` 확인
- [ ] `DATABASE_URL`의 호스트가 `db`인지 확인 (Docker 환경)
- [ ] 웹 컨테이너 **캐시 없이** 재빌드
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache web
  ```
- [ ] 컨테이너 재시작
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
  ```

### 새 환경 배포 시

- [ ] 해당 환경의 `.env` 파일 작성
- [ ] `NEXT_PUBLIC_API_URL`이 HTTPS인지 확인 (프로덕션)
- [ ] 빌드 전 환경 변수 로드 확인
  ```bash
  # .env 파일 확인
  cat .env | grep NEXT_PUBLIC_API_URL
  ```

### 코드 변경 시

- [ ] `api.ts`에 하드코딩된 URL 추가하지 않기
- [ ] 새로운 API 클라이언트는 `API_BASE_URL` 사용
- [ ] 환경 변수는 반드시 `NEXT_PUBLIC_` 접두사 사용 (클라이언트 사이드)

## 검증 방법

### 1. 빌드된 파일에서 하드코딩된 URL 확인
```bash
# 프로덕션 빌드에서 하드코딩된 IP 검색 (결과 없어야 함)
docker exec zero-factory-web-1 sh -c 'grep -r "43.201.190.116" .next/ 2>/dev/null'

# 올바른 API URL 확인
docker exec zero-factory-web-1 sh -c 'grep -o "https://zeromap.store/api" .next/server/app/page.js | head -1'
```

### 2. 브라우저 개발자 도구에서 확인
1. F12를 눌러 개발자 도구 열기
2. Network 탭으로 이동
3. 페이지 새로고침 (Ctrl+Shift+R)
4. API 요청 확인:
   - URL이 `https://zeromap.store/api/...`로 시작하는지 확인
   - Mixed Content 경고가 없는지 확인

### 3. 환경 변수 누락 시 빌드 실패 테스트
```bash
# NEXT_PUBLIC_API_URL 없이 빌드 시도 (실패해야 정상)
docker build --build-arg NEXT_PUBLIC_KAKAO_MAP_KEY="test" -f web/Dockerfile web
# Expected: Error: NEXT_PUBLIC_API_URL environment variable is not set...
```

## 트러블슈팅

### 문제: Mixed Content 오류 여전히 발생

**원인**: 브라우저 캐시 또는 이전 빌드 사용

**해결**:
```bash
# 1. 웹 컨테이너 캐시 없이 재빌드
docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache web

# 2. 컨테이너 재시작
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 3. 브라우저 캐시 강제 새로고침
# Chrome/Firefox: Ctrl+Shift+R (또는 Cmd+Shift+R on Mac)
```

### 문제: 빌드는 성공했지만 여전히 잘못된 URL 사용

**원인**: 환경 변수가 빌드 시점에 전달되지 않음

**해결**:
```bash
# 1. .env 파일에서 환경 변수 확인
cat .env | grep NEXT_PUBLIC_API_URL

# 2. docker-compose 파일 확인
grep -A 5 "NEXT_PUBLIC_API_URL" docker-compose.prod.yml

# 3. 빌드 시 환경 변수 명시적으로 전달
docker compose -f docker-compose.yml -f docker-compose.prod.yml build \
  --build-arg NEXT_PUBLIC_API_URL="https://zeromap.store/api" web
```

### 문제: DATABASE_URL 관련 오류

**원인**: Docker 환경에서 `localhost` 대신 `db` 사용해야 함

**해결**:
```bash
# .env 파일 수정
DATABASE_URL="postgresql://postgres:postgres@db:5432/zerowaste_dev"
```

## 관련 문서

- [환경 설정 가이드](./ENVIRONMENT_SETUP.md)
- [트러블슈팅](./troubleshooting.md)
- [Docker 가이드](./docker.md)

## 참고사항

### Next.js 환경 변수 규칙

- **클라이언트에서 접근 가능**: `NEXT_PUBLIC_*` 접두사 필요
- **서버에서만 접근 가능**: 접두사 없이 사용
- **빌드 타임**: `NEXT_PUBLIC_*` 변수는 빌드 시점에 번들에 포함됨
- **런타임**: 서버 환경 변수는 런타임에 읽음

### Mixed Content 정책

현대 브라우저는 보안을 위해 HTTPS 페이지에서 HTTP 리소스 로드를 차단합니다:
- ✅ HTTPS → HTTPS: 허용
- ✅ HTTP → HTTP: 허용
- ❌ HTTPS → HTTP: **차단 (Mixed Content)**
- ✅ HTTP → HTTPS: 허용

따라서 프로덕션 환경에서는 반드시 모든 API 엔드포인트가 HTTPS를 사용해야 합니다.
