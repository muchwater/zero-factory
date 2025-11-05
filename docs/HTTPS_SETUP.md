# HTTPS 설정 가이드

이 문서는 Nginx + Let's Encrypt를 사용하여 HTTPS를 설정하는 방법을 설명합니다.

## 아키텍처

```
Internet (HTTPS/HTTP)
         ↓
    Nginx (포트 80, 443)
         ↓
    ┌────┴────┐
    ↓         ↓
 API:3000  Web:3001
(내부 전용) (내부 전용)
```

- **Nginx**: 리버스 프록시 및 SSL 종료 지점
- **Certbot**: Let's Encrypt SSL 인증서 자동 발급 및 갱신
- **API/Web**: 내부 네트워크에서만 접근 가능 (포트 노출 안 함)

## 주요 URL 구조

- `https://zeromap.store` → Web (Next.js)
- `https://zeromap.store/api` → API (NestJS)
- HTTP는 자동으로 HTTPS로 리디렉션

## 초기 설정 (첫 배포 시)

### 1단계: 도메인 DNS 설정 확인

도메인이 서버 IP로 연결되어 있는지 확인:

```bash
nslookup zeromap.store
# 결과: 43.201.190.116
```

### 2단계: 기존 컨테이너 중지

```bash
cd /home/ubuntu/zero-factory
docker-compose down
```

### 3단계: SSL 인증서 발급

`init-letsencrypt.sh` 스크립트를 실행하여 인증서를 발급받습니다:

```bash
# 스크립트 실행 권한 확인
chmod +x init-letsencrypt.sh

# 인증서 발급 (실제 인증서)
./init-letsencrypt.sh
```

**중요 참고사항:**
- 스크립트 실행 중 이메일 주소를 확인하세요 (기본값: admin@zeromap.store)
- Let's Encrypt는 도메인당 주간 발급 제한이 있습니다 (주 50개)
- 테스트 시에는 `staging=1`로 설정하여 제한을 피할 수 있습니다

### 4단계: 이메일 주소 변경 (선택사항)

`init-letsencrypt.sh` 파일을 열고 이메일 주소를 변경:

```bash
email="your-email@example.com"  # 실제 이메일로 변경
```

### 5단계: 전체 서비스 시작

```bash
docker-compose up -d --build
```

### 6단계: 확인

```bash
# HTTPS 접속 확인
curl -I https://zeromap.store
curl -I https://zeromap.store/api/health

# 또는 브라우저에서:
# https://zeromap.store
```

## 인증서 갱신

Let's Encrypt 인증서는 **90일**마다 만료됩니다. Certbot 컨테이너가 **자동으로 12시간마다** 갱신을 시도합니다.

### 수동 갱신 (필요한 경우)

```bash
# Certbot 컨테이너로 갱신 실행
docker-compose run --rm certbot renew

# Nginx 재시작
docker-compose exec nginx nginx -s reload
```

### 갱신 상태 확인

```bash
# 인증서 만료일 확인
docker-compose run --rm certbot certificates
```

## 파일 구조

```
zero-factory/
├── nginx/
│   └── nginx.conf              # Nginx 설정 파일
├── certbot/
│   ├── conf/                   # SSL 인증서 저장소
│   └── www/                    # ACME 챌린지 파일
├── init-letsencrypt.sh         # 인증서 초기 발급 스크립트
├── docker-compose.yml          # Nginx, Certbot 서비스 포함
└── .env                        # HTTPS URL 설정
```

## Nginx 설정 주요 내용

### HTTP → HTTPS 리디렉션

모든 HTTP 트래픽은 자동으로 HTTPS로 리디렉션됩니다:

```nginx
server {
    listen 80;
    server_name zeromap.store www.zeromap.store;

    location / {
        return 301 https://$host$request_uri;
    }
}
```

### SSL/TLS 설정

- **프로토콜**: TLSv1.2, TLSv1.3
- **키 크기**: RSA 4096-bit
- **HSTS**: 활성화 (1년)
- **보안 헤더**: X-Frame-Options, X-Content-Type-Options 등

### 프록시 설정

```nginx
# API 프록시
location /api/ {
    rewrite ^/api/(.*) /$1 break;  # /api 제거
    proxy_pass http://api:3000;
}

# Web 프록시
location / {
    proxy_pass http://web:3001;
}
```

## 환경변수 설정

`.env` 파일에서 HTTPS URL 사용:

```env
NEXT_PUBLIC_API_URL=https://zeromap.store/api
```

**중요:**
- Docker 빌드 시 환경변수가 포함되므로, 변경 후 반드시 재빌드 필요
- `docker-compose up -d --build` 실행

## 보안 고려사항

### 1. 방화벽 설정

AWS Security Group에서 다음 포트 허용:
- **80/tcp** - HTTP (HTTPS 리디렉션용)
- **443/tcp** - HTTPS

### 2. Rate Limiting

Nginx에서 요청 제한 설정:
- API: 초당 10건 (버스트 20건)
- Web: 초당 30건 (버스트 50건)

### 3. CORS 설정

[server/src/main.ts:10-21](server/src/main.ts#L10-L21)에서 허용된 도메인만 접근 가능:

```typescript
app.enableCors({
  origin: [
    'https://zeromap.store',
    'https://www.zeromap.store',
    // ...
  ],
  credentials: true,
});
```

## 트러블슈팅

### 인증서 발급 실패

**증상**: `init-letsencrypt.sh` 실행 시 오류

**해결 방법**:
1. 도메인 DNS 연결 확인: `nslookup zeromap.store`
2. 포트 80이 열려있는지 확인: `curl http://zeromap.store`
3. Staging 모드로 테스트: `staging=1` 설정

### Nginx 시작 실패

**증상**: `docker-compose up` 시 nginx 컨테이너 종료

**해결 방법**:
1. Nginx 설정 문법 확인:
   ```bash
   docker-compose run --rm nginx nginx -t
   ```
2. 로그 확인:
   ```bash
   docker-compose logs nginx
   ```

### Mixed Content 경고

**증상**: 브라우저에서 일부 리소스가 로드되지 않음

**해결 방법**:
- 모든 외부 리소스(이미지, API 등)를 HTTPS로 로드
- Kakao Map API가 HTTPS를 지원하는지 확인

### 인증서 갱신 실패

**증상**: 90일 후 HTTPS 접속 불가

**해결 방법**:
1. 수동 갱신 시도:
   ```bash
   docker-compose run --rm certbot renew --dry-run
   ```
2. Certbot 컨테이너 재시작:
   ```bash
   docker-compose restart certbot
   ```

## 성능 최적화

### 1. HTTP/2 활성화

이미 Nginx 설정에서 활성화됨:
```nginx
listen 443 ssl http2;
```

### 2. Gzip 압축

텍스트 기반 콘텐츠 자동 압축 (70-90% 절약)

### 3. SSL Session 캐싱

SSL 핸드셰이크 성능 향상:
```nginx
ssl_session_cache shared:SSL:50m;
ssl_session_timeout 1d;
```

## 배포 워크플로우

### 코드 변경 후 배포

```bash
cd /home/ubuntu/zero-factory

# 최신 코드 가져오기
git pull origin main

# 컨테이너 재빌드 및 시작
docker-compose up -d --build

# 로그 확인
docker-compose logs -f
```

### 설정 변경 후 Nginx 재시작

```bash
# Nginx 설정 문법 확인
docker-compose exec nginx nginx -t

# Nginx 리로드 (다운타임 없음)
docker-compose exec nginx nginx -s reload
```

## 추가 정보

- **Let's Encrypt**: https://letsencrypt.org/
- **Certbot 문서**: https://certbot.eff.org/
- **Nginx 문서**: https://nginx.org/en/docs/
- **SSL Labs 테스트**: https://www.ssllabs.com/ssltest/

## 관련 문서

- [배포 가이드](DEPLOYMENT_SETUP.md)
- [Docker 가이드](DOCKER_GUIDE.md)
