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

### 데이터베이스 설정
- `POSTGRES_USER`: PostgreSQL 사용자명
- `POSTGRES_PASSWORD`: PostgreSQL 비밀번호
- `POSTGRES_DB`: 데이터베이스 이름
- `DATABASE_URL`: 데이터베이스 연결 URL

### API Key 설정

#### Kakao Map API Key

1. [Kakao Developers](https://developers.kakao.com/)에 접속하여 로그인
2. 내 애플리케이션 > 애플리케이션 추가하기
3. 생성된 애플리케이션의 **앱 키 > JavaScript 키** 복사
4. **플랫폼 설정** (매우 중요!)
   - "플랫폼" 메뉴 클릭
   - "Web 플랫폼 등록" 클릭
   - 사이트 도메인 등록:
     - 개발: `http://localhost:3001`
     - 프로덕션: 실제 서버 주소 (예: `http://43.201.190.116:3001` 또는 도메인)
5. `.env` 파일에 다음과 같이 설정:

```bash
NEXT_PUBLIC_KAKAO_MAP_KEY=your_kakao_map_key_here
```

6. Docker 재빌드 (환경 변수 변경 시 필수):

```bash
docker compose down
docker compose up -d --build
```

**중요**:
- Kakao Map API는 등록된 도메인에서만 작동합니다
- 새로운 도메인/IP를 추가할 때마다 Kakao Developers에서 플랫폼 등록 필요
- 환경 변수 변경 후 반드시 Docker 재빌드 필요 (Next.js는 빌드 시점에 환경 변수를 코드에 삽입)

## 트러블슈팅

### Kakao Map API 연결 실패

**증상**: 지도가 로드되지 않거나 콘솔에 Kakao API 관련 에러 발생

**해결 방법**:

1. **플랫폼 등록 확인**
   ```bash
   # 서버 IP 확인
   curl ifconfig.me
   ```
   - [Kakao Developers 콘솔](https://developers.kakao.com/)에서 해당 IP 또는 도메인이 등록되어 있는지 확인
   - 형식: `http://YOUR_IP:3001` 또는 `http://localhost:3001`

2. **환경 변수 확인**
   ```bash
   # .env 파일에 API 키가 있는지 확인
   cat .env | grep KAKAO

   # Docker 컨테이너에 환경 변수가 전달되었는지 확인
   docker logs zero-factory-web-1 2>&1 | head -20
   ```

3. **빌드된 파일에 API 키 포함 확인**
   ```bash
   # 빌드된 HTML에 API 키가 포함되어 있는지 확인
   curl -s http://localhost:3001 | grep -o "appkey=[^&\"]*"
   ```
   - API 키가 보이지 않으면 Docker 재빌드 필요

4. **브라우저 콘솔 확인**
   - F12 키를 눌러 개발자 도구 열기
   - Console 탭에서 에러 메시지 확인
   - 일반적인 에러:
     - `Failed to load resource`: 네트워크 문제 또는 플랫폼 미등록
     - `Kakao Map API error`: API 키 오류 또는 플랫폼 미등록

5. **Docker 재빌드**
   ```bash
   docker compose down
   docker compose up -d --build
   ```

### 데이터베이스 연결 실패

```bash
# 데이터베이스 컨테이너 상태 확인
docker compose ps

# 데이터베이스 로그 확인
docker compose logs db

# API 서버 로그 확인
docker compose logs api
```

## 라이선스

MIT
