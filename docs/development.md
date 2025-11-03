# 로컬 개발 환경

Docker를 사용하지 않고 로컬에서 각 서비스를 개별적으로 실행하는 방법을 안내합니다.

## Backend (Server)

### 필수 요구사항

- Node.js 22.19.0
- pnpm

### 설치 및 실행

#### 1. pnpm 설치

```bash
npm i -g pnpm
```

#### 2. 환경 변수 설정

루트 디렉토리의 `.env.example` 파일을 `server/.env`로 복사합니다:

```bash
cd server
cp ../.env.example .env
```

#### 3. 의존성 설치 및 DB 설정

```bash
pnpm install
pnpm db:setup  # DB 실행 + 초기 마이그레이션 + Prisma Client 생성
```

#### 4. 개발 서버 실행

```bash
pnpm dev
```

서버는 http://localhost:3000 에서 실행됩니다.

### 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `pnpm dev` | 개발 서버 실행 |
| `pnpm build` | 프로덕션 빌드 |
| `pnpm start` | 프로덕션 서버 실행 |
| `pnpm db:setup` | DB 실행 + 초기 마이그레이션 + Client 생성 |
| `pnpm prisma:migrate --name <name>` | 마이그레이션 생성 및 실행 |
| `pnpm prisma:generate` | Prisma Client 재생성 |
| `pnpm prisma:studio` | Prisma Studio 실행 |

## Frontend (Web)

### 필수 요구사항

- Node.js (권장: 18.x 이상)
- npm

### 설치 및 실행

#### 1. 환경 변수 설정

루트 디렉토리의 `.env.example` 파일을 `web/.env.local`로 복사합니다:

```bash
cd web
cp ../.env.example .env.local
```

#### 2. 의존성 설치

```bash
npm install
```

#### 3. 개발 서버 실행

```bash
npm run dev
```

웹 서버는 http://localhost:3001 에서 실행됩니다.

> 참고: 로컬 개발과 Docker 환경 모두 3001번 포트를 사용합니다. Backend는 3000번 포트를 사용하므로 포트 충돌 없이 동시 실행이 가능합니다.

### 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `npm run dev` | 개발 서버 실행 (http://localhost:3001) |
| `npm run build` | 프로덕션 빌드 |
| `npm run start` | 프로덕션 서버 실행 |
| `npm run lint` | 린팅 |

## 데이터베이스 관리

### Prisma Studio

데이터베이스를 GUI로 관리하려면 Prisma Studio를 사용할 수 있습니다:

```bash
cd server
pnpm prisma:studio
```

브라우저에서 http://localhost:5555 로 접속하여 데이터베이스를 확인할 수 있습니다.

### 마이그레이션

스키마 변경 후 마이그레이션을 생성하려면:

```bash
cd server
pnpm prisma:migrate --name <migration_name>
```

예시:
```bash
pnpm prisma:migrate --name add_user_table
```

## 개발 워크플로우

1. **새로운 기능 개발 시작**
   - 새 브랜치 생성: `git checkout -b feature/new-feature`
   - 로컬 개발 서버 실행

2. **DB 스키마 변경**
   - `prisma/schema.prisma` 수정
   - 마이그레이션 생성 및 적용
   - Prisma Client 재생성

3. **코드 작성 및 테스트**
   - 기능 구현
   - 로컬에서 테스트

4. **커밋 및 푸시**
   - 변경사항 커밋
   - GitHub에 푸시
   - Pull Request 생성

## 다음 단계

- [Docker 명령어 가이드](./docker.md)
- [트러블슈팅](./troubleshooting.md)
