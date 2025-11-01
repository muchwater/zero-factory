# Zero Factory Backend API

NestJS 기반의 제로웨이스트 가게 위치 정보 API 서버입니다.

## 기술 스택

- **NestJS** - Node.js 프레임워크
- **Prisma ORM** - 데이터베이스 ORM
- **PostgreSQL with PostGIS** - 공간 데이터베이스
- **TypeScript** - 타입 안전성

## Docker로 실행 (권장)

프로젝트 루트 디렉토리에서 실행하세요:

```bash
# 프로젝트 루트로 이동
cd ..

# Docker Compose로 전체 서비스 실행
docker compose up -d
```

자세한 내용은 [루트 디렉토리의 README.md](../README.md)를 참조하세요.

## 로컬 개발 환경 실행

Node 버전 22.19.0
pnpm 사용 (미설치시 아래 명령어 입력)

```bash
npm i -g pnpm
```

**중요: .env.example 파일을 복사해서 .env 파일로 변경**

```bash
pnpm i
pnpm db:setup
pnpm dev
```

# DB Commands

## 전체 초기 세팅 (DB 실행 + 초기 마이그레이션 + Client 생성)

```bash
pnpm db:setup
```

## 마이그레이션 실행 (예시)

```bash
pnpm prisma:migrate --name migration-file
```

## Prisma Client 재생성

```bash
pnpm prisma:generate
```
