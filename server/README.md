# TUTORIAL

## How to start

Node 버전 22.19.0  
pnpm 사용 (미설치시 아래 명령어 입력)

```bash
npm i -g pnpm
```

docker desktop 추천  
[window설치](https://docs.docker.com/desktop/setup/install/windows-install/)  
[mac설치](https://docs.docker.com/desktop/setup/install/mac-install/)

**중요: .env.example 파일을 복사해서 .env파일로 변경**

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
