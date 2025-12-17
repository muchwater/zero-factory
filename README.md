# Zero Factory

> 제로웨이스트 서비스의 위치 정보 제공과 리워드 제공의 통합 서비스
> Integrated service providing location information and rewards for zero-waste services

---

## 📌 풀고자 하는 사회 문제 | Social Problem

현대 사회는 일회용품 사용 증가로 인한 환경 오염이 심각한 수준에 이르렀습니다. 특히 일회용 컵과 포장재 사용은 매년 급증하고 있으며, 이는 플라스틱 쓰레기 문제를 가중시키고 있습니다. 제로웨이스트를 실천하고자 하는 사람들은 다음과 같은 어려움에 직면합니다:

- 다회용기 사용 가능한 가게를 찾기 어려움
- 다회용기 반납 장소에 대한 정보 부족
- 친환경 실천에 대한 동기 부여 및 리워드 시스템 미흡
- 산재된 제로웨이스트 정보로 인한 접근성 문제

Modern society faces severe environmental pollution due to increased disposable product usage. Single-use cups and packaging waste are growing exponentially each year, intensifying plastic waste problems. People trying to practice zero-waste face challenges such as:

- Difficulty finding stores accepting reusable containers
- Lack of information about return locations for reusable containers
- Insufficient motivation and reward systems for eco-friendly practices
- Poor accessibility due to scattered zero-waste information

---

## 💡 솔루션 개요 | Solution Overview

### 핵심 특장점 | Key Features

Zero Factory는 제로웨이스트 실천의 두 가지 핵심 요소를 통합한 플랫폼입니다:

1. **위치 기반 정보 제공**: 카카오맵 기반으로 다회용컵 사용 가능한 카페, 반납함, 텀블러 포인트 적립 장소 등을 한눈에 확인
2. **리워드 시스템 통합**: 제로웨이스트 실천에 대한 즉각적인 보상으로 지속 가능한 행동 변화 유도
3. **통합 플랫폼**: 산재된 정보를 하나의 서비스로 통합하여 사용자 편의성 극대화

Zero Factory is an integrated platform combining two core elements of zero-waste practice:

1. **Location-Based Information**: Kakao Map-based service to easily find cafes accepting reusable cups, return boxes, and tumbler point locations
2. **Integrated Reward System**: Encouraging sustainable behavior change through immediate rewards for zero-waste practices
3. **Unified Platform**: Maximizing user convenience by consolidating scattered information into one service

### 유사 솔루션 대비 특장점 | Advantages Over Similar Solutions

- **통합성**: 위치 정보 + 리워드를 하나의 플랫폼에서 제공
- **접근성**: 직관적인 지도 기반 UI로 누구나 쉽게 사용 가능
- **실시간성**: 사용자 위치 기반 실시간 정보 제공
- **확장성**: 다양한 제로웨이스트 파트너사 연동 가능한 구조

- **Integration**: Location information + rewards in one platform
- **Accessibility**: Intuitive map-based UI for easy access
- **Real-time**: Location-based real-time information
- **Scalability**: Structure enabling integration with various zero-waste partners

### 기대 효과 | Expected Impact

- 제로웨이스트 실천 접근성 향상으로 참여자 증가
- 다회용기 사용 활성화를 통한 일회용품 쓰레기 감소
- 리워드 시스템을 통한 지속 가능한 행동 변화 유도
- 제로웨이스트 생태계 활성화 및 파트너 네트워크 확장

- Increase in participants through improved accessibility to zero-waste practices
- Reduction in disposable waste through activation of reusable container usage
- Inducing sustainable behavior change through reward system
- Activation of zero-waste ecosystem and expansion of partner network

---

## 주요 기능 | Main Features

- 🗺️ **지도 기반 검색 | Map-Based Search**: 카카오맵을 활용한 제로웨이스트 장소 검색
- ♻️ **카테고리별 필터링 | Category Filtering**: 다회용컵 카페, 반납함, 텀블러 포인트 적립 등
- 📍 **근처 장소 추천 | Nearby Recommendations**: 사용자 위치 기반 제로웨이스트 장소 추천
- 🔍 **검색 기능 | Search**: 상점명 또는 지역명으로 검색
- 🎁 **리워드 시스템 | Reward System**: 제로웨이스트 실천에 대한 포인트 적립 및 보상

---

## 🚀 설치 및 실행 방법 | Installation and Setup

### 사전 요구사항 | Prerequisites

- Docker
- Docker Compose

### 설치 및 실행 | Installation

#### 개발 환경 (Development)

```bash
# 1. 저장소 클론
git clone https://github.com/muchwater/zero-factory.git
cd zero-factory

# 2. 환경 변수 설정
# .env.dev 파일에서 다음 값들을 설정하세요:
# - NEXT_PUBLIC_KAKAO_MAP_KEY: 카카오맵 API 키 (https://developers.kakao.com/)
# - ADMIN_CODE: 관리자 인증 코드

# 3. 개발 환경으로 실행
./start-dev.sh

# 4. 서비스 접속
# Frontend: http://localhost (또는 http://localhost:3001)
# Frontend-admin: http://localhost/admin
# Backend API: http://localhost:3000
```

#### 배포 환경 (Production)

```bash
# 1. 환경 변수 설정
# .env.prod 파일에서 다음 값들을 설정하세요:
# - NEXT_PUBLIC_KAKAO_MAP_KEY: 카카오맵 API 키
# - ADMIN_CODE: 보안이 강화된 관리자 인증 코드

# 2. 배포 환경으로 실행
./start-prod.sh

# 3. 서비스 접속
# Frontend: https://zeromap.store
# Backend API: https://zeromap.store/api
```

**더 자세한 설명은 [환경 설정 가이드](./docs/ENVIRONMENT_SETUP.md)와 [시작하기 가이드](./docs/getting-started.md)를 참조하세요.**

## 프로젝트 구조

```
zero-factory/
├── server/                   # Backend API (NestJS)
├── web/                      # Frontend (Next.js)
├── ai-server/                # AI Server & Label Studio
├── docs/                     # 프로젝트 문서
├── nginx/                    # Nginx 설정
│   ├── nginx.conf           # Production 설정 (HTTPS)
│   └── nginx.dev.conf       # Development 설정 (HTTP)
├── docker-compose.yml        # Base 설정 (공통)
├── docker-compose.dev.yml    # Development 오버라이드
├── docker-compose.prod.yml   # Production 오버라이드
├── .env                      # 활성 환경 변수 (사용자 생성, gitignore)
├── .env.dev                  # Development 환경 변수 템플릿
├── .env.prod                 # Production 환경 변수 템플릿
├── .env.example              # 환경 변수 예제
├── start-dev.sh              # Development 환경 실행 스크립트
├── start-prod.sh             # Production 환경 실행 스크립트
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

- **[환경 설정 가이드](./docs/ENVIRONMENT_SETUP.md)** - 개발/배포 환경 설정 및 실행 방법 ⭐ NEW
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

프로젝트는 환경별로 다른 설정을 사용합니다:

### Development (.env.dev)
```bash
NODE_ENV=development
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=zerowaste_dev
DATABASE_URL="postgresql://postgres:postgres@db:5432/zerowaste_dev"
NEXT_PUBLIC_API_URL=http://localhost:3000
NEXT_PUBLIC_KAKAO_MAP_KEY=your_kakao_map_key_here
```

### Production (.env 또는 .env.prod)
```bash
NODE_ENV=production
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=zerowaste_dev
DATABASE_URL="postgresql://postgres:postgres@db:5432/zerowaste_dev"
NEXT_PUBLIC_API_URL=https://zeromap.store/api
NEXT_PUBLIC_KAKAO_MAP_KEY=your_kakao_map_key_here
DOMAIN=zeromap.store
```

### ⚠️ 중요 사항

**NEXT_PUBLIC_API_URL 설정 필수:**
- 이 환경 변수는 빌드 시점에 반드시 설정되어야 합니다
- 설정되지 않으면 빌드가 실패합니다 (하드코딩된 폴백 URL 제거됨)
- Docker 빌드 전에 `.env` 파일에 올바른 값이 설정되어 있는지 확인하세요

**DATABASE_URL 호스트:**
- Docker 환경: `db` 사용 (컨테이너 이름)
- 로컬 개발: `localhost` 사용

**환경 변수 변경 후 재빌드 필요:**
- `NEXT_PUBLIC_*` 환경 변수를 변경한 경우 반드시 웹 컨테이너를 재빌드해야 합니다
- 빌드 명령어: `docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache web`

자세한 설정 방법은 [환경 설정 가이드](./docs/ENVIRONMENT_SETUP.md)와 [API Keys 가이드](./docs/api-keys.md)를 참조하세요.

## 배포

프로젝트는 GitHub Actions를 통해 자동으로 EC2에 배포됩니다.

`main` 브랜치에 push하면:

1. 자동으로 EC2 서버에 배포
2. Docker 이미지 빌드
3. Production 환경으로 서비스 재시작 (`./start-prod.sh`)
4. 헬스 체크 수행

### 환경별 배포 방식

- **개발 서버**: 자동으로 development 설정 사용
- **배포 서버**: 자동으로 production 설정 사용 (SSL, HTTPS)

자세한 내용은 [환경 설정 가이드](./docs/ENVIRONMENT_SETUP.md)와 [Docker 가이드 - 프로덕션 배포](./docs/docker.md#프로덕션-배포)를 참조하세요.

---

## 📚 연관 자료 | References

### 프로젝트 자료 | Project Resources
- **프로젝트 페이지 | Project Page**: [Notion](https://www.notion.so/251d5c59d5e680dc9339e33b8d16f3d1)
- **GitHub Repository**: [github.com/muchwater/zero-factory](https://github.com/muchwater/zero-factory)
- **펠로우 조직 | Fellow Organization**: [보틀팩토리 (Bottle Factory)](https://www.bottlefactory.co.kr/)

### 기술 문서 | Technical Documentation
- [환경 설정 가이드 | Environment Setup](./docs/ENVIRONMENT_SETUP.md)
- [시작하기 가이드 | Getting Started](./docs/getting-started.md)
- [API Keys 설정 | API Keys Setup](./docs/api-keys.md)
- [배포 설정 가이드 | Deployment Guide](./docs/deployment.md)

---

## 👥 팀 및 팀원 소개 | Team Members

### 팀원 | Team Members

| 이름 | 역할 | 소속 | 연락처 |
|------|------|------|--------|
| 전준형 | PM (Project Manager) | KAIST 전산학부 | muchwater@kaist.ac.kr |
| 송원태 | Front-end Developer | KAIST 전산학부 | wontae1014@kaist.ac.kr |
| 신은지 | UI/UX Designer | KAIST 산업디자인학과 | sargentz@kaist.ac.kr |
| 황현우 | Back-end Developer | KAIST 전기및전자공학부 | hwanghw001@kaist.ac.kr |
| 허재영 | Back-end Developer | KAIST 전산학부 | trick@kaist.ac.kr |

### 멘토 및 펠로우 | Mentor & Fellow

| 이름 | 역할 | 소속 | 연락처 |
|------|------|------|--------|
| 장지수 | 멘토 (Mentor) | 카카오 (Kakao) | - |
| 정다운 | 펠로우 (Fellow) | 보틀팩토리 (Bottle Factory) | dawoon@bottlefactory.co.kr |

---

## 라이선스 | License

MIT License

---

## 문의 | Contact

프로젝트에 대한 문의사항이나 버그 리포트는 [GitHub Issues](https://github.com/muchwater/zero-factory/issues)에 등록해 주세요.

For inquiries or bug reports, please submit them to [GitHub Issues](https://github.com/muchwater/zero-factory/issues).
