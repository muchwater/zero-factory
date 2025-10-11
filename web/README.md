# Zero Factory Frontend

제로웨이스트 라이프스타일을 위한 친환경 가이드 앱의 프론트엔드입니다.

## 기술 스택

- **Next.js 15.5.4** - React 프레임워크
- **TypeScript** - 타입 안전성
- **Tailwind CSS** - 스타일링
- **React 18** - UI 라이브러리

## 주요 기능

- 🗺️ **지도 기반 검색**: 카카오맵을 활용한 제로웨이스트 장소 검색
- ♻️ **카테고리별 필터링**: 다회용컵 카페, 반납함, 텀블러 포인트 적립 등
- 📍 **근처 장소 추천**: 사용자 위치 기반 제로웨이스트 장소 추천
- 🔍 **검색 기능**: 상점명 또는 지역명으로 검색

## 설치 및 실행

### 1. 의존성 설치
```bash
npm install
```

### 2. 개발 서버 실행
```bash
npm run dev
```

### 3. 브라우저에서 확인
[http://localhost:3000](http://localhost:3000)에서 앱을 확인할 수 있습니다.

## 프로젝트 구조

```
src/
├── app/
│   ├── globals.css      # 전역 스타일
│   ├── layout.tsx       # 루트 레이아웃
│   └── page.tsx         # 메인 페이지
├── components/          # 재사용 가능한 컴포넌트
└── types/              # TypeScript 타입 정의
```

## 디자인 시스템

Figma 디자인을 기반으로 구현된 모바일 우선 반응형 디자인입니다.

- **색상**: 흰색 배경, 검은색 텍스트
- **폰트**: Inter (Google Fonts)
- **아이콘**: SVG 기반 커스텀 아이콘
- **레이아웃**: 모바일 우선, 데스크톱 대응

## 개발 명령어

```bash
# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 프로덕션 서버 실행
npm run start

# 린팅
npm run lint
```

## 라이선스

MIT License
# zero-factory-frontend
