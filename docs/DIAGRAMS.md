# AI 기반 다회용기 검증 시스템 - 다이어그램

## 목차
1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [다회용기 등록 플로우](#2-다회용기-등록-플로우)
3. [사용 인증 플로우](#3-사용-인증-플로우)
4. [AI 모델 추론 프로세스](#4-ai-모델-추론-프로세스)
5. [데이터 흐름](#5-데이터-흐름)

---

## 1. 시스템 아키텍처

전체 시스템의 컴포넌트 구성 및 통신 방식

```mermaid
graph TB
    subgraph "사용자 디바이스"
        User[사용자]
        Camera[카메라]
        IMU[IMU 센서]
        GPS[GPS]
    end

    subgraph "Frontend - Next.js"
        WebApp[웹 애플리케이션<br/>포트: 3001]
        RegisterPage[다회용기 등록<br/>/register-reusable]
        VerifyPage[사용 인증<br/>/verify-usage]
        AdminPage[관리자 페이지<br/>/admin/reusables]
    end

    subgraph "Backend - NestJS"
        API[API 서버<br/>포트: 3000]
        ReusablesModule[Reusables Module]
        AiServiceModule[AI Service Module]
        PrismaModule[Prisma ORM]
    end

    subgraph "AI Server - FastAPI"
        AIServer[AI Model Server<br/>포트: 8000]
        Classifier[분류 모델<br/>ResNet50]
        Embedding[임베딩 모델<br/>CLIP]
        Beverage[음료 검증<br/>MobileNetV3]
    end

    subgraph "Database"
        PostgreSQL[(PostgreSQL<br/>+ PostGIS)]
        ReusableTable[Reusable 테이블<br/>임베딩 벡터 포함]
        VerificationTable[ReusableVerification<br/>검증 이력]
        MemberTable[Member<br/>사용자 + 포인트]
    end

    subgraph "개발 도구"
        Jupyter[Jupyter Lab<br/>포트: 8888<br/>모델 학습]
        LabelStudio[Label Studio<br/>포트: 8080<br/>데이터 어노테이션]
    end

    User --> Camera
    User --> IMU
    User --> GPS

    Camera --> WebApp
    IMU --> WebApp
    GPS --> WebApp

    WebApp --> RegisterPage
    WebApp --> VerifyPage
    WebApp --> AdminPage

    RegisterPage --> API
    VerifyPage --> API
    AdminPage --> API

    API --> ReusablesModule
    ReusablesModule --> AiServiceModule
    ReusablesModule --> PrismaModule

    AiServiceModule -->|HTTP| AIServer

    AIServer --> Classifier
    AIServer --> Embedding
    AIServer --> Beverage

    PrismaModule --> PostgreSQL
    PostgreSQL --> ReusableTable
    PostgreSQL --> VerificationTable
    PostgreSQL --> MemberTable

    Jupyter -.->|학습된 모델| AIServer
    LabelStudio -.->|어노테이션 데이터| Jupyter

    style User fill:#e1f5ff
    style WebApp fill:#ffecb3
    style API fill:#c8e6c9
    style AIServer fill:#f8bbd0
    style PostgreSQL fill:#d1c4e9
    style Jupyter fill:#ffe0b2
    style LabelStudio fill:#ffe0b2
```

---

## 2. 다회용기 등록 플로우

사용자가 다회용기를 등록하는 전체 과정

```mermaid
sequenceDiagram
    actor User as 사용자
    participant Web as Web Frontend
    participant IMU as IMU 센서
    participant API as NestJS API
    participant AI as AI Server
    participant DB as PostgreSQL

    User->>Web: 다회용기 등록 페이지 접속
    Web->>IMU: 각도 센서 데이터 수집
    IMU-->>Web: 베타/감마 각도 (±15도 체크)

    alt 각도가 범위 벗어남
        Web->>User: ⚠️ 경고 메시지 표시<br/>(촬영은 가능)
    end

    User->>Web: 정면에서 촬영
    Web->>Web: 이미지 캡처

    User->>Web: 메타데이터 입력<br/>(이름, 브랜드, 설명)
    User->>Web: 등록 버튼 클릭

    Web->>API: POST /reusables/register<br/>(이미지 + 메타데이터)

    API->>AI: POST /classify-reusable<br/>(이미지)
    AI->>AI: ResNet50 분류 모델 추론
    AI-->>API: is_reusable: true/false<br/>confidence: 0.0~1.0

    alt 일회용기로 판정
        API-->>Web: ❌ 400 Bad Request<br/>"일회용기입니다"
        Web-->>User: 등록 거부 메시지
    else 다회용기로 판정
        API->>AI: POST /generate-embedding<br/>(이미지)
        AI->>AI: CLIP 임베딩 생성<br/>(512차원 벡터)
        AI-->>API: embedding: [512 floats]

        API->>API: 이미지 저장<br/>(uploads/reusables/)

        API->>DB: INSERT INTO Reusable<br/>(imageUrl, embedding, metadata)
        DB-->>API: Reusable ID

        API-->>Web: ✅ 201 Created<br/>(등록 정보)
        Web-->>User: 등록 완료 화면<br/>(ID, 신뢰도 표시)
    end
```

---

## 3. 사용 인증 플로우

다회용기 사용을 촬영으로 검증하여 포인트를 받는 과정

```mermaid
sequenceDiagram
    actor User as 사용자
    participant Web as Web Frontend
    participant Sensors as 센서<br/>(IMU + GPS)
    participant API as NestJS API
    participant AI as AI Server
    participant DB as PostgreSQL

    User->>Web: 사용 인증 페이지 접속

    Web->>Sensors: 센서 데이터 수집
    Sensors-->>Web: 각도 + 위치정보

    User->>Web: 음료 담긴 다회용기 촬영
    Web->>Web: 이미지 캡처

    Web->>API: POST /reusables/verify-usage<br/>(이미지 + lat/lng + memberId)

    rect rgb(255, 245, 235)
        Note over API,AI: Step 1: 음료 검증
        API->>AI: POST /verify-beverage<br/>(이미지)
        AI->>AI: MobileNetV3 추론<br/>(음료 유무 판단)
        AI-->>API: has_beverage: true/false<br/>confidence: 0.0~1.0

        alt 음료 없음
            API-->>Web: ❌ 400 Bad Request<br/>"음료가 담겨있지 않습니다"
            Web-->>User: 인증 실패
        end
    end

    rect rgb(232, 245, 233)
        Note over API,AI: Step 2: 임베딩 비교
        API->>AI: POST /generate-embedding<br/>(이미지)
        AI-->>API: embedding: [512 floats]

        API->>DB: SELECT * FROM Reusable<br/>WHERE memberId = ? AND state = 'APPROVED'
        DB-->>API: 사용자 등록 다회용기 리스트

        API->>DB: SELECT * FROM Reusable<br/>WHERE ownerType = 'ADMIN' AND state = 'APPROVED'
        DB-->>API: 관리자 표준 다회용기 리스트

        API->>API: 코사인 유사도 계산<br/>사용자: threshold 0.7<br/>관리자: threshold 0.75

        alt 매칭 실패
            API-->>Web: ❌ 400 Bad Request<br/>"등록된 다회용기와 일치하지 않습니다"
            Web-->>User: 인증 실패
        end
    end

    rect rgb(227, 242, 253)
        Note over API,DB: Step 3: 중복 검증
        API->>DB: SELECT * FROM ReusableVerification<br/>WHERE memberId = ?<br/>AND createdAt > NOW() - 1 HOUR
        DB-->>API: 최근 1시간 검증 이력

        API->>API: 거리 계산 (Haversine)<br/>500m 이내 체크

        alt 중복 (1시간 이내 + 500m 이내)
            API-->>Web: ❌ 400 Bad Request<br/>"이미 인증하셨습니다"
            Web-->>User: 중복 인증 거부
        end
    end

    rect rgb(248, 231, 255)
        Note over API,DB: Step 4: 포인트 지급
        API->>API: 이미지 저장

        API->>DB: BEGIN TRANSACTION

        API->>DB: INSERT INTO ReusableVerification<br/>(memberId, imageUrl, location,<br/>similarity, hasBeverage, pointsEarned)

        API->>DB: UPDATE Member<br/>SET pointBalance = pointBalance + 10<br/>WHERE id = ?

        API->>DB: COMMIT

        DB-->>API: Success

        API-->>Web: ✅ 200 OK<br/>(포인트 지급 정보)
        Web-->>User: 🎉 인증 완료!<br/>+10 포인트
    end
```

---

## 4. AI 모델 추론 프로세스

각 AI 모델의 역할과 입출력

```mermaid
graph LR
    subgraph "1. 분류 모델 (Classifier)"
        I1[이미지 입력]
        P1[전처리<br/>224x224<br/>정규화]
        M1[ResNet50<br/>사전학습 모델]
        O1[출력<br/>is_reusable: bool<br/>confidence: float]

        I1 --> P1 --> M1 --> O1
    end

    subgraph "2. 임베딩 모델 (Embedding)"
        I2[이미지 입력]
        P2[전처리<br/>CLIP 프로세서]
        M2[CLIP ViT-B/32<br/>Vision Encoder]
        N2[L2 정규화]
        O2[출력<br/>512차원 벡터<br/>norm = 1.0]

        I2 --> P2 --> M2 --> N2 --> O2
    end

    subgraph "3. 음료 검증 모델 (Beverage)"
        I3[이미지 입력]
        P3[전처리<br/>224x224<br/>정규화]
        M3[MobileNetV3-Small<br/>경량 모델]
        O3[출력<br/>has_beverage: bool<br/>confidence: float]

        I3 --> P3 --> M3 --> O3
    end

    style M1 fill:#ffcdd2
    style M2 fill:#c5e1a5
    style M3 fill:#b3e5fc
```

### 모델 상세 정보

| 모델 | 백본 | 입력 크기 | 출력 | 추론 속도 | 용도 |
|------|------|-----------|------|-----------|------|
| **분류** | ResNet50 | 224x224 | 2 classes | ~200ms | 다회용기 vs 일회용기 |
| **임베딩** | CLIP ViT-B/32 | 224x224 | 512-dim | ~300ms | 이미지 유사도 비교 |
| **음료** | MobileNetV3-Small | 224x224 | 2 classes | <100ms | 음료 유무 판단 |

---

## 5. 데이터 흐름

시스템 전체의 데이터 이동 경로

```mermaid
flowchart TD
    Start([사용자 촬영]) --> Capture[이미지 캡처<br/>Camera API]

    Capture --> IMU{IMU 센서<br/>각도 체크}
    IMU -->|±15도 이내| Upload[이미지 업로드]
    IMU -->|범위 벗어남| Warning[⚠️ 경고]
    Warning --> Upload

    Upload --> API[NestJS API<br/>Multer 파일 수신]

    API --> Route{라우팅}

    Route -->|등록| RegFlow[등록 플로우]
    Route -->|인증| VerFlow[인증 플로우]

    RegFlow --> AI1[AI: 분류 모델]
    AI1 -->|일회용기| Reject1[❌ 거부]
    AI1 -->|다회용기| AI2[AI: 임베딩 생성]

    AI2 --> SaveImg1[이미지 저장<br/>uploads/]
    SaveImg1 --> SaveDB1[(DB 저장<br/>Reusable)]
    SaveDB1 --> Success1[✅ 등록 완료]

    VerFlow --> AI3[AI: 음료 검증]
    AI3 -->|음료 없음| Reject2[❌ 거부]
    AI3 -->|음료 있음| AI4[AI: 임베딩 생성]

    AI4 --> Compare[유사도 비교<br/>DB 임베딩과 계산]
    Compare -->|매칭 실패| Reject3[❌ 거부]
    Compare -->|매칭 성공| DupCheck{중복 체크<br/>시간 + 위치}

    DupCheck -->|중복| Reject4[❌ 거부]
    DupCheck -->|통과| SaveImg2[이미지 저장]

    SaveImg2 --> SaveDB2[(DB 저장<br/>Verification)]
    SaveDB2 --> UpdatePoints[(포인트 증가<br/>Member)]
    UpdatePoints --> Success2[🎉 인증 완료<br/>+10 포인트]

    Reject1 --> End([종료])
    Reject2 --> End
    Reject3 --> End
    Reject4 --> End
    Success1 --> End
    Success2 --> End

    style Start fill:#e1f5ff
    style Success1 fill:#c8e6c9
    style Success2 fill:#c8e6c9
    style Reject1 fill:#ffcdd2
    style Reject2 fill:#ffcdd2
    style Reject3 fill:#ffcdd2
    style Reject4 fill:#ffcdd2
    style AI1 fill:#fff9c4
    style AI2 fill:#fff9c4
    style AI3 fill:#fff9c4
    style AI4 fill:#fff9c4
```

---

## 6. 데이터베이스 ERD

주요 테이블 간 관계

```mermaid
erDiagram
    Member ||--o{ Reusable : "등록"
    Member ||--o{ ReusableVerification : "검증"
    Reusable ||--o{ ReusableVerification : "매칭"
    Member ||--o{ PointTransaction : "포인트"
    Place ||--o{ PointTransaction : "장소"

    Member {
        string id PK
        string nickname UK
        string deviceId UK
        int pointBalance
        datetime createdAt
    }

    Reusable {
        int id PK
        string memberId FK
        enum ownerType "USER or ADMIN"
        string imageUrl
        float[] embedding "512-dim vector"
        string name
        string brand
        enum state "PENDING, APPROVED, REJECTED"
        float confidence
        datetime createdAt
    }

    ReusableVerification {
        int id PK
        string memberId FK
        int reusableId FK
        string imageUrl
        float latitude
        float longitude
        float similarity
        bool hasBeverage
        float beverageConfidence
        int pointsEarned
        bool isApproved
        datetime createdAt
    }

    PointTransaction {
        int id PK
        string memberId FK
        int placeId FK
        int amount
        enum type "EARN or REDEEM"
        datetime createdAt
    }

    Place {
        int id PK
        string name
        string address
        geography location
        enum category
        enum state
    }
```

---

## 7. 임베딩 벡터 비교 프로세스

코사인 유사도 계산 방식

```mermaid
graph TB
    subgraph "쿼리 이미지"
        Q[촬영된 이미지] --> QE[임베딩 생성<br/>CLIP]
        QE --> QV[쿼리 벡터<br/>512-dim, L2 norm=1]
    end

    subgraph "데이터베이스"
        DB1[(사용자 등록<br/>다회용기)] --> UV[사용자 벡터들<br/>N개]
        DB2[(관리자 표준<br/>다회용기)] --> AV[관리자 벡터들<br/>M개]
    end

    subgraph "유사도 계산"
        QV --> CS1[코사인 유사도<br/>dot product]
        UV --> CS1
        CS1 --> S1[사용자 최고 유사도<br/>threshold: 0.7]

        QV --> CS2[코사인 유사도<br/>dot product]
        AV --> CS2
        CS2 --> S2[관리자 최고 유사도<br/>threshold: 0.75]
    end

    subgraph "판정"
        S1 --> Judge{임계값 이상?}
        S2 --> Judge
        Judge -->|Yes| Match[✅ 매칭 성공]
        Judge -->|No| NoMatch[❌ 매칭 실패]
    end

    style QV fill:#e3f2fd
    style UV fill:#fff3e0
    style AV fill:#ffe0b2
    style Match fill:#c8e6c9
    style NoMatch fill:#ffcdd2
```

### 코사인 유사도 공식

L2 정규화된 벡터의 경우:
```
similarity = v1 · v2 = Σ(v1[i] * v2[i])
```

범위: -1.0 ~ 1.0 (높을수록 유사)

---

## 8. 센서 데이터 통합

IMU 센서 + GPS 활용 방식

```mermaid
flowchart LR
    subgraph "센서 수집"
        IMU[IMU 센서<br/>DeviceOrientation API]
        GPS[GPS<br/>Geolocation API]
    end

    subgraph "IMU 데이터"
        Beta[베타 각도<br/>X축 회전<br/>-180~180°]
        Gamma[감마 각도<br/>Y축 회전<br/>-90~90°]
    end

    subgraph "각도 검증"
        Check{정면 촬영?<br/>±15도}
        Check -->|Yes| OK[✅ 촬영 가능<br/>녹색 프레임]
        Check -->|No| Warn[⚠️ 경고<br/>빨간 프레임<br/>촬영은 가능]
    end

    subgraph "GPS 데이터"
        Lat[위도]
        Lng[경도]
        Acc[정확도]
    end

    subgraph "위치 검증"
        Dist[거리 계산<br/>Haversine]
        Dist --> DupCheck{1시간 이내<br/>500m 이내?}
        DupCheck -->|Yes| Dup[❌ 중복]
        DupCheck -->|No| Allow[✅ 허용]
    end

    IMU --> Beta
    IMU --> Gamma
    Beta --> Check
    Gamma --> Check

    GPS --> Lat
    GPS --> Lng
    GPS --> Acc
    Lat --> Dist
    Lng --> Dist

    style OK fill:#c8e6c9
    style Warn fill:#fff9c4
    style Dup fill:#ffcdd2
    style Allow fill:#c8e6c9
```

---

## 9. Docker 컨테이너 구성

서비스 간 네트워크 및 볼륨 관계

```mermaid
graph TB
    subgraph "Docker Network: zero-factory_app-network"
        subgraph "기존 서비스"
            Nginx[Nginx<br/>포트: 80, 443]
            WebServer[Next.js Web<br/>포트: 3001]
            APIServer[NestJS API<br/>포트: 3000]
            Database[(PostgreSQL<br/>+ PostGIS<br/>포트: 5432)]
        end
    end

    subgraph "Docker Network: ai-network"
        subgraph "AI 서비스"
            AIServer[FastAPI AI Server<br/>포트: 8000]
            Jupyter[Jupyter Lab<br/>포트: 8888]
            LabelStudio[Label Studio<br/>포트: 8080]
        end
    end

    subgraph "볼륨 (Volumes)"
        V1[models/<br/>학습된 모델]
        V2[uploads/<br/>업로드 이미지]
        V3[data/<br/>학습 데이터]
        V4[label-studio/<br/>어노테이션]
    end

    Nginx --> WebServer
    Nginx --> APIServer
    WebServer --> APIServer
    APIServer --> Database
    APIServer -.->|HTTP| AIServer

    AIServer --> V1
    AIServer --> V2
    Jupyter --> V1
    Jupyter --> V3
    LabelStudio --> V3
    LabelStudio --> V4

    style AIServer fill:#f8bbd0
    style Jupyter fill:#ffe0b2
    style LabelStudio fill:#ffe0b2
    style V1 fill:#e1bee7
    style V2 fill:#e1bee7
    style V3 fill:#e1bee7
    style V4 fill:#e1bee7
```

---

## 10. 개발 워크플로우

모델 학습부터 배포까지

```mermaid
flowchart TD
    Start([개발 시작]) --> Setup[환경 설정<br/>Docker Compose Up]

    Setup --> Data[데이터 수집]
    Data --> Annotate[Label Studio로<br/>어노테이션]

    Annotate --> Export[데이터셋 내보내기<br/>COCO/YOLO 포맷]

    Export --> Train1[Jupyter Notebook<br/>01_classifier.ipynb]
    Train1 --> Model1[classifier.pth]

    Export --> Train2[Jupyter Notebook<br/>03_beverage.ipynb]
    Train2 --> Model2[beverage_detector.pth]

    Export --> Embed[Jupyter Notebook<br/>02_embedding.ipynb<br/>사전학습 모델 사용]

    Model1 --> Deploy[모델 배포<br/>models/weights/]
    Model2 --> Deploy

    Deploy --> Restart[AI Server 재시작<br/>docker-compose restart]

    Restart --> Test[API 테스트<br/>/docs에서 Swagger UI]

    Test --> Integrate[백엔드 통합<br/>NestJS ↔ AI Server]

    Integrate --> Frontend[프론트엔드 연동<br/>Next.js]

    Frontend --> E2E[E2E 테스트<br/>전체 플로우 검증]

    E2E --> Production[프로덕션 배포]

    Production --> Monitor[모니터링<br/>로그 + 성능]

    Monitor --> Improve{개선 필요?}
    Improve -->|Yes| Data
    Improve -->|No| End([운영])

    style Start fill:#e1f5ff
    style Train1 fill:#fff9c4
    style Train2 fill:#fff9c4
    style Embed fill:#fff9c4
    style Deploy fill:#c8e6c9
    style Production fill:#c8e6c9
    style End fill:#c8e6c9
```

---

## 요약

### 핵심 플로우
1. **등록**: 촬영 → AI 분류 → 임베딩 저장
2. **인증**: 촬영 → 음료 검증 → 임베딩 비교 → 포인트 지급

### 주요 컴포넌트
- **Frontend**: Next.js (카메라, 센서)
- **Backend**: NestJS (비즈니스 로직)
- **AI Server**: FastAPI (모델 추론)
- **Database**: PostgreSQL (데이터 저장)

### 센서 활용
- **IMU**: 정면 촬영 가이드 (±15도)
- **GPS**: 중복 방지 (500m, 1시간)

### AI 모델
- **분류**: ResNet50 (다회용기 vs 일회용기)
- **임베딩**: CLIP (512차원 유사도)
- **음료**: MobileNetV3 (음료 유무)
