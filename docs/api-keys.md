# API Keys 설정 가이드

Zero Factory 프로젝트에서 사용하는 외부 API 키 설정 방법을 안내합니다.

## Kakao Map API Key

Kakao Map API는 지도 표시 및 장소 검색 기능에 사용됩니다.

### 1. API 키 발급

#### 1.1 Kakao Developers 계정 생성

1. [Kakao Developers](https://developers.kakao.com/)에 접속
2. 카카오 계정으로 로그인

#### 1.2 애플리케이션 생성

1. **내 애플리케이션** 메뉴 클릭
2. **애플리케이션 추가하기** 클릭
3. 앱 정보 입력:
   - 앱 이름: `Zero Factory` (또는 원하는 이름)
   - 사업자명: 개인 또는 회사명
4. **저장** 클릭

#### 1.3 JavaScript 키 확인

1. 생성한 애플리케이션 선택
2. **앱 키** 탭 클릭
3. **JavaScript 키** 복사
   - 예: `1234567890abcdef1234567890abcdef`

### 2. 플랫폼 등록

**매우 중요!** Kakao Map API는 등록된 도메인에서만 작동합니다.

#### 2.1 Web 플랫폼 등록

1. 애플리케이션 설정 페이지에서 **플랫폼** 메뉴 클릭
2. **Web 플랫폼 등록** 클릭
3. 사이트 도메인 등록:

**로컬 개발 환경**:
```
http://localhost:3001
```

**프로덕션 환경 (EC2)**:
```
http://43.201.190.116:3001
```

또는 도메인이 있다면:
```
http://yourdomain.com
https://yourdomain.com
```

4. **저장** 클릭

> 주의: 여러 환경에서 사용하려면 각 도메인을 모두 등록해야 합니다.

#### 2.2 등록 확인

플랫폼 설정 페이지에서 등록된 도메인 목록 확인:

```
✓ http://localhost:3001
✓ http://43.201.190.116:3001
```

### 3. 환경 변수 설정

#### 3.1 .env 파일에 API 키 추가

프로젝트 루트의 `.env` 파일을 열고 다음 내용 추가:

```bash
NEXT_PUBLIC_KAKAO_MAP_KEY=your_javascript_key_here
```

예시:
```bash
NEXT_PUBLIC_KAKAO_MAP_KEY=1234567890abcdef1234567890abcdef
```

#### 3.2 Docker 재빌드 (필수!)

Next.js는 빌드 시점에 환경 변수를 코드에 포함시키므로, 환경 변수 변경 후 **반드시 재빌드**가 필요합니다:

```bash
# 1. 컨테이너 중지
docker compose down

# 2. 재빌드 및 시작
docker compose up -d --build
```

### 4. 동작 확인

#### 4.1 브라우저에서 확인

1. http://localhost:3001 접속
2. 지도가 정상적으로 표시되는지 확인

#### 4.2 API 키 포함 확인

빌드된 HTML에 API 키가 포함되었는지 확인:

```bash
curl -s http://localhost:3001 | grep -o "appkey=[^&\"]*"
```

출력 예시:
```
appkey=1234567890abcdef1234567890abcdef
```

#### 4.3 개발자 도구에서 확인

1. 브라우저에서 F12 키 눌러 개발자 도구 열기
2. **Console** 탭 확인
3. 에러가 없으면 정상 작동

### 5. 문제 해결

#### 문제: 지도가 로드되지 않음

**체크리스트**:

1. ✓ JavaScript 키를 올바르게 복사했는가?
2. ✓ 현재 접속 URL이 플랫폼에 등록되어 있는가?
3. ✓ `.env` 파일에 API 키가 올바르게 입력되었는가?
4. ✓ Docker를 재빌드했는가?

**디버깅**:

```bash
# 1. 환경 변수 확인
cat .env | grep KAKAO

# 2. 빌드된 파일에 API 키 포함 확인
curl -s http://localhost:3001 | grep -o "appkey=[^&\"]*"

# 3. Docker 로그 확인
docker compose logs web | grep -i kakao

# 4. 브라우저 콘솔에서 에러 확인
```

일반적인 에러 메시지:

- `Failed to load resource`: 플랫폼 미등록 또는 네트워크 문제
- `Kakao Map API error`: API 키 오류
- `CORS error`: 플랫폼 미등록

자세한 내용은 [트러블슈팅 가이드](./troubleshooting.md#kakao-map-api-관련)를 참조하세요.

### 6. 보안 주의사항

#### 6.1 API 키 관리

- ✓ `.env` 파일은 절대 Git에 커밋하지 마세요
- ✓ `.gitignore`에 `.env` 파일이 포함되어 있는지 확인
- ✓ GitHub 등 공개 저장소에 API 키가 노출되지 않도록 주의

#### 6.2 .gitignore 확인

`.gitignore` 파일에 다음 내용이 포함되어 있는지 확인:

```
.env
.env.local
.env*.local
```

#### 6.3 API 키 재발급

API 키가 노출되었다면:

1. [Kakao Developers 콘솔](https://developers.kakao.com/) 접속
2. 애플리케이션 선택
3. **앱 키** 탭에서 **키 재발급** 클릭
4. 새 키로 `.env` 파일 업데이트
5. Docker 재빌드

### 7. 프로덕션 배포

#### 7.1 EC2 서버에서 설정

```bash
# 1. EC2 서버 접속
ssh ubuntu@your-ec2-ip

# 2. 프로젝트 디렉토리로 이동
cd /home/ubuntu/zero-factory

# 3. .env 파일 수정
nano .env

# 4. NEXT_PUBLIC_KAKAO_MAP_KEY 추가/수정
# 5. 저장 (Ctrl+O, Enter, Ctrl+X)

# 6. Docker 재빌드
docker compose down
docker compose up -d --build
```

#### 7.2 자동 배포 시 주의사항

GitHub Actions를 통해 자동 배포하는 경우:

- EC2 서버의 `.env` 파일은 자동으로 업데이트되지 않습니다
- 최초 1회 수동으로 `.env` 파일을 설정해야 합니다
- 이후 코드 변경사항만 자동 배포됩니다

## 향후 추가될 수 있는 API

현재는 Kakao Map API만 사용하지만, 향후 다음 API들이 추가될 수 있습니다:

- Google Analytics
- Firebase
- AWS S3
- 기타 서드파티 API

API가 추가되면 이 문서를 업데이트하겠습니다.

## 참고 자료

- [Kakao Developers 문서](https://developers.kakao.com/docs/latest/ko/local/dev-guide)
- [Kakao Map JavaScript API](https://apis.map.kakao.com/web/)
- [Next.js 환경 변수 가이드](https://nextjs.org/docs/app/building-your-application/configuring/environment-variables)
