# 자동 배포 설정 가이드

이 문서는 GitHub에 push할 때 AWS EC2 서버에 자동으로 배포되도록 설정하는 방법을 설명합니다.

## 📋 사전 요구사항

- GitHub Repository 접근 권한 (Settings 메뉴 접근 가능)
- AWS EC2 인스턴스의 SSH 프라이빗 키 파일
- EC2 인스턴스가 실행 중이어야 함

## 🔧 설정 단계

### 1. SSH 프라이빗 키 확인

EC2 인스턴스에 접속할 때 사용하는 `.pem` 파일의 내용이 필요합니다.

```bash
# SSH 키 파일 내용 확인 (로컬 PC에서 실행)
cat ~/.ssh/your-ec2-key.pem
```

또는 EC2 인스턴스 내부에서 새로운 키를 생성할 수도 있습니다:

```bash
# EC2 서버에서 실행
ssh-keygen -t rsa -b 4096 -f ~/.ssh/github_actions_key -N ""

# 공개키를 authorized_keys에 추가
cat ~/.ssh/github_actions_key.pub >> ~/.ssh/authorized_keys

# 프라이빗 키 출력 (이것을 GitHub Secrets에 추가)
cat ~/.ssh/github_actions_key
```

### 2. GitHub Secrets 설정

1. GitHub 저장소 페이지로 이동
2. **Settings** → **Secrets and variables** → **Actions** 클릭
3. **New repository secret** 버튼 클릭
4. 다음 3개의 시크릿을 추가:

#### Secret 1: EC2_HOST
- **Name**: `EC2_HOST`
- **Value**: `43.201.190.116` (현재 EC2 퍼블릭 IP)

#### Secret 2: EC2_USERNAME
- **Name**: `EC2_USERNAME`
- **Value**: `ubuntu`

#### Secret 3: EC2_SSH_KEY
- **Name**: `EC2_SSH_KEY`
- **Value**: SSH 프라이빗 키 전체 내용 (-----BEGIN RSA PRIVATE KEY----- 부터 -----END RSA PRIVATE KEY----- 까지 전부)

예시:
```
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
(전체 키 내용)
...
-----END RSA PRIVATE KEY-----
```

### 3. 워크플로우 파일 Push

`.github/workflows/deploy.yml` 파일을 GitHub에 push합니다:

```bash
git add .github/workflows/deploy.yml
git commit -m "feat: add GitHub Actions auto-deployment workflow"
git push origin main
```

## 🚀 사용 방법

설정이 완료되면, `main` 브랜치에 push할 때마다 자동으로 배포가 실행됩니다:

```bash
# 코드 수정 후
git add .
git commit -m "your commit message"
git push origin main
```

GitHub Actions 탭에서 배포 진행 상황을 실시간으로 확인할 수 있습니다.

## 🔍 배포 프로세스

1. ✅ GitHub에 코드 push
2. ✅ GitHub Actions 워크플로우 자동 실행
3. ✅ EC2 서버에 SSH 접속
4. ✅ 최신 코드 pull (`git pull origin main`)
5. ✅ **Production 환경 설정 적용** (`.env.prod` → `.env`)
6. ✅ Docker 컨테이너 빌드
7. ✅ **Production 모드로 서비스 재시작** (`./start-prod.sh`)
   - HTTPS with SSL 적용
   - Production 최적화 설정
   - Let's Encrypt 인증서 사용
8. ✅ 헬스체크 수행
9. ✅ 배포 완료

> **참고**: 개발 서버와 배포 서버는 자동으로 환경이 구분되어 적용됩니다. 자세한 내용은 [환경 설정 가이드](./ENVIRONMENT_SETUP.md)를 참조하세요.

## 📊 배포 확인 방법

### GitHub에서 확인
- Repository → **Actions** 탭
- 최근 워크플로우 실행 상태 확인
- 로그 확인 가능

### 서버에서 확인
```bash
# EC2 서버에 접속
ssh -i your-key.pem ubuntu@43.201.190.116

# 활성 환경 확인
cat .env | head -n 1  # NODE_ENV=production 확인

# 컨테이너 상태 확인
docker compose ps

# 로그 확인
docker compose logs -f api
docker compose logs -f web
docker compose logs -f nginx

# API 헬스체크
curl http://localhost:3000/health
# 또는 Nginx를 통해
curl https://zeromap.store/api/health
```

### 브라우저에서 확인

**Production (배포 환경):**
- Web: https://zeromap.store
- API: https://zeromap.store/api

**Development (개발 환경):**
- Web: http://localhost 또는 http://localhost:3001
- API: http://localhost:3000

## 🔧 문제 해결

### 배포가 실패하는 경우

1. **SSH 연결 실패**
   - GitHub Secrets의 `EC2_SSH_KEY`가 올바른지 확인
   - EC2 Security Group에서 SSH 포트(22)가 열려있는지 확인
   - EC2 인스턴스가 실행 중인지 확인

2. **Git pull 실패**
   - EC2 서버에서 `git status` 확인
   - 로컬 변경사항이 있다면 `git stash` 실행

3. **Docker 빌드 실패**
   - EC2 서버에서 수동으로 `docker compose build` 실행
   - 디스크 공간 확인: `df -h`
   - Docker 로그 확인: `docker compose logs`

4. **권한 문제**
   ```bash
   # Docker 권한 확인
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

## 🎓 학습 포인트

이 자동 배포 설정을 통해 다음을 학습할 수 있습니다:

- ✅ **CI/CD 파이프라인**: GitHub Actions를 이용한 자동화
- ✅ **AWS EC2 관리**: 클라우드 서버 운영
- ✅ **SSH 인증**: 키 기반 보안 접속
- ✅ **Docker 컨테이너**: 컨테이너 기반 배포
- ✅ **시크릿 관리**: 민감한 정보 안전하게 관리
- ✅ **Infrastructure as Code**: 코드로 인프라 관리

## 📚 추가 개선 아이디어

더 학습하고 싶다면:

1. **알림 추가**: Slack/Discord에 배포 알림 전송
2. **테스트 통합**: 배포 전 자동 테스트 실행
3. **롤백 기능**: 배포 실패 시 이전 버전으로 자동 복구
4. **Blue-Green 배포**: 다운타임 없는 배포
5. **모니터링**: CloudWatch로 서버 상태 모니터링
6. **로드 밸런서**: ALB를 통한 트래픽 분산

## 🌍 환경별 배포 설정

프로젝트는 개발 환경과 배포 환경을 자동으로 구분합니다:

### Development (개발 서버)
```bash
# 개발 환경으로 배포
./start-dev.sh
```
- HTTP only (포트 80)
- 모든 서비스 포트 직접 노출
- Hot reload 활성화
- SSL 비활성화

### Production (배포 서버)
```bash
# 배포 환경으로 배포
./start-prod.sh
```
- HTTPS (포트 443) with SSL
- Nginx 프록시만 노출
- Production 최적화
- Let's Encrypt 자동 SSL 갱신

자세한 환경 설정 방법은 [환경 설정 가이드](./ENVIRONMENT_SETUP.md)를 참조하세요.

## 🔗 참고 자료

- [환경 설정 가이드](./ENVIRONMENT_SETUP.md) - 개발/배포 환경 상세 설정
- [GitHub Actions 문서](https://docs.github.com/en/actions)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [AWS EC2 가이드](https://docs.aws.amazon.com/ec2/)
