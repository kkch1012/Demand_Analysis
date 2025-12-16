# 유틸리티 스크립트

이 폴더에는 프로젝트 개발/관리용 유틸리티 스크립트들이 있습니다.

## 📋 스크립트 목록

### 가상환경 관리

- **`setup_venv.ps1`** - 가상환경 생성 및 패키지 설치
- **`activate.ps1`** - 가상환경 활성화
- **`deactivate.ps1`** - 가상환경 비활성화
- **`run.ps1`** - 가상환경에서 메인 스크립트 실행

## 🚀 사용 방법

### 프로젝트 루트에서 실행

```powershell
# 1. 가상환경 생성 및 패키지 설치
.\scripts\setup_venv.ps1

# 2. 가상환경 활성화
.\scripts\activate.ps1

# 3. 메인 스크립트 실행 (가상환경 자동 활성화)
.\scripts\run.ps1

# 4. 가상환경 비활성화
.\scripts\deactivate.ps1
```

### 직접 가상환경 활성화

```powershell
.\venv\Scripts\Activate.ps1
```

## ⚠️ 주의사항

- PowerShell 실행 정책 오류가 나면:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

- 모든 스크립트는 프로젝트 루트에서 실행해야 합니다.

