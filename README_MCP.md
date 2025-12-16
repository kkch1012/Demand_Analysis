# MCP 서버 자동화 가이드

## 📋 목차
1. [빠른 시작](#빠른-시작)
2. [주요 명령어](#주요-명령어)
3. [스크립트 설명](#스크립트-설명)
4. [자동화 방법](#자동화-방법)

## 🚀 빠른 시작

### 1. MCP 서버 상태 확인
```powershell
.\check_mcp.ps1
```

### 2. 데이터 분석 실행
```powershell
.\analyze_data.ps1
```

### 3. PATH 환경 변수 설정 (한 번만 실행)
```powershell
# 관리자 권한으로 실행
.\setup_path.ps1
```

## 🔧 주요 명령어

### MCP 서버 관련

#### 1. MCP 서버 상태 확인
```powershell
# Node.js 확인
node --version

# npx 확인
npx --version

# uvx 확인
uvx --version
```

#### 2. Filesystem MCP 서버 수동 실행 (테스트용)
```powershell
$env:PATH += ";C:\Program Files\nodejs"
npx -y @modelcontextprotocol/server-filesystem "C:\Users\human\project\Demand_Analysis\filesystem"
```

#### 3. YouTube MCP 서버 수동 실행 (테스트용)
```powershell
$env:PATH += ";C:\Users\human\AppData\Local\Programs\Python\Python313\Scripts"
uvx --from git+https://github.com/adhikasp/mcp-youtube mcp-youtube
```

#### 4. Playwright MCP 서버 수동 실행 (테스트용)
```powershell
$env:PATH += ";C:\Program Files\nodejs"
npx @playwright/mcp@latest
```

### 파일 작업 관련

#### 1. Filesystem 디렉토리 확인
```powershell
Get-ChildItem "C:\Users\human\project\Demand_Analysis\filesystem"
```

#### 2. CSV 파일 읽기
```powershell
Import-Csv "C:\Users\human\project\Demand_Analysis\filesystem\Train.csv" | Select-Object -First 10
```

#### 3. 파일 통계 확인
```powershell
$data = Import-Csv "C:\Users\human\project\Demand_Analysis\filesystem\Train.csv"
$data.Count
$data[0].PSObject.Properties.Name
```

## 📝 스크립트 설명

### 1. `check_mcp.ps1`
- MCP 서버 실행에 필요한 도구들 확인
- Filesystem 디렉토리 존재 여부 확인
- 파일 목록 표시

**사용법:**
```powershell
.\check_mcp.ps1
```

### 2. `analyze_data.ps1`
- Train.csv 파일 자동 분석
- 기본 통계 정보 출력
- 숫자형/범주형 컬럼 분석
- 결과를 파일로 저장

**사용법:**
```powershell
# 기본 사용
.\analyze_data.ps1

# 출력 파일 지정
.\analyze_data.ps1 -OutputFile "my_result.txt"
```

### 3. `setup_path.ps1`
- PATH 환경 변수에 Node.js와 Python Scripts 경로 추가
- 관리자 권한 필요

**사용법:**
```powershell
# 관리자 권한 PowerShell에서 실행
.\setup_path.ps1
```

## 🤖 자동화 방법

### 방법 1: 작업 스케줄러 사용

1. 작업 스케줄러 열기
2. 기본 작업 만들기
3. 트리거: 로그온 시 또는 특정 시간
4. 작업: PowerShell 스크립트 실행
   ```
   프로그램: powershell.exe
   인수: -ExecutionPolicy Bypass -File "C:\Users\human\project\Demand_Analysis\check_mcp.ps1"
   ```

### 방법 2: 배치 파일 생성

`run_analysis.bat` 파일 생성:
```batch
@echo off
cd /d "C:\Users\human\project\Demand_Analysis"
powershell.exe -ExecutionPolicy Bypass -File .\analyze_data.ps1
pause
```

### 방법 3: PowerShell 프로필에 추가

PowerShell 프로필 편집:
```powershell
notepad $PROFILE
```

다음 내용 추가:
```powershell
# MCP 관련 함수
function Check-MCP {
    & "C:\Users\human\project\Demand_Analysis\check_mcp.ps1"
}

function Analyze-Data {
    & "C:\Users\human\project\Demand_Analysis\analyze_data.ps1"
}

Set-Alias -Name mcp-check -Value Check-MCP
Set-Alias -Name analyze -Value Analyze-Data
```

사용법:
```powershell
mcp-check
analyze
```

## 📌 참고사항

1. **MCP 서버는 Cursor가 자동으로 실행합니다**
   - Cursor를 재시작하면 MCP 서버가 자동으로 시작됩니다
   - 수동 실행은 테스트 목적으로만 사용하세요

2. **PATH 환경 변수**
   - PowerShell을 재시작하면 PATH 변경사항이 적용됩니다
   - 또는 `setup_path.ps1`을 실행하여 영구적으로 설정하세요

3. **파일 권한**
   - filesystem 디렉토리에 대한 읽기/쓰기 권한이 필요합니다

## 🔍 문제 해결

### npx를 찾을 수 없을 때
```powershell
$env:PATH += ";C:\Program Files\nodejs"
```

### uvx를 찾을 수 없을 때
```powershell
$env:PATH += ";C:\Users\human\AppData\Local\Programs\Python\Python313\Scripts"
```

### MCP 서버가 작동하지 않을 때
1. Cursor 재시작
2. `check_mcp.ps1` 실행하여 상태 확인
3. MCP 설정 파일 확인: `C:\Users\human\.cursor\mcp.json`

