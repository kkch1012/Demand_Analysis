# 가상환경 설정 스크립트

Write-Host "=== Python 가상환경 설정 ===" -ForegroundColor Cyan

# 가상환경 이름
$venvName = "venv"

# 프로젝트 루트로 이동
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

# 가상환경이 이미 존재하는지 확인
if (Test-Path $venvName) {
    Write-Host "`n경고: 가상환경 '$venvName'이 이미 존재합니다." -ForegroundColor Yellow
    $response = Read-Host "기존 가상환경을 삭제하고 새로 만들까요? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "기존 가상환경 삭제 중..." -ForegroundColor Yellow
        Remove-Item -Path $venvName -Recurse -Force
    } else {
        Write-Host "기존 가상환경을 사용합니다." -ForegroundColor Green
        exit 0
    }
}

# Python 버전 확인
Write-Host "`nPython 버전 확인 중..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "  $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  오류: Python이 설치되어 있지 않습니다." -ForegroundColor Red
    Write-Host "  Python을 먼저 설치해주세요." -ForegroundColor Red
    exit 1
}

# 가상환경 생성
Write-Host "`n가상환경 생성 중..." -ForegroundColor Yellow
python -m venv $venvName

if (-not (Test-Path $venvName)) {
    Write-Host "  오류: 가상환경 생성 실패" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ 가상환경 생성 완료" -ForegroundColor Green

# 가상환경 활성화 스크립트 경로
$activateScript = Join-Path $venvName "Scripts\Activate.ps1"

# 가상환경 활성화
Write-Host "`n가상환경 활성화 중..." -ForegroundColor Yellow
& $activateScript

# pip 업그레이드
Write-Host "`npip 업그레이드 중..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# requirements.txt 설치
if (Test-Path "requirements.txt") {
    Write-Host "`n패키지 설치 중..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "  ✓ 패키지 설치 완료" -ForegroundColor Green
} else {
    Write-Host "`n경고: requirements.txt 파일을 찾을 수 없습니다." -ForegroundColor Yellow
}

Write-Host "`n=== 가상환경 설정 완료 ===" -ForegroundColor Cyan
Write-Host "`n가상환경 활성화 방법:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "`n또는" -ForegroundColor Yellow
Write-Host "  .\scripts\activate.ps1" -ForegroundColor Gray

