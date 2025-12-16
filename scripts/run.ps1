# 가상환경에서 메인 스크립트 실행

# 프로젝트 루트로 이동
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

$venvName = "venv"
$activateScript = Join-Path $venvName "Scripts\Activate.ps1"

# 가상환경이 없으면 생성
if (-not (Test-Path $activateScript)) {
    Write-Host "가상환경이 없습니다. 생성 중..." -ForegroundColor Yellow
    & .\scripts\setup_venv.ps1
}

# 가상환경 활성화
& $activateScript

# 메인 스크립트 실행
Write-Host "`n메인 스크립트 실행 중..." -ForegroundColor Cyan
python main.py $args

