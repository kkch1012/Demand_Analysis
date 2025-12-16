# 가상환경 활성화 스크립트 (간편 버전)

# 프로젝트 루트로 이동
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

$venvName = "venv"
$activateScript = Join-Path $venvName "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "가상환경이 활성화되었습니다." -ForegroundColor Green
    Write-Host "비활성화: deactivate 또는 .\scripts\deactivate.ps1" -ForegroundColor Gray
} else {
    Write-Host "오류: 가상환경을 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "먼저 다음 명령어를 실행하세요: .\scripts\setup_venv.ps1" -ForegroundColor Yellow
}

