# 가상환경 비활성화 스크립트

if (Get-Command deactivate -ErrorAction SilentlyContinue) {
    deactivate
    Write-Host "가상환경이 비활성화되었습니다." -ForegroundColor Green
} else {
    Write-Host "가상환경이 활성화되어 있지 않습니다." -ForegroundColor Yellow
}

