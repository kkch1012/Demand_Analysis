# PATH 환경 변수 영구 설정 스크립트
# 관리자 권한으로 실행 필요
# UTF-8 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "=== PATH 환경 변수 설정 ===" -ForegroundColor Cyan

$pathsToAdd = @(
    "C:\Program Files\nodejs",
    "C:\Users\human\AppData\Local\Programs\Python\Python313\Scripts"
)

$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

Write-Host "`n현재 사용자 PATH:" -ForegroundColor Yellow
$currentPath -split ';' | ForEach-Object { Write-Host "  $_" }

$updated = $false
foreach ($path in $pathsToAdd) {
    if ($currentPath -notlike "*$path*") {
        Write-Host "`n추가 중: $path" -ForegroundColor Yellow
        $currentPath += ";$path"
        $updated = $true
    } else {
        Write-Host "`n이미 존재: $path" -ForegroundColor Gray
    }
}

if ($updated) {
    [Environment]::SetEnvironmentVariable("Path", $currentPath, "User")
    Write-Host "`n✓ PATH 환경 변수가 업데이트되었습니다." -ForegroundColor Green
    Write-Host "`n주의: 변경사항을 적용하려면 PowerShell을 재시작하세요." -ForegroundColor Yellow
} else {
    Write-Host "`n모든 경로가 이미 설정되어 있습니다." -ForegroundColor Green
}

Write-Host "`n=== 설정 완료 ===" -ForegroundColor Cyan

