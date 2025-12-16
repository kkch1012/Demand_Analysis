# MCP Server Status Check Script
# UTF-8 Encoding Setup
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
try { chcp 65001 | Out-Null } catch {}

Write-Host "=== MCP 서버 상태 확인 ===" -ForegroundColor Cyan

# Node.js 경로 추가
$env:PATH += ";C:\Program Files\nodejs"

# Python Scripts 경로 추가
$env:PATH += ";C:\Users\human\AppData\Local\Programs\Python\Python313\Scripts"

Write-Host "`n1. Node.js 확인:" -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "   ✓ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Node.js가 설치되지 않았습니다." -ForegroundColor Red
}

Write-Host "`n2. npx 확인:" -ForegroundColor Yellow
try {
    $npxVersion = npx --version
    Write-Host "   ✓ npx: $npxVersion" -ForegroundColor Green
} catch {
    Write-Host "   ✗ npx를 사용할 수 없습니다." -ForegroundColor Red
}

Write-Host "`n3. uvx 확인:" -ForegroundColor Yellow
try {
    $uvxVersion = uvx --version
    Write-Host "   ✓ uvx: $uvxVersion" -ForegroundColor Green
} catch {
    Write-Host "   ✗ uvx를 사용할 수 없습니다." -ForegroundColor Red
}

Write-Host "`n4. Filesystem 디렉토리 확인:" -ForegroundColor Yellow
$fsPath = "C:\Users\human\project\Demand_Analysis\filesystem"
if (Test-Path $fsPath) {
    $files = Get-ChildItem $fsPath
    Write-Host "   ✓ 디렉토리 존재: $fsPath" -ForegroundColor Green
    Write-Host "   파일 목록:" -ForegroundColor Gray
    foreach ($file in $files) {
        Write-Host "     - $($file.Name)" -ForegroundColor Gray
    }
} else {
    Write-Host "   ✗ 디렉토리가 없습니다: $fsPath" -ForegroundColor Red
    Write-Host "   디렉토리를 생성합니다..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $fsPath -Force | Out-Null
    Write-Host "   ✓ 디렉토리 생성 완료" -ForegroundColor Green
}

Write-Host "`n=== 확인 완료 ===" -ForegroundColor Cyan
Write-Host "`n참고: MCP 서버는 Cursor가 자동으로 실행합니다." -ForegroundColor Gray
Write-Host "Cursor를 재시작하면 MCP 서버가 자동으로 시작됩니다." -ForegroundColor Gray

