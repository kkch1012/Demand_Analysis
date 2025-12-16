# Train.csv 자동 분석 스크립트
param(
    [string]$InputFile = "filesystem\Train.csv",
    [string]$OutputFile = "analysis_result.txt"
)

# UTF-8 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
try { chcp 65001 | Out-Null } catch {}

Write-Host "=== 데이터 분석 시작 ===" -ForegroundColor Cyan

# 파일 경로 확인
$fullPath = Join-Path $PSScriptRoot $InputFile
if (-not (Test-Path $fullPath)) {
    Write-Host "오류: 파일을 찾을 수 없습니다: $fullPath" -ForegroundColor Red
    exit 1
}

Write-Host "`n파일 읽는 중: $fullPath" -ForegroundColor Yellow

# CSV 파일 읽기
$data = Import-Csv $fullPath

# 결과를 저장할 문자열 배열
$result = @()

# 기본 통계
$totalRows = $data.Count
$columns = $data[0].PSObject.Properties.Name

Write-Host "`n=== 기본 정보 ===" -ForegroundColor Cyan
$result += "=== 기본 정보 ==="
Write-Host "총 행 수: $totalRows"
$result += "총 행 수: $totalRows"
Write-Host "컬럼 수: $($columns.Count)"
$result += "컬럼 수: $($columns.Count)"
Write-Host "컬럼 목록: $($columns -join ', ')"
$result += "컬럼 목록: $($columns -join ', ')"

# 숫자형 컬럼 분석
$numericColumns = @('Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales', 'Outlet_Establishment_Year')

Write-Host "`n=== 숫자형 컬럼 통계 ===" -ForegroundColor Cyan
$result += ""
$result += "=== 숫자형 컬럼 통계 ==="
foreach ($col in $numericColumns) {
    if ($columns -contains $col) {
        $values = $data | Where-Object { $_.$col -ne '' -and $_.$col -ne $null } | 
                  ForEach-Object { [double]$_.$col }
        if ($values.Count -gt 0) {
            $mean = ($values | Measure-Object -Average).Average
            $min = ($values | Measure-Object -Minimum).Minimum
            $max = ($values | Measure-Object -Maximum).Maximum
            $sum = ($values | Measure-Object -Sum).Sum
            
            Write-Host "`n$col :" -ForegroundColor Yellow
            $result += ""
            $result += "$col :"
            $meanStr = "  평균: $([math]::Round($mean, 2))"
            $minStr = "  최소: $([math]::Round($min, 2))"
            $maxStr = "  최대: $([math]::Round($max, 2))"
            $sumStr = "  합계: $([math]::Round($sum, 2))"
            Write-Host $meanStr
            Write-Host $minStr
            Write-Host $maxStr
            Write-Host $sumStr
            $result += $meanStr
            $result += $minStr
            $result += $maxStr
            $result += $sumStr
        }
    }
}

# 범주형 컬럼 분석
$categoricalColumns = @('Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type')

Write-Host "`n=== 범주형 컬럼 통계 ===" -ForegroundColor Cyan
$result += ""
$result += "=== 범주형 컬럼 통계 ==="
foreach ($col in $categoricalColumns) {
    if ($columns -contains $col) {
        $groups = $data | Group-Object $col | Sort-Object Count -Descending
        Write-Host "`n$col :" -ForegroundColor Yellow
        $result += ""
        $result += "$col :"
        foreach ($group in $groups | Select-Object -First 10) {
            $percentage = [math]::Round(($group.Count / $totalRows) * 100, 2)
            $line = "  $($group.Name): $($group.Count) ($percentage%)"
            Write-Host $line
            $result += $line
        }
    }
}

# 결과를 파일로 저장
$outputPath = Join-Path $PSScriptRoot $OutputFile
$result | Out-File -FilePath $outputPath -Encoding UTF8
Write-Host "`n=== 결과 저장 ===" -ForegroundColor Cyan
Write-Host "결과가 저장되었습니다: $outputPath" -ForegroundColor Green

Write-Host "`n=== 분석 완료 ===" -ForegroundColor Cyan

