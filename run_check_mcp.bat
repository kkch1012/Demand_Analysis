@echo off
chcp 65001 >nul
powershell.exe -ExecutionPolicy Bypass -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $OutputEncoding = [System.Text.Encoding]::UTF8; & '%~dp0check_mcp.ps1'"
pause


