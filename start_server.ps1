# PharmAgentAI — Start Server
# Run this script to start the API + Frontend on http://127.0.0.1:8080
# Usage: .\start_server.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   PharmAgentAI API Server" -ForegroundColor Cyan
Write-Host "   http://127.0.0.1:8080" -ForegroundColor Green
Write-Host "   UI: http://127.0.0.1:8080/ui/" -ForegroundColor Green
Write-Host "   Docs: http://127.0.0.1:8080/docs" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

.\venv\Scripts\uvicorn src.api.server:app --host 127.0.0.1 --port 8080 --reload
