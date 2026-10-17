# scripts/smoke_test.ps1 -- one-shot health check of the whole stack (Windows PowerShell).
#   .\scripts\smoke_test.ps1
param(
  [string]$ML   = "http://localhost:8000",
  [string]$MCP  = "http://localhost:8001",
  [string]$Orch = "http://localhost:8002",
  [string]$UI   = "http://localhost:5173"
)
$failed = $false
function Check($name, $url) {
  try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 8 | Out-Null; Write-Host "  OK    $name  $url" }
  catch { Write-Host "  FAIL  $name  $url  ($($_.Exception.Message))"; $script:failed = $true }
}
Write-Host "1) Is each service up?"
Check "ML API       " "$ML/health"
Check "MCP server   " "$MCP/health"
Check "Orchestrator " "$Orch/health"
Check "Frontend     " "$UI/"

Write-Host "`n2) Orchestrator -> LLM, MCP and ML API (deep check)"
try { Invoke-RestMethod -Uri "$Orch/health/dependencies" -TimeoutSec 15 | ConvertTo-Json -Depth 6 }
catch { Write-Host "  FAIL $($_.Exception.Message)"; $failed = $true }

Write-Host "`n3) A real question (needs a configured LLM)"
$body = @{ message = "Predict the exam score for a student who studies 12.5 hours per week, has 88% attendance, scored 72, 75 and 70 in the three mock tests, belongs to the Medium income bracket, lives in Pune, enrolled on 2025-06-01, has shoe size 9 and lucky number 42." } | ConvertTo-Json
try {
  $r = Invoke-RestMethod -Uri "$Orch/chat" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 120
  Write-Host "  kind: $($r.kind) | provider: $($r.provider.name) $($r.provider.model) | trace: $($r.trace_id)"
  foreach ($p in $r.predictions) { Write-Host "  model output: $($p.label) -> $($p.predicted_final_score)" }
  Write-Host "  assistant says: $($r.assistant_message)"
} catch {
  Write-Host "  The chat request failed: $($_.Exception.Message)"
  if ($_.ErrorDetails.Message) { Write-Host "  $($_.ErrorDetails.Message)" }
  $failed = $true
}
if ($failed) { Write-Host "`nSMOKE TEST FAILED - see docs/TROUBLESHOOTING.md"; exit 1 } else { Write-Host "`nSMOKE TEST PASSED" }
