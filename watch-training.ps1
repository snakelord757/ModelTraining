$ErrorActionPreference = 'Stop'
$statusPath = Join-Path $PSScriptRoot 'training-status.json'
function Format-Duration($seconds) {
    if ($null -eq $seconds) { return '--:--:--' }
    $span = [TimeSpan]::FromSeconds([Math]::Max(0, [double]$seconds))
    return ('{0:00}:{1:00}:{2:00}' -f [int][Math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds)
}
while ($true) {
    if (Test-Path -LiteralPath $statusPath) {
        try { $s = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json } catch { Start-Sleep -Seconds 1; continue }
        Clear-Host
        Write-Host ('Stage: {0}' -f $s.stage)
        if ($null -ne $s.percent) { Write-Host ('Progress: {0:N2}%  |  optimizer steps: {1}/{2}' -f $s.percent, $s.step, $s.total_steps) }
        Write-Host ('Training elapsed: {0}  |  estimated remaining: {1}' -f (Format-Duration $s.training_elapsed_seconds), (Format-Duration $s.eta_seconds))
        Write-Host ('Total elapsed including preparation: {0}' -f (Format-Duration $s.total_elapsed_seconds))
        Write-Host ('Loss: {0}' -f $s.loss)
        $age = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $s.updated_at
        if ($age -gt 10) { Write-Host 'Status is stale: training may have stopped.' -ForegroundColor Yellow }
        if ($s.error) { Write-Host $s.error -ForegroundColor Red }
        if ($s.stage -in @('completed','failed','interrupted')) { break }
    } else { Write-Host 'Waiting for training to start...' }
    Start-Sleep -Seconds 1
}
