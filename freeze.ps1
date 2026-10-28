# Commit every snapshot in data/ and tag the frozen price set.
# Run once by the Task Scheduler job KalshiComboFreeze; output goes to freeze.log.
param(
    [string]$Tag = 'prices-frozen-2026-11-01',
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$Email
)

Set-Location $PSScriptRoot
Start-Transcript -Path (Join-Path $PSScriptRoot 'freeze.log') -Append | Out-Null
try {
    $id = @('-c', "user.name=$Name", '-c', "user.email=$Email")

    git rev-parse -q --verify "refs/tags/$Tag" | Out-Null
    if ($LASTEXITCODE -eq 0) { throw "tag $Tag already exists" }

    git add data
    if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        $n = (Get-ChildItem data -Filter 'snapshot_*.json').Count
        git @id commit -q -m "Freeze Kalshi snapshots ($n files) for $Tag"
        if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
    }

    git @id tag -a $Tag -m "Kalshi prices frozen $(Get-Date -Format o)"
    if ($LASTEXITCODE -ne 0) { throw 'git tag failed' }
    git log -1 --oneline --decorate
} catch {
    Write-Output "FREEZE FAILED: $_"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
