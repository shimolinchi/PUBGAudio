$ErrorActionPreference = 'Stop'
$moduleRoot = 'C:\Projects\PUBGAssistant-cpp\modules\PUBGAudio'
$datasetRoot = Join-Path $moduleRoot 'datasets'
$newDataset = Join-Path $datasetRoot 'spatial-v6-24h-20260908'
$receipt = Get-Content -LiteralPath (Join-Path $moduleRoot 'outputs\reports\v6_replacement_cleanup_20260908.json') -Raw | ConvertFrom-Json
$verification = Get-Content -LiteralPath (Join-Path $newDataset 'verification.json') -Raw | ConvertFrom-Json
$coverage = Get-Content -LiteralPath (Join-Path $newDataset 'coverage_verification.json') -Raw | ConvertFrom-Json
if ($verification.status -ne 'passed' -or $verification.clips -ne 720 -or $verification.seconds -ne 86400) {
    throw 'The full new 24-hour dataset must pass verification before cleanup.'
}
if ($coverage.status -ne 'passed' -or -not $coverage.every_evaluation_gun_type_and_role_seen_in_training -or
    -not $coverage.all_available_gun_type_roles_audible_in_training -or
    -not $coverage.all_selected_external_target_types_audible_in_training -or
    -not $coverage.training_external_types_cover_four_quadrants) {
    throw 'New dataset coverage has not passed.'
}
foreach ($split in @('train','validation','test')) {
    $hash = (Get-FileHash -LiteralPath (Join-Path $newDataset "manifest_$split.jsonl") -Algorithm SHA256).Hash
    if ($hash -ne $verification.manifests_sha256.$split) { throw "New $split manifest changed." }
}
$archive = 'C:\Projects\PUBGAssistant-cpp\modules\PUBGAudio\datasets\registry\v5_retired_training_metadata_20260908.zip'
if ($receipt.archive -ne $archive -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $receipt.archive_sha256) {
    throw 'Verified retirement archive is missing or changed.'
}
$names = @('light-v5_3-20260907','light-v5_3-gunvehicle-20260907',
    'cache-light-v5_3-20260907','cache-light-v5_3-gunvehicle-20260907','scenario-small-20260907','scenario-smoke-v4')
if (@($receipt.targets.PSObject.Properties).Count -ne $names.Count) { throw 'Unexpected cleanup target count.' }
$pythonProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'")
foreach ($name in $names) {
    $path = Join-Path $datasetRoot $name
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if ($resolved -ne $path -or (Split-Path -Parent $resolved) -ne $datasetRoot) { throw "Unexpected target: $resolved" }
    $saved = $receipt.targets.$name
    if ($null -eq $saved -or $saved.path -ne $resolved) { throw 'Target is not in the verified receipt.' }
    $items = @((Get-Item -LiteralPath $resolved -Force)) + @(Get-ChildItem -LiteralPath $resolved -Recurse -Force)
    if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Reparse point in target.' }
    $files = @($items | Where-Object { -not $_.PSIsContainer })
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($files.Count -ne $saved.files -or $bytes -ne $saved.bytes) { throw "Old data changed since archive: $name" }
    foreach ($process in $pythonProcesses) {
        if ($process.CommandLine -and $process.CommandLine.Contains($name)) { throw "A Python process still uses $name" }
    }
}
foreach ($item in $receipt.retained_models.PSObject.Properties) {
    $path = Join-Path $moduleRoot $item.Name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.Value.sha256) { throw 'A retained model changed.' }
}
[pscustomobject]@{status='cleanup_targets_verified_no_deletion';targets=$names;bytes=$receipt.total_bytes_to_delete;
    archive_sha256=$receipt.archive_sha256;retained_models=$receipt.retained_models.PSObject.Properties.Count} | ConvertTo-Json -Compress
