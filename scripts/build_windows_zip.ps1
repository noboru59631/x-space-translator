[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$version = "v0.1.3-beta"
$assetName = "X-Space-Translator-Windows-$version.zip"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distDir = Join-Path $projectRoot "dist"
$assetPath = Join-Path $distDir $assetName
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) (
    "x-space-translator-package-" + [Guid]::NewGuid().ToString("N")
)
$stagingDir = Join-Path $temporaryRoot "staging"
$verificationDir = Join-Path $temporaryRoot "verify"

try {
    New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null
    New-Item -ItemType Directory -Path $distDir -Force | Out-Null

    $includes = @(
        ".env.example",
        "app",
        "LICENSE",
        "README.md",
        "README_FIRST.txt",
        "requirements.txt",
        "requirements-gpu.txt",
        "setup.bat",
        "start.bat"
    )
    $trackedFiles = @(& git -C $projectRoot ls-files -- $includes)
    if ($LASTEXITCODE -ne 0 -or $trackedFiles.Count -eq 0) {
        throw "Could not read tracked distribution files from Git."
    }

    foreach ($relativePath in $trackedFiles) {
        $sourcePath = Join-Path $projectRoot $relativePath
        $destinationPath = Join-Path $stagingDir $relativePath
        $destinationDir = Split-Path -Parent $destinationPath
        New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    }

    foreach ($requiredFile in @("setup.bat", "start.bat", "README_FIRST.txt")) {
        if (-not (Test-Path -LiteralPath (Join-Path $stagingDir $requiredFile))) {
            throw "Required distribution file is missing: $requiredFile"
        }
    }

    if (Test-Path -LiteralPath $assetPath) {
        Remove-Item -LiteralPath $assetPath -Force
    }
    Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $assetPath -CompressionLevel Optimal

    Expand-Archive -LiteralPath $assetPath -DestinationPath $verificationDir
    $forbiddenNames = @(
        ".env", ".git", ".venv", "__pycache__", ".pytest_cache",
        ".ruff_cache", "data", "temp", "logs"
    )
    $forbiddenExtensions = @(
        ".db", ".sqlite", ".sqlite3", ".mp3", ".wav", ".m4a", ".mp4",
        ".webm", ".flac", ".ogg", ".bin", ".safetensors", ".pt", ".pth",
        ".ckpt", ".log"
    )
    $forbiddenItems = @(
        Get-ChildItem -LiteralPath $verificationDir -Recurse -Force | Where-Object {
            $_.Name -in $forbiddenNames -or $_.Extension -in $forbiddenExtensions
        }
    )
    if ($forbiddenItems.Count -gt 0) {
        throw "Forbidden files are present in the ZIP: $($forbiddenItems.FullName -join ', ')"
    }

    $secretFiles = @()
    foreach ($file in Get-ChildItem -LiteralPath $verificationDir -Recurse -File) {
        $content = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content -match "hf_[A-Za-z0-9]{20,}|gh[opusr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----") {
            $secretFiles += $file.FullName
        }
    }
    if ($secretFiles.Count -gt 0) {
        throw "Possible secrets are present in the ZIP: $($secretFiles -join ', ')"
    }

    $asset = Get-Item -LiteralPath $assetPath
    Write-Host "Windows ZIP created: $($asset.FullName)"
    Write-Host ("ZIP size: {0:N2} MB" -f ($asset.Length / 1MB))
}
finally {
    $resolvedTemp = [IO.Path]::GetFullPath($temporaryRoot)
    $systemTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($resolvedTemp.StartsWith($systemTemp, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedTemp).StartsWith("x-space-translator-package-")) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
