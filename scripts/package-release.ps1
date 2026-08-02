$ErrorActionPreference = 'Stop'

$ReleaseTag = '0.2.0-rtx4090-v1'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$DistRoot = Join-Path $RepoRoot 'dist'
$LinuxName = "ninfer-rtx4090-qwen3.6-27b-linux-x64-$ReleaseTag"
$LinuxDir = Join-Path $DistRoot $LinuxName
$LinuxArchive = Join-Path $DistRoot "$LinuxName.tar.gz"
$LinuxBuild = Join-Path $RepoRoot 'build-sm89'

function Reset-PackageDirectory([string] $Path) {
    $absolute = [System.IO.Path]::GetFullPath($Path)
    $distAbsolute = [System.IO.Path]::GetFullPath($DistRoot) +
        [System.IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($distAbsolute, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to reset a path outside dist: $absolute"
    }
    if (Test-Path -LiteralPath $absolute) {
        Remove-Item -LiteralPath $absolute -Recurse -Force
    }
    New-Item -ItemType Directory -Path $absolute | Out-Null
}

function Write-PackageHashes([string] $Directory) {
    $lines = Get-ChildItem -LiteralPath $Directory -File |
        Sort-Object Name |
        ForEach-Object {
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            "$hash  $($_.Name)"
        }
    Set-Content -LiteralPath (Join-Path $Directory 'SHA256SUMS.txt') `
        -Value $lines -Encoding ascii
}

$products = @(
    (Join-Path $LinuxBuild 'apps/ninfer'),
    (Join-Path $LinuxBuild 'apps/ninfer-serve'),
    (Join-Path $LinuxBuild 'bench/ninfer_bench')
)
foreach ($required in $products) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required RTX 4090 Linux product is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
Get-ChildItem -LiteralPath $DistRoot |
    Where-Object {
        $_.Name -like 'ninfer-rtx4090-qwen3.6-27b-linux-x64-*' -and
        $_.Name -notin @($LinuxName, "$LinuxName.tar.gz")
    } |
    ForEach-Object {
        $absolute = [System.IO.Path]::GetFullPath($_.FullName)
        $distAbsolute = [System.IO.Path]::GetFullPath($DistRoot) +
            [System.IO.Path]::DirectorySeparatorChar
        if (-not $absolute.StartsWith($distAbsolute, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove a stale product outside dist: $absolute"
        }
        Remove-Item -LiteralPath $absolute -Recurse -Force
    }

Reset-PackageDirectory $LinuxDir
Copy-Item -LiteralPath $products[0] -Destination $LinuxDir
Copy-Item -LiteralPath $products[1] -Destination $LinuxDir
Copy-Item -LiteralPath $products[2] -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'VERSION') -Destination $LinuxDir
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs/rtx-4090-linux.md') `
    -Destination (Join-Path $LinuxDir 'README.md')
Write-PackageHashes $LinuxDir

if (Test-Path -LiteralPath $LinuxArchive) {
    Remove-Item -LiteralPath $LinuxArchive -Force
}
& tar -C $DistRoot -czf $LinuxArchive $LinuxName
if ($LASTEXITCODE -ne 0) {
    throw 'Linux archive creation failed.'
}

$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $LinuxArchive).Hash.ToLowerInvariant()
Set-Content -LiteralPath (Join-Path $DistRoot 'SHA256SUMS.txt') `
    -Value "$archiveHash  $LinuxName.tar.gz" -Encoding ascii

Get-Item -LiteralPath $LinuxArchive |
    Select-Object Name, @{Name = 'SizeMB'; Expression = {[math]::Round($_.Length / 1MB, 2)}}
