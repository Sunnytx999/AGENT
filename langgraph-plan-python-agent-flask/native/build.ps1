$ErrorActionPreference = "Stop"
$nativeRoot = $PSScriptRoot
$buildDirectory = Join-Path $nativeRoot "build"
$zigAvailable = $false
python -c "import ziglang" 2>$null
if ($LASTEXITCODE -eq 0) {
    $zigAvailable = $true
}

if (Get-Command cmake -ErrorAction SilentlyContinue) {
    cmake -S $nativeRoot -B $buildDirectory -A x64
    cmake --build $buildDirectory --config Release
} elseif ($zigAvailable) {
    $outputDirectory = Join-Path $nativeRoot "bin"
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    $env:ZIG_GLOBAL_CACHE_DIR = Join-Path $nativeRoot ".zig-global-cache"
    $env:ZIG_LOCAL_CACHE_DIR = Join-Path $nativeRoot ".zig-local-cache"
    python -m ziglang c++ -std=c++17 -O2 -w -shared `
        "-I$(Join-Path $nativeRoot 'include')" `
        (Join-Path $nativeRoot "src\agent_tools.cpp") `
        -o (Join-Path $outputDirectory "agent_tools_v2.dll")
} else {
    throw "No compiler was found. Install CMake plus Visual Studio Build Tools, or run: python -m pip install ziglang"
}

Write-Host "Built native library in $(Join-Path $nativeRoot 'bin')"
