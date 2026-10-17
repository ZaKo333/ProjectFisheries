# -*- coding: ascii -*-
<#
Windows one-shot deployment for ProjectFisheries (aquaculture API).

Run on the target Windows box from an elevated PowerShell:

    git clone --depth 1 https://github.com/ZaKo333/ProjectFisheries.git C:\fisheries\app
    powershell -NoProfile -ExecutionPolicy Bypass -File C:\fisheries\app\scripts\deploy_windows.ps1

What it does (idempotent, safe to re-run):
  1. prepares C:\fisheries\{app,data\uploads,logs}
  2. makes sure a CPython 3.11 exists (downloads the official installer if not)
  3. creates C:\fisheries\venv and pip installs the project (editable) with [dev,rag,api]
  4. builds the RAG knowledge base (fastembed; falls back to --embedder hash)
  5. writes C:\fisheries\run-server.bat (env vars + python -m fisheries.api.app)
  6. opens the firewall port and registers a scheduled task that starts on boot
  7. waits for /api/v1/health and prints DEPLOY_OK / DEPLOY_FAIL

Options:
    -Port 8000            listen port (default 8000)
    -Extras "dev,rag,api" pip extras (default dev,rag,api)
    -SkipPython           do not download Python, only use what is present
    -NoTask               do not register the scheduled task, just print the command
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$Extras = "dev,rag,api",
    [switch]$SkipPython,
    [switch]$NoTask
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = "C:\fisheries"
$App = Join-Path $Root "app"
$Venv = Join-Path $Root "venv"
$LogDir = Join-Path $Root "logs"
$DataDir = Join-Path $Root "data"
$UploadDir = Join-Path $DataDir "uploads"
$StateDb = Join-Path $DataDir "api_state.sqlite3"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$RunBat = Join-Path $Root "run-server.bat"
$ServerLog = Join-Path $LogDir "api.log"
$DeployLog = Join-Path $LogDir "deploy.log"
$PipIndex = "https://pypi.tuna.tsinghua.edu.cn/simple"
$PyUrls = @(
    "https://registry.npmmirror.com/-/binary/python/3.11.9/python-3.11.9-amd64.exe",
    "https://mirrors.huaweicloud.com/python/3.11.9/python-3.11.9-amd64.exe",
    "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
)
$PyExe = Join-Path $env:TEMP "python-3.11.9-amd64.exe"

function Step([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
}

function Fail([string]$msg) {
    Write-Host "DEPLOY_FAIL: $msg" -ForegroundColor Red
    if (Test-Path $ServerLog) {
        Write-Host "---- last 40 lines of $ServerLog ----" -ForegroundColor Yellow
        Get-Content $ServerLog -Tail 40 | ForEach-Object { Write-Host $_ }
    }
    exit 1
}

New-Item -ItemType Directory -Force -Path $Root, $LogDir, $DataDir, $UploadDir | Out-Null
Start-Transcript -Path $DeployLog -Append | Out-Null

Step "host: $env:COMPUTERNAME / user: $env:USERNAME / port: $Port"

# --- 1. project sources --------------------------------------------------
if (Test-Path (Join-Path $App ".git")) {
    Step "updating sources in $App"
    & git -C $App fetch --depth 1 origin main 2>&1 | Write-Host
    & git -C $App reset --hard FETCH_HEAD 2>&1 | Write-Host
} elseif (Test-Path (Join-Path $App "pyproject.toml")) {
    Step "using existing sources in $App (no .git)"
} else {
    Step "cloning sources into $App"
    New-Item -ItemType Directory -Force -Path (Split-Path $App) | Out-Null
    & git clone --depth 1 https://github.com/ZaKo333/ProjectFisheries.git $App 2>&1 | Write-Host
}
if (-not (Test-Path (Join-Path $App "pyproject.toml"))) {
    Fail "pyproject.toml not found under $App (git clone failed?)"
}

# --- 2. python -----------------------------------------------------------
function Resolve-Python {
    foreach ($cand in @("py -3", "python")) {
        try {
            $out = & cmd /c "$cand -c `"import sys;print(sys.executable)`"" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
        } catch { }
    }
    return $null
}

$sysPy = Resolve-Python
if (-not $sysPy) {
    if ($SkipPython) { Fail "no CPython found and -SkipPython was given" }
    Step "no CPython found, downloading Python 3.11.9 (python.org may be slow outside CN, mirrors first)"
    $downloaded = $false
    foreach ($u in $PyUrls) {
        Write-Host "  trying $u"
        & curl.exe -L --fail --silent --show-error --max-time 600 -o $PyExe $u
        if ($LASTEXITCODE -eq 0 -and (Test-Path $PyExe) -and ((Get-Item $PyExe).Length -gt 20MB)) {
            $downloaded = $true
            break
        }
        Write-Host "  -> failed (exit $LASTEXITCODE)" -ForegroundColor Yellow
    }
    if (-not $downloaded) { Fail "python installer download failed from every mirror" }
    Step "installing Python 3.11.9 silently"
    $proc = Start-Process -FilePath $PyExe -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1", "Include_test=0" -Wait -PassThru
    if ($proc.ExitCode -ne 0) { Fail "python installer exited with $($proc.ExitCode)" }
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $sysPy = Resolve-Python
    if (-not $sysPy) { Fail "python installed but still not on PATH" }
}
Step "system python: $sysPy"
& $sysPy --version

# --- 3. venv + editable install -----------------------------------------
if (-not (Test-Path $VenvPy)) {
    Step "creating venv $Venv"
    & $sysPy -m venv $Venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Fail "venv creation failed" }
}
Step "installing pip requirements (index: $PipIndex)"
& $VenvPy -m pip install --quiet --upgrade pip -i $PipIndex
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed" }
& $VenvPy -m pip install -e "$App[$Extras]" -i $PipIndex
if ($LASTEXITCODE -ne 0) { Fail "pip install -e .[$Extras] failed" }
& $VenvPy -c "import fisheries, fastapi; print('import ok:', fisheries.__file__)"
if ($LASTEXITCODE -ne 0) { Fail "the package does not import inside the venv" }

# --- 4. knowledge base ---------------------------------------------------
$Embedder = "fastembed"
$EmbedderFile = Join-Path $Root "embedder.txt"
if (Test-Path $EmbedderFile) {
    $Embedder = (Get-Content $EmbedderFile -Raw).Trim()
    if (-not $Embedder) { $Embedder = "fastembed" }
}
$kbManifest = Join-Path $App "knowledge\chroma\fisheries_manifest.json"
if (Test-Path $kbManifest) {
    Step "knowledge base already present"
} else {
    Step "building knowledge base with embedder=$Embedder"
    # HuggingFace is unreachable from many CN hosts; hf-mirror keeps fastembed usable.
    if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
    Step "HF_ENDPOINT=$env:HF_ENDPOINT"
    Push-Location $App
    try {
        & $VenvPy "scripts\build_knowledge_base.py" --embedder $Embedder
        if ($LASTEXITCODE -ne 0) {
            Step "fastembed build failed, retrying with --embedder hash (offline pseudo-vectors)"
            & $VenvPy "scripts\build_knowledge_base.py" --embedder hash
            if ($LASTEXITCODE -ne 0) { Fail "knowledge base build failed for both embedders" }
            $Embedder = "hash"
            Set-Content -Path $EmbedderFile -Value "hash" -Encoding ascii
        }
    } finally { Pop-Location }
}
Step "embedder for the service: $Embedder"

# --- 5. launcher ---------------------------------------------------------
$bat = @"
@echo off
rem generated by scripts\deploy_windows.ps1 - do not edit by hand
setlocal
set "FISHERIES_API_HOST=0.0.0.0"
set "FISHERIES_API_PORT=$Port"
set "FISHERIES_API_LOG_LEVEL=warning"
set "FISHERIES_API_UPLOAD_DIR=$UploadDir"
set "FISHERIES_API_STATE_DB=$StateDb"
set "FISHERIES_API_MAX_UPLOAD_MB=10"
set "FISHERIES_API_ALLOW_LOCAL_PATH=0"
set "FISHERIES_API_ENABLE_CORS=0"
set "FISHERIES_RAG_EMBEDDER=$Embedder"
set "PYTHONIOENCODING=utf-8"
cd /d "$App"
if not exist "$ServerLog" type nul > "$ServerLog"
"$VenvPy" -m fisheries.api.app >> "$ServerLog" 2>&1
"@
Set-Content -Path $RunBat -Value $bat -Encoding ascii
Step "wrote $RunBat"

# --- 6. firewall + scheduled task ---------------------------------------
$ruleName = "FisheriesAPI-$Port"
& netsh advfirewall firewall delete rule name="$ruleName" 2>&1 | Out-Null
& netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP localport=$Port 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "WARN: firewall rule not created (continuing)" -ForegroundColor Yellow }
else { Step "firewall rule $ruleName allows TCP $Port" }

if (-not $NoTask) {
    Step "registering scheduled task FisheriesAPI (on boot, as SYSTEM)"
    # schtasks writes "not found" to stderr for a missing task; do not let that abort the run.
    $ErrorActionPreference = "Continue"
    $null = & schtasks /Query /TN "FisheriesAPI" 2>&1
    if ($LASTEXITCODE -eq 0) { $null = & schtasks /Delete /TN "FisheriesAPI" /F 2>&1 }
    $null = & schtasks /Create /TN "FisheriesAPI" /TR "`"$RunBat`"" /SC ONSTART /RU SYSTEM /RL HIGHEST /F 2>&1
    $taskOk = ($LASTEXITCODE -eq 0)
    if ($taskOk) {
        Step "task registered, starting it now"
        $null = & schtasks /Run /TN "FisheriesAPI" 2>&1
        if ($LASTEXITCODE -ne 0) { $taskOk = $false }
    }
    if (-not $taskOk) {
        Write-Host "WARN: autostart task not available, starting the server for this session" -ForegroundColor Yellow
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$RunBat`"" -WindowStyle Hidden
    }
    $ErrorActionPreference = "Stop"
}

# --- 7. wait for health -------------------------------------------------
Step "waiting for http://127.0.0.1:$Port/api/v1/health"
$ok = $false
for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:$Port/api/v1/health"
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) { Fail "health endpoint did not answer within 90s" }

$body = ($r.Content | Out-String)
Step "health response: $body"
Write-Host ""
Write-Host "DEPLOY_OK  local: http://127.0.0.1:$Port/app/   public: map a port to $Port" -ForegroundColor Green
try { Stop-Transcript | Out-Null } catch { }
exit 0
