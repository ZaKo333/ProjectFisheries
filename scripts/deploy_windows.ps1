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
    -Token "..."          access token; generated once and kept in C:\fisheries\token.txt when omitted
    -NoAuth               serve without a token (only for a machine that is not reachable from the internet)
    -LlmProvider ...      real LLM backend (deepseek / openai / dashscope / aqua-7b-ollama ...)
    -LlmApiKey ...        its key; kept in C:\fisheries\llm.txt and reused by later runs
    -LlmModel / -LlmBaseUrl   override the preset model / endpoint
    -NoLlm                drop the LLM config and fall back to the mock advisor
    -CodeSha <sha>        download that exact commit archive from a mirror and copy
                          it over the app dir (immutable, so a cached mirror cannot
                          serve a stale tree); data/ and models/ are kept
    -Embedder fastembed   embedding backend for the knowledge base (default fastembed)
    -Embedder hash        offline pseudo-vectors (no model download, 256-dim)
    -RebuildKb            rebuild the knowledge base even if it is already there
    -SkipPython           do not download Python, only use what is present
    -NoTask               do not register the scheduled task, just print the command
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$Extras = "dev,rag,api",
    [string]$Token,
    [switch]$NoAuth,
    [string]$LlmProvider,
    [string]$LlmApiKey,
    [string]$LlmModel,
    [string]$LlmBaseUrl,
    [switch]$NoLlm,
    [string]$Embedder,
    [switch]$RebuildKb,
    [switch]$Vision,
    [string]$WeightsFrom,
    [string]$CodeSha,
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
        Get-Content $ServerLog -Tail 40 -Encoding UTF8 | ForEach-Object { Write-Host $_ }
    }
    exit 1
}

function Warn([string]$msg) {
    Write-Host ("[ {0} ] WARN {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) -ForegroundColor Yellow
}

New-Item -ItemType Directory -Force -Path $Root, $LogDir, $DataDir, $UploadDir | Out-Null
Start-Transcript -Path $DeployLog -Append | Out-Null

Step "host: $env:COMPUTERNAME / user: $env:USERNAME / port: $Port"

# --- 1. project sources --------------------------------------------------
# -CodeSha pulls the archive of that exact commit from a CN mirror and copies it
# over the previous tree. A commit archive is immutable, so the mirror cache can
# never hand back a stale copy - which is exactly what bit us with
# archive/refs/heads/main.zip (the mirror kept serving the old tree, and tar
# happily re-extracted it while the deploy still printed DEPLOY_OK).
if ($CodeSha) {
    $short = $CodeSha.Substring(0, [Math]::Min(7, $CodeSha.Length))
    $codeZip = Join-Path $Root "code-$short.zip"
    $codeUrls = @(
        "https://ghfast.top/https://github.com/ZaKo333/ProjectFisheries/archive/$CodeSha.zip",
        "https://ghproxy.net/https://github.com/ZaKo333/ProjectFisheries/archive/$CodeSha.zip"
    )
    Remove-Item -LiteralPath $codeZip -ErrorAction SilentlyContinue
    $got = $false
    foreach ($u in $codeUrls) {
        & curl.exe -L --fail --silent --show-error --max-time 900 -o $codeZip $u
        if ($LASTEXITCODE -eq 0 -and (Test-Path $codeZip) -and ((Get-Item $codeZip).Length -gt 500KB)) {
            $got = $true
            break
        }
        Warn "code archive download failed from $u"
    }
    if (-not $got) { Fail "cannot download code archive for $CodeSha" }
    $codeTmp = Join-Path $Root "code-$short"
    Remove-Item -LiteralPath $codeTmp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $codeTmp | Out-Null
    & tar.exe -xf $codeZip -C $codeTmp
    if ($LASTEXITCODE -ne 0) { Fail "tar.exe could not extract $codeZip" }
    $inner = @(Get-ChildItem -Path $codeTmp -Directory)
    if ($inner.Count -ne 1) { Fail "archive $codeZip should hold exactly one root folder, found $($inner.Count)" }
    # GitHub names the folder after the FULL sha (ProjectFisheries-<40 chars>),
    # so a 7-char short sha never matches - accept either form.
    $rootName = $inner[0].Name
    $rootOk = $rootName.StartsWith("ProjectFisheries-") -and
        ($rootName.EndsWith($CodeSha) -or $rootName.EndsWith($short))
    if (-not $rootOk) {
        Fail "archive root is '$rootName' but the requested commit is $CodeSha"
    }
    New-Item -ItemType Directory -Force -Path $App | Out-Null
    # /E copies without deleting: data/ and models/ inside $App must survive.
    & robocopy $inner[0].FullName $App /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "robocopy failed with code $LASTEXITCODE" }
    Remove-Item -LiteralPath $codeTmp -Recurse -Force -ErrorAction SilentlyContinue
    Step "code updated to $short from $codeZip"
} elseif (Test-Path (Join-Path $App ".git")) {
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

# --- 3b. optional vision (ultralytics + .pt weights) ---------------------
# Species / lesion models are optional on purpose: without them the service
# answers 503 from /api/v1/species/detect and /api/v1/ingest/image and the web
# UI hides those modules. Enabling them needs two independent halves:
#   * the runtime  -> ultralytics (which drags torch/torchvision along),
#   * the weights  -> training products, so they are not in git.
# Both are opt-in because the install is ~1 GB and the files live elsewhere.
if ($Vision) {
    Step "installing vision extras (ultralytics; pulls torch, this can take a while)"
    & $VenvPy -m pip install ultralytics -i $PipIndex
    if ($LASTEXITCODE -ne 0) { Fail "pip install ultralytics failed" }
    & $VenvPy -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
    if ($LASTEXITCODE -ne 0) { Fail "ultralytics does not import inside the venv" }
}

if ($WeightsFrom) {
    $modelsDir = Join-Path $App "models"
    New-Item -ItemType Directory -Force $modelsDir | Out-Null
    if (-not (Test-Path $WeightsFrom)) {
        Warn "weights source not found: $WeightsFrom (skipped; /api/v1/species/detect keeps reporting the missing model)"
    } else {
        Step "copying *.pt weights from $WeightsFrom"
        $copied = 0
        Get-ChildItem -Path $WeightsFrom -Filter *.pt -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $modelsDir $_.Name) -Force
            $copied++
        }
        if ($copied -eq 0) {
            Warn "no .pt file under $WeightsFrom"
        } else {
            Step "copied $copied weight file(s) into $modelsDir"
        }
    }
}

# --- 4. knowledge base ---------------------------------------------------
# The store records which embedder produced it; the service refuses to load a
# store built by a different one (StoreMismatchError). So switching embedders
# must rebuild, and the choice has to be persisted for the launcher.
$EmbedderFile = Join-Path $Root "embedder.txt"
if (-not $Embedder) {
    if (Test-Path $EmbedderFile) { $Embedder = (Get-Content $EmbedderFile -Raw).Trim() }
    # Default: offline pseudo-vectors. fastembed has to download its ONNX
    # model, which times out on many CN hosts and stalls the deploy for
    # minutes; ask for it explicitly with -Embedder fastembed.
    if (-not $Embedder) { $Embedder = "hash" }
}
$kbDir = Join-Path $App "knowledge\chroma"
$kbManifest = Join-Path $kbDir "fisheries_manifest.json"
$kbEmbedder = ""
if (Test-Path $kbManifest) {
    try { $kbEmbedder = [string](Get-Content $kbManifest -Raw | ConvertFrom-Json).embedder } catch { $kbEmbedder = "" }
}
$kbMatches = $kbEmbedder.ToLower().StartsWith($Embedder.ToLower())
if ((Test-Path $kbManifest) -and $kbMatches -and (-not $RebuildKb)) {
    Step "knowledge base already present (embedder=$kbEmbedder)"
} else {
    if (Test-Path $kbManifest) {
        Step "rebuilding knowledge base: have '$kbEmbedder', want '$Embedder'"
    } else {
        Step "building knowledge base with embedder=$Embedder"
    }
    # Release the store before deleting it: a running service keeps it open.
    $ErrorActionPreference = "Continue"
    $null = & schtasks /End /TN "FisheriesAPI" 2>&1
    Start-Sleep -Seconds 2
    $ErrorActionPreference = "Stop"
    if (Test-Path $kbDir) { Remove-Item -Path $kbDir -Recurse -Force -ErrorAction SilentlyContinue }
    # HuggingFace is unreachable from many CN hosts; hf-mirror keeps fastembed usable.
    if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
    Step "HF_ENDPOINT=$env:HF_ENDPOINT"
    Push-Location $App
    try {
        & $VenvPy "scripts\build_knowledge_base.py" --embedder $Embedder
        if ($LASTEXITCODE -ne 0) {
            if ($Embedder -eq "hash") { Fail "knowledge base build failed with --embedder hash" }
            Step "$Embedder build failed, retrying with --embedder hash (offline pseudo-vectors)"
            & $VenvPy "scripts\build_knowledge_base.py" --embedder hash
            if ($LASTEXITCODE -ne 0) { Fail "knowledge base build failed for both embedders" }
            $Embedder = "hash"
        }
    } finally { Pop-Location }
}
Set-Content -Path $EmbedderFile -Value $Embedder -Encoding ascii
Step "embedder for the service: $Embedder"

# --- 4b. optional LLM ----------------------------------------------------
# The API ships with a deterministic mock advisor: /chat then answers with the
# retrieved text verbatim and flags answer_is_raw_retrieval=true. Supplying a
# provider here switches it to a real OpenAI-compatible backend. The choice is
# persisted in llm.txt (provider / model / base_url / api_key) so a re-deploy
# without parameters keeps it - and so the key does not live in the .bat only.
$LlmFile = Join-Path $Root "llm.txt"
$llm = @{}
if (Test-Path $LlmFile) {
    foreach ($line in Get-Content $LlmFile) {
        if ($line -match "^\s*([A-Za-z_]+)\s*=\s*(.*)$") { $llm[$Matches[1]] = $Matches[2].Trim() }
    }
}
if ($NoLlm) {
    $llm = @{}
    Remove-Item -Path $LlmFile -ErrorAction SilentlyContinue
} else {
    if ($LlmProvider) { $llm["provider"] = $LlmProvider }
    if ($LlmApiKey) { $llm["api_key"] = $LlmApiKey }
    if ($LlmModel) { $llm["model"] = $LlmModel }
    if ($LlmBaseUrl) { $llm["base_url"] = $LlmBaseUrl }
    if ($llm["provider"] -and -not $llm["model"]) {
        # The OpenAI-compatible SDK rejects an empty model, and the failure only
        # shows up later as a ValidationError inside /chat, so a
        # missing -LlmModel is filled with the provider default instead.
        $modelDefaults = @{
            "dashscope"      = "qwen-plus"
            "deepseek"       = "deepseek-chat"
            "openai"         = "gpt-4o-mini"
            "aqua-7b-ollama" = "aqua-7b"
            "aqua-7b-vllm"   = "KurmaAI/AQUA-7B"
        }
        $fallbackModel = $modelDefaults[$llm["provider"]]
        if ($fallbackModel) {
            $llm["model"] = $fallbackModel
            Step "llm: -LlmModel not given, using '$fallbackModel' for provider '$($llm['provider'])'"
        }
    }
    if ($llm["provider"] -and -not $llm["api_key"]) {
        Warn "llm provider '$($llm['provider'])' has no api key; /chat will fall back to raw retrieval"
    }
    if ($llm["provider"] -eq "mock") {
        $llm = @{}
        Remove-Item -Path $LlmFile -ErrorAction SilentlyContinue
    } elseif ($llm.Count -gt 0) {
        $llm.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" } |
            Set-Content -Path $LlmFile -Encoding ascii
    }
}
$llmLines = ""
if ($llm["provider"]) { $llmLines += "set `"FISHERIES_LLM_PROVIDER=$($llm['provider'])`"`r`n" }
if ($llm["model"]) { $llmLines += "set `"FISHERIES_LLM_MODEL=$($llm['model'])`"`r`n" }
if ($llm["base_url"]) { $llmLines += "set `"FISHERIES_LLM_BASE_URL=$($llm['base_url'])`"`r`n" }
if ($llm["api_key"]) { $llmLines += "set `"FISHERIES_LLM_API_KEY=$($llm['api_key'])`"`r`n" }
if ($llm["provider"]) {
    Step "llm: provider=$($llm['provider']) model=$($llm['model']) (config kept in $LlmFile)"
} else {
    Step "llm: none (mock advisor - /chat returns raw retrieval and says so)"
}

# --- 5. launcher ---------------------------------------------------------
# The port is usually mapped to the public internet, so the generated launcher
# carries an access token unless -NoAuth was given. Reusing the same value on a
# re-deploy keeps existing browser sessions working.
$TokenFile = Join-Path $Root "token.txt"
if (-not $NoAuth) {
    $Token = "$Token".Trim()
    if ($Token) {
        if ($Token -match "[^\x20-\x7E]") {
            Fail "access token must be printable ASCII (it is written into the launcher and token.txt)"
        }
        # An explicit -Token wins and replaces token.txt, so a later deploy
        # without -Token does not resurrect the old secret.
        Set-Content -Path $TokenFile -Value $Token -Encoding ascii
        Step "access token taken from -Token (saved to $TokenFile)"
    } else {
        if (Test-Path $TokenFile) { $Token = (Get-Content $TokenFile -Raw).Trim() }
        if (-not $Token) {
            $rnd = New-Object byte[] 16
            [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($rnd)
            $Token = ([System.BitConverter]::ToString($rnd) -replace "-", "").ToLower()
            Set-Content -Path $TokenFile -Value $Token -Encoding ascii
            Step "access token generated (saved to $TokenFile)"
        } else {
            Step "access token loaded from $TokenFile"
        }
    }
    if ($Token.Length -lt 8) {
        Warn "access token is short ($($Token.Length) chars) and the port may be public; 12+ random chars is safer"
    }
    Step "access token in use (see $TokenFile); browsers land on /login"
} else {
    $Token = ""
    Step "COMPAT: -NoAuth given, the service will be reachable without a password"
}

$bat = @"
@echo off
rem generated by scripts\deploy_windows.ps1 - do not edit by hand
setlocal
set "FISHERIES_API_HOST=0.0.0.0"
set "FISHERIES_API_PORT=$Port"
set "FISHERIES_API_LOG_LEVEL=warning"
set "FISHERIES_API_UPLOAD_DIR=$UploadDir"
set "FISHERIES_API_STATE_DB=$StateDb"
set "FISHERIES_API_TOKEN=$Token"
set "FISHERIES_API_MAX_UPLOAD_MB=10"
set "FISHERIES_API_ALLOW_LOCAL_PATH=0"
set "FISHERIES_API_ENABLE_CORS=0"
set "FISHERIES_RAG_EMBEDDER=$Embedder"
$llmLines
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
if ($Token) {
    Write-Host "LOGIN      token: $Token" -ForegroundColor Green
    Write-Host "           (also saved in $TokenFile; open /login once and the browser remembers it)"
} else {
    Write-Host "LOGIN      no token (-NoAuth): anyone who can reach the port can use the service"
}
if ($llm["provider"]) {
    Write-Host "LLM        provider=$($llm['provider']) model=$($llm['model'])" -ForegroundColor Green
    Write-Host "           (set in $LlmFile; /chat now answers with the model, answer_is_raw_retrieval=false)"
} else {
    Write-Host "LLM        none: /chat returns the retrieved text verbatim and flags answer_is_raw_retrieval=true"
}
$ptCount = @(Get-ChildItem (Join-Path $App "models") -Filter *.pt -File -ErrorAction SilentlyContinue).Count
Write-Host ("VISION     {0} weight file(s) in {1}" -f $ptCount, (Join-Path $App "models"))
if ($ptCount -gt 0 -and -not $Vision) {
    Write-Host "           (files present but ultralytics may be missing: re-run with -Vision)"
}
try { Stop-Transcript | Out-Null } catch { }
exit 0
