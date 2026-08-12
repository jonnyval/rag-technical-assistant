[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Query,

    [string]$UsageFile = ""
)

$HermesExecutable = Join-Path $env:LOCALAPPDATA "hermes\hermes-agent\venv\Scripts\hermes.exe"
$PromptPath = Join-Path $PSScriptRoot "SEARCH_AGENT_PROMPT.md"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectPython = "C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe"
$McpPort = 8765

if (-not (Test-Path -LiteralPath $HermesExecutable)) {
    throw "Hermes executable not found: $HermesExecutable"
}

& $ProjectPython (Join-Path $PSScriptRoot "configure_provider.py")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply the active Hermes LLM profile"
}

$McpReady = Test-NetConnection -ComputerName "127.0.0.1" -Port $McpPort -InformationLevel Quiet -WarningAction SilentlyContinue
if (-not $McpReady) {
    $StdoutLog = Join-Path $PSScriptRoot "mcp_http_stdout.log"
    $StderrLog = Join-Path $PSScriptRoot "mcp_http_stderr.log"
    Start-Process -FilePath $ProjectPython `
        -ArgumentList @("-m", "ticket_clustering.hermes_agent.server", "--transport", "streamable-http") `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog | Out-Null
    $Deadline = (Get-Date).AddSeconds(180)
    do {
        Start-Sleep -Seconds 2
        $McpReady = Test-NetConnection -ComputerName "127.0.0.1" -Port $McpPort -InformationLevel Quiet -WarningAction SilentlyContinue
    } while ((-not $McpReady) -and ((Get-Date) -lt $Deadline))
    if (-not $McpReady) {
        throw "RegLab MCP server did not start on port $McpPort. See $StderrLog"
    }
}

$Instructions = Get-Content -LiteralPath $PromptPath -Raw -Encoding UTF8
$Request = "$Instructions`n`n# Вопрос пользователя`n`n$Query"
if (-not $UsageFile) {
    $UsageDirectory = Join-Path $PSScriptRoot "usage"
    New-Item -ItemType Directory -Path $UsageDirectory -Force | Out-Null
    $UsageFile = Join-Path $UsageDirectory ("search_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}

& $HermesExecutable `
    --oneshot $Request `
    --toolsets "reglab_search_local" `
    --ignore-rules `
    --usage-file $UsageFile

Write-Host "`nUsage: $UsageFile"
