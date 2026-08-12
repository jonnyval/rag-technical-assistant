[CmdletBinding(DefaultParameterSetName = "Cluster")]
param(
    [Parameter(Mandatory = $true, ParameterSetName = "Cluster")]
    [string]$ClusterId,

    [Parameter(Mandatory = $true, ParameterSetName = "Topic")]
    [ValidateNotNullOrEmpty()]
    [string]$Topic,

    [int]$MaxTurns = 80
)

$HermesExecutable = Join-Path $env:LOCALAPPDATA "hermes\hermes-agent\venv\Scripts\hermes.exe"
$PromptPath = Join-Path $PSScriptRoot "ARTICLE_AGENT_PROMPT.md"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectPython = "C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe"
$McpPort = 8765

if (-not (Test-Path -LiteralPath $HermesExecutable)) {
    throw "Hermes executable not found: $HermesExecutable"
}

# Apply the project-owned llm.active profile before every run. This keeps the
# user Hermes config in sync with ticket_clustering/hermes_agent/config.yaml.
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

$ResearchInstructions = Get-Content -LiteralPath $PromptPath -Raw -Encoding UTF8
$ResearchTarget = if ($PSCmdlet.ParameterSetName -eq "Topic") {
@"
Research the freely supplied topic exactly as written: $Topic
First call create_research_seed with this topic. Use its returned research_id as cluster_id when saving.
"@
} else {
@"
Research the candidate with cluster_id: $ClusterId
"@
}
$ResearchRequest = @"
$ResearchInstructions

$ResearchTarget
Use only the reglab_articles MCP tools. Do not use Hermes memory, web, terminal, or file tools.
"@

& $HermesExecutable chat `
    --toolsets "reglab_articles" `
    --ignore-rules `
    --max-turns $MaxTurns `
    -q $ResearchRequest
