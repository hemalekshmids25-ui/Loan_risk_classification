$projectPath = $PSScriptRoot
$url = "http://127.0.0.1:8501"

Set-Location $projectPath
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 6
    Start-Process "http://127.0.0.1:8501"
} | Out-Null

python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
