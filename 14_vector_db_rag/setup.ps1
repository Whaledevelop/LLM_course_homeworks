$ErrorActionPreference = "Stop"

$runtimePython = "C:\Users\whale\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path $runtimePython) {
    $python = $runtimePython
} else {
    $python = "python"
}

& $python --version
& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe --version
