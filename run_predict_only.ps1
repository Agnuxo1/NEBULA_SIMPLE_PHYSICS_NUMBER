param(
  [int]$TimeoutSeconds = 300,
  [int]$TestSamples = 2000
)

Write-Host "[run_predict_only] Closing previous SIMPLE_* processes..."
$names = @('SIMPLE_FISIC_NUMBER','SIMPLE_FISIC_NUMBER_NEW','SIMPLE_PHYSICS_NUMBERS','SIMPLE_PHYSICS_NUMBERS_NEW','SIMPLE_PHYSICS_NUMBERS01_NEW')
foreach ($n in $names) { Try { Get-Process -Name $n -ErrorAction Stop | Stop-Process -Force -ErrorAction SilentlyContinue } Catch {} }

Write-Host "[run_predict_only] Setting environment (predict only)..."
Remove-Item Env:MAX_EPOCHS -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:MAX_BATCHES -ErrorAction SilentlyContinue | Out-Null
$env:PREDICT_ONLY = "1"
Remove-Item Env:SKIP_SUBMISSION -ErrorAction SilentlyContinue | Out-Null
$env:MAX_TEST_SAMPLES = "$TestSamples"
Remove-Item Env:MAX_TRAIN_SAMPLES -ErrorAction SilentlyContinue | Out-Null

$workdir = $PSScriptRoot
$exe = Join-Path $workdir 'SIMPLE_PHYSICS_NUMBERS01_NEW.exe'
if (-not (Test-Path $exe)) { $exe = Join-Path $workdir 'SIMPLE_PHYSICS_NUMBERS_NEW.exe' }
if (-not (Test-Path $exe)) { $exe = Join-Path $workdir '..\SIMPLE_PHYSICS_NUMBERS01_NEW.exe' }
if (-not (Test-Path $exe)) { throw "Executable not found next to project root." }

Write-Host "[run_predict_only] Launching: $exe (predict-only) with timeout $TimeoutSeconds s"
$p = Start-Process -FilePath $exe -WorkingDirectory $workdir -PassThru -WindowStyle Hidden

try {
  Wait-Process -Id $p.Id -Timeout $TimeoutSeconds -ErrorAction SilentlyContinue
} catch {}

if (-not $p.HasExited) {
  Write-Warning "[run_predict_only] Timeout reached. Killing process..."
  try { Stop-Process -Id $p.Id -Force } catch {}
  exit 124
} else {
  Write-Host "[run_predict_only] Finished. ExitCode=$($p.ExitCode)"
  if (Test-Path (Join-Path $root 'submission.csv')) {
    Write-Host "[run_predict_only] submission.csv generated."
  } else {
    Write-Warning "[run_predict_only] submission.csv not found."
  }
  exit $p.ExitCode
}
