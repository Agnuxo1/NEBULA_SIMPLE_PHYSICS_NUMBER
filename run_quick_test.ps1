param(
  [int]$TimeoutSeconds = 240,
  [int]$MaxEpochs = 1,
  [int]$MaxBatches = 3,
  [int]$TrainSamples = 1024,
  [int]$TestSamples = 512
)

Write-Host "[run_quick_test] Closing previous SIMPLE_* processes..."
$names = @('SIMPLE_FISIC_NUMBER','SIMPLE_FISIC_NUMBER_NEW','SIMPLE_PHYSICS_NUMBERS','SIMPLE_PHYSICS_NUMBERS_NEW','SIMPLE_PHYSICS_NUMBERS01_NEW')
foreach ($n in $names) { Try { Get-Process -Name $n -ErrorAction Stop | Stop-Process -Force -ErrorAction SilentlyContinue } Catch {} }

Write-Host "[run_quick_test] Setting environment for short run..."
$env:MAX_EPOCHS = "$MaxEpochs"
$env:MAX_BATCHES = "$MaxBatches"
$env:SKIP_SUBMISSION = "1"
$env:MAX_TRAIN_SAMPLES = "$TrainSamples"
$env:MAX_TEST_SAMPLES  = "$TestSamples"

$workdir = $PSScriptRoot
$exe = Join-Path $workdir 'SIMPLE_PHYSICS_NUMBERS01_NEW.exe'
if (-not (Test-Path $exe)) { $exe = Join-Path $workdir 'SIMPLE_PHYSICS_NUMBERS_NEW.exe' }
if (-not (Test-Path $exe)) { $exe = Join-Path $workdir '..\SIMPLE_PHYSICS_NUMBERS01_NEW.exe' }
if (-not (Test-Path $exe)) { throw "Executable not found next to project root." }

Write-Host "[run_quick_test] Launching: $exe with timeout $TimeoutSeconds s"
$p = Start-Process -FilePath $exe -WorkingDirectory $workdir -PassThru -WindowStyle Hidden

try {
  Wait-Process -Id $p.Id -Timeout $TimeoutSeconds -ErrorAction SilentlyContinue
} catch {}

if (-not $p.HasExited) {
  Write-Warning "[run_quick_test] Timeout reached. Killing process..."
  try { Stop-Process -Id $p.Id -Force } catch {}
  exit 124
} else {
  Write-Host "[run_quick_test] Finished. ExitCode=$($p.ExitCode)"
  exit $p.ExitCode
}
