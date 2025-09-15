param(
  [int]$TimeoutSeconds = 3600,
  [int]$MaxEpochs = 30,
  [switch]$ShowAcc
)

Write-Host "[run_train_full] Closing previous SIMPLE_* processes..."
$names = @('SIMPLE_FISIC_NUMBER','SIMPLE_FISIC_NUMBER_NEW','SIMPLE_PHYSICS_NUMBERS','SIMPLE_PHYSICS_NUMBERS_NEW','SIMPLE_PHYSICS_NUMBERS01_NEW')
foreach ($n in $names) { Try { Get-Process -Name $n -ErrorAction Stop | Stop-Process -Force -ErrorAction SilentlyContinue } Catch {} }

Write-Host "[run_train_full] Setting environment for full training..."
Remove-Item Env:MAX_TRAIN_SAMPLES -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:MAX_TEST_SAMPLES -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:PREDICT_ONLY -ErrorAction SilentlyContinue | Out-Null
Remove-Item Env:SKIP_SUBMISSION -ErrorAction SilentlyContinue | Out-Null
$env:MAX_EPOCHS = "$MaxEpochs"
Remove-Item Env:MAX_BATCHES -ErrorAction SilentlyContinue | Out-Null
if ($ShowAcc) { $env:SHOW_BATCH_ACC = '1' } else { Remove-Item Env:SHOW_BATCH_ACC -ErrorAction SilentlyContinue | Out-Null }
Remove-Item Env:DEBUG_OPTICS -ErrorAction SilentlyContinue | Out-Null

$workdir = $PSScriptRoot
$exe = Join-Path $workdir 'SIMPLE_PHYSICS_NUMBERS01_NEW.exe'
if (-not (Test-Path $exe)) { $exe = Join-Path $workdir 'SIMPLE_PHYSICS_NUMBERS_NEW.exe' }
if (-not (Test-Path $exe)) { $exe = Join-Path $workdir '..\SIMPLE_FISICS_NUMBERS01_NEW.exe' }
if (-not (Test-Path $exe)) { throw "Executable not found next to project root." }

Write-Host "[run_train_full] Launching: $exe for $MaxEpochs epochs with timeout $TimeoutSeconds s"
$p = Start-Process -FilePath $exe -WorkingDirectory $workdir -PassThru -WindowStyle Hidden

try {
  Wait-Process -Id $p.Id -Timeout $TimeoutSeconds -ErrorAction SilentlyContinue
} catch {}

if (-not $p.HasExited) {
  Write-Warning "[run_train_full] Timeout reached. Killing process..."
  try { Stop-Process -Id $p.Id -Force } catch {}
  exit 124
} else {
  Write-Host "[run_train_full] Finished. ExitCode=$($p.ExitCode)"
  if (Test-Path (Join-Path $root 'submission.csv')) {
    Write-Host "[run_train_full] submission.csv generated."
  } else {
    Write-Warning "[run_train_full] submission.csv not found."
  }
  exit $p.ExitCode
}
