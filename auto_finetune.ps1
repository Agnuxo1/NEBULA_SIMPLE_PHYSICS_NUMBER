param(
  [int]$MaxTrials = 24,
  [int]$TimeoutSeconds = 300,
  [int]$MaxEpochs = 1,
  [int]$MaxBatches = 6,
  [int]$TrainSamples = 1024,
  [switch]$SingleSample,
  [int]$Seed = 42,
  [double]$LR = [double]::NaN,
  [double]$TEMP = [double]::NaN,
  [double]$MOM = [double]::NaN,
  [string]$GainList = $null,
  [string]$PhaseList = $null
)

Write-Host "[auto_finetune] Starting hyperparameter sweep ($MaxTrials trials)"

$workdir = $PSScriptRoot
$exe = Join-Path $workdir 'SIMPLE_PHYSICS_NUMBERS01_NEW.exe'
if (-not (Test-Path $exe)) { throw "Executable not found: $exe" }

# Candidate grids (overridable)
if ([double]::IsNaN($LR))    { $lrList    = @(0.10, 0.15, 0.20, 0.25) } else { $lrList = @($LR) }
if ([double]::IsNaN($TEMP))  { $tempList  = @(3.0, 3.5, 4.0, 5.0) }   else { $tempList = @($TEMP) }
if ([double]::IsNaN($MOM))   { $momList   = @(0.85, 0.90) }            else { $momList = @($MOM) }
if ($null -ne $GainList -and $GainList.Trim() -ne '') {
  $tmp = @()
  foreach ($g in $GainList.Split(',')) { $tmp += [double]::Parse($g.Trim(), [System.Globalization.CultureInfo]::InvariantCulture) }
  $gainList = $tmp
} else { $gainList = @(1.5, 2.0, 3.0) }
if ($null -ne $PhaseList -and $PhaseList.Trim() -ne '') {
  $tmp2 = @()
  foreach ($p in $PhaseList.Split(',')) { $tmp2 += [double]::Parse($p.Trim(), [System.Globalization.CultureInfo]::InvariantCulture) }
  $phaseList = $tmp2
} else { $phaseList = @(0.0005, 0.0010, 0.0020) }

# Basic env
$env:MAX_EPOCHS = "$MaxEpochs"
$env:MAX_BATCHES = "$MaxBatches"
$env:MAX_TRAIN_SAMPLES = "$TrainSamples"
$env:SKIP_SUBMISSION = "1"

function ToInvStr([double]$x) {
  return [System.String]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $x)
}

if ($SingleSample) {
  # Use only 1 sample (first), avoid shuffle to calibrate a fixed case
  $env:MAX_TRAIN_SAMPLES = "1"
  $env:NO_SHUFFLE = "1"
}

# Random generator
$rng = New-Object System.Random($Seed)

$best = $null

for ($t = 1; $t -le $MaxTrials; $t++) {
  $lr    = $lrList[ $rng.Next(0, $lrList.Count) ]
  $temp  = $tempList[ $rng.Next(0, $tempList.Count) ]
  $mom   = $momList[ $rng.Next(0, $momList.Count) ]
  $gain  = $gainList[ $rng.Next(0, $gainList.Count) ]
  $phase = $phaseList[ $rng.Next(0, $phaseList.Count) ]

  $env:LEARNING_RATE_INIT = (ToInvStr $lr)
  $env:BASE_TEMPERATURE   = (ToInvStr $temp)
  $env:MOMENTUM           = (ToInvStr $mom)
  $env:LOGIT_GAIN         = (ToInvStr $gain)
  $env:MZI_PHASE_LR       = (ToInvStr $phase)

  Write-Host ("`n[auto_finetune] Trial {0}/{1} => LR={2} TEMP={3} MOM={4} GAIN={5} PHASE_LR={6}" -f $t,$MaxTrials,(ToInvStr $lr),(ToInvStr $temp),(ToInvStr $mom),(ToInvStr $gain),(ToInvStr $phase))

  $pinfo = New-Object System.Diagnostics.ProcessStartInfo
  $pinfo.FileName = $exe
  $pinfo.WorkingDirectory = $workdir
  $pinfo.UseShellExecute = $false
  $pinfo.RedirectStandardOutput = $true
  $pinfo.RedirectStandardError = $true
  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $pinfo
  [void]$p.Start()

  if (-not $p.WaitForExit($TimeoutSeconds*1000)) { try { $p.Kill() } catch {}; Write-Warning "Timeout"; continue }
  $out = $p.StandardOutput.ReadToEnd()
  $err = $p.StandardError.ReadToEnd()

  # Parse EpochAvgLoss line
  $avgLoss = $null
  foreach ($line in ($out -split "`n")) {
    if ($line -match 'EpochAvgLoss:\s*([0-9]+\.[0-9]+)') { $avgLoss = [double]$Matches[1] }
  }
  if ($null -eq $avgLoss) {
    # fallback: try last progress Loss: value
    foreach ($line in (($out -split "`n") | Where-Object { $_ -match 'Loss:\s*([0-9]+\.[0-9]+)' })) {
      $m = [regex]::Match($line, 'Loss:\s*([0-9]+\.[0-9]+)')
      if ($m.Success) { $avgLoss = [double]$m.Groups[1].Value }
    }
  }

  if ($null -eq $avgLoss) { Write-Warning "Could not parse loss. Skipping."; continue }

  Write-Host "[auto_finetune] Result: EpochAvgLoss=$avgLoss"

  if ($null -eq $best -or $avgLoss -lt $best.loss) {
    $best = [pscustomobject]@{ loss=[double]$avgLoss; lr=[double]$lr; temp=[double]$temp; mom=[double]$mom; gain=[double]$gain; phase=[double]$phase }
  }
}

if ($best -ne $null) {
  Write-Host "\n[auto_finetune] Best => Loss=$($best.loss) LR=$($best.lr) TEMP=$($best.temp) MOM=$($best.mom) GAIN=$($best.gain) PHASE_LR=$($best.phase)"
  # Store to file
  $best | ConvertTo-Json | Set-Content -Path (Join-Path $workdir 'auto_finetune_best.json')
} else {
  Write-Warning "No successful trials."
}
