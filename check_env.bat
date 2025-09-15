@echo off
echo ================================================
echo  Environment Variables Potentially Limiting Run
echo ================================================
setlocal enabledelayedexpansion
set _VARS=MAX_EPOCHS MAX_BATCHES MAX_TRAIN_SAMPLES MAX_TEST_SAMPLES PREDICT_ONLY NO_SHUFFLE ^
          RSNA_TRAIN_MIPS RSNA_MIPS_CSV MULTIEXPERT_4 ^
          CKPT_DIR CKPT_DIR_FRONT CKPT_DIR_BACK CKPT_DIR_LEFT CKPT_DIR_RIGHT ^
          USE_MAX_PLUS USE_HADAMARD USE_KWTA KWTA_FRAC KWTA_KEEP LI_ALPHA DISABLE_EARLY_STOP

for %%V in (%_VARS%) do (
  call :printvar %%V
)
goto :eof

:printvar
set NAME=%1
for /f "tokens=1* delims==" %%a in ('set %NAME% 2^>nul') do (
  echo %%a=%%b
)
exit /b 0

