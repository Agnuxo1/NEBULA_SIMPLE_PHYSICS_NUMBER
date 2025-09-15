@echo off
echo ==================================================
echo   WORKFLOW COMPLETO RSNA - DESPUES DE ACTUALIZAR DRIVERS
echo ==================================================
echo.

echo Este script ejecuta el flujo completo:
echo 1. Verificacion CUDA
echo 2. Compilacion del modelo
echo 3. Prueba de entrenamiento
echo 4. Procesamiento del dataset completo
echo.

set /p continue="¿Continuar con el workflow completo? (s/n): "
if /i not "%continue%"=="s" (
    echo Workflow cancelado.
    pause
    exit /b 0
)

echo.
echo PASO 1: Verificando setup CUDA...
call verify_cuda_setup.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Verificacion CUDA fallo
    pause
    exit /b 1
)

echo.
echo PASO 2: Compilando modelo principal...
call compile_cuda.cmd
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilacion fallo
    pause
    exit /b 1
)

echo.
echo PASO 3: Prueba de entrenamiento rapida...
echo Configurando variables de entorno para prueba...
set RSNA_TRAIN_MIPS=1
set RSNA_MIPS_CSV=train_mips.csv
set MULTIEXPERT_4=0
set MAX_EPOCHS=1
set MAX_BATCHES=5
set RSNA_DEBUG_SAMPLES=1

echo Ejecutando prueba de entrenamiento...
main01.exe
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Prueba de entrenamiento fallo
    pause
    exit /b 1
)

echo.
echo PASO 4: Procesamiento del dataset completo...
set /p process_full="¿Procesar el dataset completo (300GB)? Esto puede tomar varias horas. (s/n): "
if /i "%process_full%"=="s" (
    echo.
    echo Iniciando procesamiento completo del dataset...
    echo Esto procesara el 100%% del dataset (puede tomar varias horas)
    echo.
    call process_large_dataset.bat rsna-intracranial-aneurysm-detection\train.csv rsna-intracranial-aneurysm-detection\series train_mips_full.csv 1.0
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ==================================================
        echo   PROCESAMIENTO COMPLETO EXITOSO
        echo ==================================================
        echo.
        echo Dataset procesado: train_mips_full.csv
        echo Para entrenar con el dataset completo:
        echo   set RSNA_MIPS_CSV=train_mips_full.csv
        echo   run_rsna_train.bat
        echo.
    ) else (
        echo ERROR: Procesamiento completo fallo
    )
) else (
    echo Procesamiento completo omitido.
)

echo.
echo ==================================================
echo   WORKFLOW COMPLETADO
echo ==================================================
echo.
echo Sistema listo para:
echo - Entrenamiento RSNA completo
echo - Inferencia con 4 vistas MIP
echo - Participacion en competencia Kaggle
echo.

pause






