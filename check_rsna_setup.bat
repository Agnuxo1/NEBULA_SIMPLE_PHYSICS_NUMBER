@echo off
echo ========================================
echo   Verificacion de configuracion RSNA
echo ========================================
echo.

set ERROR_COUNT=0

REM Verificar ejecutable
echo [1/5] Verificando ejecutable...
if exist "SIMPLE_PHYSICS_NUMBERS01_NEW.exe" (
    echo   OK: SIMPLE_PHYSICS_NUMBERS01_NEW.exe encontrado
) else (
    echo   ERROR: SIMPLE_PHYSICS_NUMBERS01_NEW.exe NO encontrado
    echo          Solucion: Ejecuta compile_main01.bat
    set /a ERROR_COUNT+=1
)

REM Verificar train.csv en carpeta de dataset
echo [2/5] Verificando datos originales RSNA...
if exist "rsna-intracranial-aneurysm-detection\train.csv" (
    echo   OK: rsna-intracranial-aneurysm-detection\train.csv encontrado
) else (
    echo   ERROR: train.csv NO encontrado en la carpeta de dataset
    echo          Esperado en: rsna-intracranial-aneurysm-detection\train.csv
    set /a ERROR_COUNT+=1
)

REM Verificar train_mips.csv
echo [3/5] Verificando train_mips.csv...
if exist "train_mips.csv" (
    echo   OK: train_mips.csv encontrado
) else (
    echo   AVISO: train_mips.csv NO encontrado
    echo          Generelo con create_train_mips.py o process_large_dataset.bat
)

REM Verificar carpeta de MIPs
echo [4/5] Verificando imagenes MIP...
if exist "mips" (
    echo   OK: Carpeta mips encontrada
) else (
    echo   AVISO: Carpeta mips NO encontrada
    echo          Se creara al generar los MIPs
)

REM Verificar carpeta de series DICOM
echo [5/5] Verificando series DICOM...
if exist "rsna-intracranial-aneurysm-detection\series" (
    echo   OK: Carpeta rsna-intracranial-aneurysm-detection\series encontrada
) else (
    echo   AVISO: Carpeta series no encontrada (opcional para entrenamiento)
    echo          Nota: Solo necesaria para generar nuevos MIPs
)

echo.
echo ========================================
if %ERROR_COUNT%==0 (
    echo   CONFIGURACION CORRECTA
    echo.
    echo Todo listo para entrenar! Ejecuta:
    echo   run_rsna_train.bat          ^(basico^)
    echo   run_rsna_train_full.bat     ^(completo^)
) else (
    echo   ENCONTRADOS %ERROR_COUNT% PROBLEMAS
    echo.
    echo Corrige los errores mostrados arriba antes de continuar.
)
echo ========================================

pause

