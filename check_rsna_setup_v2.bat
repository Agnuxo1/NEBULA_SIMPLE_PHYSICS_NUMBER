@echo off
setlocal enabledelayedexpansion
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
    echo          Solucion: Ejecuta compile_main03.bat
    set /a ERROR_COUNT+=1
)

REM Verificar train.csv en carpeta de dataset
echo [2/5] Verificando datos originales RSNA...
if exist "rsna-intracranial-aneurysm-detection\train.csv" (
    echo   OK: rsna-intracranial-aneurysm-detection\train.csv encontrado
    REM Verificar que no este vacio
    for %%F in (rsna-intracranial-aneurysm-detection\train.csv) do (
        if %%~zF==0 (
            echo     WARNING: train.csv esta vacio
            set /a ERROR_COUNT+=1
        ) else (
            echo     Tamano: %%~zF bytes
        )
    )
) else (
    echo   ERROR: train.csv NO encontrado (rsna-intracranial-aneurysm-detection\train.csv)
    set /a ERROR_COUNT+=1
)

REM Verificar train_mips.csv
echo [3/5] Verificando train_mips.csv...
if exist "train_mips.csv" (
    echo   OK: train_mips.csv encontrado
    REM Verificar que tenga el header correcto
    set "first_line="
    for /f "usebackq delims=" %%A in ("train_mips.csv") do (
        set "first_line=%%A"
        goto :check_header
    )
    :check_header
    echo !first_line! | findstr /C:"SeriesInstanceUID,label,front,back,left,right" >nul
    if !errorlevel! == 0 (
        echo     OK: Header correcto
        for %%F in (train_mips.csv) do echo     Tamano: %%~zF bytes
    ) else (
        echo     ERROR: Header incorrecto en train_mips.csv
        echo     Esperado: SeriesInstanceUID,label,front,back,left,right
        echo     Encontrado: !first_line!
        set /a ERROR_COUNT+=1
    )
) else (
    echo   AVISO: train_mips.csv NO encontrado
    echo          Generelo con create_train_mips.py o process_large_dataset.bat
)

REM Verificar carpeta de MIPs
echo [4/5] Verificando imagenes MIP...
if exist "mips" (
    echo   OK: Carpeta mips encontrada
    REM Contar carpetas en mips
    set FOLDER_COUNT=0
    for /d %%D in (mips\*) do set /a FOLDER_COUNT+=1
    echo     Series encontradas: !FOLDER_COUNT!
    
    REM Verificar que al menos una carpeta tenga los 4 archivos PGM
    set PGM_CHECK=0
    for /d %%D in (mips\*) do (
        if exist "%%D\front.pgm" if exist "%%D\back.pgm" if exist "%%D\left.pgm" if exist "%%D\right.pgm" (
            set PGM_CHECK=1
            goto :pgm_found
        )
    )
    :pgm_found
    if !PGM_CHECK! == 1 (
        echo     OK: Archivos PGM verificados
    ) else (
        echo     AVISO: No se encontraron archivos PGM completos
        echo     Cada serie necesita: front.pgm, back.pgm, left.pgm, right.pgm
    )
) else (
    echo   AVISO: Carpeta mips NO encontrada
    echo          Se creara al generar los MIPs
)

REM Verificar carpeta de series DICOM
echo [5/5] Verificando series DICOM...
if exist "rsna-intracranial-aneurysm-detection\series" (
    set SERIES_COUNT=0
    for /d %%D in (rsna-intracranial-aneurysm-detection\series\*) do set /a SERIES_COUNT+=1
    echo   OK: Carpeta series encontrada (!SERIES_COUNT! series DICOM)
) else (
    echo   AVISO: Carpeta series no encontrada (opcional para entrenamiento)
    echo          Nota: Solo necesaria para generar nuevos MIPs
)

echo.
echo ========================================
if !ERROR_COUNT!==0 (
    echo   CONFIGURACION CORRECTA
    echo.
    echo Todo listo para entrenar! Ejecuta:
    echo   run_rsna_train.bat          ^(basico^)
    echo   run_rsna_train_full.bat     ^(completo^)
) else (
    echo   ENCONTRADOS !ERROR_COUNT! PROBLEMAS
    echo.
    echo Corrige los errores mostrados arriba antes de continuar.
)

endlocal
echo ========================================

pause

