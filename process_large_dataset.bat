@echo off
echo ==================================================
echo   PROCESAMIENTO DATASET RSNA GRANDE (300GB)
echo ==================================================
echo.

REM Configurar parámetros para dataset grande
set TRAIN_CSV=%1
set IMAGES_DIR=%2
set OUTPUT_CSV=%3
set SAMPLE_RATE=%4

REM Valores por defecto si no se especifican
if "%TRAIN_CSV%"=="" set TRAIN_CSV=rsna-intracranial-aneurysm-detection\train.csv
if "%IMAGES_DIR%"=="" set IMAGES_DIR=rsna-intracranial-aneurysm-detection\series
if "%OUTPUT_CSV%"=="" set OUTPUT_CSV=train_mips.csv
if "%SAMPLE_RATE%"=="" set SAMPLE_RATE=0.05

echo Configuración:
echo   Train CSV: %TRAIN_CSV%
echo   Images Dir: %IMAGES_DIR%
echo   Output CSV: %OUTPUT_CSV%
echo   Sample Rate: %SAMPLE_RATE% (5%% por defecto para dataset grande)
echo.

REM Verificar que existen los archivos/directorios
if not exist "%TRAIN_CSV%" (
    echo ERROR: No se encontró %TRAIN_CSV%
    echo Uso: process_large_dataset.bat [train.csv] [images_dir] [output.csv] [sample_rate]
    pause
    exit /b 1
)

if not exist "%IMAGES_DIR%" (
    echo ERROR: No se encontró directorio %IMAGES_DIR%
    echo Uso: process_large_dataset.bat [train.csv] [images_dir] [output.csv] [sample_rate]
    pause
    exit /b 1
)

echo Iniciando procesamiento con parámetros optimizados para dataset grande...
echo.

REM Ejecutar con parámetros optimizados para dataset grande
python create_train_mips.py ^
    --train_csv "%TRAIN_CSV%" ^
    --images_dir "%IMAGES_DIR%" ^
    --output "%OUTPUT_CSV%" ^
    --sample_rate %SAMPLE_RATE% ^
    --chunk_size 50 ^
    --memory_limit 3.0 ^
    --skip_existing ^
    --resume

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================================
    echo   PROCESAMIENTO COMPLETADO
    echo ==================================================
    echo Archivo generado: %OUTPUT_CSV%
    echo MIPs guardados en: mips/
    echo.
    echo Para procesar más muestras, ejecute con mayor sample_rate:
    echo   process_large_dataset.bat %TRAIN_CSV% %IMAGES_DIR% %OUTPUT_CSV% 0.1
    echo.
) else (
    echo.
    echo ==================================================
    echo   ERROR EN EL PROCESAMIENTO
    echo ==================================================
    echo Código de error: %ERRORLEVEL%
    echo.
    echo Para reanudar el procesamiento:
    echo   process_large_dataset.bat %TRAIN_CSV% %IMAGES_DIR% %OUTPUT_CSV% %SAMPLE_RATE%
    echo.
)

pause






