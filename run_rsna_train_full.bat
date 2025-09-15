@echo off
echo ==================================================
echo   ENTRENAMIENTO RSNA COMPLETO - OPTICAL NEURAL NETWORK
echo ==================================================
echo.

REM Verificar que existe el archivo train_mips_full.csv
if not exist "train_mips_full.csv" (
    echo ERROR: No se encontro train_mips_full.csv
    echo.
    echo Para generar train_mips_full.csv, ejecute primero:
    echo   python create_train_mips.py --train_csv rsna-intracranial-aneurysm-detection\train.csv --images_dir rsna-intracranial-aneurysm-detection\series --output train_mips_full.csv --sample_rate 1.0 --memory_limit 12.0
    echo.
    pause
    exit /b 1
)

echo Archivo train_mips_full.csv encontrado.
echo Validando rutas y PGM...
python validate_mips_csv.py train_mips_full.csv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Validacion de CSV fallida. Corrige rutas/PGM antes de entrenar.
    pause
    exit /b 2
)
echo Iniciando entrenamiento RSNA con dataset completo...
echo.

REM Configurar variables de entorno para el entrenamiento RSNA
set RSNA_TRAIN_MIPS=1
set RSNA_MIPS_CSV=train_mips_full.csv
set MULTIEXPERT_4=1
set USE_MAX_PLUS=1
set USE_HADAMARD=1
set USE_KWTA=1
set KWTA_FRAC=0.15
set LI_ALPHA=0.08
set DISABLE_EARLY_STOP=1
set RSNA_NO_SHUFFLE=0
set RSNA_DEBUG_SAMPLES=0
set BATCH_SIZE=128
set CKPT_DIR_FRONT=ckpt_front
set CKPT_DIR_BACK=ckpt_back
set CKPT_DIR_LEFT=ckpt_left
set CKPT_DIR_RIGHT=ckpt_right

REM Configuraciones para dataset grande optimizado
set MAX_EPOCHS=100
set MAX_BATCHES=2000

REM Crear directorios de checkpoint si no existen
if not exist "ckpt_front" mkdir ckpt_front
if not exist "ckpt_back" mkdir ckpt_back
if not exist "ckpt_left" mkdir ckpt_left
if not exist "ckpt_right" mkdir ckpt_right

echo Variables de entorno configuradas:
echo   RSNA_TRAIN_MIPS=%RSNA_TRAIN_MIPS%
echo   RSNA_MIPS_CSV=%RSNA_MIPS_CSV%
echo   MULTIEXPERT_4=%MULTIEXPERT_4%
echo   MAX_EPOCHS=%MAX_EPOCHS%
echo   MAX_BATCHES=%MAX_BATCHES%
echo   BATCH_SIZE=%BATCH_SIZE%
echo   CKPT_DIR_FRONT=%CKPT_DIR_FRONT%
echo   CKPT_DIR_BACK=%CKPT_DIR_BACK%
echo   CKPT_DIR_LEFT=%CKPT_DIR_LEFT%
echo   CKPT_DIR_RIGHT=%CKPT_DIR_RIGHT%
echo.

REM Ejecutar el programa principal
echo Ejecutando main01.exe...
main01.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================================
    echo   ENTRENAMIENTO COMPLETADO EXITOSAMENTE
    echo ==================================================
    echo Los checkpoints se han guardado en:
    echo   - ckpt_front/
    echo   - ckpt_back/
    echo   - ckpt_left/
    echo   - ckpt_right/
    echo.
    echo Para realizar inferencia, use: run_rsna_infer.bat
) else (
    echo.
    echo ==================================================
    echo   ERROR EN EL ENTRENAMIENTO
    echo ==================================================
    echo Codigo de error: %ERRORLEVEL%
    echo Revise los mensajes de error anteriores.
)

echo.
pause
