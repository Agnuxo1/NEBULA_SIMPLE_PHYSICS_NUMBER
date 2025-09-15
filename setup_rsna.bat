@echo off
echo ==================================================
echo   CONFIGURACION INICIAL RSNA
echo ==================================================
echo.

echo Este script configura el entorno para el entrenamiento RSNA.
echo.

REM Crear directorios necesarios
echo Creando directorios...
if not exist "mips" mkdir mips
if not exist "ckpt_front" mkdir ckpt_front
if not exist "ckpt_back" mkdir ckpt_back
if not exist "ckpt_left" mkdir ckpt_left
if not exist "ckpt_right" mkdir ckpt_right
if not exist "data" mkdir data

echo Directorios creados:
echo   - mips/ (para imagenes MIP)
echo   - ckpt_front/ (checkpoints red front)
echo   - ckpt_back/ (checkpoints red back)
echo   - ckpt_left/ (checkpoints red left)
echo   - ckpt_right/ (checkpoints red right)
echo   - data/ (para datos temporales)
echo.

REM Verificar dependencias
echo Verificando dependencias...

REM Verificar Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python no encontrado. Instale Python 3.6 o superior.
    pause
    exit /b 1
)

REM Verificar pydicom
python -c "import pydicom" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: pydicom no encontrado. Instalando...
    pip install pydicom
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: No se pudo instalar pydicom.
        pause
        exit /b 1
    )
)

REM Verificar PIL
python -c "from PIL import Image" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Pillow no encontrado. Instalando...
    pip install Pillow
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: No se pudo instalar Pillow.
        pause
        exit /b 1
    )
)

echo Dependencias verificadas.
echo.

REM Verificar si existe main01.exe
if not exist "main01.exe" (
    echo WARNING: main01.exe no encontrado.
    echo Asegurese de compilar el codigo CUDA antes de continuar.
    echo.
    echo Para compilar:
    echo   nvcc -o main01.exe main01.cu -lcublas -lcurand
    echo.
)

REM Crear archivo de ejemplo train_mips.csv si no existe
if not exist "train_mips.csv" (
    echo Creando archivo de ejemplo train_mips.csv...
    echo SeriesInstanceUID,label,front,back,left,right > train_mips.csv
    echo # Agregue aqui las filas con sus datos reales >> train_mips.csv
    echo.
    echo Archivo de ejemplo creado: train_mips.csv
    echo Edite este archivo con sus datos reales antes de entrenar.
    echo.
)

echo ==================================================
echo   CONFIGURACION COMPLETADA
echo ==================================================
echo.
echo Prximos pasos:
echo.
echo 1. Descargue el dataset RSNA desde Kaggle:
echo    https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data
echo.
echo 2. Genere train_mips.csv con el dataset local del repo:
echo    python create_train_mips.py --train_csv rsna-intracranial-aneurysm-detection\train.csv --images_dir rsna-intracranial-aneurysm-detection\series --output train_mips.csv
echo.
echo 3. Compile el codigo (si no lo ha hecho):
echo    nvcc -o main01.exe main01.cu -lcublas -lcurand
echo.
echo 4. Ejecute el entrenamiento:
echo    run_rsna_train.bat
echo.
echo 5. Para inferencia:
echo    run_rsna_infer.bat ^<front.pgm^> ^<back.pgm^> ^<left.pgm^> ^<right.pgm^>
echo.
echo ==================================================

pause



