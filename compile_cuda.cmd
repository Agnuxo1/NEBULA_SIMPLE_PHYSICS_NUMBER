@echo off
echo ==================================================
echo   COMPILACION CUDA - OPTICAL NEURAL NETWORK
echo ==================================================
echo.

REM Configurar entorno de Visual Studio
echo Configurando entorno de Visual Studio 2022...
call "E:\VS2022\VC\Auxiliary\Build\vcvars64.bat"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: No se pudo configurar el entorno de Visual Studio.
    echo Verifique que Visual Studio 2022 Build Tools está instalado.
    pause
    exit /b 1
)

echo Entorno configurado correctamente.
echo.

REM Verificar que nvcc está disponible
nvcc --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvcc no encontrado después de configurar el entorno.
    echo Verifique que CUDA Toolkit está instalado.
    pause
    exit /b 1
)

echo Compilador CUDA encontrado.
echo.

REM Verificar que el archivo fuente existe
if not exist "main01.cu" (
    echo ERROR: main01.cu no encontrado en el directorio actual.
    pause
    exit /b 1
)

echo Compilando main01.cu...
echo.

REM Compilar con optimizaciones
nvcc -O3 -std=c++17 -o main01.exe main01.cu -lcublas -lcurand

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================================
    echo   COMPILACION EXITOSA
    echo ==================================================
    echo Ejecutable creado: main01.exe
    echo.
    echo Para ejecutar el entrenamiento RSNA:
    echo   run_rsna_train.bat
    echo.
    echo Para realizar inferencia:
    echo   run_rsna_infer.bat ^<front.pgm^> ^<back.pgm^> ^<left.pgm^> ^<right.pgm^>
    echo.
) else (
    echo.
    echo ==================================================
    echo   ERROR EN LA COMPILACION
    echo ==================================================
    echo Codigo de error: %ERRORLEVEL%
    echo Revise los mensajes de error anteriores.
    echo.
    echo Posibles soluciones:
    echo 1. Verifique que CUDA Toolkit está instalado correctamente
    echo 2. Verifique que las librerias cublas y curand están disponibles
    echo 3. Revise la sintaxis del codigo CUDA
    echo.
)

echo.
pause
