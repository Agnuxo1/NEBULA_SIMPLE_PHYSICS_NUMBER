@echo off
echo ==================================================
echo   INFERENCIA RSNA - OPTICAL NEURAL NETWORK
echo ==================================================
echo.

REM Verificar que se proporcionaron las rutas de las imágenes
if "%1"=="" (
    echo Uso: run_rsna_infer.bat ^<ruta_front^> ^<ruta_back^> ^<ruta_left^> ^<ruta_right^>
    echo.
    echo Ejemplo:
    echo   run_rsna_infer.bat mips\series001_front.pgm mips\series001_back.pgm mips\series001_left.pgm mips\series001_right.pgm
    echo.
    echo Las rutas deben apuntar a archivos PGM de las 4 vistas MIP.
    pause
    exit /b 1
)

if "%2"=="" (
    echo ERROR: Faltan argumentos. Se requieren 4 rutas de imagen.
    pause
    exit /b 1
)

if "%3"=="" (
    echo ERROR: Faltan argumentos. Se requieren 4 rutas de imagen.
    pause
    exit /b 1
)

if "%4"=="" (
    echo ERROR: Faltan argumentos. Se requieren 4 rutas de imagen.
    pause
    exit /b 1
)

REM Verificar que existen los archivos de imagen
if not exist "%1" (
    echo ERROR: No se encontro el archivo front: %1
    pause
    exit /b 1
)

if not exist "%2" (
    echo ERROR: No se encontro el archivo back: %2
    pause
    exit /b 1
)

if not exist "%3" (
    echo ERROR: No se encontro el archivo left: %3
    pause
    exit /b 1
)

if not exist "%4" (
    echo ERROR: No se encontro el archivo right: %4
    pause
    exit /b 1
)

echo Archivos de imagen verificados:
echo   Front:  %1
echo   Back:   %2
echo   Left:   %3
echo   Right:  %4
echo.

REM Configurar variables de entorno para la inferencia RSNA
set RSNA_INFER=1
set RSNA_FRONT=%1
set RSNA_BACK=%2
set RSNA_LEFT=%3
set RSNA_RIGHT=%4

REM Configurar checkpoints (opcional, usar los mejores disponibles)
set CKPT_FRONT=ckpt_front\best.bin
set CKPT_BACK=ckpt_back\best.bin
set CKPT_LEFT=ckpt_left\best.bin
set CKPT_RIGHT=ckpt_right\best.bin

REM Si no existen los checkpoints "best", usar los de la ultima epoca
if not exist "%CKPT_FRONT%" (
    set "_last="
    for /f "delims=" %%i in ('dir /b /od ckpt_front\*.bin 2^>nul') do set "_last=%%i"
    if defined _last set "CKPT_FRONT=ckpt_front\%_last%"
)

if not exist "%CKPT_BACK%" (
    set "_last="
    for /f "delims=" %%i in ('dir /b /od ckpt_back\*.bin 2^>nul') do set "_last=%%i"
    if defined _last set "CKPT_BACK=ckpt_back\%_last%"
)

if not exist "%CKPT_LEFT%" (
    set "_last="
    for /f "delims=" %%i in ('dir /b /od ckpt_left\*.bin 2^>nul') do set "_last=%%i"
    if defined _last set "CKPT_LEFT=ckpt_left\%_last%"
)

if not exist "%CKPT_RIGHT%" (
    set "_last="
    for /f "delims=" %%i in ('dir /b /od ckpt_right\*.bin 2^>nul') do set "_last=%%i"
    if defined _last set "CKPT_RIGHT=ckpt_right\%_last%"
)

echo Checkpoints configurados:
echo   Front:  %CKPT_FRONT%
echo   Back:   %CKPT_BACK%
echo   Left:   %CKPT_LEFT%
echo   Right:  %CKPT_RIGHT%
echo.

REM Ejecutar la inferencia
echo Ejecutando inferencia...
main01.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================================
    echo   INFERENCIA COMPLETADA EXITOSAMENTE
    echo ==================================================
    echo El resultado se ha mostrado en formato JSON.
    echo Las probabilidades representan la deteccion de aneurisma
    echo en cada una de las 14 arterias + deteccion global.
) else (
    echo.
    echo ==================================================
    echo   ERROR EN LA INFERENCIA
    echo ==================================================
    echo Codigo de error: %ERRORLEVEL%
    echo Revise los mensajes de error anteriores.
)

echo.
pause



