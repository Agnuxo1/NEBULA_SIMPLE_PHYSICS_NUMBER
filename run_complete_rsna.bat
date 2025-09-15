@echo off
echo ==================================================
echo   SISTEMA COMPLETO RSNA - OPTICAL NEURAL NETWORK
echo ==================================================
echo.
echo Este script ejecuta el proceso completo para el RSNA:
echo 1. Configuracion inicial
echo 2. Compilacion del codigo
echo 3. Prueba rapida
echo 4. Entrenamiento (opcional)
echo.

REM Paso 1: Configuracion inicial
echo [PASO 1] Configuracion inicial...
call setup_rsna.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Fallo en la configuracion inicial.
    pause
    exit /b 1
)

echo.
echo [PASO 2] Compilacion del codigo...
call compile.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Fallo en la compilacion.
    pause
    exit /b 1
)

echo.
echo [PASO 3] Prueba rapida del sistema...
call test_rsna.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Fallo en la prueba rapida.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   SISTEMA LISTO PARA RSNA
echo ==================================================
echo.
echo El sistema ha sido configurado y probado exitosamente.
echo.

REM Preguntar si quiere proceder con el entrenamiento
set /p choice="¿Desea proceder con el entrenamiento completo? (s/n): "
if /i "%choice%"=="s" (
    echo.
    echo [PASO 4] Iniciando entrenamiento completo...
    echo.
    echo IMPORTANTE: Asegurese de que:
    echo - Ha descargado el dataset RSNA desde Kaggle
    echo - Ha generado train_mips.csv con sus datos reales
    echo - Tiene suficiente espacio en disco para los checkpoints
    echo.
    pause
    
    call run_rsna_train.bat
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ==================================================
        echo   ENTRENAMIENTO COMPLETADO
        echo ==================================================
        echo.
        echo El modelo ha sido entrenado exitosamente.
        echo Los checkpoints estan disponibles en:
        echo   - ckpt_front/
        echo   - ckpt_back/
        echo   - ckpt_left/
        echo   - ckpt_right/
        echo.
        echo Para realizar inferencia, use:
        echo   run_rsna_infer.bat ^<front.pgm^> ^<back.pgm^> ^<left.pgm^> ^<right.pgm^>
        echo.
    ) else (
        echo.
        echo ERROR: Fallo en el entrenamiento.
        echo Revise los mensajes de error anteriores.
        echo.
    )
) else (
    echo.
    echo Entrenamiento omitido.
    echo.
    echo Para entrenar mas tarde, ejecute:
    echo   run_rsna_train.bat
    echo.
    echo Para realizar inferencia, ejecute:
    echo   run_rsna_infer.bat ^<front.pgm^> ^<back.pgm^> ^<left.pgm^> ^<right.pgm^>
    echo.
)

echo ==================================================
echo   PROCESO COMPLETO FINALIZADO
echo ==================================================
echo.
echo Archivos creados:
echo   - main01.exe (ejecutable principal)
echo   - train_mips.csv (dataset procesado)
echo   - mips/ (imagenes MIP)
echo   - ckpt_*/ (checkpoints del modelo)
echo.
echo Scripts disponibles:
echo   - setup_rsna.bat (configuracion inicial)
echo   - compile.bat (compilacion)
echo   - test_rsna.bat (prueba rapida)
echo   - run_rsna_train.bat (entrenamiento)
echo   - run_rsna_infer.bat (inferencia)
echo   - create_train_mips.py (procesamiento de datos)
echo.
echo Documentacion: README_RSNA.md
echo.

pause



