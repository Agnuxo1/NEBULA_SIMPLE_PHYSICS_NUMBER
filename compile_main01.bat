@echo off
REM Ensure working directory is this script's folder
pushd "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc" -O3 --use_fast_math -std=c++17 ^
  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 ^
  -o "%~dp0SIMPLE_PHYSICS_NUMBERS01_NEW.exe" "%~dp0main01.cu" -lcurand -lcublas
echo Compilation finished
dir "%~dp0*.exe"
popd
pause
