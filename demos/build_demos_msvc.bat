@echo off

:: Copyright (C) 2018-2019 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::      http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.


setlocal EnableDelayedExpansion
set "ROOT_DIR=%~dp0"

set "SOLUTION_DIR64=%USERPROFILE%\Documents\Intel\OpenVINO\omz_demos_build"

set MSBUILD_BIN=
set VS_PATH=
set VS_VERSION=
set EXTRA_CMAKE_OPTS=

:argParse
if not "%1" == "" (
    rem cmd.exe mangles -DENABLE_PYTHON=YES into -DENABLE_PYTHON YES,
    rem so it gets split into two arguments
    if "%1" == "-DENABLE_PYTHON" (
        set EXTRA_CMAKE_OPTS=%EXTRA_CMAKE_OPTS% %1=%2
        shift & shift
        goto argParse
    )

    if not "%VS_VERSION%" == "" (
        echo Unexpected argument: "%1"
        goto errorHandling
    )

    if "%1"=="VS2015" (
        set "VS_VERSION=2015"
    ) else if "%1"=="VS2017" (
        set "VS_VERSION=2017"
    ) else if "%1"=="VS2019" (
        set "VS_VERSION=2019"
    ) else (
        echo Unrecognized Visual Studio version specified: "%1"
        echo Supported versions: VS2015, VS2017, VS2019
        goto errorHandling
    )

    shift
    goto argparse
)

if "%INTEL_OPENVINO_DIR%"=="" (
    if exist "%ROOT_DIR%\..\..\bin\setupvars.bat" (
        call "%ROOT_DIR%\..\..\bin\setupvars.bat"
    ) else if exist "%ROOT_DIR%\..\..\..\bin\setupvars.bat" (
        call "%ROOT_DIR%\..\..\..\bin\setupvars.bat"
    ) else (
        echo Failed to set the environment variables automatically
        echo To fix, run the following command: ^<INSTALL_DIR^>\bin\setupvars.bat
        echo where INSTALL_DIR is the OpenVINO installation directory.
        goto errorHandling
    )
)

if "%PROCESSOR_ARCHITECTURE%" == "AMD64" (
    set "PLATFORM=x64"
) else (
    set "PLATFORM=Win32"
)

set VSWHERE="false"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    set VSWHERE="true"
    cd "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" (
    set VSWHERE="true"
    cd "%ProgramFiles%\Microsoft Visual Studio\Installer"
) else (
    echo "vswhere tool is not found"
)

if !VSWHERE! == "true" (
    if "!VS_VERSION!"=="" (
        echo Searching the latest Visual Studio...
        for /f "usebackq tokens=*" %%i in (`vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
            set VS_PATH=%%i
        )
    ) else (
        echo Searching Visual Studio !VS_VERSION!...
        for /f "usebackq tokens=*" %%i in (`vswhere -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
            set CUR_VS=%%i
            if not "!CUR_VS:%VS_VERSION%=!"=="!CUR_VS!" (
                set VS_PATH=!CUR_VS!
            )
        )
    )
    if exist "!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe"
    )
    if exist "!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe"
    )
    if exist "!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe"
    )
)

if "!MSBUILD_BIN!" == "" (
    if "!VS_VERSION!"=="2015" (
        if exist "C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe" (
            set "MSBUILD_BIN=C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
            set "MSBUILD_VERSION=14 2015"
        )
    ) else if "!VS_VERSION!"=="2017" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe" (
            set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe"
            set "MSBUILD_VERSION=15 2017"
        ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe" (
            set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
            set "MSBUILD_VERSION=15 2017"
        ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe" (
            set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
            set "MSBUILD_VERSION=15 2017"
        )
    )
) else (
    if not "!MSBUILD_BIN:2019=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=16 2019"
    if not "!MSBUILD_BIN:2017=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=15 2017"
    if not "!MSBUILD_BIN:2015=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=14 2015"
)

if "!MSBUILD_BIN!" == "" (
    echo Build tools for Microsoft Visual Studio !VS_VERSION! cannot be found. If you use Visual Studio 2017, please download and install build tools from https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
    goto errorHandling
)

if exist "%SOLUTION_DIR64%\CMakeCache.txt" del "%SOLUTION_DIR64%\CMakeCache.txt"

echo Creating Visual Studio %MSBUILD_VERSION% %PLATFORM% files in %SOLUTION_DIR64%... && ^
cd "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR64%"

cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio !MSBUILD_VERSION!" -A %PLATFORM% %EXTRA_CMAKE_OPTS% "%ROOT_DIR%"

echo.
echo ###############^|^| Build Inference Engine Demos using MS Visual Studio (MSBuild.exe) ^|^|###############
echo.
echo "!MSBUILD_BIN!" Demos.sln /p:Configuration=Release
"!MSBUILD_BIN!" Demos.sln /p:Configuration=Release
if ERRORLEVEL 1 goto errorHandling

echo Done.
goto :eof

:errorHandling
echo Error
exit /B 1
