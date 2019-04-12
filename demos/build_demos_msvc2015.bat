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


@setlocal
set "ROOT_DIR=%~dp0"

set "SOLUTION_DIR64=%USERPROFILE%\Documents\Intel\OpenVINO\omz_demos_build_2015"
if exist "%SOLUTION_DIR64%" rd /s /q "%SOLUTION_DIR64%"
if "%InferenceEngine_DIR%"=="" set "InferenceEngine_DIR=%ROOT_DIR%\..\share"
if exist "%ROOT_DIR%\..\..\bin\setupvars.bat" call "%ROOT_DIR%\..\..\bin\setupvars.bat"
if exist "%ROOT_DIR%\..\..\..\bin\setupvars.bat" call "%ROOT_DIR%\..\..\..\bin\setupvars.bat"


if exist "C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
   set "VS_VERSION=14 2015"
)

if exist "%SOLUTION_DIR64%\CMakeCache.txt" del "%SOLUTION_DIR64%\CMakeCache.txt"

echo Creating Visual Studio 2015 (x64) files in %SOLUTION_DIR64%... && ^
cd "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR64%" && cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio 14 2015 Win64" "%ROOT_DIR%"

if "%VS_VERSION%" == "" (
   echo Build tools for Visual Studio 2015 or 2017 cannot be found. If you use Visual Studio 2017, please download and install build tools from https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
   GOTO errorHandling
)

echo.
echo ###############^|^| Build Open Model Zoo demos using MS Visual Studio (MSBuild.exe) ^|^|###############
echo.
echo %MSBUILD_BIN%" Demos.sln /p:Configuration=Release
"%MSBUILD_BIN%" Demos.sln /p:Configuration=Release
if ERRORLEVEL 1 GOTO errorHandling


echo Done.
goto :eof

:errorHandling
echo Error
