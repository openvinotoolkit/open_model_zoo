@echo off

:: Copyright (C) 2018 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0
::


@setlocal
set "ROOT_DIR=%~dp0"

set "SOLUTION_DIR64=%USERPROFILE%\Documents\Intel\OpenVINO\inference_engine_demos_2017"
if exist "%SOLUTION_DIR64%" rd /s /q "%SOLUTION_DIR64%"
if "%InferenceEngine_DIR%"=="" set "InferenceEngine_DIR=%ROOT_DIR%\..\share"
if exist "%ROOT_DIR%\..\..\bin\setupvars.bat" call "%ROOT_DIR%\..\..\bin\setupvars.bat"
if exist "%ROOT_DIR%\..\..\..\bin\setupvars.bat" call "%ROOT_DIR%\..\..\..\bin\setupvars.bat"

echo Creating Visual Studio 2017 (x64) files in %SOLUTION_DIR64%... && ^
cd "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR64%" && cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio 15 2017 Win64" "%ROOT_DIR%"

echo Done.
pause
