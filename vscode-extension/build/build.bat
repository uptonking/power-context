@echo off
setlocal EnableExtensions
pushd "%~dp0"

for %%I in ("..\context-engine-uploader") do set "EXT_DIR=%%~fI"
for %%I in ("..\out") do set "OUT_DIR=%%~fI"
for %%I in ("..\..\scripts\standalone_upload_client.py") do set "SRC_SCRIPT=%%~fI"
set "CLIENT=standalone_upload_client.py"
set "STAGE_DIR=%OUT_DIR%\extension-stage"
set "BUILD_RESULT=0"

echo Building clean Context Engine Uploader extension...

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if errorlevel 1 (
    echo Failed to create output directory.
    set "BUILD_RESULT=1"
    goto cleanup
)

REM Ensure the source extension directory has no lingering upload client
if exist "%EXT_DIR%\%CLIENT%" del /f /q "%EXT_DIR%\%CLIENT%"

copy /Y "%SRC_SCRIPT%" "%OUT_DIR%\%CLIENT%" >nul
if errorlevel 1 (
    echo Failed to copy upload client from scripts directory.
    set "BUILD_RESULT=1"
    goto cleanup
)

if exist "%STAGE_DIR%" rd /s /q "%STAGE_DIR%"
mkdir "%STAGE_DIR%"
if errorlevel 1 (
    echo Failed to create staging directory.
    set "BUILD_RESULT=1"
    goto cleanup
)

robocopy "%EXT_DIR%" "%STAGE_DIR%" /E /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 (
    echo Failed to copy extension into staging directory.
    set "BUILD_RESULT=1"
    goto cleanup
)

copy /Y "%OUT_DIR%\%CLIENT%" "%STAGE_DIR%\%CLIENT%" >nul
if errorlevel 1 (
    echo Failed to place upload client into staging directory.
    set "BUILD_RESULT=1"
    goto cleanup
)

pushd "%STAGE_DIR%"
echo Packaging extension...
npx @vscode/vsce package --no-dependencies --out "%OUT_DIR%"
if errorlevel 1 (
    echo Packaging failed.
    set "BUILD_RESULT=1"
    popd
    goto cleanup
)
popd

echo Build complete! Check the /out directory for .vsix files.
dir "%OUT_DIR%\*.vsix"

:cleanup
if exist "%STAGE_DIR%" rd /s /q "%STAGE_DIR%"
popd
endlocal & exit /b %BUILD_RESULT%