@echo off
echo configuring jamlang...

REM Persist install root
setx JAM_INSTALL_DIR %~dp0

echo %JAM_INSTALL_DIR%

REM add binaries to PATH
:: setx PATH "%PATH%;%~dp0\bin"

echo Done. Please restart your terminal to apply changes.
pause