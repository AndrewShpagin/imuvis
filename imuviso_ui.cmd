@echo off
setlocal
rem Double-click to use height.csv, or drop one CSV file onto this command file.
pushd "%~dp0"

py -3 -c "import sys" >nul 2>nul
if errorlevel 1 goto check_python
py -3 "%~dp0imuviso_ui.py" %*
set "imuviso_exit=%errorlevel%"
goto finished

:check_python
python -c "import sys" >nul 2>nul
if errorlevel 1 goto find_local_python
python "%~dp0imuviso_ui.py" %*
set "imuviso_exit=%errorlevel%"
goto finished

:find_local_python
set "imuviso_python="
for /d %%D in ("%LocalAppData%\Python\pythoncore-*") do if exist "%%~fD\python.exe" set "imuviso_python=%%~fD\python.exe"
for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do if exist "%%~fD\python.exe" set "imuviso_python=%%~fD\python.exe"
if not defined imuviso_python goto no_python
"%imuviso_python%" "%~dp0imuviso_ui.py" %*
set "imuviso_exit=%errorlevel%"
goto finished

:no_python
echo Python 3 was not found. Install Python, then run this launcher again.
set "imuviso_exit=1"

:finished
if not "%imuviso_exit%"=="0" pause
popd
exit /b %imuviso_exit%
