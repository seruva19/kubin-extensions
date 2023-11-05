for /r %%i in (.installed) do if exist "%%i" del "%%i"
git pull
PAUSE
