@REM Installed using python 3.9, 
@REM latest currently supported version by pytorch
py -3.9 -m venv ..\env
call ..\env\scripts\activate.bat
python -m pip install -r requirements.txt
ECHO Virtual environment installed
PAUSE
