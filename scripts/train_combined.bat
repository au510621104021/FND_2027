@echo off
setlocal

cd /d "%~dp0.."
python scripts\train.py --config config\config.yaml %*

