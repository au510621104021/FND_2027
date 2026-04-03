@echo off
setlocal

cd /d "%~dp0.."
python scripts\evaluate.py --config config\config.yaml --checkpoint checkpoints\best_model.pt %*

