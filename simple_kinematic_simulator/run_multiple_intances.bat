@echo off
setlocal enabledelayedexpansion

set /p times="Enter the number of times to run: "

for /l %%x in (1, 1, %times%) do (
   timeout /t 0.1
   start python train.py
)