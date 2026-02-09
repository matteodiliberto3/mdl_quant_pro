@echo off
title FINANCE PRO DASHBOARD - WATCHDOG
echo Avvio del Guardiano del Bot...
:loop
echo [ %DATE% %TIME% ] Avvio MDL Quant Pro...
py mdl_quant_pro.py
echo.
echo ⚠️ ATTENZIONE: Il Bot si e' chiuso! Riavvio in 5 secondi...
timeout /t 5
goto loop