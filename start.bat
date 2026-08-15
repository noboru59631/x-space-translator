@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title X Space Translator

if not exist ".venv\Scripts\python.exe" (
  echo [エラー] 初回セットアップが完了していません。
  echo 先に setup.bat をダブルクリックしてください。
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [エラー] FFmpegが見つかりません。
  echo setup.batをもう一度実行し、表示される案内を確認してください。
  pause
  exit /b 1
)

if not exist ".env" copy ".env.example" ".env" >nul
if not exist ".env" (
  echo [エラー] 設定ファイル.envを作成できませんでした。
  echo このフォルダへの書き込み権限を確認してください。
  pause
  exit /b 1
)

set "APP_PORT="
for /f %%P in ('powershell.exe -NoProfile -Command "$port=8765; while ($port -le 8775) { try { $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$port); $listener.Start(); $listener.Stop(); Write-Output $port; exit 0 } catch { $port++ } }; exit 1"') do set "APP_PORT=%%P"
if not defined APP_PORT (
  echo [エラー] 使用できるポートが見つかりませんでした。
  echo 8765から8775を使用中のアプリを閉じて、もう一度お試しください。
  pause
  exit /b 1
)
set "APP_URL=http://127.0.0.1:%APP_PORT%"

echo X Space Translatorを起動しています...
echo URL: %APP_URL%
echo 準備ができるとブラウザが自動で開きます。
echo 終了するには、この画面を閉じるか Ctrl+C を押してください。
echo.

start "" /min powershell.exe -NoProfile -WindowStyle Hidden -Command "$url='%APP_URL%'; for ($i=0; $i -lt 90; $i++) { try { Invoke-WebRequest -UseBasicParsing -Uri ($url + '/api/health') -TimeoutSec 1 | Out-Null; Start-Process $url; exit 0 } catch { Start-Sleep -Seconds 1 } }"
call .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port %APP_PORT%

echo.
echo X Space Translatorを終了しました。
pause
