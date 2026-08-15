@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title X Space Translator - 初回セットアップ

echo ==============================================
echo  X Space Translator 初回セットアップ
echo ==============================================
echo この処理には数分から十数分かかる場合があります。
echo.

where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)
if not defined PYTHON_CMD (
  where python >nul 2>nul
  if errorlevel 1 goto :python_error
  set "PYTHON_CMD=python"
)

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
if errorlevel 1 goto :python_error

echo 使用するPython:
%PYTHON_CMD% --version
echo.

if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
  if errorlevel 1 goto :venv_error
) else (
  echo [1/5] 専用のPython環境を作成しています...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :failed
)

echo [2/5] pipを更新しています...
call .venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo [3/5] 必要なパッケージをインストールしています...
echo       ダウンロードのためインターネット接続が必要です。
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [4/5] 設定ファイルを準備しています...
if not exist ".env" copy ".env.example" ".env" >nul
if not exist ".env" goto :failed

echo [5/5] FFmpegを確認しています...
where ffmpeg >nul 2>nul
if errorlevel 1 goto :ffmpeg_error

echo FFmpegが見つかりました。
echo.
echo ==============================================
echo  セットアップが完了しました。
echo  次に start.bat をダブルクリックしてください。
echo ==============================================
pause
exit /b 0

:python_error
echo.
echo [エラー] Python 3.11が見つかりません。
echo Python 3.11をインストールし、Windowsを再起動してから
echo setup.batをもう一度実行してください。
echo.
echo インストール例:
echo   winget install -e --id Python.Python.3.11
echo 公式サイト: https://www.python.org/downloads/
pause
exit /b 1

:venv_error
echo.
echo [エラー] 既存の.venvがPython 3.11用ではありません。
echo .venvフォルダを削除または別名へ変更してから、
echo setup.batをもう一度実行してください。
pause
exit /b 1

:ffmpeg_error
echo.
echo [エラー] FFmpegが見つかりません。
echo 以下のコマンドでインストールできます。
echo   winget install -e --id Gyan.FFmpeg
echo.
echo インストール後にWindowsを再起動し、setup.batを
echo もう一度実行してください。
echo 公式サイト: https://ffmpeg.org/download.html
pause
exit /b 1

:failed
echo.
echo [エラー] セットアップを完了できませんでした。
echo 上に表示された内容を確認してください。
echo インターネット接続を確認し、setup.batをもう一度実行してください。
echo 「ファイル名または拡張子が長すぎます」やWinError 206の場合は、
echo フォルダを C:\XSpaceTranslator のような短い場所へ移して再実行してください。
pause
exit /b 1
