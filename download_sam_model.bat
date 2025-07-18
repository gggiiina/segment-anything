@echo off
setlocal

echo 📦 開始下載 SAM 模型 checkpoint...
set URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
set OUTPUT=sam_vit_h_4b8939.pth

:: 檢查檔案是否已存在
if exist "%OUTPUT%" (
    echo ✅ 模型已存在：%OUTPUT%
    goto :done
)

:: 使用 PowerShell 下載
echo 🌐 正在下載...
powershell -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%OUTPUT%'"

:: 檢查是否成功
if exist "%OUTPUT%" (
    echo ✅ 下載成功：%OUTPUT%
) else (
    echo ❌ 下載失敗！請手動下載：
    echo %URL%
)

:done
echo 完成。
pause
