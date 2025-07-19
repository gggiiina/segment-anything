@echo off
setlocal

:: 建立虛擬環境
python -m venv venv
call venv\Scripts\activate.bat

:: 使用清華大學 Python 鏡像 + 最快安裝
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --prefer-binary
