Read me is pending, i spent the morning reinstalling all of WSL + python, tensorflow, magenta, BUT i did get a solid like process down to run everything locally that should be relatively painless for yall in WSL2 which is also pending ;-;

Structure your files as follows:
```
<name of working directory>/
├─ lmd_wan/
│  ├─ eval/
│  │  ├─ original (contains midi files)/
│  ├─ test/
│  │  ├─ original (contains midi files)/
│  ├─ train/
│  │  ├─ original (contains midi files)/
├─ cat-mel_2bar_big.tar (download this from magenta)
├─ <other code files on this level>
├─ <such as config.py, data_processing.ipynb etc>
├─ <other project folders are here too>/
```
