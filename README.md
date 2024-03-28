embeddings are pending ;-; 

Read me is pending, i spent the morning reinstalling all of WSL + python, tensorflow, magenta.

Process to install and run on WSL2:

Windows 10 must be build 20145 or later.
Windows 11 is all okay.

Get latest driver for your gpu: https://www.nvidia.com/download/index.aspx (Yinas def has one, kevin i am not sure)

In 'cmd' as Admin:
```
wsl --install
wsl --install -d Ubuntu-20.04
#check kernel version, it must be at least 4.19.121
wsl -v
```

In wsl:
```
#if this fails, your driver is not done correctly
nvidia-smi
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-wsl-ubuntu-11-2-local_11.2.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-2-local_11.2.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-2-local/7fa2af80.pub 
sudo apt-get update
sudo apt-get -y install cuda-11-2

echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```
Now download cuDNN Library for Linux (x86_64) from https://developer.nvidia.com/rdp/cudnn-archive
Place .tar in \\wsl.localhost\Ubuntu-20.04\home\yournamefromsetup
In wsl:
```
CUDNN_TAR_FILE="cudnn-11.2-linux-x64-v8.1.1.33.tgz"
tar -xzvf ${CUDNN_TAR_FILE}

sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.2/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64/
sudo chmod a+r /usr/local/cuda-11.2/lib64/libcudnn*

#to verify, check:
nvidia-smi
nvcc -V
```
Cuda and CUDNN are now installed
To install dependencies:
```
#check to ensure version is 3.8.10
python3 -V

sudo apt-get install python3-pip
pip install tensorflow==2.9.1
pip apt-get install tensorflow-gpu==2.9.1
sudo apt-get install build-essential libasound2-dev libjack-dev portaudio19-dev
pip install magenta

pip install -U apache-beam==2.22.0
pip install apache-beam[interactive]
pip install -U resampy==0.3.1
pip install -U Jinja2==3.1.2
pip install -U protobuf==3.20.3
pip install -U tensorflow-metadata==1.13.0
pip install -U MarkupSafe==2.1.1

#check tensorflow install, this should show your GPU(s)
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

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
