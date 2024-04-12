# Music Generation Using Autoencoders and Transformer Mixture Distribution Models
Prepared for ECE324H1-S. Companion project report can be found in 'Music_Generation_Using_Autoencoders_and_Transformer_Mixture_Distribution_Models.pdf'

Authors: Kevin Gao 1007638790, Yina Gao 1008084485, David Guo 1007677275

## Installation
All code is written in Python 3. Development was done on WSL2 in Ubuntu 20.04.

To install WSL2:

Windows 10 must be build 20145 or later.

Windows 11 is all okay.

Get latest driver for your gpu: https://www.nvidia.com/download/index.aspx 

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
Now download cuDNN v8.1.1: 'cuDNN Library for Linux (x86_64)' from https://developer.nvidia.com/rdp/cudnn-archive

Place .tar in \home\yournamefromsetup

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
Cuda and cuDNN are now installed

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
pip install -U apache-beam[interactive]
pip install -U resampy==0.3.1
pip install -U Jinja2==3.1.2
pip install -U protobuf==3.20.3
pip install -U tensorflow-metadata==1.13.0
pip install -U MarkupSafe==2.1.1
pip install -U jax==0.2.8
pip install -U jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -U flax==0.3.0
pip install -U ray==1.1.0
pip install -U aiohttp==3.6.2
pip install -U redis==3.5.3
pip install -U aiohttp-cors==0.7.0
pip install -U aioredis===1.3.1
pip install pretty-midi
pip install pyfluidsynth
pip install pypianoroll

#check tensorflow install, this should show your GPU(s)
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Data Pre-processing and Setup
To start, download [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) and a subset of [Classical Archive's MIDI Database](https://www.classicalarchives.com/midi.html). You will have to split them into training, testing, and validation sets.

Download the MusicVAE checkpoint 'cat-mel_2bar_big' [here](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar).

In 'data_processing.ipynb' follow all steps sequentially for both datasets (thus denoted as 'lmd_wan' and 'classical_midi_split'), replacing all file paths as needed. Note that it is not recommended to process or train on the full Lahk MIDI dataset as it is very large and will multiple days on lower end systems.

## Training

Run 'training.ipynb'; hyperparameters, training settings, checkpoint frequency, epoch number and similar settigns can be configured via flags in 'train_autoregressive.py'

To resume training, save the checkpoint to resume from in './save/mdn/checkpoint_#', set 'resume' to 'True' and set 'ckpt_dir' to 'checkpoint_#'. 

Note that 'checkpoint_#' is a placeholder, and anything will work as long as the checkpoint is in the correct folder under './save/mdn/' and the name of that folder is passed via 'ckpt_dir' 

## Generation

After training, place the desired checkpoint to generate off of in './save/mdn/checkpoint_#' with the same setup as in training. 

Run 'results.ipynb' to generate results and audio samples. Samples are saved to './audio'. 

**Note: alteratively, since running the results notebook is time-consuming (and requires proper set up of the environment), to listen to the generated samples directly, open the MIDI files located in './audio/gen/audio'. The random and testing MIDI samples are located in './audio/prior/audio' and './audio/real/audio' respectively, if you wish to compare.**


