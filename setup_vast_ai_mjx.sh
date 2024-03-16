cd ~/

sudo apt update -y
sudo apt upgrade -y

sudo apt install -y fish htop nano git python3.9-full
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install mujoco mujoco-mjx

# set shell to fish
chsh -s /usr/bin/fish

# install croc
curl https://getcroc.schollz.com | bash

# git clone https://github.com/nikisalli/nightmare_rl
cd nightmare_rl
pip install -r requirements.txt
pip uninstall -y torch
pip install torch

# install rsl_rl
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout tags/v1.0.2
pip install .
cd ..