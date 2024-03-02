sudo apt update -y
sudo apt upgrade -y

sudo apt install -y fish htop nano git
sudo apt install -y python3-pip
sudo apt install -y python3-venv
sudo apt install -y zstd

# set shell to fish
chsh -s /usr/bin/fish

# install croc
curl https://getcroc.schollz.com | bash

# install rsl_rl
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout tags/v1.0.2
pip install -e .
cd ..

# git clone https://github.com/nikisalli/nightmare_rl
cd nightmare_rl
pip install -r requirements.txt
