#!/bin/bash
sudo apt install -y curl
curl -sSL https://dist.crystal-lang.org/apt/setup.sh | sudo bash
sudo apt install -y crystal cmake libeigen3-dev gnuplot
./build.sh
