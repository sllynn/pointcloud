#!/bin/bash
# --
# This is for Ubuntu 22.04 (Jammy)
# [1] corresponds to DBR 13+
# [2] jammy offers GDAL 3.4.1 and PDAL 2.3.0
# Author: Stuart Lynn | stuart@databricks.com
# Last Modified: 05 Aug, 2024

# refresh package info
# 0.4.2 - added "-y"
sudo apt-add-repository -y "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc)-backports main universe multiverse restricted"
sudo apt-add-repository -y "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc)-updates main universe multiverse restricted"
sudo apt-add-repository -y "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc)-security main multiverse restricted universe"
sudo apt-add-repository -y "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) main multiverse restricted universe"
sudo apt-get update -y

# install natives
sudo apt-get -o DPkg::Lock::Timeout=-1 install -y unixodbc libcurl3-gnutls libsnappy-dev libopenjp2-7
sudo apt-get -o DPkg::Lock::Timeout=-1 install -y gdal-bin libgdal-dev pdal libpdal-dev python3-numpy python3-gdal

# pip install gdal
# matches jammy version
pip install --upgrade pip
pip install gdal==3.4.1