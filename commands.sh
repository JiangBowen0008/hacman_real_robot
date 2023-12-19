rsync --delete -avz /home/bowen/Projects/hacman_real_robot bowen@192.168.1.106:~/Projects
rsync --delete -avz /home/bowen/Projects/Anaconda3-2023.09-0-Linux-x86_64.sh bowen@192.168.1.106:~/Projects

# Source code from from protobuf-3.13.x
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure --prefix=/home/bowen/protobuf-3.13.x
make
make check
sudo make install
sudo ldconfig


# Setting to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
