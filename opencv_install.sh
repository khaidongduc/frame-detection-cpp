sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-3.4/modules ../opencv-3.4
cmake --build .
sudo make install