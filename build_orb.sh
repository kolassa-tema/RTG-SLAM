#!/bin/bash  
set -e

mkdir -p thirdParty && cd thirdParty
install_path=$(pwd)/install
mkdir -p ${install_path}

python_prefix=$(python -c "import sys; print(sys.prefix)")  
python_include=${python_prefix}/include/python3.9/
python_lib=${python_prefix}/lib/libpython3.9.so
python_exe=${python_prefix}/bin/python
python_env=${python_prefix}/lib/python3.9/site-packages/
numpy_include=$(python -c "import numpy; print(numpy.get_include())")  

echo ${python_env}

# # build pangolin
if [ ! -d "Pangolin" ]; then
    git clone -b v0.6 https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=${install_path}
    make install -j
    cd ../../
else
    echo "Pangolin directory already exists, skipping this step."
fi

opencv_version="4.2.0"
# build opencv
# if [ ! -d "opencv-${opencv_version}" ]; then
#     wget https://github.com/opencv/opencv/archive/${opencv_version}.zip
#     unzip ${opencv_version}.zip
#     cd opencv-${opencv_version}
#     mkdir -p build && cd build
#     cmake .. -DCMAKE_INSTALL_PREFIX=${install_path}
#     make install -j
#     cd ../../
# else
#     echo "opencv-${opencv_version} directory already exists, skipping this step."
# fi

opencv_dir=/usr/lib/x86_64-linux-gnu/cmake/opencv4/

# build orbslam2
if [ ! -d "ORB-SLAM2-PYBIND/build" ]; then
    cd ORB-SLAM2-PYBIND
    bash build.sh ${opencv_dir} ${install_path}
    cd ../
else
    echo "ORB-SLAM2-PYBIND/build directory already exists, skipping this step."
fi


# # build pybind
# # build boost
if [ ! -d "boost_1_80_0" ]; then
    wget -t 999 -c https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.zip
    unzip boost_1_80_0.zip
fi

if [ ! -d "install/include/boost" ]; then
    cd boost_1_80_0
    ./bootstrap.sh --with-libraries=python --prefix=${install_path} --with-python=${python_exe}

# # ./b2
    ./b2 install --with-python include=${python_include} --prefix=${install_path}
    cd ../

fi



# # build orbslam_pybind
cd pybind
if [ ! -d "build" ]; then
    mkdir -p build && cd build

    cmake .. -DPYTHON_INCLUDE_DIRS=${python_include} \
            -DPYTHON_LIBRARIES=${python_lib} \
            -DPYTHON_EXECUTABLE=${python_exe} \
            -DBoost_INCLUDE_DIRS=${install_path}/include/ \
            -DBoost_LIBRARIES=${install_path}/lib/libboost_python39.so \
            -DORB_SLAM2_INCLUDE_DIR=${install_path}/include/ORB_SLAM2 \
            -DORB_SLAM2_LIBRARIES=${install_path}/lib/libORB_SLAM2.so \
            -DOpenCV_DIR=${opencv_dir} \
            -DPangolin_DIR=${install_path}/lib/cmake/Pangolin \
            -DPangolin_LIBRARIES=${install_path}/lib/libpangolin.so \
            -DPangolin_INCLUDE_DIRS=${install_path}/include/pangolin \
            -DPYTHON_NUMPY_INCLUDE_DIR=${numpy_include} \
            -DCMAKE_INSTALL_PREFIX=${python_env}

    cd ..
else
    echo "pybind/build directory already exists, skipping this step."
fi
cd build
make install -j
cd ../../



export LD_LIBRARY_PATH=/workspace/thirdParty/install/lib/:$LD_LIBRARY_PATH