FROM python:3.7.8

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install nano && apt-get install git-lfs 
RUN pip install --upgrade pip

# install libs
RUN apt-get install gfortran libgtk-3-0 -y && apt-get install libhdf5-dev libc-ares-dev libeigen3-dev ffmpeg -y \
    && apt-get install libatlas-base-dev libopenblas-dev libblas-dev -y \
    && apt-get install openmpi-bin libopenmpi-dev -y \
    && apt-get install liblapack-dev cython -y \
    && pip3 install keras_applications==1.0.8 --no-deps \
    && pip3 install keras_preprocessing==1.1.0 --no-deps \
    && pip3 install -U --user six wheel mock \
    && pip3 install pybind11 \
    && pip install https://www.piwheels.org/simple/numpy/numpy-1.20.3-cp37-cp37m-linux_armv7l.whl \
    && pip3 install https://www.piwheels.org/simple/h5py/h5py-2.10.0-cp37-cp37m-linux_armv7l.whl \
    && pip3 install --upgrade setuptools
    
RUN pip install https://www.piwheels.org/simple/pillow/Pillow-8.2.0-cp37-cp37m-linux_armv7l.whl \
    && pip install https://www.piwheels.org/simple/scipy/scipy-1.4.1-cp37-cp37m-linux_armv7l.whl \
    && pip install https://www.piwheels.org/simple/lxml/lxml-4.6.3-cp37-cp37m-linux_armv7l.whl \
    && pip install https://www.piwheels.org/simple/pywavelets/PyWavelets-1.1.1-cp37-cp37m-linux_armv7l.whl \
    && pip install https://www.piwheels.org/simple/kiwisolver/kiwisolver-1.3.1-cp37-cp37m-linux_armv7l.whl \
    && pip install https://www.piwheels.org/simple/matplotlib/matplotlib-3.4.2-cp37-cp37m-linux_armv7l.whl \
    && pip install https://www.piwheels.org/simple/scikit-image/scikit_image-0.18.1-cp37-cp37m-linux_armv7l.whl \
    && pip install https://www.piwheels.org/simple/opencv-python/opencv_python-4.5.1.48-cp37-cp37m-linux_armv7l.whl

RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg \
    && pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl

RUN pip install tqdm absl-py easydict imutils keras

ENTRYPOINT ["python", "./DetectFromStream.py"]
