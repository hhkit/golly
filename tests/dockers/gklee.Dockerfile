FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
USER root 
WORKDIR /

ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y g++ cmake libboost-all-dev bison flex perl git wget ninja-build patch
RUN git clone https://github.com/Geof23/Gklee.git
ADD gklee.patch gklee.patch
RUN cd Gklee && git apply ../gklee.patch
RUN mkdir Gklee/build

WORKDIR /Gklee/build
RUN apt install -y python2
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2 9
RUN cmake .. -G Ninja
RUN cmake --build .

ENV KLEE_HOME_DIR=/Gklee
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV C_INCLUDE_PATH="/usr/local/cuda/include:$C_INCLUDE_PATH"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda/include:$CPLUS_INCLUDE_PATH"
ENV PATH="/usr/local/cuda/bin:$KLEE_HOME_DIR/bin:$PATH"

RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
WORKDIR /workspace