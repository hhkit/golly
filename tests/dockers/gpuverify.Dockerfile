FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
USER root 
WORKDIR /

RUN apt update
RUN apt install -y git build-essential python3 python3-pip python3-distutils wget unzip llvm clang libclang-dev
RUN apt install ca-certificates gnupg
RUN gpg --homedir /tmp --no-default-keyring --keyring /usr/share/keyrings/mono-official-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
RUN echo "deb [signed-by=/usr/share/keyrings/mono-official-archive-keyring.gpg] https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list
RUN apt update
RUN apt install -y mono-devel
RUN pip3 install psutil

# RUN git clone https://github.com/mc-imperial/gpuverify
RUN wget https://github.com/mc-imperial/gpuverify/releases/download/2018-03-22/GPUVerifyLinux64.zip
RUN unzip GPUVerifyLinux64 -d gpuverify
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt install -y libncurses5 libc6-dev-i386
ENV PATH=$PATH:/gpuverify/2018-03-22
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV PATH="/usr/local/cuda/bin:$PATH"

ADD kernel_extractor kernel_extractor