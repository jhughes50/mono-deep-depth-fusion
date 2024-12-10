FROM nvidia/cuda:12.6.1-base-ubuntu24.04

#Run the frontend first so it doesn't throw an error later
RUN apt-get update \
  && export TZ="America/New_York" \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
  && ln -fs "/usr/share/zoneinfo/$TZ" /etc/localtime \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && apt-get clean

# General dependencies for development
RUN apt-get update \
  && apt-get install -y \
  build-essential \
  cmake \
  cppcheck \
  gdb \
  git \
  libeigen3-dev \
  g++ \
  libbluetooth-dev \
  libcwiid-dev \
  libgoogle-glog-dev \
  libspnav-dev \
  libusb-dev \
  libpcl-dev \
  lsb-release \
  mercurial \
  python3-dbg \
  python3-empy \
  python3-pip \
  python3-venv \
  software-properties-common \
  sudo \
  wget \
  curl \
  cmake-curses-gui \
  geany \
  tmux \
  dbus-x11 \
  iputils-ping \
  default-jre \
  iproute2 \
  vim \
  && apt-get clean

# remove ubuntu user
RUN userdel -f ubuntu

# add a user
ENV user_id=1000
ARG USER jason
RUN useradd -U --uid ${user_id} -ms /bin/bash $USER \
 && echo "$USER:$USER" | chpasswd \
 && adduser $USER sudo \
 && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER

# Set locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
  locale-gen
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=n_US:en

# Commands below run as the developer user
USER $USER

WORKDIR /home/$USER

COPY ./ml-depth-pro ./ml-depth-pro
RUN cd ml-depth-pro \
 && sudo pip3 install . --break-system-packages

#RUN git clone https://github.com/isl-org/Open3D \
# && cd Open3D \
# && yes yes | ./util/install_deps_ubuntu.sh \
# && mkdir build \
# && cd build \
# && cmake .. \
# && make -j$(nproc) \
# && sudo make install \
# && sudo make pip-package 
#
#RUN cd Open3D \
# && pip3 install --break-system-package build/lib/python_package/pip_package/open3d_cpu-0.18.0+b7e5f163e-cp312-cp312-manylinux_2_39_x86_64.whl

RUN echo 'export PS1="\[$(tput setaf 2; tput bold)\]\u\[$(tput setaf 7)\]@\[$(tput setaf 3)\]\h\[$(tput setaf 7)\]:\[$(tput setaf 4)\]\W\[$(tput setaf 7)\]$ \[$(tput sgr0)\]"' >> ~/.bashrc

