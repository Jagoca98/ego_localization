FROM osrf/ros:noetic-desktop-full

# Set environment variable to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    wget \
    nano \
    sudo \
    git-all \
    libgl1 \
    sysvbanner \
    figlet
    
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python3 get-pip.py && \
    rm -f /usr/bin/pip && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    python3 -m pip install --upgrade pip setuptools
    
# Create a user with the specified UID
ARG USER_ID
ARG USER_NAME
ARG GROUP_ID
ARG GROUP_NAME
ARG WORKSPACE

# Remove the default user and create a new one
RUN groupadd -g $GROUP_ID $GROUP_NAME && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USER_NAME && \
    echo "$USER_NAME:$USER_NAME" | chpasswd && adduser $USER_NAME sudo

RUN mkdir /$WORKSPACE && \
    chown ${USER_NAME}:${GROUP_NAME} /${WORKSPACE} -R
    
# Switch to the new user
USER ${USER_ID}

RUN export PATH=$PATH:/home/$USER_NAME/.local/bin && \
echo "export PATH=$PATH:/home/$USER_NAME/.local/bin" >> /home/$USER_NAME/.bashrc

WORKDIR /$WORKSPACE

RUN pip install numpy --break-system-packages && \
    pip install tqdm --break-system-packages && \
    pip install scikit-learn --break-system-packages && \
    pip install matplotlib --break-system-packages && \
    pip install filterpy --break-system-packages && \
    pip install opencv-python --break-system-packages && \
    pip install numba --break-system-packages

# Add the figlet command to the bashrc to display a welcome message
RUN echo "figlet -f slant 'Ego localization!'" >> ~/.bashrc

USER root

# Grant permissions to the workspace directory
RUN mkdir -p /catkin_ws/src && \
    chown -R ${USER_NAME}:${GROUP_NAME} /catkin_ws/src && \
    chmod -R 775 /catkin_ws/src
RUN chown -R ${USER_NAME}:${GROUP_NAME} /catkin_ws && \
    chmod -R 775 /catkin_ws

WORKDIR /catkin_ws

RUN apt-get update && apt-get install -y \
    python3-catkin-tools


USER ${USER_ID}

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

# Set the default entry point to start the ROS environment
CMD ["tail", "-f", "/dev/null"]
