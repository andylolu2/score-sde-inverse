# Ubuntu 22.04 comes with Python 3.10
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG USERNAME=dockeruser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEBIAN_FRONTEND=noninteractive

# Install apt packages
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-add-repository ppa:fish-shell/release-3 && \
    apt-add-repository ppa:neovim-ppa/unstable && \
    apt update && \
    apt install -y \
        build-essential \
        curl \
        fish \
        git \
        neovim \
        python3-pip \
        ripgrep \
        sudo \
        tar \
        tmux \
        wget && \
    ln -s /usr/bin/python3 /usr/bin/python

# Intall python packages
ADD requirements.txt /tmp/requirements.txt
RUN pip install -U -r /tmp/requirements.txt

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /usr/bin/fish \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# (Optional) Setup dotfiles
ADD https://api.github.com/repos/andylolu2/dotfiles/git/refs/heads/main /tmp/version.json
RUN git clone https://github.com/andylolu2/dotfiles $HOME/.dotfiles && \
    $HOME/.dotfiles/main.fish
