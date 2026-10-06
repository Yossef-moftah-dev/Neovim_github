#!/usr/bin/env bash
set -euo pipefail

TARGET_USER="${SUDO_USER:-$USER}"

echo "Detecting operating system..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID:-}"
    OS_LIKE="${ID_LIKE:-}"
else
    echo "Unsupported distribution: /etc/os-release not found." >&2
    exit 1
fi

case "$OS_ID" in
    ubuntu|debian)
        echo "Installing Docker via official Apt repository on $PRETTY_NAME..."
        sudo apt-get update -y
        sudo apt-get install -y ca-certificates curl gnupg
        sudo install -m 0755 -d /etc/apt/keyrings
        
        sudo curl -fsSL "https://download.docker.com/linux/$OS_ID/gpg" -o /etc/apt/keyrings/docker.asc
        sudo chmod a+r /etc/apt/keyrings/docker.asc

        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$OS_ID $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
            sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        sudo apt-get update -y
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        ;;

    amzn)
        echo "Installing Docker on Amazon Linux..."
        if [ "$VERSION_ID" = "2023" ]; then
            sudo dnf update -y
            sudo dnf install -y docker
        else
            sudo amazon-linux-extras install -y docker || sudo yum install -y docker
        fi
        ;;

    rhel|centos|rocky|almalinux|fedora)
        echo "Installing Docker on RPM-based distribution..."
        PKG_MGR=$(command -v dnf >/dev/null 2>&1 && echo "dnf" || echo "yum")
        sudo $PKG_MGR install -y dnf-plugins-core
        sudo $PKG_MGR config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        sudo $PKG_MGR install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        ;;

    *)
        echo "Distribution $OS_ID ($OS_LIKE) not directly matched. Attempting official get.docker script..."
        curl -fsSL https://get.docker.com | sudo sh
        ;;
esac

# Start and enable Docker service
echo "Enabling and starting Docker service..."
sudo systemctl enable --now docker

# Allow target user to run docker without sudo
if ! getent group docker >/dev/null; then
    sudo groupadd docker
fi
sudo usermod -aG docker "$TARGET_USER"

# Configure 2GB swap space if no swap is active (vital for builds on t2/t3 instances)
if [ "$(swapon --show | wc -l)" -le 1 ]; then
    echo "Configuring 2GB swap space..."
    sudo fallocate -l 2G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    echo "✓ 2GB swap file enabled."
fi

echo "=== Docker & Environment installed successfully ==="
echo "Run 'newgrp docker' or log out and back in to apply group permissions."
docker --version