#!/bin/bash

set -e

echo "Installing Python 3.9 and pip..."
sudo apt-get update && sudo apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Step 4: Upgrade pip
echo "Upgrading pip..."
python3.9 -m pip install --upgrade pip

# Step 5: Copy requirements.txt and install dependencies
if [ -f "requirements.txt" ]; then
    echo "Copying requirements.txt..."
    cp requirements.txt ./
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found in the current directory."
    exit 1
fi

# Step 6: Install hf_transfer
echo "Installing hf_transfer..."
pip install hf_transfer

# Step 7: Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 8: Copy the main.py script
if [ -f "main.py" ]; then
    echo "Copying main.py..."
    cp main.py ./
else
    echo "main.py not found in the current directory."
    exit 1
fi

# Step 9: Expose port 8002 (for information only, not applicable in a script)
PORT=8002
echo "The application will run on port $PORT."

# Step 10: Run the application
echo "Starting the application..."
python3.9 main.py
