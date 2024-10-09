#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Install Go if it's not already installed
if ! command -v go &> /dev/null; then
    echo "Go is not installed. Installing Go..."
    wget https://golang.org/dl/go1.23.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go1.23.linux-amd64.tar.gz
    export PATH=$PATH:/usr/local/go/bin
    echo "Go installed successfully."
else
    echo "Go is already installed."
fi

# Step 2: Set the working directory
mkdir -p /app
cd /app

# Step 3: Copy go.mod and go.sum
# Note: Adjust the paths as necessary to your project structure
if [ -f "../go.mod" ] && [ -f "../go.sum" ]; then
    cp ../go.mod ./
    cp ../go.sum ./
else
    echo "go.mod and go.sum not found in the parent directory."
    exit 1
fi

# Step 4: Download and verify Go modules
go mod download && go mod verify

# Step 5: Copy the remaining application files
cp -r ../* ./

# Step 6: Build the Go application
go build -o main .

# Step 7: Expose the port (not applicable in a script but for information)
PORT=9092
echo "Application will run on port $PORT."

# Step 8: Run the Go application
./main
