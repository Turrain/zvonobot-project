#!/bin/bash

# Function to create a virtual environment and install dependencies
setup_project() {
  project_dir="$1"
  has_requirements=false
  has_python_files=false

  echo "------------------------------------"
  echo "Checking directory: $project_dir"

  if [ -d "$project_dir" ]; then
    # Check for requirements.txt file
    if [ -f "$project_dir/requirements.txt" ]; then
      echo "Found requirements.txt in $project_dir"
      has_requirements=true
    else
      echo "No requirements.txt found in $project_dir"
    fi

    # Check for any Python files (.py)
    if ls "$project_dir"/*.py >/dev/null 2>&1; then
      echo "Found Python files in $project_dir"
      has_python_files=true
    else
      echo "No Python files found in $project_dir"
    fi


    if [ "$has_requirements" = true ] && [ "$has_python_files" = true ]; then
      echo "Setting up environment for $project_dir"
      cd "$project_dir" || { echo "Failed to change directory to $project_dir"; exit 1; }
      echo "Creating virtual environment..."
      python3 -m venv venv
      echo "Virtual environment created at $project_dir/venv"
      echo "Activating virtual environment..."
      source venv/bin/activate
      echo "Installing dependencies from requirements.txt..."
      pip install -r requirements.txt
      echo "Dependencies installed successfully"
      echo "Deactivating virtual environment..."
      deactivate
      cd - || { echo "Failed to return to starting directory"; exit 1; }
    else
      echo "Skipping $project_dir: requirements.txt or Python files not found"
    fi
  else
    echo "Directory $project_dir does not exist"
  fi
  echo "------------------------------------"
}

for dir in ./services/*/; do
  dir=${dir%/}
  setup_project "$dir"
done