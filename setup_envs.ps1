# Define a function to set up the project
function Setup-Project {
    param (
        [string]$ProjectDir
    )

    # Variables to check for the presence of required files
    $requirementsFileExists = Test-Path "$ProjectDir\requirements.txt"
    $pythonFilesExist = @(Get-ChildItem -Path $ProjectDir -Filter *.py).Count -gt 0

    Write-Host "------------------------------------"
    Write-Host "Checking directory: $ProjectDir"

    if ($requirementsFileExists) {
        Write-Host "Found requirements.txt in $ProjectDir"
    } else {
        Write-Host "No requirements.txt found in $ProjectDir"
    }

    if ($pythonFilesExist) {
        Write-Host "Found Python files in $ProjectDir"
    } else {
        Write-Host "No Python files found in $ProjectDir"
    }

    # Proceed only if both a requirements.txt and Python files exist
    if ($requirementsFileExists -and $pythonFilesExist) {
        Write-Host "Setting up environment for $ProjectDir"
        
        # Navigate to project directory
        Push-Location $ProjectDir
    
        # Create virtual environment
        Write-Host "Creating virtual environment..."
        python -m venv venv
        Write-Host "Virtual environment created at $ProjectDir\venv"
        
        # Activate virtual environment
        Write-Host "Activating virtual environment..."
        .\venv\Scripts\Activate.ps1 

        # Install dependencies
        Write-Host "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
        Write-Host "Dependencies installed successfully"

        # Deactivate virtual environment
        Write-Host "Deactivating virtual environment..."
        Deactivate

        # Return to original directory
        Pop-Location
    } else {
        Write-Host "Skipping $ProjectDir: requirements.txt or Python files not found"
    }
    Write-Host "------------------------------------"
}

# Determine the base directory
$RelativePath = ".\services"  # Change this to your desired relative path
$BaseDirectory = Resolve-Path -Path $RelativePath

# Get all subdirectories within the base directory
$Subdirectories = Get-ChildItem -Path $BaseDirectory -Directory

# Run the setup for each subdirectory
foreach ($Dir in $Subdirectories) {
    Setup-Project -ProjectDir $Dir.FullName
}