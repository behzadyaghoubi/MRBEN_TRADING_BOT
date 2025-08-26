# MR BEN AI System - PowerShell Execution Script
# Handles encoding issues and runs the comprehensive update

Write-Host "🎯 MR BEN AI System - PowerShell Execution" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Set encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Function to clean commands
function Clean-Command {
    param([string]$Command)

    $unwantedPrefixes = @('رز', 'ز', 'ر', 'زpython', 'رpython')

    foreach ($prefix in $unwantedPrefixes) {
        if ($Command.StartsWith($prefix)) {
            $Command = $Command.Substring($prefix.Length)
            Write-Host "Removed unwanted prefix: $prefix" -ForegroundColor Yellow
        }
    }

    return $Command.Trim()
}

# Function to run command safely
function Invoke-CommandSafely {
    param(
        [string]$Command,
        [string]$Description = ""
    )

    Write-Host "`n📊 Executing: $Description" -ForegroundColor Cyan

    # Clean the command
    $cleanCmd = Clean-Command $Command
    Write-Host "Command: $cleanCmd" -ForegroundColor Gray

    try {
        # Run the command
        $result = Invoke-Expression $cleanCmd 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $Description completed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ $Description failed with exit code: $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Error executing $Description`: $_" -ForegroundColor Red
        return $false
    }
}

# Main execution
Write-Host "Starting MR BEN AI System execution..." -ForegroundColor Yellow

# Step 1: Test system components
Write-Host "`n📊 Step 1: Testing system components..." -ForegroundColor Cyan
$success = Invoke-CommandSafely -Command "python test_system_update.py" -Description "System component test"

if (-not $success) {
    Write-Host "❌ System component test failed. Please check the logs." -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

# Step 2: Run comprehensive update
Write-Host "`n🚀 Step 2: Running comprehensive system update..." -ForegroundColor Cyan
$success = Invoke-CommandSafely -Command "python fixed_comprehensive_update.py" -Description "Comprehensive system update"

if ($success) {
    Write-Host "`n✅ MR BEN AI System update completed successfully!" -ForegroundColor Green
    Write-Host "📋 Check the logs/ directory for detailed reports" -ForegroundColor Yellow
} else {
    Write-Host "`n❌ System update failed. Please check the logs for details." -ForegroundColor Red
}

Write-Host "`nMR BEN AI System execution completed" -ForegroundColor Green
Read-Host "Press Enter to continue"
