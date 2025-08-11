# Test script to compile and run Aether examples

Write-Host "Testing Aether Examples" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green

$compiler = ".\target\release\aetherc.exe"

# Test hello_world.ae
Write-Host "`nTesting: hello_world.ae" -ForegroundColor Yellow
if (Test-Path "hello_world.ae") {
    Write-Host "Compiling..." -ForegroundColor Cyan
    & $compiler build hello_world.ae -o hello_test.exe
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Compilation successful" -ForegroundColor Green
        Write-Host "Running..." -ForegroundColor Cyan
        & .\hello_test.exe
        Remove-Item hello_test.exe -ErrorAction SilentlyContinue
    }
}

# Test loops_example.ae
Write-Host "`nTesting: examples\loops_example.ae" -ForegroundColor Yellow
if (Test-Path "examples\loops_example.ae") {
    Write-Host "Compiling..." -ForegroundColor Cyan
    & $compiler build examples\loops_example.ae -o loops_test.exe
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Compilation successful" -ForegroundColor Green
        Write-Host "Running..." -ForegroundColor Cyan
        & .\loops_test.exe
        Remove-Item loops_test.exe -ErrorAction SilentlyContinue
    }
}

# Test tensor_operations.ae
Write-Host "`nTesting: examples\tensor_operations.ae" -ForegroundColor Yellow
if (Test-Path "examples\tensor_operations.ae") {
    Write-Host "Compiling..." -ForegroundColor Cyan
    & $compiler build examples\tensor_operations.ae -o tensor_test.exe
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Compilation successful" -ForegroundColor Green
        Write-Host "Running..." -ForegroundColor Cyan
        & .\tensor_test.exe
        Remove-Item tensor_test.exe -ErrorAction SilentlyContinue
    }
}

Write-Host "`nTesting complete!" -ForegroundColor Green