echo "Removing old build files..."
rm -rf dist/ build/ plan4grid.egg-info/
echo ""

echo "Uninstalling the package..."
pip uninstall plan4grid -y
echo ""

echo "Building the package..."
python -m build --sdist --wheel
echo ""

echo "Checking the package..."
twine check dist/*
