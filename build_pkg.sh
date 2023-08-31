echo "Removing old build files..."
rm -rf dist/ build/ plan4grid.egg-info/

echo "Uninstalling the package..."
pip uninstall plan4grid -y

echo "Building the package..."
python setup.py sdist bdist_wheel

echo "Installing the package..."
pip install dist/plan4grid-0.0.1-py3-none-any.whl

echo "Checking the package..."
twine check dist/*
