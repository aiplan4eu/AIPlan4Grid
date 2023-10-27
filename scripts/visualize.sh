if [ $# -ne 1 ]; then
    echo "Usage: ./visualize.sh <env_path>"
    exit 1
fi

cd ..
echo "Installing grid2viz..."
pip install grid2viz
echo "Launching grid2viz web app..."
grid2viz --agents_path results --env_path $1
