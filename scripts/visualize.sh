if [ $# -ne 2 ]; then
    echo "Usage: ./visualize.sh <results_path> <env_path>"
    exit 1
fi

cd ..
echo "Installing grid2viz..."
pip install grid2viz
echo "Launching grid2viz web app..."
grid2viz --agents_path $1 --env_path $2
