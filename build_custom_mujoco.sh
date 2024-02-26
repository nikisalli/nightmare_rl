# check we are in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please run this script in a virtual environment"
    exit 1
fi

cd /tmp
git clone https://github.com/google-deepmind/mujoco.git
cd mujoco
git switch 3.1.2
python -m pip install --upgrade --require-hashes -r python/build_requirements.txt

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:STRING=/tmp/mujoco_install -DMUJOCO_BUILD_EXAMPLES:BOOL=OFF -DCMAKE_CXX_FLAGS="-Wno-array-bounds -Wno-stringop-overread" -DCMAKE_C_FLAGS="-Wno-array-bounds -Wno-stringop-overread" ..
make -j$(nproc)

# build python bindings
cd ../python
bash make_sdist.sh
cd dist

MUJOCO_PATH=/tmp/mujoco_install MUJOCO_PLUGIN_PATH=/tmp/mujoco_install/mujoco_plugin MUJOCO_CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF" pip wheel -v --no-deps mujoco-*.tar.gz