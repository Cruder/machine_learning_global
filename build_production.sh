mkdir -p neuratron/build
cd neuratron/build
cmake ..
make

LD_LIBRARY_PATH=`pwd`/
export LD_LIBRARY_PATH

cd ../../

cd matrix
crystal build --release src/matrix.cr
