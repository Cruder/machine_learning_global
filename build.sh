set -e

echo -e "\e[36mCreate folder\e[39m"
mkdir -p neuratron/build
cd neuratron/build
echo -e "\e[36mCreate cmake artefacts\e[39m"
cmake ..
echo -e "\e[36mBuild lib\e[39m"
sudo make install

LD_LIBRARY_PATH=`pwd`/
echo -e "\e[36mExport LD_LIBRARY\e[39m"
echo -e "\e[35mValue " $LD_LIBRARY_PATH "\e[39m"
export LD_LIBRARY_PATH

cd ../../

cd matrix
echo -e "\e[36mBuild Crystal\e[39m"
crystal build src/matrix.cr

echo -e "\e[32mAll done\e[39m"
