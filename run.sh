# embeddings
curl -O http://www-nlp.stanford.edu/data/glove.6B.50d.txt.gz 
gzip -d glove.6B.50d.txt.gz

# dataset
curl -O http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip trainDevTestTrees_PTB.zip

# Eigen
curl -L http://bitbucket.org/eigen/eigen/get/3.2.4.tar.gz -o eigen.tar.gz
tar -xzvf eigen.tar.gz --strip-components=1 eigen-eigen-10219c95fe65/Eigen

# compile & run
g++ drsv_main.cpp -I ./Eigen/ -std=c++11 -O3 -fopenmp -o drsv
./drsv
