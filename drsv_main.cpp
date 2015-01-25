#include <string>
#include <vector>
#include "Eigen/Dense"
#include "drsv.cpp"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  vector<Node*> tra, dev, test;
  
  srand(13457);
  
  readTrees(tra, "trees/train.txt");
  readTrees(dev, "trees/dev.txt");
  readTrees(test, "trees/test.txt");

  LookupTable *lt = new LookupTable();
  lt->load("/home/oirsoy/googlenews-mikolov-300", 200000, 300, true);
  Node::LT = lt;

  if (ADAGRAD)
    Node::lr = 0.01;
  else
    Node::lr = 0.002;
  Node::la = 0.0001;
  Node::mr = 0.9;

  Node::nx = 300;
  Node::nh = 50;
  Node::ny = 5;

  train(tra, dev, test);
  return 0;
}
