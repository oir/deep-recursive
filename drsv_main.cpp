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
  cout << "Loading word vectors..." << flush;
  // i used 300d word2vec in my own experiments.
  lt->load("glove.6B.50d.txt", 400000, 50, true);
  cout << " Done." << endl;
  Node::LT = lt;

  if (ADAGRAD)
    Node::lr = 0.01;
  else
    Node::lr = 0.002;
  Node::la = 0.0001;
  Node::mr = 0.9;

  Node::nx = 50;
  Node::nh = 50;
  Node::ny = 5;

  train(tra, dev, test);
  return 0;
}
