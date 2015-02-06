#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <iterator>
#include <cassert>
#include "Eigen/Dense"

#define uint unsigned int

using namespace Eigen;
using namespace std;

VectorXd relu(const VectorXd &x) {
  return x.array().max(0);
}

VectorXd relup(const VectorXd &y) {
  return (y.array() > 0).cast<double>();
}

VectorXd softmax(const VectorXd &x) {
  double m = x.maxCoeff();
  VectorXd v = (x.array() - m).exp();
  return v.array() / v.sum();
}

VectorXd smxntp(const VectorXd &y, const VectorXd &r) {
  return y-r;
}

double str2double(const string& s) {
  istringstream i(s);
  double x;
  if (!(i >> x))
    return 0;
  return x;
}

class LookupTable {
  public:
    void load(string fname, uint n, uint d, bool noUnknown=false);
    //ColXpr operator[](string word);
    VectorXd operator[](string word);
    void gradAdd(string word, VectorXd v);
    void update();
    void clear();

  private:
    double lr;
    unordered_map<string, uint> table; // word -> index
    MatrixXd data;            // index -> vector representation
    MatrixXd gdata;           // gradients
    MatrixXd adata;           // adagrad past
    set<uint> modifiedCols;
};

void LookupTable::clear() {
  data = gdata = adata = MatrixXd();
}

void LookupTable::load(string fname, uint n, uint d, bool noUnknown) {
  ifstream in(fname.c_str());
  assert(in.is_open());
  string line;
  if (noUnknown) n++;
  data = MatrixXd(d,n);
  gdata = adata = MatrixXd::Zero(d,n);
  adata.fill(1e-6);
  uint j=0;
  while(std::getline(in, line)) {
    std::istringstream ss(line);
    std::istream_iterator<std::string> begin(ss), end;

    //putting all the tokens in the vector
    std::vector<std::string> tokens(begin, end); 
    for (uint i=0; i<d; i++)
      data(i,j) = str2double(tokens[i+1]);
    table[tokens[0]] = j;
    j++;
    if (j == n) break;
  }
  
  if (noUnknown) {
    VectorXd v = data.rowwise().mean();
    data.col(n-1) = v;
    table["*UNKNOWN*"] = n-1;
  }

  double min = data.minCoeff();
  //cout << "Lookup table min: " << min << endl;
  //data = data.array()-min;
}

VectorXd LookupTable::operator[](string word) {
  unordered_map<string,uint>::iterator it;

  // this might not be the best place for this,
  // if i'm calling this frequently
  if (word == "-LRB-")
    word = "(";
  else if (word == "-RRB-")
    word = ")";
  else if (word == "-LSB-")
    word = "(";
  else if (word == "-RSB-")
    word = ")";
  else if (word == "-LCB-")
    word = "(";
  else if (word == "-RCB-")
    word = ")";

  it = table.find(word);
  if (it != table.end()) // exists
    return data.col(table[word]);
  else
    return data.col(table["*UNKNOWN*"]);
}

void LookupTable::gradAdd(string word, VectorXd v) {
  unordered_map<string,uint>::iterator it;
  if (word == "-LRB-")
    word = "(";
  else if (word == "-RRB-")
    word = ")";
  else if (word == "-LSB-")
    word = "(";
  else if (word == "-RSB-")
    word = ")";
  else if (word == "-LCB-")
    word = "(";
  else if (word == "-RCB-")
    word = ")";
 
  it = table.find(word);
  if (it != table.end()) {// exists
    gdata.col(table[word]) += v;
    modifiedCols.insert(table[word]);
  } else {
    gdata.col(table["*UNKNOWN*"]) += v;
    modifiedCols.insert(table[word]);
  }
}

void LookupTable::update() {
  lr = 0.001;
  for (auto i : modifiedCols) {
    adata.col(i) = (adata.col(i).cwiseProduct(adata.col(i)) + 
                  gdata.col(i).cwiseProduct(gdata.col(i))).cwiseSqrt();
    data.col(i) -= lr*gdata.col(i).cwiseQuotient(adata.col(i));
    gdata.col(i).setZero();
  }
  modifiedCols.clear();
}

// index of max in a vector
uint argmax(const VectorXd& x) {
  double max = x(0);
  uint maxi = 0;
  for (uint i=1; i<x.size(); i++) {
    if (x(i) > max) {
      max = x(i);
      maxi = i;
    }
  }
  return maxi;
}

// this is used for randomly initializing an Eigen matrix
double urand(double dummy) {
  double min = -1, max = 1;
  return (double(rand())/RAND_MAX)*(max-min) + min;
        //+ int((double(rand())/RAND_MAX) < 0.5);
}

// KFY shuffle (uniformly randomly) of a vector
template <class T>
void shuffle(vector<T>& v) {
  if (v.size() == 0) return;
  for (uint i=v.size()-1; i>0; i--) {
    uint j = (rand() % i);
    T tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
  }
}

template <class T, class T2>
void shuffle(vector<T>& v, vector<T2>& w) {
  assert(w.size() == v.size());
  if (w.size() == 0) return;
  for (uint i=v.size()-1; i>0; i--) {
    uint j = (rand() % i);
    T tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
    T2 tmp2 = w[i];
    w[i] = w[j];
    w[j] = tmp2;
  }
}

bool isWhitespace(std::string str) {
  for(uint i=0; i<str.size(); i++) {
    if (!isspace(str[i]))
      return false;
  }
  return true;
}

vector<string> split(const string &s, char delim) {
  stringstream ss(s);
  string item;
  vector<string> elems;
  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

