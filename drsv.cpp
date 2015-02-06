#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include "Eigen/Dense"
#include "utils.cpp"

#define uint unsigned int

#define layers 3
#define MAXEPOCH 200
#define MINIBATCH 20
#define NORMALIZE true // relevant only with momentum
#define ADAGRAD true // adagrad or momentum

#define WI true // initialize W as 0.5*I
#define WIr true // regularize W to 0.5*I instead of 0

using namespace std;
using namespace Eigen;

double DROP = 0.1;

Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p=DROP) {
  for (uint i=0; i<x.size(); i++) {
    if ((double)rand()/RAND_MAX < p)
      x(i) = 0;
  }
  return x;
}

VectorXd (*f)(const VectorXd& x) = &relu;
VectorXd (*fp)(const VectorXd& y) = &relup;

class Node {
  public:
    Node();
    void forward(bool test);
    void backward();
    static void init();
    static void update();
    static void save(string);
    static void load(string);
    void print(string indent="");
    uint read(string &treeText, uint index, bool init);
  
    VectorXd x[layers], dx[layers], y, r;
    string word;
    // last dimensions are param, gradient, velocity/adapast respectively
    static MatrixXd Whhl[layers][3], Whhr[layers][3], Wxhl[layers][3], 
                    Wxhr[layers][3], Vxh[3], Vhh[layers][3], Uhy[layers][3],
                    Uxy[3];
    static VectorXd b[layers][3], c[3];

    static LookupTable* LT;
    static double lr, la, mr;
    static uint ny, nh, nx, fold;
    static double nnorm;

    MatrixXd& W(const uint& l, uint type=0);
    MatrixXd& V(const uint& l, uint type=0);
    MatrixXd& U(const uint& l, uint type=0);

    Node *left, *right;
    bool isLeft;
};

MatrixXd Node::Whhl[layers][3], Node::Whhr[layers][3], Node::Wxhl[layers][3], 
         Node::Wxhr[layers][3], Node::Vxh[3], Node::Vhh[layers][3], 
         Node::Uhy[layers][3], Node::Uxy[3];
VectorXd Node::b[layers][3], Node::c[3];
LookupTable* Node::LT;
double Node::lr, Node::la, Node::mr;
uint Node::ny, Node::nh, Node::nx, Node::fold;
double Node::nnorm = 0; // this is to test for explosion

Node::Node() {
  left = right = NULL;
  isLeft = false;
}


void Node::print(string indent) {
  cout << indent;
  if (!isLeft) { // root or right
    cout << "\\-";
    indent += " ";
  } else {    // left child
    cout << "|-";
    indent += "| ";
  }

  if (left == NULL) {
    cout << word << " : " << argmax(r) << " : " << argmax(y);
  } else 
    cout << " " << " : " << argmax(r) << " : " << argmax(y);
  cout << endl;

  if (left != NULL) {
    left->print(indent);
    right->print(indent);
  }
}

MatrixXd& Node::W(const uint& l, uint type) {
  if (l > 0 || left != NULL) {
    if (isLeft)
      return Whhl[l][type];
    else
      return Whhr[l][type];
  } else if (l==0 && left == NULL) {
    if (isLeft)
      return Wxhl[l][type];
    else
      return Wxhr[l][type];
  } else
    assert(false);
}
MatrixXd& Node::V(const uint& l, uint type) {
  assert(l > 0);
  if (left == NULL && l == 1)
    return Vxh[type];
  else
    return Vhh[l][type];
}
MatrixXd& Node::U(const uint& l, uint type) {
  if (left == NULL && l == 0)
    return Uxy[type];
  else
    return Uhy[l][type];
}

// for a root node, read a PTB tree and construct
// the tree recursively
uint Node::read(string &treeText, uint index, bool init) {
  char c; // current char
  string b; // buffer string
  uint numChild =0;

  for (uint i=index; i<treeText.size(); i++) {
    c = treeText[i];
    if (c == '(') {
      if (init) { // initial '(' is omitted since already root
        init = false;
        continue;
      }
      b = "";
      if (numChild==0) {
        left = new Node();
        left->isLeft = true;
        i = left->read(treeText, i+1, false);
      } else if (numChild==1) {
        right = new Node();
        right->isLeft = false;
        i = right->read(treeText, i+1, false);
      } else
        assert(false);
      numChild++;
    } else if (c == ')') {
      word = b;
      assert(numChild == 2 || numChild == 0);
      return i;
    } else if (isspace(c)) {
      if (numChild == 0) {
        r = VectorXd::Zero(5);  // buffer is label
        r[(int)str2double(b)] = 1;
      }
      b = ""; // reset buffer
    } else {
      b += c;
    }
  }
}

void Node::forward(bool test) {
  if (left != NULL) { // not a leaf
    left->forward(test);
    right->forward(test);
  } else {
    if (x[0].size() != nx)
      x[0] = (*LT)[word]; // don't have to repeat this if no finetuning
  }

  for (uint l=0; l<layers; l++) {
    VectorXd dropper = dropout(VectorXd::Ones(nh));    
    VectorXd v = b[l][0];
    if (left != NULL) { 
      v.noalias() += (left->W(l))*(left->x[l]) + (right->W(l))*(right->x[l]);
    }
    if (l > 0) 
      v.noalias() += V(l)*x[l-1];
    if (l > 0 || left != NULL) { // layer 0 leaves already have their x!!
      if (!test)
        x[l] = f(v).cwiseProduct(dropper);
      else
        x[l] = f(v)*(1-DROP);
    }
    dx[l] = VectorXd::Zero(x[l].size());
  }

  VectorXd v = c[0];
  for (uint l=layers-1; l<layers; l++)
    v += U(l)*x[l];
  y = softmax(v);
}

void Node::backward() {
  // unit regularize
  for (uint l=0; l<layers; l++)
    dx[l].noalias() += la*x[l];

  VectorXd gpyd = smxntp(y,r); 
  for (uint l=layers-1; l<layers; l++) {
    dx[l].noalias() += U(l).transpose() * gpyd;
    U(l,1).noalias() += gpyd * x[l].transpose();
  }
  c[1] += gpyd;

  for (int l=layers-1; l>=0; l--) {
    VectorXd fpxd = fp(x[l]).cwiseProduct(dx[l]);
    if (left != NULL) {
      left->dx[l].noalias() += (left->W(l)).transpose() * fpxd;
      right->dx[l].noalias() += (right->W(l)).transpose() * fpxd;
      left->W(l,1).noalias() += fpxd * (left->x[l]).transpose();
      right->W(l,1).noalias() += fpxd * (right->x[l]).transpose();
    }
    if (l > 0 || left != NULL)
      b[l][1].noalias() += fpxd;
    if (l > 0) {
      dx[l-1].noalias() += V(l).transpose() * fpxd;
      V(l,1).noalias() += fpxd * x[l-1].transpose();
    }
  }

  if (left != NULL) {
    left->backward();
    right->backward();
  } else {
    ; // word vector fine tuning can go here
  }
}

void Node::save(string fname) {
  ofstream out(fname.c_str());
  out << ny << " " << nh << " " << nx << " " << layers << endl;
  for (uint l=0; l<layers; l++) {
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nh; j++)
        out << Whhl[l][0](i,j) << " ";
    out << endl;
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nh; j++)
        out << Whhr[l][0](i,j) << " ";
    out << endl;
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nx; j++)
        out << Wxhl[l][0](i,j) << " ";
    out << endl;
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nx; j++)
        out << Wxhr[l][0](i,j) << " ";
    out << endl;
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nh; j++)
        out << Vhh[l][0](i,j) << " ";
    out << endl;
    for (uint i=0; i<ny; i++)
      for (uint j=0; j<nh; j++)
        out << Uhy[l][0](i,j) << " ";
    out << endl;
    for (uint i=0; i<nh; i++)
      out << b[l][0](i) << " ";
    out << endl;
  }
  for (uint i=0; i<nh; i++)
    for (uint j=0; j<nx; j++)
      out << Vxh[0](i,j) << " ";
  out << endl;
  for (uint i=0; i<ny; i++)
    for (uint j=0; j<nx; j++)
      out << Uxy[0](i,j) << " ";
  out << endl;
  for (uint i=0; i<ny; i++)
    out << c[0](i) << " ";
  out << endl;
}

void Node::load(string fname) {
  uint layers_;
  ifstream in(fname.c_str());
  in >> ny >> nh >> nx >> layers_;
  assert(layers == layers_);
  Node::init();
  for (uint l=0; l<layers; l++) {
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nh; j++)
        in >> Whhl[l][0](i,j);
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nh; j++)
        in >> Whhr[l][0](i,j);
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nx; j++)
        in >> Wxhl[l][0](i,j);
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nx; j++)
        in >> Wxhr[l][0](i,j);
    for (uint i=0; i<nh; i++)
      for (uint j=0; j<nh; j++)
        in >> Vhh[l][0](i,j);
    for (uint i=0; i<ny; i++)
      for (uint j=0; j<nh; j++)
        in >> Uhy[l][0](i,j);
    for (uint i=0; i<nh; i++)
      in >> b[l][0](i);
  }
  for (uint i=0; i<nh; i++)
    for (uint j=0; j<nx; j++)
      in >> Vxh[0](i,j);
  for (uint i=0; i<ny; i++)
    for (uint j=0; j<nx; j++)
      in >> Uxy[0](i,j);
  for (uint i=0; i<ny; i++)
    in >> c[0](i);
}

void Node::init() {
  for (uint l=0; l<layers; l++) {
    Whhl[l][0] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand))*(1/sqrt(nh));
    Whhr[l][0] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand))*(1/sqrt(nh));
    if (WI) {
      Whhl[l][0] += 0.5*MatrixXd::Identity(nh,nh);
      Whhr[l][0] += 0.5*MatrixXd::Identity(nh,nh);
    }
    Wxhl[l][0] = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand))*(1/sqrt(nx));
    Wxhr[l][0] = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand))*(1/sqrt(nx));
    Vhh[l][0] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand))*(1/sqrt(nh));// + MatrixXd::Identity(nh,nh);
    Uhy[l][0] = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand))*(1/sqrt(nh));
    b[l][0] = VectorXd(nh).unaryExpr(ptr_fun(urand))*(1/sqrt(nh));

    Whhl[l][1] = Whhr[l][1] = MatrixXd::Zero(nh, nh);
    Wxhl[l][1] = Wxhr[l][1] = MatrixXd::Zero(nh, nx);
    Vhh[l][1] = MatrixXd::Zero(nh, nh);
    Uhy[l][1] = MatrixXd::Zero(ny, nh);
    b[l][1] = VectorXd::Zero(nh);
   
    if (ADAGRAD) { 
      Whhl[l][2] = Whhr[l][2] = MatrixXd::Zero(nh, nh).array() + 0.001;
      Wxhl[l][2] = Wxhr[l][2] = MatrixXd::Zero(nh, nx).array() + 0.001;
      Vhh[l][2] = MatrixXd::Zero(nh, nh).array() + 0.001;
      Uhy[l][2] = MatrixXd::Zero(ny, nh).array() + 0.001;
      b[l][2] = VectorXd::Zero(nh).array() + 0.001;
    } else {
      Whhl[l][2] = Whhr[l][2] = MatrixXd::Zero(nh, nh);
      Wxhl[l][2] = Wxhr[l][2] = MatrixXd::Zero(nh, nx);
      Vhh[l][2] = MatrixXd::Zero(nh, nh);
      Uhy[l][2] = MatrixXd::Zero(ny, nh);
      b[l][2] = VectorXd::Zero(nh);

    }
  }
  Vxh[0] = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand))*(1/sqrt(nx));
  Uxy[0] = MatrixXd(ny,nx).unaryExpr(ptr_fun(urand))*(1/sqrt(nx));
  c[0] = VectorXd(ny).unaryExpr(ptr_fun(urand))*(1/sqrt(ny));
  Vxh[1] = MatrixXd::Zero(nh, nx);
  Uxy[1] = MatrixXd::Zero(ny,nx);
  c[1] = VectorXd::Zero(ny);

  if (ADAGRAD) {
    Vxh[2] = MatrixXd::Zero(nh, nx).array()+0.001;
    Uxy[2] = MatrixXd::Zero(ny,nx).array()+0.001;
    c[2] = VectorXd::Zero(ny).array()+0.001;
  } else {
    Vxh[2] = MatrixXd::Zero(nh, nx);
    Uxy[2] = MatrixXd::Zero(ny,nx);
    c[2] = VectorXd::Zero(ny);
  }
}

void Node::update() {
  double norm = 0;

  // regularize
  
  for (uint l=0; l<layers; l++) {
    if (WIr) {
      Whhl[l][1].noalias() += la * (Whhl[l][0] - 0.5*MatrixXd::Identity(nh,nh));
      Whhr[l][1].noalias() += la * (Whhr[l][0] - 0.5*MatrixXd::Identity(nh,nh));
    } else {
      Whhl[l][1].noalias() += la * Whhl[l][0];
      Whhr[l][1].noalias() += la * Whhr[l][0];
    }
    Wxhl[l][1].noalias() += la * Wxhl[l][0];
    Wxhr[l][1].noalias() += la * Wxhr[l][0];
    Vhh[l][1].noalias() += la * Vhh[l][0];
    Uhy[l][1].noalias() += la * Uhy[l][0];
    b[l][1].noalias() += la * b[l][0];

    norm += Whhl[l][1].squaredNorm() + Whhr[l][1].squaredNorm()
          + Wxhl[l][1].squaredNorm() + Wxhr[l][1].squaredNorm()
          + Vhh[l][1].squaredNorm() + b[l][1].squaredNorm()
          + Uhy[l][1].squaredNorm();
  }
  Vxh[1].noalias() +=  la * Vxh[0];
  Uxy[1].noalias() += la * Uxy[0];
  c[1].noalias() += la * c[0];

  norm += Vxh[1].squaredNorm() + Uxy[1].squaredNorm() + c[1].squaredNorm();
  
  double cap = 1; 
  if (NORMALIZE)
    norm = (norm > cap*cap) ? sqrt(norm/(cap*cap)) : 1;
  else
    norm = 1;

  if (!ADAGRAD) {
    // update velocities
    for (uint l=0; l<layers; l++) {
      Whhl[l][2] = lr * Whhl[l][1]/norm + mr * Whhl[l][2];
      Whhr[l][2] = lr * Whhr[l][1]/norm + mr * Whhr[l][2];
      Wxhl[l][2] = lr * Wxhl[l][1]/norm + mr * Wxhl[l][2];
      Wxhr[l][2] = lr * Wxhr[l][1]/norm + mr * Wxhr[l][2];
      Vhh[l][2] = lr * Vhh[l][1]/norm + mr * Vhh[l][2];
      b[l][2] = lr * b[l][1]/norm + mr * b[l][2];
    }
    Vxh[2] = lr * Vxh[1] + mr * Vxh[2];
    Uxy[2] = 0.5 * lr * Uxy[1] + mr * Uxy[2];
    c[2] = 0.5 * lr * c[1] + mr * c[2];
    
    // update & reset
    for (uint l=0; l<layers; l++) {
      Whhl[l][0] -= Whhl[l][2]; 
      Whhr[l][0] -= Whhr[l][2];
      Wxhl[l][0] -= Wxhl[l][2];
      Wxhr[l][0] -= Wxhr[l][2];
      Vhh[l][0] -= Vhh[l][2]; 
      Uhy[l][0] -= Uhy[l][2]; 
      
      Whhl[l][1].setZero(); 
      Whhr[l][1].setZero();
      Wxhl[l][1].setZero();
      Wxhr[l][1].setZero();
      Vhh[l][1].setZero(); 
      Uhy[l][1].setZero(); 
      
      b[l][0] -= b[l][2];
      b[l][1].setZero();
    }
    Vxh[0] -= Vxh[2];
    Vxh[1].setZero();
    Uxy[0] -= Uxy[2];
    Uxy[1].setZero();
    c[0] -= c[2];
    c[1].setZero();
  } else {
  // update velocities
    for (uint l=0; l<layers; l++) {
      Whhl[l][2] = (Whhl[l][1].array().square() +  Whhl[l][2].array().square()).cwiseSqrt();
      Whhr[l][2] = (Whhr[l][1].array().square() + Whhr[l][2].array().square()).cwiseSqrt();
      Wxhl[l][2] = (Wxhl[l][1].array().square() + Wxhl[l][2].array().square()).cwiseSqrt();
      Wxhr[l][2] = (Wxhr[l][1].array().square() + Wxhr[l][2].array().square()).cwiseSqrt();
      Vhh[l][2] = (Vhh[l][1].array().square() + Vhh[l][2].array().square()).cwiseSqrt();
      Uhy[l][2] = (Uhy[l][1].array().square() + Uhy[l][2].array().square()).cwiseSqrt();
      b[l][2] = (b[l][1].array().square() + b[l][2].array().square()).cwiseSqrt();
    }
    Vxh[2] = (Vxh[1].array().square() + Vxh[2].array().square()).cwiseSqrt();
    Uxy[2] = (Uxy[1].array().square() + Uxy[2].array().square()).cwiseSqrt();
    c[2] = (c[1].array().square() + c[2].array().square()).cwiseSqrt();
    
    // update & reset
    for (uint l=0; l<layers; l++) {
      Whhl[l][0].noalias() -= lr*Whhl[l][1].cwiseQuotient(Whhl[l][2]); 
      Whhr[l][0].noalias() -= lr*Whhr[l][1].cwiseQuotient(Whhr[l][2]);
      Wxhl[l][0].noalias() -= lr*Wxhl[l][1].cwiseQuotient(Wxhl[l][2]);
      Wxhr[l][0].noalias() -= lr*Wxhr[l][1].cwiseQuotient(Wxhr[l][2]);
      Vhh[l][0].noalias() -= lr*Vhh[l][1].cwiseQuotient(Vhh[l][2]); 
      Uhy[l][0].noalias() -= lr*Uhy[l][1].cwiseQuotient(Uhy[l][2]); 
      
      Whhl[l][1].setZero(); 
      Whhr[l][1].setZero();
      Wxhl[l][1].setZero();
      Wxhr[l][1].setZero();
      Vhh[l][1].setZero(); 
      Uhy[l][1].setZero(); 
      
      b[l][0].noalias() -= lr*b[l][1].cwiseQuotient(b[l][2]);
      b[l][1].setZero();
    }
    Vxh[0].noalias() -= lr*Vxh[1].cwiseQuotient(Vxh[2]);
    Vxh[1].setZero();
    Uxy[0].noalias() -= 0.5*lr*Uxy[1].cwiseQuotient(Uxy[2]);
    Uxy[1].setZero();
    c[0].noalias() -= 0.5*lr*c[1].cwiseQuotient(c[2]);
    c[1].setZero();
  }
}

void readTrees(std::vector<Node*> &trees, std::string fname) { 
  ifstream in(fname.c_str());
  string line;
  while(std::getline(in, line)) {
    std::istringstream ss(line);
    if (!isWhitespace(line)) {
      Node* t = new Node();
      t->read(line, 0, true);
      trees.push_back(t);
    }
  }
}

VectorXd error(vector<Node*>& trees) {
  // root level
  double err=0,errbin=0,nbin=0;
  double onenorm=0;
  for (uint i=0; i<trees.size(); i++) {
    trees[i]->forward(true);
    onenorm += trees[i]->x[layers-1].sum() / Node::nh;
    uint r = argmax(trees[i]->r);
    uint y = argmax(trees[i]->y);
    if (r != y)
      err++;
    if (r != 2) {
      nbin++;
      if (trees[i]->y(0) + trees[i]->y(1) >
          trees[i]->y(3) + trees[i]->y(4)) {
        if (r == 3 || r == 4)
          errbin++;
      } else {
        if (r == 0 || r == 1)
          errbin++;
      }
    }
    
  }
  Node::nnorm = onenorm / trees.size();
  //cout << onenorm / trees.size() << " ";
  VectorXd v(2); v << err / trees.size(), errbin / nbin; 
  return v;
}

void train(vector<Node*>& tra, vector<Node*> dev, vector<Node*> test) {
  ostringstream strS;
  strS << "models/drsv_" << layers << "_" << Node::nh << "_"
        << (int)ADAGRAD << "_" << DROP << "_"
        << (int)NORMALIZE << "_"
        << MAXEPOCH << "_" << Node::lr << "_" << Node::la << "_"
        << Node::mr << "_" << Node::fold;
  string fname = strS.str();
  cout << tra.size() << " " << dev.size() << " " << test.size() << endl;
  vector<uint> perm;
  for (uint i=0; i<tra.size(); i++)
    perm.push_back(i);

  Node::init();

  VectorXd bestDev, bestTest;
  VectorXd devAcc, testAcc;
  bestDev = bestTest = VectorXd::Zero(2);
  for (uint e=0; e<MAXEPOCH; e++) {
    shuffle(perm);
    for (uint i=0; i<tra.size(); i++) {
      uint j = perm[i];
      tra[j]->forward(false);
      tra[j]->backward();
      if ((i+1) % MINIBATCH == 0 || i+1 == tra.size()) {
        Node::update();
      }
    }
    devAcc = 1-error(dev).array();
    testAcc = 1-error(test).array();
    cout << 1-error(tra).array().transpose() 
       << " " << devAcc.transpose() 
       << " " << testAcc.transpose() << endl;

    // diagnostic
    /*
    for (uint l=0; l<layers; l++) {
      cout << Node::Vhh[l][0].norm() << " ";
      cout << Node::Whhl[l][0].norm() << " ";
      cout << Node::Whhr[l][0].norm() << " ";
      cout << Node::b[l][0].norm() << " ";
      cout << Node::Uhy[l][0].norm() << " ";
    }
    cout << Node::Vxh[0].norm() << " ";
    cout << Node::Wxhl[0][0].norm() << " ";
    cout << Node::Wxhr[0][0].norm() << " ";
    cout << Node::Uxy[0].norm() << " ";
    cout << Node::c[0].norm() << " ";
    cout << endl;*/
    if (devAcc[0] > bestDev[0]) {
      bestDev = devAcc;
      bestTest = testAcc;
      cout << "New Best: " << bestDev.transpose()
        << " " << bestTest.transpose() << endl;
      Node::save(fname);
    }

    if (Node::nnorm > 1e6) break; // exploded, do not continue training
    if (Node::LT != NULL) {
      delete Node::LT; // to save memory. don't do if finetuning
      Node::LT = NULL;
    }
  }
  cout << "best:" << endl << bestDev.transpose() << " " 
    << bestTest.transpose() << endl;
}

