#include <iostream>
#include "btree.h"
using namespace std;

int main() {
    B_Tree fp1(0.5);
    fp1.read("../../../data/ami33");
    fp1.init();
    cout << "fp1:" << endl;
    cout << "area:" << fp1.getArea();
    cout << ", wirelength:" << fp1.getWireLength();
    cout << endl << endl;

    B_Tree fp2(0.5);
    fp2.read("../../../data/ami49");
    fp2.init();
    cout << "fp2:" << endl;
    cout << "area:" << fp2.getArea();
    cout << ", wirelength:" << fp2.getWireLength();
    cout << endl << endl;

    // fp2.perturb();
    // fp2.packing();

    // cout << "fp1:" << endl;
    // cout << "area:" << fp1.getArea();
    // cout << ", wirelength:" << fp1.getWireLength();
    // cout << endl << endl;

    // cout << "fp2:" << endl;
    // cout << "area:" << fp2.getArea();
    // cout << ", wirelength:" << fp2.getWireLength();
    // cout << endl << endl;
}