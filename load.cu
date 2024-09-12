#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>
#include "phantom.h"
#include "util.cuh"
#include <cstdlib>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

int main() {
    // * Load ciphertext
	cout << "Loading ciphertext" << endl;
	ifstream infile("./tmp/ciphertext.txt", ifstream::binary);
	PhantomCiphertext loaded_ciphertext;
	loaded_ciphertext.load(infile);
	infile.close();

    return 0;
}