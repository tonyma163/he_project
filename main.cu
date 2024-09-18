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
#include <sys/stat.h>
#include <filesystem>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

//Set to True to test the program on the IDE
bool IDE_MODE = true;

string input_folder;

//Argument
string text;

void setup_environment(int argc, char *argv[]);

int main(int argc, char *argv[]) {
    // ./app_name.cu "{input_text}"
    setup_environment(argc, argv);

    // Encryption parameters
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    // Coefficient modulus
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60}
    ));

    // Context
    PhantomContext context(parms);

    // Keys
    // secret & public keys
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    // relinearization keys for multiplication
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    // galois keys for rotation
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    // Encoder
    PhantomCKKSEncoder encoder(context);

    // Scale parameter
    double scale = pow(2.0, 40);

    // Load embeddings
    vector<double> input_embeddings;

    // no. of files in specified directory
    int inputs_count = 0;
    filesystem::path path { "../python/tmp_embeddings" };
    for (__attribute__((unused)) auto& p : filesystem::directory_iterator(path)) {
        ++inputs_count;
    }

    for (int i=0; i<inputs_count; i++ ) {
        string path = "../python/tmp_embeddings/input_"+to_string(i)+".txt";

        ifstream file(path);

        string row;
        while (getline(file, row)) {
            istringstream stream(row);
            string value;
            while (getline(stream, value, ',')) {
                try {
                    double num = stod(value);
                    input_embeddings.push_back(num * scale);
                } catch (const invalid_argument e) {
                    cerr << "Cannot convert: " << value << endl;
                }
            }
        }
        file.close();

        vector<double> repeated;

        for (int j=0; j<128; j++) { // 128 bert-tiny hidden layer
            for (int i=0; i<128; i++) {
                repeated.push_back(input_embeddings[j]);
            }
        }

        int size = static_cast<int>(repeated.size());

        if (scale != 1) {
            for (int i=0; i<size; i++) {
                repeated[i] = repeated[i] * scale;
            }
        }

        // encrypt it
        cout << "repeated" << repeated[0]; // test
    }
    
    

    // Encoder1

    return 0;
}

void setup_environment(int argc, char *argv[]) {
    string command;

    cout << "argc: " << argc;

    if (IDE_MODE) {
        //Removing any previous embedding
        filesystem::remove_all("../python/tmp_embeddings");

        input_folder = "../python/tmp_embeddings/";

        //text = "This is a bad movie.";
        text = argv[1];
        command = "python3 ../python/ExtractEmbeddings.py \"" + text + "\"";

        system(command.c_str());

        return;
    }
}