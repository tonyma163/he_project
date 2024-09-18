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
#include <cmath>

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
static inline vector<double> read_values_from_file(const string& filename, double scale);
static inline vector<double> read_plain_repeated_input(const string& filename, double scale);
vector<PhantomCiphertext> matmulRE(PhantomContext &context, const vector<PhantomCiphertext> rows, const PhantomPlaintext &weight, const PhantomPlaintext &bias, double scale, PhantomRelinKey &relin_keys, PhantomGaloisKey &galois_keys);

int main(int argc, char *argv[]) {
    // ./app_name.cu "{input_text}"
    setup_environment(argc, argv);

    // Encryption parameters
    EncryptionParameters parms(scheme_type::ckks);
    //size_t poly_modulus_degree = 16384;
    size_t poly_modulus_degree = 32768; // 16384*2
    parms.set_poly_modulus_degree(poly_modulus_degree);

    // Coefficient modulus
    parms.set_coeff_modulus(CoeffModulus::Create(
        //poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60} // 40*14 max for my laptop
        poly_modulus_degree, {60, 40, 60}
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
    //vector<double> input_embeddings;

    // no. of files in specified directory
    int inputs_count = 0;
    //filesystem::path path { "../python/tmp_embeddings" };
    /*
    for (__attribute__((unused)) auto& p : filesystem::directory_iterator("../python/tmp_embeddings")) {
        ++inputs_count;
    }
    */
    for (const auto& entry : filesystem::directory_iterator("../python/tmp_embeddings")) {
        ++inputs_count;
    }

    vector<PhantomCiphertext> inputs;
    for (int i=0; i<inputs_count; i++ ) {
        string filename = "../python/tmp_embeddings/input_"+to_string(i)+".txt";
        vector<double> input_embeddings = read_values_from_file(filename, scale);
        vector<double> repeated;

        // check
        if (input_embeddings.size() < 128) {
            cerr << "Not enough embeddings in file: " << endl;
            continue;
        }        

        for (int j=0; j<128; j++) { // 128 bert-tiny hidden layer
            for (int k=0; k<128; k++) {
                repeated.push_back(input_embeddings[j]);
            }
        }

        //cout << "size: " << repeated.size() << endl;

        // encrypt input embeddings
        //cout << "repeated" << repeated[0]; // test
        PhantomPlaintext plaintext;
        encoder.encode(context, repeated, scale, plaintext);
        
        PhantomCiphertext ciphertext;
        public_key.encrypt_asymmetric(context, plaintext, ciphertext);

        inputs.push_back(ciphertext);
    }

    // Encoder1
    // query
    vector<double> query_weight_vec = read_values_from_file("../weights-sst2/layer0_attself_query_weight.txt", scale);
    PhantomPlaintext query_weight_pt;
    encoder.encode(context, query_weight_vec, scale, query_weight_pt);

    vector<double> query_bias_vec = read_plain_repeated_input("../weights-sst2/layer0_attself_query_bias.txt", scale);
    PhantomPlaintext query_bias_pt;
    encoder.encode(context, query_bias_vec, scale, query_bias_pt);

    // key
    vector<double> key_weight_vec = read_values_from_file("../weights-sst2/layer0_attself_key_weight.txt", scale);
    PhantomPlaintext key_weight_pt;
    encoder.encode(context, key_weight_vec, scale, key_weight_pt);

    vector<double> key_bias_vec = read_plain_repeated_input("../weights-sst2/layer0_attself_key_bias.txt", scale);
    PhantomPlaintext key_bias_pt;
    encoder.encode(context, key_bias_vec, scale, key_bias_pt);

    vector<PhantomCiphertext> Q = matmulRE(context, inputs, query_weight_pt, query_bias_pt, scale, relin_keys, galois_keys);

    return 0;
}

void setup_environment(int argc, char *argv[]) {
    string command;

    //cout << "argc: " << argc;

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

static inline vector<double> read_values_from_file(const string& filename, double scale) {
    vector<double> values;
    
    ifstream file(filename);
    /*
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return values;
    }
    */

    string row;
    while (getline(file, row)) {
        istringstream stream(row);
        string value;
        while (getline(stream, value, ',')) {
            try {
                double num = stod(value);
                values.push_back(num * scale);
            } catch (const invalid_argument& e) {
                cerr << "Cannot convert: " << value << endl;
            }
        }
    }
    file.close();
    return values;
}

static inline vector<double> read_plain_repeated_input(const string& filename, double scale) {
    vector<double> input = read_values_from_file(filename, scale);

    vector<double> repeated;
    /*
    // check
    if (input.size() < 128) {
        cerr << "Not enough embeddings in file: " << endl;
        return;
    }
    */

    for (int j=0; j<128; j++) { // 128 bert-tiny hidden layer
        for (int k=0; k<128; k++) {
            repeated.push_back(input[j]);
        }
    }

    return repeated;
}

void rotsum(PhantomContext &context, PhantomCiphertext &ciphertext, int slots, PhantomGaloisKey &galois_keys) {
    //PhantomCiphertext result = in->Clone();

    for (int i=0; i<log2(slots); i++) {
        int steps = 1 << i; // 2^i

        // rotate ciphertext
        PhantomCiphertext rotated_ciphertext = ciphertext;
        rotate_inplace(context, rotated_ciphertext, steps, galois_keys);

        // add
        add_inplace(context, ciphertext, rotated_ciphertext);
    }
}

vector<PhantomCiphertext> matmulRE(PhantomContext &context, const vector<PhantomCiphertext> rows, const PhantomPlaintext &weight, const PhantomPlaintext &bias, double scale, PhantomRelinKey &relin_keys, PhantomGaloisKey &galois_keys) {
    vector<PhantomCiphertext> columns;

    for (int i=0; i<rows.size(); i++) {
        // multipy the encrypted row with the plaintext weight
        PhantomCiphertext product_ciphertext = multiply_plain(context, rows[i], weight);
        relinearize_inplace(context, product_ciphertext, relin_keys);
        rescale_to_next_inplace(context, product_ciphertext);
        product_ciphertext.set_scale(scale);

        // rotsum
        rotsum(context, product_ciphertext, rows[i].size(), galois_keys);

        // add bias
        add_plain_inplace(context, product_ciphertext, bias);

        //
        columns.push_back(product_ciphertext);
    }

    return columns;
}