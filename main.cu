#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>
#include "util.cuh"
#include <cstdlib>
#include <sys/stat.h>
#include <filesystem>
#include <cmath>
#include <iostream>
#include <vector>
#include "HEaaN/HEaaN.hpp"

using namespace std;
using namespace HEaaN;

//Set to True to test the program on the IDE
bool IDE_MODE = true;

string input_folder;

int scale = 1;

int num_slots = 16384;

//Argument
string text;

void setup_environment(int argc, char *argv[]);
static inline vector<double> read_values_from_file(const string& filename, double scale);
Ciphertext encrypt_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale);
Ciphertext encrypt_expanded_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale);
vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, const Ciphertext &bias, double scale);

inline size_t log2ceil(size_t x) {
    size_t y = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(x))));
    return y;
}

int main(int argc, char *argv[]) {
    // ./app_name.cu "{input_text}"
    setup_environment(argc, argv);

    // Initialize context
    Context context = makeContext(ParameterPreset::FGa); // FGa - Precision optimal FG parameter

    // Initialize keys
    SecretKey sk(context); // generate secret key
    KeyGenerator keygen(context, sk); // generate public key
    keygen.genEncryptionKey(); // generate encryption key
    keygen.genMultiplicationKey(); // generate multiplication key
    keygen.genRotationKeyBundle();
    KeyPack keypack = keygen.getKeyPack();

    // Initialize encryptor, decryptor, encoder, evaluator
    Encryptor encryptor(context);
    Decryptor decryptor(context);
    EnDecoder encoder(context);
    HomEvaluator evaluator(context, keypack);

    cout << "Loaded." << endl;

    // Load embeddings
    // no. of files in specified directory
    int inputs_count = 0;
    for (const auto& entry : filesystem::directory_iterator("../python/tmp_embeddings")) {
        ++inputs_count;
    }

    vector<Ciphertext> inputs;
    for (int i=0; i<inputs_count; i++ ) {
        string filename = "../python/tmp_embeddings/input_"+to_string(i)+".txt";
        Ciphertext ctxt = encrypt_expanded_input(context, encryptor, keypack, filename, scale);
        inputs.push_back(ctxt);
    }
    //cout << "inputs: " << inputs.size() << endl;

    // Encoder1
    // query
    Ciphertext query_weight = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_attself_query_weight.txt", scale);
    //cout << "query_weight" << endl;
    //PhantomPlaintext query_weight_pt;
    //encoder.encode(context, query_weight_vec, scale, query_weight_pt);

    Ciphertext query_bias = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_attself_query_bias.txt", scale);
    //cout << "query_bias" << endl;
    //PhantomPlaintext query_bias_pt;
    //encoder.encode(context, query_bias_vec, scale, query_bias_pt);

    // key
    Ciphertext key_weight = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_attself_key_weight.txt", scale);
    //cout << "key_weight" << endl;
    //PhantomPlaintext key_weight_pt;
    //encoder.encode(context, key_weight_vec, scale, key_weight_pt);

    Ciphertext key_bias = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_attself_key_bias.txt", scale);
    //cout << "key_bias" << endl;
    //PhantomPlaintext key_bias_pt;
    //encoder.encode(context, key_bias_vec, scale, key_bias_pt);

    //vector<PhantomCiphertext> Q = matmulRE(context, inputs, query_weight_pt, query_bias_pt, scale, relin_keys, galois_keys);
    vector<Ciphertext> Q = matmulRE(context, evaluator, inputs, query_weight, query_bias, scale);
    cout << "MatMulRE Q" << endl;
    vector<Ciphertext> K = matmulRE(context, evaluator, inputs, key_weight, key_bias, scale);
    cout << "MatMulRE K" << endl;

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

Ciphertext encrypt_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale) {
    vector<double> input_embeddings = read_values_from_file(filename, scale);

    // check
    if (input_embeddings.size() < 128) {
        cerr << "Not enough embeddings in file: " << endl;
    }        

    // pad the vector to match num_slots
    if (input_embeddings.size() < static_cast<size_t>(num_slots)) {
        input_embeddings.resize(num_slots, 0.0); // pad with zeros
    } else if (input_embeddings.size() > static_cast<size_t>(num_slots)) {
        input_embeddings.resize(num_slots);
    }
    //cout << "size: " << input_embeddings.size() << endl;

    // encrypt input embeddings
    Message tmp_msg(log2ceil(num_slots));
    for (int j=0; j<input_embeddings.size(); ++j) {
        tmp_msg[j] = Complex(input_embeddings[j], 0.0);
    }
    //cout << "converted." << endl;

    Ciphertext ctxt(context);
    encryptor.encrypt(tmp_msg, keypack, ctxt);
    //cout << "encrypted." << endl;

    return ctxt;
}

Ciphertext encrypt_expanded_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale) {
    vector<double> input_embeddings = read_values_from_file(filename, scale);

    vector<double> repeated;
    // check
    if (input_embeddings.size() < 128) {
        cerr << "Not enough embeddings in file: " << endl;
    }        

    for (int j=0; j<128; j++) { // 128 bert-tiny hidden layer
        for (int k=0; k<128; k++) {
            repeated.push_back(input_embeddings[j]);
        }
    }

    // pad the vector to match num_slots
    if (repeated.size() < static_cast<size_t>(num_slots)) {
        repeated.resize(num_slots, 0.0); // pad with zeros
    } else if (repeated.size() > static_cast<size_t>(num_slots)) {
        repeated.resize(num_slots);
    }
    //cout << "size: " << repeated.size() << endl;

    // encrypt input embeddings
    Message tmp_msg(log2ceil(num_slots));
    for (int j=0; j<repeated.size(); ++j) {
        tmp_msg[j] = Complex(repeated[j], 0.0);
    }
    //cout << "converted." << endl;

    Ciphertext ctxt(context);
    encryptor.encrypt(tmp_msg, keypack, ctxt);
    //cout << "encrypted." << endl;

    return ctxt;
}

Ciphertext rotsum(Context context, HomEvaluator evaluator, Ciphertext &in, int slots, int padding) {
    Ciphertext result = in;

    for (int i=0; i<log2ceil(slots); i++) {
        // calculate rotation steps: padding * 2^i
        int rotation_steps = static_cast<int>(padding * pow(2, i));

        Ciphertext rotated(context);
        evaluator.leftRotate(result, rotation_steps, rotated);

        evaluator.add(result, rotated, result);
    }

    return result;
}

vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, const Ciphertext &bias, double scale) {
    vector<Ciphertext> columns;

    for (int i=0; i<rows.size(); i++) {
        // multipy the encrypted row with the plaintext weight
        Ciphertext product_ciphertext(context);
        evaluator.mult(rows[i], weight, product_ciphertext);
        //cout << "mult" << endl;

        // rotsum
        Ciphertext rotated = rotsum(context, evaluator, product_ciphertext, 128, 128);
        //cout << "rotsum" << endl;

        // add bias
        Ciphertext addedBias(context);
        evaluator.add(rotated, bias, addedBias);
        //cout << "bias" << endl;

        //
        columns.push_back(addedBias);
    }

    return columns;
}