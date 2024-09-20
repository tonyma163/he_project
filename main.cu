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
Ciphertext encrypt_expanded_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale);

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

    cout << "inputs: " << inputs.size() << endl;

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