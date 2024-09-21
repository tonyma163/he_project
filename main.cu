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

//
#include <functional>

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
Ciphertext encrypt_expanded_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale, int num_inputs);
Ciphertext encrypt_repeated_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale);
Ciphertext rotsum(Context context, HomEvaluator evaluator, Ciphertext &in, int slots, int padding);
vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, const Ciphertext &bias);
vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, const Ciphertext &bias, int row_size, int padding);
vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, int row_size, int padding);
vector<Ciphertext> matmulRElarge(Context &context, HomEvaluator &evaluator, vector<Ciphertext>& inputs, const vector<Ciphertext> &weights, const Ciphertext &bias, double mask_val);
vector<Ciphertext> matmulCR(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext& matrix);
vector<Ciphertext> matmulCR(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext& weight, const Ciphertext& bias);
Ciphertext wrapUpRepeated(Context context, EnDecoder encoder, HomEvaluator evaluator, vector<Ciphertext> vectors);
Ciphertext matmulScores(Context context, HomEvaluator evaluator, vector<Ciphertext> queries, const Ciphertext &key);
Ciphertext eval_exp(Context context, HomEvaluator evaluator, Bootstrapper bootstrapper, Ciphertext &c, int inputs_number);
Ciphertext eval_inverse_naive(Context context, HomEvaluator evaluator, Bootstrapper bootstapper, const Ciphertext &c, double min, double max);
vector<Ciphertext> unwrapScoresExpanded(Context context, HomEvaluator evaluator, Ciphertext c, int inputs_num);
void print(Context context, Decryptor decryptor, SecretKey sk, const Ciphertext &c, int slots, string prefix);
void print_expanded(Context context, Decryptor decryptor, SecretKey sk, const Ciphertext &c, int slots, int expansion_factor, string prefix);
Ciphertext wrapUpExpanded(Context context, EnDecoder encoder, HomEvaluator evaluator, vector<Ciphertext> vectors);
vector<Ciphertext> unwrapExpanded(Context context, HomEvaluator evaluator, Ciphertext c, int inputs_num);
Ciphertext mask_first_n(Context context, HomEvaluator evaluator, const Ciphertext &c, int n, double mask_value);
vector<Ciphertext> generate_containers(HomEvaluator evaluator, vector<Ciphertext> inputs, const double& bias);
Ciphertext eval_gelu_function(Context context, HomEvaluator evaluator, Bootstrapper bootstrapper, const Ciphertext &c, double min, double max, double mult, int degree);
vector<vector<Ciphertext>> unwrapRepeatedLarge(Context context, EnDecoder encoder, HomEvaluator evaluator, vector<Ciphertext> containers, int input_number);
vector<Ciphertext> matmulCRlarge(Context context, HomEvaluator evaluator, vector<vector<Ciphertext>> rows, vector<Ciphertext> weights, const Ciphertext &bias);

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
    keygen.genConjugationKey();
    keygen.genRotKeysForBootstrap(log2ceil(num_slots));
    KeyPack keypack = keygen.getKeyPack();

    // Initialize encryptor, decryptor, encoder, evaluator
    Encryptor encryptor(context);
    Decryptor decryptor(context);
    EnDecoder encoder(context);
    HomEvaluator evaluator(context, keypack);
    Bootstrapper bootstrapper(evaluator, log2ceil(num_slots));

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
    vector<Ciphertext> Q = matmulRE(context, evaluator, inputs, query_weight, query_bias);
    //cout << "MatMulRE Q" << endl;
    vector<Ciphertext> K = matmulRE(context, evaluator, inputs, key_weight, key_bias);
    //cout << "MatMulRE K" << endl;

    Ciphertext K_wrapped = wrapUpRepeated(context, encoder, evaluator, K);
    //cout << "wrapped K" << endl;

    Ciphertext scores = matmulScores(context, evaluator, Q, K_wrapped);
    //cout << "scores" << endl;

    // section 5 in BertSelfAttention layer
    // Eval e^x[i]
    scores = eval_exp(context, evaluator, bootstrapper, scores, inputs.size());
    ///cout << "eval_exp" << endl;

    Ciphertext scores_sum = rotsum(context, evaluator, scores, 128, 128);
    //cout << "rotsum" << endl;

    // section 6 Eval 1/x
    // Using Chebyshev Polynomial Approximation
    Ciphertext scores_denominator = eval_inverse_naive(context, evaluator, bootstrapper, scores_sum, 2, 5000);
    //cout << "eval 1/x" << endl;

    // section 7 EvalMult
    evaluator.mult(scores, scores_denominator, scores);
    //cout << "scores" << endl;

    // section 8 Unwrap
    vector<Ciphertext> unwrapped_scores = unwrapScoresExpanded(context,evaluator, scores, inputs.size());
    //cout << "unwrap" << endl;

    // section 2
    // load weight & bias of value
    Ciphertext value_weight = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_attself_value_weight.txt", scale);
    value_weight.setLevel(scores.getLevel()-2);
    //cout << "value_weight" << endl;

    Ciphertext value_bias = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_attself_value_bias.txt", scale);
    value_bias.setLevel(scores.getLevel()-1);
    //cout << "value_bias" << endl;

    // VecMatER(weight & bias)
    vector<Ciphertext> V = matmulRE(context, evaluator, inputs, value_weight, value_bias);
    //cout << "VecMatER" << endl;

    // WrapRep
    Ciphertext V_wrapped = wrapUpRepeated(context, encoder, evaluator, V);
    //cout << "wrapRep" << endl;

    // section 9 VecMatER
    vector<Ciphertext> output = matmulRE(context, evaluator, unwrapped_scores, V_wrapped, 128, 128);
    //cout << "VecMatER" << endl;

    //print(context, decryptor, sk, output[0], 128, "Self-Attention (Repeated)");

    // Bert Self-Attention END
    cout << "Bert Self-Attention END" << endl;

    // Bert SelfOutputLayer
    // load weight & bias of dense
    Ciphertext dense_weight = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_selfoutput_weight.txt", scale);
    dense_weight.setLevel(output[0].getLevel());
    //cout << "dense_weight" << endl;

    Ciphertext dense_bias = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_selfoutput_bias.txt", scale);
    dense_bias.setLevel(output[0].getLevel()+1);
    //cout << "dense_bias" << endl;

    // VecMatRC
    output = matmulCR(context, evaluator, output, dense_weight, dense_bias);
    //cout << "VecMatRC" << endl;

    vector<Ciphertext> output2 = output;
    for (int i=0; i<output.size(); i++) {
        Ciphertext tmp(context);
        evaluator.add(output[i], inputs[i], tmp);
        output2[i] = tmp;
    }

    // WrapExp xi
    Ciphertext wrappedOutput = wrapUpExpanded(context, encoder, evaluator, output);
    //cout << "wrapUpExpanded" << endl;

    // Eval sub E
    Ciphertext precomputed_mean = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_selfoutput_mean.txt", -1);
    precomputed_mean.setLevel(wrappedOutput.getLevel());
    evaluator.add(wrappedOutput, precomputed_mean, wrappedOutput);
    //cout << "sub E" << endl;

    // Eval Mult V(p)Y
    Ciphertext vy = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_selfoutput_vy.txt", scale);
    vy.setLevel(wrappedOutput.getLevel());
    evaluator.mult(wrappedOutput, vy, wrappedOutput);
    //cout << "Mult V(p)Y" << endl;

    // Eval Add theta
    Ciphertext bias = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_selfoutput_normbias.txt", 1, inputs.size());
    bias.setLevel(wrappedOutput.getLevel());
    evaluator.add(wrappedOutput, bias, wrappedOutput);
    //cout << "Add theta" << endl;

    wrappedOutput.setLevel(3); // error -> bootstrap must >= 3
    //cout << "Level: " << wrappedOutput.getLevel() << endl;
    bootstrapper.bootstrap(wrappedOutput, wrappedOutput);

    // Clone for layerNorm
    Ciphertext output_copy = wrappedOutput;

    // Unwrap
    output = unwrapExpanded(context, evaluator, wrappedOutput, inputs.size());
    //cout << "unwrapExpanded" << endl;

    //
    //print_expanded(context, decryptor, sk, output[0], 0, 128, "Self-Output (Expanded)");

    cout << "Bert Self-Output END" << endl;

    // Bert Intermediate Layer
    double GELU_max_abs_value = 1 / 13.5;

    // load 4 weights
    Ciphertext intermediate_weight_1 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_intermediate_weight1.txt", GELU_max_abs_value);
    intermediate_weight_1.setLevel(wrappedOutput.getLevel());
    Ciphertext intermediate_weight_2 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_intermediate_weight2.txt", GELU_max_abs_value);
    intermediate_weight_2.setLevel(wrappedOutput.getLevel());
    Ciphertext intermediate_weight_3 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_intermediate_weight3.txt", GELU_max_abs_value);
    intermediate_weight_3.setLevel(wrappedOutput.getLevel());
    Ciphertext intermediate_weight_4 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_intermediate_weight4.txt", GELU_max_abs_value);
    intermediate_weight_4.setLevel(wrappedOutput.getLevel());

    vector<Ciphertext> dense_weights {intermediate_weight_1, intermediate_weight_2, intermediate_weight_3, intermediate_weight_4};

    Ciphertext intermediate_bias = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_intermediate_bias.txt", GELU_max_abs_value);
    intermediate_bias.setLevel(output[0].getLevel()+1);
    //cout << "loaded" << endl;

    output = matmulRElarge(context, evaluator, output, dense_weights, dense_bias, 1); // mask_value = 1
    //cout << "matmulRElarge" << endl;

    // Concat blocks
    // wrapup in 4 containers
    output = generate_containers(evaluator, output, 0); // bias = nullptr -> 0
    //cout << "generated containers" << endl;

    // eval GELU(x)
    for (int i=0; i<output.size(); i++) {
        output[i] = eval_gelu_function(context, evaluator, bootstrapper, output[i], -1, 1, GELU_max_abs_value, 119);
        bootstrapper.bootstrap(output[i], output[i]);
    }
    //cout << "eval GELU(x)" << endl;

    // unWrap
    vector<vector<Ciphertext>> unwrappedLargeOutput = unwrapRepeatedLarge(context, encoder, evaluator, output, inputs.size());
    //cout << "unwrappedLargeOutput" << endl;

    //print(context, decryptor, sk, unwrappedLargeOutput[0][0], 128, "Intermediate (Containers)");
    cout << "Bert Intermediate END" << endl;

    // Bert Output
    Ciphertext output_weight_1 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_output_weight1.txt", scale);
    output_weight_1.setLevel(unwrappedLargeOutput[0][0].getLevel());
    Ciphertext output_weight_2 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_output_weight2.txt", scale);
    output_weight_2.setLevel(unwrappedLargeOutput[0][0].getLevel());
    Ciphertext output_weight_3 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_output_weight3.txt", scale);
    output_weight_3.setLevel(unwrappedLargeOutput[0][0].getLevel());
    Ciphertext output_weight_4 = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_output_weight4.txt", scale);
    output_weight_4.setLevel(unwrappedLargeOutput[0][0].getLevel());

    Ciphertext output_bias = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_output_bias.txt", scale);
    output_bias.setLevel(unwrappedLargeOutput[0][0].getLevel()+1);
    cout << "loaded" << endl;

    output = matmulCRlarge(context, evaluator, unwrappedLargeOutput, {output_weight_1, output_weight_2, output_weight_3, output_weight_4}, output_bias);
    cout << "matmulCRlarge" << endl;

    wrappedOutput = wrapUpExpanded(context, encoder, evaluator, output);
    cout << "wrappedOutput" << endl;

    evaluator.add(wrappedOutput, output_copy, wrappedOutput);
    cout << "add" << endl;

    precomputed_mean = encrypt_repeated_input(context, encryptor, keypack, "../weights-sst2/layer0_output_mean.txt", -1);
    precomputed_mean.setLevel(wrappedOutput.getLevel());
    evaluator.add(wrappedOutput, precomputed_mean, wrappedOutput);
    cout << "add" << endl;

    vy = encrypt_input(context, encryptor, keypack, "../weights-sst2/layer0_output_vy.txt", 1);
    vy.setLevel(wrappedOutput.getLevel());
    evaluator.mult(wrappedOutput, vy, wrappedOutput);
    cout << "mult" << endl;

    bias = encrypt_expanded_input(context, encryptor, keypack, "../weights-sst2/layer0_output_normbias.txt", 1, inputs.size());
    bias.setLevel(wrappedOutput.getLevel());
    evaluator.add(wrappedOutput, inputs.size(), wrappedOutput);
    cout << "add" << endl;

    output = unwrapExpanded(context, evaluator, wrappedOutput, inputs.size());
    cout << "unwrapExpanded" << endl;

    print_expanded(context, decryptor, sk, output[0], 0, 128, "Output (Expanded)");
    cout << "Bert Output END" << endl;


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

Ciphertext encrypt_expanded_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale, int num_inputs) {
    vector<double> input_embeddings = read_values_from_file(filename, scale);

    vector<double> repeated;
    // check
    if (input_embeddings.size() < 128) {
        cerr << "Not enough embeddings in file: " << endl;
    }        

    for (int j=0; j<128; j++) { // 128 bert-tiny hidden layer
        for (int k=0; k<num_inputs; k++) {
            repeated.push_back(input_embeddings[j]);
        }
        for (int l=0; l<128-num_inputs; l++) {
            repeated.push_back(0);
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


Ciphertext encrypt_repeated_input(Context context, Encryptor encryptor, KeyPack keypack, const string& filename, double scale) {
    vector<double> input_embeddings = read_values_from_file(filename, scale);

    vector<double> repeated;
    // check
    if (input_embeddings.size() < 128) {
        cerr << "Not enough embeddings in file: " << endl;
    }        

    for (int j=0; j<128; j++) { // 128 bert-tiny hidden layer
        for (int k=0; k<128; k++) {
            repeated.push_back(input_embeddings[k]);
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

vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, const Ciphertext &bias) {
    vector<Ciphertext> columns;

    for (int i=0; i<rows.size(); i++) {
        // multipy the encrypted row with the plaintext weight
        Ciphertext product_ciphertext(context);
        evaluator.mult(rows[i], weight, product_ciphertext);
        //cout << "mult" << endl;

        // rotsum
        Ciphertext rotated = rotsum(context, evaluator, product_ciphertext, 128, 1);
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

vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, const Ciphertext &bias, int row_size, int padding) {
    vector<Ciphertext> columns;

    for (int i=0; i<rows.size(); i++) {
        // multipy the encrypted row with the plaintext weight
        Ciphertext product_ciphertext(context);
        evaluator.mult(rows[i], weight, product_ciphertext);
        //cout << "mult" << endl;

        // rotsum
        Ciphertext rotated = rotsum(context, evaluator, product_ciphertext, row_size, padding);
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

vector<Ciphertext> matmulRElarge(Context &context, HomEvaluator &evaluator, vector<Ciphertext>& inputs, const vector<Ciphertext> &weights, const Ciphertext &bias, double mask_val) {
    vector<Ciphertext> densed;

    for (int i=0; i<inputs.size(); i++) {
        Ciphertext i_th_result(context);
        for (int j=weights.size()-1; j>=0; j--) {
            Ciphertext out(context);
            evaluator.mult(inputs[i], weights[j], out);
            out = rotsum(context, evaluator, out, 128, 128);

            out = mask_first_n(context, evaluator, out, 128, mask_val);

            if (j == weights.size() - 1)
                i_th_result = out;
            else {
                //i_th_result = rotate(i_th_result, -128);
                evaluator.rightRotate(i_th_result, 64, i_th_result); //-64
                evaluator.rightRotate(i_th_result, 64, i_th_result); //-64
                
                evaluator.add(i_th_result, out, i_th_result);
            }
        }

        evaluator.add(i_th_result, bias, i_th_result);

        densed.push_back(i_th_result);
    }

    return densed;
}

vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, int row_size, int padding) {
    vector<Ciphertext> columns;

    for (int i=0; i<rows.size(); i++) {
        // multipy the encrypted row with the plaintext weight
        Ciphertext product_ciphertext(context);
        evaluator.mult(rows[i], weight, product_ciphertext);
        //cout << "mult" << endl;

        // rotsum
        Ciphertext rotated = rotsum(context, evaluator, product_ciphertext, row_size, padding);
        //cout << "rotsum" << endl;

        //
        columns.push_back(rotated);
    }

    return columns;
}

vector<Ciphertext> matmulCR(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext& matrix) {
    vector<Ciphertext> columns;

    for (int i=0; i<rows.size(); i++) {
        // multipy the encrypted row with the plaintext weight
        Ciphertext product_ciphertext(context);
        evaluator.mult(rows[i], matrix, product_ciphertext);
        //cout << "mult" << endl;

        // rotsum
        Ciphertext rotated = rotsum(context, evaluator, product_ciphertext, 64, 1);
        //cout << "rotsum" << endl;

        //
        columns.push_back(rotated);
    }

    return columns;
}

vector<Ciphertext> matmulCR(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext& weight, const Ciphertext& bias) {
    vector<Ciphertext> columns;

    for (int i=0; i<rows.size(); i++) {
        // multipy the encrypted row with the plaintext weight
        Ciphertext product_ciphertext(context);
        evaluator.mult(rows[i], weight, product_ciphertext);
        //cout << "mult" << endl;

        // rotsum
        Ciphertext rotated = rotsum(context, evaluator, product_ciphertext, 64, 1);
        //cout << "rotsum" << endl;

        //
        Ciphertext added(context);
        evaluator.add(rotated, bias, added);

        //
        columns.push_back(added);
    }

    return columns;
}

Ciphertext mask_block(Context context, EnDecoder encoder, HomEvaluator evaluator, Ciphertext& c, int from, int to, double mask_value) {

    vector<complex<double>> mask(num_slots, 0.0);

    for (int i=from; i<to && i<num_slots; ++i) {
        mask[i] = mask_value;
    }

    Message tmp_msg(log2ceil(num_slots));
    for (size_t i=0; i<num_slots; ++i) {
        tmp_msg[i] = Complex(mask[i].real(), 0.0);
    }

    // Multiply the ciphertext with the mask plaintext
    Ciphertext masked_ctxt(context);
    evaluator.mult(c, tmp_msg, masked_ctxt);

    return masked_ctxt;
}

Ciphertext wrapUpRepeated(Context context, EnDecoder encoder, HomEvaluator evaluator, vector<Ciphertext> vectors) {
    vector<Ciphertext> masked;

    for (int i=0; i<vectors.size(); i++) {
        int from = 128 * i;
        int to = 128 * (i+1);
        Ciphertext masked_ctxt = mask_block(context, encoder, evaluator, vectors[i], from, to, 1.0);
        masked.push_back(masked_ctxt);
    }

    // aggregate all masked ciphertexts by adding them together
    Ciphertext aggregated(context);
    if (!masked.empty()) {
        aggregated = masked[0];
        for (size_t i=1; i<masked.size(); ++i) {
            evaluator.add(aggregated, masked[i], aggregated);
        }
    }

    return aggregated;
}

Ciphertext mask_heads(Context context, HomEvaluator evaluator, Ciphertext& c, double mask_value) {
    vector<double> mask;

    for (int i=0; i<num_slots; i++) {
        if (i%64 == 0) {
            mask.push_back(mask_value);
        } else {
            mask.push_back(0);
        }
    }

    Message tmp_msg(log2ceil(num_slots));
    for (size_t i=0; i<num_slots; ++i) {
        tmp_msg[i] = Complex(mask[i], 0.0);
    }

    // Multiply the ciphertext with the mask plaintext
    Ciphertext masked_ctxt(context);
    evaluator.mult(c, tmp_msg, masked_ctxt);

    return masked_ctxt;
}

Ciphertext matmulScores(Context context, HomEvaluator evaluator, vector<Ciphertext> queries, const Ciphertext &key) {
    vector<Ciphertext> scores = matmulCR(context, evaluator, queries, key);

    double r = 1/8.0; //Later corrected with e^(x/r)

    Ciphertext scores_wrapped = mask_heads(context, evaluator, scores[scores.size()-1], 1/8.0*r);
    Ciphertext rotated(context);
    evaluator.rightRotate(scores_wrapped, 1, rotated);

    for (int i=scores.size()-2; i>=0; i--) {
        Ciphertext mask_headed =  mask_heads(context, evaluator, scores[i], 1/8.0*r);
        evaluator.add(rotated, mask_headed, rotated);

        if (i>0) evaluator.rightRotate(scores_wrapped, 1, rotated);
    }

    return rotated;
}


// Taylor Series
// requires 21 multiplications
// Compute c^i
// Multiply by coefficient
// Add to result
/*
Ciphertext eval_exp(Context context, HomEvaluator evaluator, Bootstrapper bootstrapper, Ciphertext &c, int inputs_number) {
    // Coefficients of Taylor series for exp(x)
    vector<double> coefficients = {1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0, 1.0/720.0};

    // Initialize res with the constant term (1)
    Message msg(log2ceil(num_slots));
    msg[0] = Complex(coefficients[0], 0.0); // start from the highest degree
    //cout << "Coefficients[0]: " << coefficients[0] << endl;
    for (size_t i=1; i<num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }
    Ciphertext res(context);
    evaluator.add(c, msg, res);
    cout << "added" << endl;
    
    int counter = 1;
    // Iterate through the coefficients to build the polynomial
    for (size_t i=1; i<coefficients.size(); ++i) {
        // Multiply c by itself i times to get c^i
        Ciphertext c_power = c;
        for (size_t j=1; j<i; ++j) {
            evaluator.mult(c, c_power, c_power);
            cout << "mult " << counter++ << endl;
        }
        //cout << "mult Level:" << c_power.getLevel() << endl;

        // Encode the coefficient
        Message coef_msg(log2ceil(num_slots));
        for (size_t k=0; k<num_slots; ++k) {
            coef_msg[k] = Complex(coefficients[i], 0.0);
        }
        //cout << "encoded" << endl;

        // Multiply c^i with the coefficient
        Ciphertext term(context);
        evaluator.mult(c_power, coef_msg, term);
        cout << "mult " << counter++ << endl;
        //cout << "mult2 Level:" << c_power.getLevel() << endl;

        // Add the term to res
        evaluator.add(res, term, res);
        //cout << "add2" << endl;
        cout << "loop" << endl;
    }
    
    //cout << "ready for boostrapping" << endl;
    // bootstrapping
    bootstrapper.bootstrap(res, res);
    cout << "boostrapped" << endl;
    
        
    // Perform EvalMultMany equivalent: res^8 (multiply res by itself 7 times)
    for (int i=0; i<7; i++) {
        evaluator.mult(res, res, res);
        cout << "mult " << counter++;
    }

    // Create the mask vector
    vector<double> mask(num_slots, -1.0); // Initialize all to -1
    for (int i=0; i<num_slots; i++) {
        // Assuming 128 * inputs_number as the upper bound
        if (i % 64 < inputs_number && i < (128 * inputs_number)) {
            mask[i] = 0.0;
        }
    }

    //
    Message mask_msg(log2ceil(num_slots));
    for (size_t i=0; i<num_slots; ++i) {
        mask_msg[i] = Complex(mask[i], 0.0);
    }

    // Add mask to res
    Ciphertext final_res(context);
    evaluator.add(res, mask_msg, final_res);

    return final_res;
}
*/

// Horner's Method
// requires 6 multiplications
Ciphertext eval_exp(Context context, HomEvaluator evaluator, Bootstrapper bootstrapper, Ciphertext &c, int inputs_number) {
    vector<double> coefficients = {1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0, 1.0/720.0};

    // Initialize res with the constant term (1)
    Message msg(log2ceil(num_slots));
    msg[0] = Complex(coefficients.back(), 0.0); // start from the highest degree
    //cout << "Coefficients[0]: " << coefficients[0] << endl;
    for (size_t i=1; i<num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }

    //int counter = 1;
    Ciphertext res(context);
    // Interate from a^n-1 down to a^0
    for (int i=coefficients.size()-2; i>=0; --i) {
        Ciphertext tmp(context);
        evaluator.mult(c, msg, tmp);
        //cout << "mult " << counter++ << endl;

        //bootstrapper.bootstrap(tmp, tmp);
        //cout << "bootstrap" << endl;

        //Encode the coefficient a_i
        Message coef_msg(log2ceil(num_slots));
        for (size_t k=0; k<num_slots; ++k) {
            coef_msg[k] = Complex(coefficients[i], 0.0);
        }
        evaluator.add(tmp, coef_msg, res);
        //cout << "add" << endl;

        //cout << "loop" << endl;
    }
    
    // In Horner's Method, we may not require to boostrap the ciphertext for further operations since there are operated only 6 multiplications.
    //cout << "ready for boostrapping" << endl;
    // bootstrapping
    bootstrapper.bootstrap(res, res);
    //cout << "boostrapped" << endl;

    // Create the mask vector
    vector<double> mask(num_slots, -1.0); // Initialize all to -1
    for (int i=0; i<num_slots; i++) {
        // Assuming 128 * inputs_number as the upper bound
        if (i % 64 < inputs_number && i < (128 * inputs_number)) {
            mask[i] = 0.0;
        }
    }

    //
    Message mask_msg(log2ceil(num_slots));
    for (size_t i=0; i<num_slots; ++i) {
        mask_msg[i] = Complex(mask[i], 0.0);
    }

    // Add mask to res
    Ciphertext final_res(context);
    evaluator.add(res, mask_msg, final_res);

    return final_res;
}

vector<double> EvalChebyshevCoefficients(function<double(double)> func, double a, double b, uint32_t degree) { //openfhe-src/core/lib/math/chebyshev.cpp
    // the number of coefficients to be generated should be degree+1 as zero is also included
    size_t coeffTotal{degree + 1};
    double bMinusA = 0.5 * (b-a);
    double bPlusA = 0.5 * (b+a);
    double PiByDeg = M_PI / static_cast<double>(coeffTotal);
    vector<double> functionPoints(coeffTotal);
    for (size_t i=0; i< coeffTotal; ++i)
        functionPoints[i] = func(cos(PiByDeg * (i+0.5)) * bMinusA + bPlusA);

    double multFactor = 2.0 / static_cast<double>(coeffTotal);
    vector<double> coefficients(coeffTotal);
    for (size_t i=0; i<coeffTotal; ++i) {
        for (size_t j=0; j<coeffTotal; ++j)
            coefficients[i] += functionPoints[j] * cos(PiByDeg * i * (j+0.5));
        coefficients[i] *= multFactor;
    }

    return coefficients;
}

Ciphertext eval_inverse_naive(Context context, HomEvaluator evaluator, Bootstrapper bootstrapper, const Ciphertext &c, double min, double max) {
    // Get the coefficients
    vector<double> coefficients = EvalChebyshevCoefficients([](double x) -> double {return 1 / x; }, min, max, 119);

    //
    size_t degree = coefficients.size() - 1;

    Message msg(log2ceil(num_slots));
    msg[0] = Complex(coefficients[degree], 0.0); // start from the highest degree
    //cout << "Coefficients[0]: " << coefficients[0] << endl;
    for (size_t i=1; i<num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }

    //int counter = 1;
    Ciphertext product(context);
    // Interate from a^n-1 down to a^0
    for (int i=degree-1; i>=0; --i) {
        // Multiply msg by ciphertext (c)
        Ciphertext tmp(context);
        evaluator.mult(c, msg, tmp);
        //cout << "mult " << counter++ << endl;

        //bootstrapper.bootstrap(tmp, tmp);
        //cout << "bootstrap" << endl;

        // Add the current coefficient a^i
        Message coef_msg(log2ceil(num_slots));
        for (size_t j=0; j<num_slots; ++j) {
            coef_msg[j] = Complex(coefficients[i], 0.0);
        }
        evaluator.add(tmp, coef_msg, product);
        //cout << "add" << endl;

        //cout << "loop" << endl;
    }

    bootstrapper.bootstrap(product, product);
    //cout << "bootstrap" << endl;

    return product;

}

Ciphertext mask_mod_n(Context context, HomEvaluator evaluator, const Ciphertext& c, int n) {
    vector<double> mask;
    for (int i=0; i<num_slots; i++) {
        if (i%n == 0) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }
    }

    Message tmp(log2ceil(num_slots));
    for (size_t i=0; i<num_slots; ++i) {
        tmp[i] = Complex(mask[i], 0.0);
    }

    Ciphertext product(context);
    evaluator.mult(c, tmp, product);
    
    return product;
}

Ciphertext mask_mod_n(Context context, HomEvaluator evaluator, const Ciphertext& c, int n, int padding, int max_slots) {
    vector<double> mask;
    for (int i=0; i<num_slots; i++) {
        if (i%n == padding) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }
    }

    Message tmp(log2ceil(num_slots));
    for (size_t i=0; i<num_slots; ++i) {
        tmp[i] = Complex(mask[i], 0.0);
    }

    Ciphertext product(context);
    evaluator.mult(c, tmp, product);
    
    return product;
}

Ciphertext repeat(Context context, HomEvaluator evaluator, const Ciphertext &in, int slots) {
    Ciphertext res = in;

    for (int i=0; i<log2(slots); i++) {
        Ciphertext rotated(context);

        u64 value = -pow(2, i); // -pow(2,i)
        if (value>0)
            evaluator.leftRotate(res, value, rotated);
        else
            evaluator.rightRotate(res, value, rotated);

        evaluator.add(res, rotated, res);
    }

    return res;
}

Ciphertext repeat(Context context, HomEvaluator evaluator, const Ciphertext &in, int slots, int padding) {
    Ciphertext res = in;

    for (int i=0; i<log2(slots); i++) {
        Ciphertext rotated(context);

        u64 value = padding*-pow(2, i); // -pow(2,i)
        if (value>0)
            evaluator.leftRotate(res, value, rotated);
        else
            evaluator.rightRotate(res, value, rotated);
            
        evaluator.add(res, rotated, res);
    }

    return res;
}

vector<Ciphertext> unwrapScoresExpanded(Context context, HomEvaluator evaluator, Ciphertext c, int inputs_num) {
    vector<Ciphertext> result;

    for (int i=0; i<inputs_num; i++) {
        Ciphertext i_th_1 = mask_mod_n(context, evaluator, c, 128, 0, inputs_num*128);
        Ciphertext i_th_2 = mask_mod_n(context, evaluator, c, 128, 64, inputs_num*128);
        i_th_1 = repeat(context, evaluator, i_th_1, 64);
        i_th_2 = repeat(context, evaluator, i_th_2, 64);

        if (i<inputs_num-1)
            evaluator.leftRotate(c, 1, c);
        
        Ciphertext product(context);
        evaluator.add(i_th_1, i_th_2, product);

        result.push_back(product);
    }

    return result;
}

void print(Context context, Decryptor decryptor, SecretKey sk, const Ciphertext &c, int slots, string prefix) {
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix << " Lv. " << c.getLevel() << ") " << endl;

    // Decrypt the result
    Message decrypted_result;
    decryptor.decrypt(c, sk, decrypted_result);

    // Extract real parts into a vector<double>
    vector<double> real_values;
    real_values.reserve(num_slots);
    for (size_t i=0; i<decrypted_result.getSize(); ++i) {
        real_values.push_back(decrypted_result[i].real());
    }

    // Set precision and fixed format
    cout << fixed << setprecision(4);
    cout << "[ ";

    for (int i=0; i<slots; ++i) {
        string segno = "";
        double val = real_values[i];
        if (val > 0) {
            segno = " ";
        } else {
            segno = "-";
            val = -val;
        }

        if (i == slots - 1) {
            cout << segno << val << " ]";
        } else {
            if (abs(val) < 1e-8)
                cout << " 0.0000" << ", ";
            else
                cout << segno << val << ", ";
        }
    }

    cout << endl;
}

void print_expanded(Context context, Decryptor decryptor, SecretKey sk, const Ciphertext &c, int slots, int expansion_factor, string prefix) {
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix << " Lv. " << c.getLevel() << ") " << endl;

    // Decrypt the result
    Message decrypted_result;
    decryptor.decrypt(c, sk, decrypted_result);

    // Extract real parts into a vector<double>
    vector<double> real_values;
    real_values.reserve(num_slots);
    for (size_t i=0; i<decrypted_result.getSize(); ++i) {
        real_values.push_back(decrypted_result[i].real());
    }

    // Set precision and fixed format
    cout << fixed << setprecision(4);
    cout << "[ ";

    for (int i=0; i<slots; ++i) {
        if (i % expansion_factor != 0) {
            continue;
        }
        string segno = "";
        double val = real_values[i];
        if (val > 0) {
            segno = " ";
        } else {
            segno = "-";
            val = -val;
        }

        if (i == slots - 1) {
            cout << segno << val << " ]";
        } else {
            if (abs(val) < 1e-8)
                cout << " 0.0000" << ", ";
            else
                cout << segno << val << ", ";
        }
    }

    cout << " ]";

    cout << endl;
}

Ciphertext wrapUpExpanded(Context context, EnDecoder encoder, HomEvaluator evaluator, vector<Ciphertext> vectors) {
    Ciphertext masked(context);

    masked = mask_mod_n(context, evaluator, vectors[vectors.size() -1], 128);
    if (vectors.size() > 1) {
        evaluator.rightRotate(masked, 1, masked);
    }

    for (int i=vectors.size()-2; i>=0; i--) {
        evaluator.add(masked, mask_mod_n(context, evaluator, vectors[i], 128), masked);

        if (i > 0)
            evaluator.rightRotate(masked, 1, masked);
    }

    return masked;
}

vector<Ciphertext> unwrapExpanded(Context context, HomEvaluator evaluator, Ciphertext c, int inputs_num) {
    vector<Ciphertext> result;

    for (int i=0; i<inputs_num; i++) {
        Ciphertext out = mask_mod_n(context, evaluator, c, 128, 0, inputs_num*128);
        out = repeat(context, evaluator, out, 128);

        if (i < inputs_num-1)
            evaluator.leftRotate(c, 1, c);

        result.push_back(out);
    }

    return result;
}

Ciphertext mask_first_n(Context context, HomEvaluator evaluator, const Ciphertext &c, int n, double mask_value) {
    // need to verify
    vector<double> mask;

    for (int i=0; i<num_slots; i++) {
        if (i<n) {
            mask.push_back(mask_value);
        } else {
            mask.push_back(0);
        }
    }

    Message tmp(log2ceil(num_slots));
    for (size_t i=0; i<num_slots; ++i) {
        tmp[i] = Complex(mask[i], 0.0);
    }

    Ciphertext product(context);
    evaluator.mult(c, tmp, product);

    return product;
}

vector<Ciphertext> slicing(const vector<Ciphertext> &arr, int X, int Y) {
    if (Y - X >= arr.size())
        return arr;

    if (Y > arr.size()) {
        Y = arr.size();
    }

    // Starting and Ending iterators
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y;

    // To store the sliced vector
    //vector<Ciphertext> result(Y-X); // HEaaN Ciphertext without default construction
    //copy(start, end, result.begin());

    // alternative
    vector<Ciphertext> result;
    result.reserve(Y-X); // Reserve space to optimize performance

    // Iterate through the specified range and add to result
    for(auto it = start; it != end; ++it) {
        result.emplace_back(*it); // copy or move the ciphertext
    }

    // Return the final sliced vector
    return result;
}

Ciphertext wrap_container(HomEvaluator evaluator, vector<Ciphertext> c, int inputs_number) {
    Ciphertext result = c[0];

    for (int i=1; i<inputs_number; i++) {
        evaluator.rightRotate(result, 512, result); //-512
        evaluator.add(result, c[i], result);
    }

    return result;
}

vector<Ciphertext> generate_containers(HomEvaluator evaluator, vector<Ciphertext> inputs, const double& bias) {
    vector<Ciphertext> containers;
    vector<int> quantities;

    //This reverse is not fine
    //reverse(inputs.begin(), inputs.end());

    for (int i=0; i<inputs.size()/32.0; i++) {
        int quantity = 32;
        if ((i+1)*32 > inputs.size()) {
            quantity = inputs.size() - (i * 32);
        }

        quantities.push_back(quantity);

        vector<Ciphertext> sliced_input = slicing(inputs, (i) * 32, (i + 1) * 32);
        reverse(sliced_input.begin(), sliced_input.end());

        Ciphertext partial_container = wrap_container(evaluator, sliced_input, quantity);

        if (bias != 0)
            evaluator.add(partial_container, bias, partial_container);
        
        containers.push_back(partial_container);
    }

    return containers;
}

Ciphertext eval_gelu_function(Context context, HomEvaluator evaluator, Bootstrapper bootstrapper, const Ciphertext &c, double min, double max, double mult, int degree) {
    // Get the coefficients
    //vector<double> coefficients = EvalChebyshevCoefficients([](double x) -> double {return (0.5 * (x * (1/mult)) * (1 + erf((x * (1/mult)) / 1.41421356237))); }, min, max, degree);

    //
    auto gelu_scaled = [mult](double x) -> double {
        double scaled_x = x / mult;
        return 0.5 * scaled_x * (1.0 + erf(scaled_x / sqrt(2.0)));
    };

    vector<double> coefficients = EvalChebyshevCoefficients(gelu_scaled, min, max, degree);

    Message msg(log2ceil(num_slots));
    msg[0] = Complex(coefficients[degree], 0.0); // start from the highest degree
    //cout << "Coefficients[0]: " << coefficients[0] << endl;
    for (size_t i=1; i<num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }

    //int counter = 1;
    Ciphertext product(context);
    // Interate from a^n-1 down to a^0
    for (int i=degree-1; i>=0; --i) {
        // Multiply msg by ciphertext (c)
        Ciphertext tmp(context);
        evaluator.mult(c, msg, tmp);
        //cout << "mult " << counter++ << endl;

        //bootstrapper.bootstrap(tmp, tmp);
        //cout << "bootstrap" << endl;

        // Add the current coefficient a^i
        Message coef_msg(log2ceil(num_slots));
        for (size_t j=0; j<num_slots; ++j) {
            coef_msg[j] = Complex(coefficients[i], 0.0);
        }
        evaluator.add(tmp, coef_msg, product);
        //cout << "add" << endl;

        //cout << "loop" << endl;
    }

    bootstrapper.bootstrap(product, product);
    //cout << "bootstrap" << endl;

    return product;
}

vector<Ciphertext> unwrap_512_in_4_128(Context context, EnDecoder encoder, HomEvaluator evaluator, Ciphertext &c, int index) {
    vector<Ciphertext> result;

    int shift = index * 512;

    Ciphertext score1 = mask_block(context, encoder, evaluator, c, shift+0, shift+128, 1);
    score1 = repeat(context, evaluator, score1, 128, -128);
    Ciphertext score2 = mask_block(context, encoder, evaluator, c, shift+128, shift+256, 1);
    score2 = repeat(context, evaluator, score2, 128, -128);
    Ciphertext score3 = mask_block(context, encoder, evaluator, c, shift+256, shift+384, 1);
    score3 = repeat(context, evaluator, score3, 128, -128);
    Ciphertext score4 = mask_block(context, encoder, evaluator, c, shift+384, shift+512, 1);
    score4 = repeat(context, evaluator, score4, 128, -128);

    result.push_back(score1);
    result.push_back(score2);
    result.push_back(score3);
    result.push_back(score4);

    return result;

}

vector<vector<Ciphertext>> unwrapRepeatedLarge(Context context, EnDecoder encoder, HomEvaluator evaluator, vector<Ciphertext> containers, int input_number) {
    vector<vector<Ciphertext>> unwrapped_output;
    vector<int> quantities;

    for (int i=0; i<input_number/32.0; i++) {
        int quantity = 32;
        if ((i + 1) * 32 > input_number) {
            quantity = input_number - (i * 32);
        }

        quantities.push_back(quantity);
    }

    for (int i=0; i<containers.size(); i++) {
        for (int j=0; j<quantities[i]; j++) {
            vector<Ciphertext> unwrapped_container = unwrap_512_in_4_128(context, encoder, evaluator, containers[i], j);
            unwrapped_output.push_back(unwrapped_container);
        }
    }

    return unwrapped_output;
}

Ciphertext addVectors(Context context, HomEvaluator evaluator, vector<Ciphertext> c) {
    // add all ctxt vectors together
    Ciphertext product = c[0];
    for (int i=1; i<c.size(); ++i) {
        evaluator.add(product, c[i], product);
    }

    return product;
}

vector<Ciphertext> matmulCRlarge(Context context, HomEvaluator evaluator, vector<vector<Ciphertext>> rows, vector<Ciphertext> weights, const Ciphertext &bias) {
    vector<Ciphertext> output;

    for (int i=0; i<rows.size(); i++) {
        Ciphertext p1(context);
        evaluator.mult(rows[i][0], weights[0], p1);
        Ciphertext p2(context);
        evaluator.mult(rows[i][0], weights[0], p2);
        Ciphertext p3(context);
        evaluator.mult(rows[i][0], weights[0], p3);
        Ciphertext p4(context);
        evaluator.mult(rows[i][0], weights[0], p4);

        Ciphertext res = addVectors(context, evaluator, {p1, p2, p3, p4});
        res = rotsum(context, evaluator, res, 128, 1);

        evaluator.add(res, bias, res);

        output.push_back(res);
    }

    return output;
}

