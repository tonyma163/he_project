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
Ciphertext rotsum(Context context, HomEvaluator evaluator, Ciphertext &in, int slots, int padding);
vector<Ciphertext> matmulRE(Context &context, HomEvaluator &evaluator, vector<Ciphertext> rows, const Ciphertext &weight, const Ciphertext &bias, double scale);
Ciphertext wrapUpRepeated(Context context, EnDecoder encoder, HomEvaluator evaluator, vector<Ciphertext> vectors);
Ciphertext matmulScores(Context context, HomEvaluator evaluator, vector<Ciphertext> queries, const Ciphertext &key);
Ciphertext eval_exp(Context context, HomEvaluator evaluator, Bootstrapper bootstrapper, Ciphertext &c, int inputs_number);
Ciphertext eval_inverse_naive(Context context, HomEvaluator evaluator, const Ciphertext &c, double min, double max);
vector<Ciphertext> unwrapScoresExpanded(Context context, HomEvaluator evaluator, Ciphertext c, int inputs_num);

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
    vector<Ciphertext> Q = matmulRE(context, evaluator, inputs, query_weight, query_bias, scale);
    //cout << "MatMulRE Q" << endl;
    vector<Ciphertext> K = matmulRE(context, evaluator, inputs, key_weight, key_bias, scale);
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
    Ciphertext scores_denominator = eval_inverse_naive(context, evaluator, scores_sum, 2, 5000);
    cout << "eval 1/x" << endl;

    // section 7 EvalMult
    evaluator.mult(scores, scores_denominator, scores);
    cout << "scores" << endl;

    // section 8 Unwrap
    vector<Ciphertext> unwrapped_scores = unwrapScoresExpanded(context,evaluator, scores, inputs.size());
    cout << "unwrap" << endl;

    // section 9 VecMatER

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
    // Coefficients of Taylor series for exp(x)
    vector<double> coefficients = {1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0, 1.0/720.0};

    // Initialize res with the constant term (1)
    Message msg(log2ceil(num_slots));
    msg[0] = Complex(coefficients.back(), 0.0); // start from the highest degree
    //cout << "Coefficients[0]: " << coefficients[0] << endl;
    for (size_t i=1; i<num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }

    int counter = 1;
    Ciphertext res(context);
    for (int i=coefficients.size()-2; i>=0; --i) {
        Ciphertext tmp(context);
        evaluator.mult(c, msg, tmp);
        cout << "mult " << counter++ << endl;

        //bootstrapper.bootstrap(tmp, tmp);
        //cout << "bootstrap" << endl;

        //Encode the coefficient a_i
        Message coef_msg(log2ceil(num_slots));
        for (size_t k=0; k<num_slots; ++k) {
            coef_msg[k] = Complex(coefficients[i], 0.0);
        }
        evaluator.add(tmp, coef_msg, res);
        //cout << "add" << endl;

        cout << "loop" << endl;
    }
    
    // In Horner's Method, we may not require to boostrap the ciphertext for further operations since there are operated only 6 multiplications.
    //cout << "ready for boostrapping" << endl;
    // bootstrapping
    //bootstrapper.bootstrap(res, res);
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

Ciphertext eval_inverse_naive(Context context, HomEvaluator evaluator, const Ciphertext &c, double min, double max) {
    // Eval ChebyshevFunction

    vector<double> coefficients = EvalChebyshevCoefficients([](double x) -> double {return 1 / x; }, min, max, 119);

    //
    size_t degree = coefficients.size() - 1;

    //
    Message msg(log2ceil(num_slots));
    msg[0] = Complex(coefficients[degree], 0.0); // start from the highest degree
    //cout << "Coefficients[0]: " << coefficients[0] << endl;
    for (size_t i=1; i<num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }

    int counter = 0;
    Ciphertext product(context);
    // Interate from a^n-1 down to a^0
    for (int i=degree-1; i>=0; --i) {
        // Multiply P by ciphertext (x)
        Ciphertext tmp(context);
        evaluator.mult(c, msg, tmp);
        cout << "mult " << counter++ << endl;

        //bootstrapper.bootstrap(tmp, tmp);
        //cout << "bootstrap" << endl;

        // Add the current coefficient a^i
        Message coef_msg(log2ceil(num_slots));
        for (size_t j=0; j<num_slots; ++j) {
            coef_msg[j] = Complex(coefficients[i], 0.0);
        }
        evaluator.add(tmp, coef_msg, product);
        cout << "add" << endl;

        cout << "loop" << endl;
    }

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
        //res = add(res, rotate(res, -pow(2, i)));
        //Ciphertext rotated(context);
        //evaluator.leftRotate(result, rotation_steps, rotated);

        Ciphertext rotated(context);
        evaluator.leftRotate(res, -pow(2, i), rotated);

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