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

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

int main() {
	std::cout << "phantom-fhe" << endl;
	
	EncryptionParameters parms(scheme_type::ckks);
	
	size_t poly_modulus_degree = 8192;
	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 60}));
	
	PhantomContext context(parms);

	// Create secret & public keys
	PhantomSecretKey secret_key(context);
	PhantomPublicKey public_key = secret_key.gen_publickey(context);
	
	// Encoder & encryptor setup
	PhantomCKKSEncoder encoder(context);
	double scale = pow(2.0, 40); // Scale factor

	// Input data
	vector<double> input_vector = {1.1, 2.2, 3.3};
	std::cout << "input_vector: " << input_vector[0] << endl;	

	// Encode & encrypt input data
	PhantomPlaintext plaintext;
	encoder.encode(context, input_vector, scale, plaintext);
	PhantomCiphertext ciphertext;
	public_key.encrypt_asymmetric(context, plaintext, ciphertext);

	// Decrypt and decode ciphertext
	PhantomPlaintext decrypted_plaintext;
	secret_key.decrypt(context, ciphertext, decrypted_plaintext);
	vector<double> output_vector;
	encoder.decode(context, decrypted_plaintext, output_vector);

	// Print result
	std::cout << "output_vector" << output_vector[0] <<endl;
	
	return 0;
}

