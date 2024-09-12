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

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

int main() {

	// Call test.py
	system("mkdir ../python/outputs");
	string command = "python3 ../python/test.py";
	system(command.c_str());

	// Fetch the array data from the output.txt
	string output_folder = "../python/outputs";
	ifstream file(output_folder+"/output.txt");

	vector<double> values;
	string row;

	double num_scale = 1;

	if (file.is_open()) {
		while (getline(file, row)) {
			
			istringstream stream(row);
			string value;

			while (getline(stream, value, ',')) {
				double num = stod(value);
				values.push_back(num * num_scale);
			}
		}

	}
	file.close();

	// print the arrary data
	for (double val : values) {
		std::cout << val << ",";
	}
	std::cout << endl;

	std::cout << "phantom-fhe" << endl;
	
	EncryptionParameters parms(scheme_type::ckks);
	
	size_t poly_modulus_degree = 8192;
	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 60}));
	
	PhantomContext context(parms);

	// Create secret & public keys
	PhantomSecretKey secret_key(context);
	PhantomPublicKey public_key = secret_key.gen_publickey(context);
	
	// Create relinear key for post-multiplication
	PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

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

	// * Save ciphertext
	cout << "Saving ciphertext" << endl;
	struct stat info;
	if( stat( "./tmp", &info ) != 0 )
	system("mkdir tmp");
	ofstream outfile("./tmp/process_ciphertext.txt", ofstream::binary);
    ciphertext.save(outfile);
    outfile.close();
	
	// * Load ciphertext - May Cause CUDA Memory Not Enough
	cout << "Loading ciphertext" << endl;
	ifstream infile("./tmp/process_ciphertext.txt", ifstream::binary);
	PhantomCiphertext loaded_process_ciphertext;
	loaded_process_ciphertext.load(infile);
	infile.close();

	// Multiplication - origin * origin
	PhantomCiphertext product_ciphertext = multiply(context, loaded_process_ciphertext, loaded_process_ciphertext);
	relinearize_inplace(context, product_ciphertext, relin_keys);
	rescale_to_next_inplace(context, product_ciphertext);

	/*
	// Addition - product * product
	add_inplace(context, product_ciphertext, ciphertext); // add itself
	*/

	/*
	// Addition - product * origin
	product_ciphertext.set_scale(scale);
	mod_switch_to_next_inplace(context, ciphertext);
	add_inplace(context, product_ciphertext, ciphertext);
	*/

	// Multiplication - product * origin
	product_ciphertext.set_scale(scale);
	mod_switch_to_next_inplace(context, loaded_process_ciphertext);
	PhantomCiphertext product_ciphertext2 = multiply(context, product_ciphertext, loaded_process_ciphertext);
	relinearize_inplace(context, product_ciphertext2, relin_keys);
	rescale_to_next_inplace(context, product_ciphertext2);

	// Multiplication - product * origin
	product_ciphertext2.set_scale(scale);
	mod_switch_to_next_inplace(context, loaded_process_ciphertext);
	PhantomCiphertext product_ciphertext3 = multiply(context, product_ciphertext2, loaded_process_ciphertext);
	relinearize_inplace(context, product_ciphertext3, relin_keys);
	rescale_to_next_inplace(context, product_ciphertext3);

	// * Save ciphertext
	cout << "Saving ciphertext" << endl;
	//struct stat info;
	if( stat( "./tmp", &info ) != 0 )
	system("mkdir tmp");
	ofstream outfile2("./tmp/final_ciphertext.txt", ofstream::binary);
    product_ciphertext3.save(outfile2);
    outfile2.close();
	
	// * Load ciphertext - May Cause CUDA Memory Not Enough
	cout << "Loading ciphertext" << endl;
	ifstream infile2("./tmp/final_ciphertext.txt", ifstream::binary);
	PhantomCiphertext loaded_final_ciphertext;
	loaded_final_ciphertext.load(infile2);
	infile2.close();

	// Decrypt and decode ciphertext
	PhantomPlaintext decrypted_plaintext;
	secret_key.decrypt(context, loaded_final_ciphertext, decrypted_plaintext);
	vector<double> output_vector;
	encoder.decode(context, decrypted_plaintext, output_vector);

	// Print result
	std::cout << "output_vector: " << output_vector[0] <<endl;
	
	return 0;
}

