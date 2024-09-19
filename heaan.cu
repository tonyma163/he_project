#include <iostream>
#include <vector>
//#include <complex>
#include "HEaaN/HEaaN.hpp"

using namespace std;
using namespace HEaaN;

int main() {
    Context context = makeContext(ParameterPreset::FVa);

    SecretKey sk(context);

    KeyGenerator keygen(context, sk);
    keygen.genEncryptionKey();
    keygen.genMultiplicationKey();
    KeyPack keypack = keygen.getKeyPack();

    Encryptor encryptor(context);
    Decryptor decryptor(context);
    EnDecoder encoder(context);
    HomEvaluator evaluator(context, keypack);

    // input data
    vector<Complex> vec1 = { Complex{1.0, 0.0}, Complex{2.0, 0.0} };
    vector<Complex> vec2 = { Complex{2.0, 0.0}, Complex{3.0, 0.0} };

    //
    Message msg1(2);
    Message msg2(2);

    msg1[0] = vec1[0]; msg1[1] = vec1[1];
    msg2[0] = vec2[0]; msg2[1] = vec2[1];

    // Encode the message
    Plaintext ptxt1 = encoder.encode(msg1);
    Plaintext ptxt2 = encoder.encode(msg2);

    // Encrypt the plaintext
    Ciphertext ctxt1(context);
    Ciphertext ctxt2(context);
    encryptor.encrypt(msg1, keypack, ctxt1);
    encryptor.encrypt(msg2, keypack, ctxt2);

    cout << "Scale Factor: " << ctxt1.getCurrentScaleFactor() << endl;
    cout << "Rescale Counter: " << ctxt1.getRescaleCounter() << endl;

    // Multiplication
    Ciphertext ctxt_result(context);
    evaluator.mult(ctxt1, ctxt2, ctxt_result);

    cout << "Scale Factor: " << ctxt_result.getCurrentScaleFactor() << endl;
    cout << "Rescale Counter: " << ctxt1.getRescaleCounter() << endl;

    //evaluator.rescale(ctxt_result);

    /*
    // Square
    Ciphertext squaredCiphertext(context);
    evaluator.square(ciphertext, squaredCiphertext);
    evaluator.rescale(squaredCiphertext);

    // squared level
    cout << "squared level: " << squaredCiphertext.getLevel() << endl;

    if (squaredCiphertext.getLevel() < 1) {
        cerr << "Not enough levels for further multiplication" << endl;
        return -1;
    }

    // Multiplication
    Ciphertext resultCiphertext(context);
    evaluator.mult(squaredCiphertext, ciphertext, resultCiphertext);
    evaluator.rescale(resultCiphertext);
    */

    // Decrypt the result
    Message decrypted;
    decryptor.decrypt(ctxt_result, sk, decrypted);

    // Print the result
    /*
    for (size_t i=0; decryptedMessage.getSize(); ++i) {
        cout << decryptedMessage[i] << endl;
    }
    */
    cout << decrypted[0].real() << endl;
    cout << decrypted[1].real() << endl;
    cout << decrypted[2].real() << endl;
    cout << decrypted[3].real() << endl;



    return 0;
}