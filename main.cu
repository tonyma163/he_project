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
    setup_environment(argc, argv);
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