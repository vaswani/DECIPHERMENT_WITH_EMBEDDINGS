#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/unordered_map.hpp>
#include "lm.h"
#include <boost/random.hpp>
#include <math.h>
#include <vector>
#include "slice_with_embeddings_x.h"

using namespace std;


int main(int argc, char** argv) {
    /*boost::mt19937 int_gen;
    boost::uniform_real<double> real_distribution(0,1);
    int i = 0;
    while(true){
        double random = real_distribution(int_gen);
        if(random == 1) {
            cout << "pass" << endl;
            break;
        }
    }*/
    LM test_lm;
    int iteration = atoi(argv[1]);
    test_lm.load_lm(argv[2]);
    int embedding_dimension = atoi(argv[10]);
    int opt_itr = atoi(argv[11]);
    int interval = atoi(argv[12]);
    float alpha = atof(argv[13]);
    int base_scale = atoi(argv[14]);
    int num_threads = atoi(argv[7]);
    Decipherment decipherer(test_lm, atoi(argv[5]), "/scratch/base", argv[8], argv[9], 
    embedding_dimension, opt_itr, interval, alpha, base_scale, num_threads);
    decipherer.loadSeedTable(argv[6]);
    decipherer.loadSlice(argv[3]);
    decipherer.loadCipherBigrams(argv[4]);
    decipherer.shuffle();
    decipherer.initSamples();
    for(int i = 0; i < iteration; i += interval) {
        vector<ThreadData> targs(num_threads);
        vector<pthread_t> t_handler(num_threads);
        for(int tid = 0; tid < num_threads; tid++) {
            targs[tid] = ThreadData(tid, interval, num_threads, &decipherer);
            int rc = pthread_create(&t_handler[tid], 0, &ThreadWrapper::runSampling, (void*)(&targs[tid]));
            if(rc != 0) {
                cout << "error creating thread" << endl;
                exit(0);
            }
        } 
        for(int tid = 0; tid < num_threads; tid++) {
            pthread_join(t_handler[tid], 0);
        }
        // update mapping matrix m
        if(i + interval < iteration){
            cout << "building counts matrix" << endl;
            decipherer.buildCountsMatrix();
            decipherer.doMappingOptimization();
            decipherer.updateCache();
        }
    }
    string tmp_dir = getenv("TMPDIR");
    decipherer.printTTable((tmp_dir + "/cipher.id.ptable.final").c_str());
    decipherer.printBase(tmp_dir + "/base");
    //decipherer.printAccCounts((string("") + getenv("TMPDIR") + "/cipher.id.counts.final").c_str());
    return 0;
}

