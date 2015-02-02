// Copyright 2013 by Chris Dyer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <fstream>
#include <getopt.h>

#include <boost/unordered_map.hpp> 
#include <boost/functional.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/functional/hash.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "maybe_omp.h"
#include <tclap/CmdLine.h>

#include "model.h"
#include "propagator.h"
#include "param.h"
#include "neuralClasses.h"
#include "graphClasses.h"
#include "util.h"
#include "multinomial.h"
//#include "port.h"
#include "corpus.h"
#include "ttables.h"
#include "da.h"


using namespace std;
using namespace boost::random;
using namespace nplm;

//typedef boost::unordered_map<Matrix<int,Dynamic,1>, double> vector_map;
//typedef boost::unordered_map<string,unsigned> word_to_int_map;
//typedef boost::unordered_map<unsigned,vector<unsigned> > int_to_vector_map;
//typedef boost::unordered_map<unsigned,unsigned > int_to_int_map;

struct PairHash {
  size_t operator()(const pair<short,short>& x) const {
    return (unsigned short)x.first << 16 | (unsigned)x.second;
  }
};

Dict d; // integerization map

void ParseLine(const string& line,
              vector<unsigned>* src,
              vector<unsigned>* trg) {
  static const unsigned kDIV = d.Convert("|||");
  static vector<unsigned> tmp;
  src->clear();
  trg->clear();
  d.ConvertWhitespaceDelimitedLine(line, &tmp);
  unsigned i = 0;
  while(i < tmp.size() && tmp[i] != kDIV) {
    src->push_back(tmp[i]);
    ++i;
  }
  if (i < tmp.size() && tmp[i] == kDIV) {
    ++i;
    for (; i < tmp.size() ; ++i)
      trg->push_back(tmp[i]);
  }
}

void ParseLineAndConvertToNNData(const string& line,
              vector<unsigned>* src,
              vector<unsigned>* trg,
              int_to_vector_map *src_word_to_nn_input_tuple,
              int_to_int_map *trg_word_to_nn_output_word,
              word_to_int_map& src_word_to_int,
              word_to_int_map& trg_word_to_int) {
  static const unsigned kDIV = d.Convert("|||");
  static vector<unsigned> tmp;
  static vector<string> tmp_words,word_tuple;
  src->clear();
  trg->clear();
  d.ConvertWhitespaceDelimitedLine(line, &tmp, &tmp_words);
  unsigned i = 0;
  while(i < tmp.size() && tmp[i] != kDIV) {
    src->push_back(tmp[i]);

    if (src_word_to_nn_input_tuple->find(tmp[i]) == 
        src_word_to_nn_input_tuple->end()) {
      getTuple(tmp_words[i],&word_tuple);
      vector<unsigned> word_int_tuple;
      for (unsigned q=0; q<word_tuple.size(); q++){
        //cerr<<"Word in tuple was "<<word_tuple[q]<<endl;
        word_int_tuple.push_back(src_word_to_int[word_tuple[q]]);
      }
      //getchar();
      //cerr<<"src word was "<<tmp[i]<<endl;
      src_word_to_nn_input_tuple->insert(make_pair(tmp[i],word_int_tuple));
    }
    ++i;
  }
  if (i < tmp.size() && tmp[i] == kDIV) {
    ++i;
    for (; i < tmp.size() ; ++i) {
      trg->push_back(tmp[i]);
      if (trg_word_to_nn_output_word->find(tmp[i]) ==
          trg_word_to_nn_output_word->end()) {

        //cerr<<"output word was "<<tmp_words[i]<<endl;
        //cerr<<"trg word was "<<tmp[i]<<endl;
        trg_word_to_nn_output_word->
          insert(make_pair(tmp[i],trg_word_to_int[tmp_words[i]]));
        //getchar();
      }
    }
  }
}

string input;
string conditional_probability_filename = "";
int is_reverse = 0;
int ITERATIONS = 5;
int favor_diagonal = 0;
double prob_align_null = 0.08;
double diagonal_tension = 4.0;
int optimize_tension = 0;
int variational_bayes = 0;
int vanilla_m_step = 0;
double alpha = 0.01;
int no_null_word = 0;
int hidden_layer_size = 100;
int minibatch_size = 1000;
int validation_minibatch_size = 100;
int num_epochs = 10;
double learning_rate = 1.;
int input_embedding_dimension=150;
int output_embedding_dimension=150;
int ngram_size=1;
int input_vocab_size=100000;
int output_vocab_size=100000;
int num_noise_samples=100;
int num_training_samples=10E7;
int num_validation_samples=100000;
int test_minibatch_size =10;
string src_vocab_file = "vocab.fr";
string trg_vocab_file = "vocab.en";
string training_samples_file = "training.samples";
string validation_samples_file = "validation.samples";
string input_embeddings_file = "";
string output_embeddings_file = "";
string output_biases_file = "";
int use_mixture=0;
int num_threads=0;
string neural_network_model="";

struct option options[] = {
    {"input",             required_argument, 0,                  'i'},
    {"reverse",           no_argument,       &is_reverse,        1  },
    {"iterations",        required_argument, 0,                  'I'},
    {"favor_diagonal",    no_argument,       &favor_diagonal,    0  },
    {"p0",                required_argument, 0,                  'p'},
    {"diagonal_tension",  required_argument, 0,                  'T'},
    {"optimize_tension",  no_argument,       &optimize_tension,  1  },
    {"variational_bayes", no_argument,       &variational_bayes, 1  },
    {"vanilla_m_step", no_argument,       &vanilla_m_step, 1  },
    {"alpha",             required_argument, 0,                  'a'},
    {"no_null_word",      no_argument,       &no_null_word,      1  },
    {"hidden_layer_size", required_argument, 0,      'h'},
    {"minibatch-size", required_argument, 0,      'b'},
    {"validation_minibatch_size", required_argument, 0,      'B'},
    {"test_minibatch_size", required_argument, 0,    't'},
    {"num_epochs", required_argument, 0,      'e'},
    {"input_embedding_dimension", required_argument, 0,      'q'},
    {"output_embedding_dimension", required_argument, 0,      'Q'},
    {"ngram_size", required_argument, 0,      'n'},
    {"input_vocab_size",required_argument, 0,      's'},
    {"output_vocab_size", required_argument, 0,      'S'},
    {"num_noise_samples", required_argument, 0,      'k'},
    {"num_training_samples", required_argument, 0,      'z'},
    {"num_validation_samples", required_argument, 0,      'Z'},
    {"trg_vocab_file", required_argument, 0,      'f'},
    {"src_vocab_file", required_argument, 0,      'F'},
    {"use_mixture", no_argument, &use_mixture,      1},
    {"neural_network_model",        required_argument, 0,                  'P'},
    {"num_threads",        num_threads, 0,                  'u'},
    {"input_embeddings_file",        required_argument, 0,                  'D'},
    {"output_embeddings_file",        required_argument, 0,                  'C'},
    {"output_biases_file",        required_argument, 0,                  'G'},
    /*
    {"minibatch_size", no_argument, 64,          'm'},
    {"num_noise_samples", no_argument, 100,          'c'},
    {"embedding_dimension", no_argument, 150,          'd'},
    {"nn train epochs", no_argument, 20,          'e'},
    */
    {0,0,0,0}
};

bool InitCommandLine(int argc, char** argv) {
  while (1) {
    int oi;
    int c = getopt_long(argc, argv, "i:rI:dp:T:ovma:Nc:h:b:B:t:l:e:q:Q:n:s:S:k:z:Z:f:F:M:P:u:D:C:G:", options, &oi);
    if (c == -1) break;
    switch(c) {
      case 'i': input = optarg; break;
      case 'r': is_reverse = 1; break;
      case 'I': ITERATIONS = atoi(optarg); break;
      case 'd': favor_diagonal = 1; break;
      case 'p': prob_align_null = atof(optarg); break;
      case 'T': diagonal_tension = atof(optarg); break;
      case 'o': optimize_tension = 1; break;
      case 'v': variational_bayes = 1; break;
      case 'm': vanilla_m_step = 1; break;
      case 'a': alpha = atof(optarg); break;
      case 'N': no_null_word = 1; break;
      case 'c': conditional_probability_filename = optarg; break;
      case 'h': hidden_layer_size = atoi(optarg); break;
      case 'b': minibatch_size = atoi(optarg); break;
      case 'B': validation_minibatch_size = atoi(optarg); break;
      case 't': test_minibatch_size = atoi(optarg); break;
      case 'l': learning_rate = atof(optarg); break;
      case 'e': num_epochs= atoi(optarg); break;
      case 'q': input_embedding_dimension = atoi(optarg); break;
      case 'Q': output_embedding_dimension = atoi(optarg); break;
      case 'n': ngram_size = atoi(optarg); break;
      case 's': input_vocab_size = atoi(optarg); break;
      case 'S': output_vocab_size = atoi(optarg); break;
      case 'k': num_noise_samples= atoi(optarg); break;
      case 'z': num_training_samples= atoi(optarg); break;
      case 'Z': num_validation_samples= atoi(optarg); break;
      case 'f': trg_vocab_file = optarg; break;
      case 'F': src_vocab_file = optarg; break;
      case 'M': use_mixture = 1; break;
      case 'P': neural_network_model = optarg; break;
      case 'u': num_threads = atoi(optarg); break;
      case 'D': input_embeddings_file = optarg; break;
      case 'C': output_embeddings_file = optarg; break;
      case 'G': output_biases_file = optarg; break;
 
      default: return false;
    }
  }
  if (input.size() == 0) return false;
  return true;
}

int main(int argc, char** argv) {
  if (!InitCommandLine(argc, argv)) {
    cerr << "Usage: " << argv[0] << " -i file.fr-en\n"
         << " Standard options ([USE] = strongly recommended):\n"
         << "  -i: [REQ] Input parallel corpus\n"
         << "  -v: [USE] Use Dirichlet prior on lexical translation distributions\n"
         << "  -m: [USE] Run vanilla M-step\n"
         << "  -d: [USE] Favor alignment points close to the monotonic diagonoal\n"
         << "  -o: [USE] Optimize how close to the diagonal alignment points should be\n"
         << "  -r: Run alignment in reverse (condition on target and predict source)\n"
         << "  -c: Output conditional probability table\n"
         << " Advanced options:\n"
         << "  -I: number of iterations in EM training (default = 5)\n"
         << "  -p: p_null parameter (default = 0.08)\n"
         << "  -N: No null word\n"
         << "  -a: alpha parameter for optional Dirichlet prior (default = 0.01)\n"
         << "  -T: starting lambda for diagonal distance parameter (default = 4)\n"
         << "  -h: hidden layer size (default = 100)\n"
         << "  -b: training minibatch size (default = 1000)\n"
         << "  -B: validation minibatch size (default = 100)\n"
         << "  -t: test minibatch size (default = 10)\n"
         << "  -e: num epochs for neural network (default = 10)\n"
         << "  -l: learning rate (default = 1.)\n"
         << "  -q: input embedding dimension (default = 150)\n"
         << "  -Q: output embedding dimension (default = 150)\n"
         << "  -n: ngram size (default = 1)\n"
         << "  -s: input vocab size (default = 100000)\n"
         << "  -S: output vocab size (default = 100000)\n"
         << "  -k: num noise samples (default = 100)\n"
         << "  -z: num training samples (default = 10E7)\n"
         << "  -Z: num validation samples (default = 10E5)\n"
         << "  -f: target vocab file (default = vocab.en)\n"
         << "  -F: src vocab file (default = vocab.fr)\n"
         << "  -M: use_mixture (default = false)\n"
         << "  -P: neural_network_model (default = )\n"
         << "  -u: number of threads to use (default = all)\n"
         << "  -D: input embeddings file (default = "")\n"
         << "  -C: output embeddings file (default = "")\n"
         << "  -G: output weights file (default = "")\n";
    return 1;
  }
  cerr<<"read the options"<<endl;
  //Setting up the threads
  setup_threads(num_threads);

  bool use_null = !no_null_word;
  if (variational_bayes && alpha <= 0.0) {
    cerr << "--alpha must be > 0\n";
    return 1;
  }
  double prob_align_not_null = 1.0 - prob_align_null;
  const unsigned kNULL = d.Convert("<eps>");
  TTable s2t, t2s,s2t_mixture;
  boost::unordered_map<pair<short, short>, unsigned, PairHash> size_counts;
  double tot_len_ratio = 0;
  double mean_srclen_multiplier = 0;
  vector<double> probs;
  vector<double> mixture_al_probs;

  /////CREATE AND INITIALIZE THE NEURAL NETWORK AND
  //ASSOCIATED PROPAGATORS.
  //unsigned seed = std::time(0);
  unsigned seed = 1234; //for testing only
  mt19937 rng(seed);
  int last_src_vocab_id,last_trg_vocab_id;
  last_src_vocab_id = last_trg_vocab_id = -1;
  //Reading the output vocabulary
  word_to_int_map trg_word_to_int,
              src_word_to_int;
  getWordToInt(src_word_to_int,
    src_vocab_file,
    last_src_vocab_id);
  getWordToInt(trg_word_to_int,
    trg_vocab_file,
    last_trg_vocab_id);
  cerr<<"The source vocab size is "<<last_src_vocab_id+1<<endl;
  cerr<<"The target vocab size is "<<last_trg_vocab_id+1<<endl;
  
  //cerr<<"The size of the target vocab is "<<trg_word_to_int.size()<<endl;
  //cerr<<"The size of the src vocab is "<<src_word_to_int.size()<<endl;

  param myParam;
  myParam.ngram_size = ngram_size;
  myParam.input_vocab_size = last_src_vocab_id+1; //input_vocab_size;
  myParam.output_vocab_size = last_trg_vocab_id+1; //output_vocab_size;
  myParam.input_embedding_dimension = input_embedding_dimension;
  myParam.num_hidden = hidden_layer_size;
  myParam.output_embedding_dimension = output_embedding_dimension;
  myParam.share_embeddings = false;
  myParam.init_normal = true;
  myParam.init_range = 0.05;
  myParam.parameter_update = "SGD";
  myParam.adagrad_epsilon = 0.001;
  myParam.minibatch_size = minibatch_size;
  myParam.validation_minibatch_size = validation_minibatch_size;
  myParam.activation_function = "rectifier";
  myParam.loss_function = "nce";
  myParam.num_epochs=num_epochs;
  myParam.learning_rate=learning_rate;
  myParam.num_noise_samples=num_noise_samples;
  myParam.test_minibatch_size = test_minibatch_size;
  myParam.model_prefix = "nn";
  myParam.model_file = neural_network_model;

  cerr<<"Printing neural network options..."<<endl;
  cerr<<"ngram size:"<<myParam.ngram_size<<endl;
  cerr<<"input vocab size:"<<myParam.input_vocab_size<<endl;
  cerr<<"output vocab size:"<<myParam.output_vocab_size<<endl;
  cerr<<"input embedding dimension: "<<myParam.input_embedding_dimension<<endl;
  cerr<<"output embedding dimension: "<<myParam.input_embedding_dimension<<endl;
  cerr<<"num hidden: "<<myParam.num_hidden<<endl;
  cerr<<"learning_rate: "<<myParam.learning_rate<<endl;
  cerr<<"num noise samples: "<<myParam.num_noise_samples<<endl;
  cerr<<"num epochs: "<<myParam.num_epochs<<endl;
  cerr<<"minibatch size: "<<myParam.minibatch_size<<endl;
  cerr<<"validation minibatch size: "<<myParam.validation_minibatch_size<<endl;
  cerr<<"test minibatch size: "<<myParam.test_minibatch_size<<endl;

  /*
  //myParam.ngram_size=2;
  cerr<<"Creating the network..."<<endl;
  model nn(myParam.ngram_size,
      myParam.input_vocab_size,
      myParam.output_vocab_size,
      myParam.input_embedding_dimension,
      myParam.num_hidden,
      myParam.output_embedding_dimension,
      myParam.share_embeddings);

  nn.initialize(rng,
      myParam.init_normal,
      myParam.init_range,
      -log(myParam.output_vocab_size),
      myParam.parameter_update,
      myParam.adagrad_epsilon);
    nn.set_activation_function(string_to_activation_function(myParam.activation_function));
    loss_function_type loss_function = string_to_loss_function(myParam.loss_function);

  propagator prop(nn, myParam.minibatch_size);
  propagator prop_validation(nn, myParam.validation_minibatch_size);
  //SoftmaxNCELoss<multinomial<data_size_t> > softmax_loss(unigram);
  // normalization parameters
  vector_map c_h, c_h_running_gradient;
  
  cerr<<"Finished creating the network..."<<endl;
  */
  neuralNetworkTrainer trainer(myParam,
  	rng,
		input_embeddings_file,
    output_embeddings_file,
		output_biases_file);
  //PARSE SENTENCES TO GET WORD ID's AND LOAD THE NEURAL
  //NETWORK
  cerr<<"Preparing the neural network"<<endl;
  ifstream in(input.c_str());
  if (!in) {
    cerr << "Can't read " << input << endl;
    return 1;
  }
  /*
  while(true) {
    int lc = 0;
    bool flag = false;
    string line;
    string ssrc, strg;
    vector<unsigned> src, trg;
    double toks = 0;

    getline(in, line);
    if (!in) break;
    ++lc;
    if (lc % 1000 == 0) { cerr << '.'; flag = true; }
    if (lc %50000 == 0) { cerr << " [" << lc << "]\n" << flush; flag = false; }
    src.clear(); trg.clear();
    ParseLine(line, &src, &trg);
    if (is_reverse) swap(src, trg);
    if (src.size() == 0 || trg.size() == 0) {
      cerr << "Error in line " << lc << "\n" << line << endl;
      return 1;
    }
    toks += trg.size();
  }
  */
  int_to_vector_map src_word_to_nn_input_tuple;
  int_to_int_map trg_word_to_nn_output_word;
  boost::unordered_map<unsigned,vector<double> > mixture_probs;
  ////////COMPUTING EXPECTED COUNTS/////////////////
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const bool final_iteration = (iter == (ITERATIONS - 1));
    cerr << "ITERATION " << (iter + 1) << (final_iteration ? " (FINAL)" : "") << endl;
    ifstream in(input.c_str());
    if (!in) {
      cerr << "Can't read " << input << endl;
      return 1;
    }
    double likelihood = 0;
    double denom = 0.0;
    int lc = 0;
    bool flag = false;
    string line;
    string ssrc, strg;
    vector<unsigned> src, trg;
    double c0 = 0;
    double emp_feat = 0;
    double toks = 0;
    while(true) {
      getline(in, line);
      if (!in) break;
      ++lc;
      if (lc % 1000 == 0) { cerr << '.'; flag = true; }
      if (lc %50000 == 0) { cerr << " [" << lc << "]\n" << flush; flag = false; }
      src.clear(); trg.clear();
      if (iter == 0){
        //CONVERT THE SOURCE AND TARGET WORDS TO NN WORDS 
        //AND NN TUPLES
        ParseLineAndConvertToNNData(line,
            &src,
            &trg,
            &src_word_to_nn_input_tuple,
            &trg_word_to_nn_output_word,
            src_word_to_int,
            trg_word_to_int);
        //FILLING THE MIXTURE PROBS DICTIONARY
        if (use_mixture == true) {
        for (int i=0;i<src.size();i++){
            if (mixture_probs.find(src[i]) ==
                mixture_probs.end()){
              mixture_probs.insert(make_pair(src[i],vector<double>(2,0.5)));
            }
          }
        }

      } else {
        ParseLine(line, &src, &trg);
      }
      if (is_reverse) swap(src, trg);
      if (src.size() == 0 || trg.size() == 0) {
        cerr << "Error in line " << lc << "\n" << line << endl;
        return 1;
      }
      if (iter == 0) {
        tot_len_ratio += static_cast<double>(trg.size()) / static_cast<double>(src.size());
      }
      denom += trg.size();
      probs.resize(src.size() + 1);
      if (use_mixture == true) {
        mixture_al_probs.resize(src.size()+1);
      }
      if (iter == 0)
        ++size_counts[make_pair<short,short>(trg.size(), src.size())];
      bool first_al = true;  // used when printing alignments
      toks += trg.size();
      for (unsigned j = 0; j < trg.size(); ++j) {
        const unsigned& f_j = trg[j];
        double sum = 0;
        double prob_a_i = 1.0 / (src.size() + use_null);  // uniform (model 1)
        if (use_null) {
          if (favor_diagonal) prob_a_i = prob_align_null;
          probs[0] = s2t.prob(kNULL, f_j) * prob_a_i;
          sum += probs[0];
        }
        double az = 0;
        if (favor_diagonal)
          az = DiagonalAlignment::ComputeZ(j+1, trg.size(), src.size(), diagonal_tension) / prob_align_not_null;
        for (unsigned i = 1; i <= src.size(); ++i) {
		  cerr<<"The source word is "<<d.Convert(src[i]);
		  getchar();
          if (favor_diagonal) {
            prob_a_i = DiagonalAlignment::UnnormalizedProb(j + 1, i, trg.size(), src.size(), diagonal_tension) / az;
          }
          if (use_mixture == true) {
            probs[i] = mixture_probs[src[i-1]][0] * 
              s2t.prob(src[i-1], f_j) *prob_a_i ;
            mixture_al_probs[i] =  mixture_probs[src[i-1]][1] * 
              s2t_mixture.prob(src[i-1],f_j)*prob_a_i;
            sum += (probs[i]+mixture_al_probs[i]);
            //sum += probs[i];
          } else {
            probs[i] = s2t.prob(src[i-1], f_j) * prob_a_i;
            sum += probs[i];
          }
        }
        if (final_iteration) {
          double max_p = -1;
          int max_index = -1;
          if (use_null) {
            max_index = 0;
            max_p = probs[0];
          }
          for (unsigned i = 1; i <= src.size(); ++i) {
            //QUESTION: DURING VITERBI, SHOULD WE ALSO TAKE THE 
            //MAX OVER THE NN t-table VS standard t-table OR SHOULD
            //WE MARGINALIZE THE t-table out. I guess marginalize
            if (use_mixture == true) {
              
              if (max(probs[i],mixture_al_probs[i]) > max_p){
                max_index = i;
                max_p = max(probs[i],mixture_al_probs[i]);
              }
              
			  /*
              if (probs[i] + mixture_al_probs[i] > max_p){
                max_index = i;
                max_p = probs[i] +mixture_al_probs[i];
              }
			  */

            } else {
              if (probs[i] > max_p) {
                max_index = i;
                max_p = probs[i];
              }
            }
          }
          if (max_index > 0) {
            if (first_al) first_al = false; else cout << ' ';
            if (is_reverse)
              cout << j << '-' << (max_index - 1);
            else
              cout << (max_index - 1) << '-' << j;
          }
        } else {
          if (use_null) {
            double count = probs[0] / sum;
            c0 += count;
            s2t.Increment(kNULL, f_j, count);
          }
          for (unsigned i = 1; i <= src.size(); ++i) {
              const double p = probs[i] / sum;
              s2t.Increment(src[i-1], f_j, p);
            if (use_mixture == true) {
              const double mixture_p = mixture_al_probs[i] / sum;
              s2t_mixture.Increment(src[i-1],f_j,mixture_p);
              emp_feat += 
                DiagonalAlignment::Feature(j, i, trg.size(), src.size()) * (p + mixture_p);
            } else {
              emp_feat += DiagonalAlignment::Feature(j, i, trg.size(), src.size()) * p;
            }
          }
        }
        likelihood += log(sum);
      }
      if (final_iteration) cout << endl;
    }
    if (iter == 0){
      //CREATING THE TEST DATA
      trainer.createTestData(src_word_to_nn_input_tuple,myParam);
    }
    // log(e) = 1.0
    double base2_likelihood = likelihood / log(2);

    if (flag) { cerr << endl; }
    if (iter == 0) {
      mean_srclen_multiplier = tot_len_ratio / lc;
      cerr << "expected target length = source length * " << mean_srclen_multiplier << endl;
    }
    emp_feat /= toks;
    cerr << "  log_e likelihood: " << likelihood << endl;
    cerr << "  log_2 likelihood: " << base2_likelihood << endl;
    cerr << "     cross entropy: " << (-base2_likelihood / denom) << endl;
    cerr << "        perplexity: " << pow(2.0, -base2_likelihood / denom) << endl;
    cerr << "      posterior p0: " << c0 / toks << endl;
    cerr << " posterior al-feat: " << emp_feat << endl;
    //cerr << "     model tension: " << mod_feat / toks << endl;
    cerr << "       size counts: " << size_counts.size() << endl;
    if (!final_iteration) {
      if (favor_diagonal && optimize_tension && iter > 0) {
        for (int ii = 0; ii < 8; ++ii) {
          double mod_feat = 0;
          boost::unordered_map<pair<short,short>,unsigned,PairHash>::iterator it = size_counts.begin();
          for(; it != size_counts.end(); ++it) {
            const pair<short,short>& p = it->first;
            for (short j = 1; j <= p.first; ++j)
              mod_feat += it->second * DiagonalAlignment::ComputeDLogZ(j, p.first, p.second, diagonal_tension);
          }
          mod_feat /= toks;
          cerr << "  " << ii + 1 << "  model al-feat: " << mod_feat << " (tension=" << diagonal_tension << ")\n";
          diagonal_tension += (emp_feat - mod_feat) * 20.0;
          if (diagonal_tension <= 0.1) diagonal_tension = 0.1;
          if (diagonal_tension > 14) diagonal_tension = 14;
        }
        cerr << "     final tension: " << diagonal_tension << endl;
      }
      if (variational_bayes) {
        s2t.NormalizeVB(alpha);
      } else if (vanilla_m_step) {
        s2t.Normalize();
      } else {
        cerr<<"neural network optimize"<<endl;
        if (use_mixture == true) {
          s2t.Normalize();
          s2t_mixture.NormalizeNN(trainer,
          src_word_to_int,
          trg_word_to_int,
          d,
          rng,
          myParam,
          src_word_to_nn_input_tuple,
          trg_word_to_nn_output_word,
          num_training_samples,
          num_validation_samples,
          iter);
        } else {
          s2t.NormalizeNN(trainer,
            src_word_to_int,
            trg_word_to_int,
            d,
            rng,
            myParam,
            src_word_to_nn_input_tuple,
            trg_word_to_nn_output_word,
            num_training_samples,
            num_validation_samples,
            iter);
        }
        //IF USE MIXTURE IS TRUE, WE MUST 
        //RENORMALIZE THE MIXTURE PROBS TOO
        if (use_mixture == true){
          boost::unordered_map<unsigned,vector<double> >::iterator mixture_itr;
          for (mixture_itr=mixture_probs.begin();
              mixture_itr!=mixture_probs.end(); 
              mixture_itr++) {
            double mixture_sum = mixture_itr->second[0]+mixture_itr->second[1];
            mixture_itr->second[0] /= mixture_sum;
            mixture_itr->second[1] /= mixture_sum;
          }
        }
      }

      //prob_align_null *= 0.8; // XXX
      //prob_align_null += (c0 / toks) * 0.2;
      prob_align_not_null = 1.0 - prob_align_null;
    }
  }
  if (!conditional_probability_filename.empty()) {
    cerr << "conditional probabilities: " << conditional_probability_filename << endl;
    s2t.ExportToFile(conditional_probability_filename.c_str(), d);
  }
  return 0;
}
