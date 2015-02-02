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
#ifndef _TTABLES_H_
#define _TTABLES_H_

#include <cmath>
#include <fstream>

//#include "port.h"
#include "model.h"
#include "propagator.h"
#include "corpus.h"
#include "trainNeuralNetwork.h"

//typedef boost::unordered_map<string,unsigned> word_to_int_map;
//typedef boost::unordered_map<unsigned,vector<unsigned> > int_to_vector_map;
//typedef boost::unordered_map<unsigned,unsigned > int_to_int_map;

using namespace std;
using namespace nplm;
using namespace boost::random;

//typedef long long int data_size_t; // training data can easily exceed 2G instances

struct Md {
  static double digamma(double x) {
    double result = 0, xx, xx2, xx4;
    for ( ; x < 7; ++x)
      result -= 1/x;
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
    return result;
  }
};

class TTable {
 public:
  TTable() {}
  typedef boost::unordered_map<unsigned, double> Word2Double;
  typedef std::vector<Word2Double> Word2Word2Double;
  inline double prob(unsigned e, unsigned f) const {
    if (e < ttable.size()) {
      const Word2Double& cpd = ttable[e];
      const Word2Double::const_iterator it = cpd.find(f);
      if (it == cpd.end()) return 1e-9;
      return it->second;
    } else {
      return 1e-9;
    }
  }
  inline void Increment(unsigned e, unsigned f) {
    if (e >= counts.size()) counts.resize(e + 1);
    counts[e][f] += 1.0;
  }
  inline void Increment(unsigned e, unsigned f, double x) {
    if (e >= counts.size()) counts.resize(e + 1);
    counts[e][f] += x;
  }
  void NormalizeVB(const double alpha) {
    ttable.swap(counts);
    for (unsigned i = 0; i < ttable.size(); ++i) {
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        tot += it->second + alpha;
      if (!tot) tot = 1;
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second = exp(Md::digamma(it->second + alpha) - Md::digamma(tot));
    }
    counts.clear();
  }

  void Normalize() {
    ttable.swap(counts);
    for (unsigned i = 0; i < ttable.size(); ++i) {
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it) {
        tot += it->second;
      }
      if (!tot) tot = 1;
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second /= tot;
    }
    counts.clear();
  }

  void NormalizeNN(neuralNetworkTrainer &trainer,
      word_to_int_map &src_word_to_int,
      word_to_int_map &trg_word_to_int,
      Dict &d,
      mt19937 &rng,
      param &myParam,
      int_to_vector_map& src_word_to_nn_input_tuple,
      int_to_int_map& trg_word_to_nn_output_word,
      int num_training_samples,
      int num_validation_samples,
      int outer_iteration) {
    ttable.swap(counts);
    vector<pair<unsigned,unsigned> > word_pairs;
    vector<double> probs;
    double joint_prob_sum = 0.;
    
    //FROM THE EXPECTED COUNTS, GENERATE THE TRAINING DATA
    //FOR THE NEURAL NETWORK
    int greater_one = 0;
    cerr<<"Generating the training data for neural network"<<endl;
    for (unsigned i = 0; i < ttable.size(); ++i) {
      //cerr<<"The conditioning id is "<<i<<endl;
      //std::cerr<<"The conditioning word is "<<d.Convert(i)<<std::endl;
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it){
        //cerr<<"The conditioned id is "<<it->first<<endl;
        //std::cerr<<"The conditioned word is "<<d.Convert(it->first)<<std::endl;
        tot += it->second;
        //cerr<<"Added a pair"<<endl;
        //word_pairs.push_back(make_pair(src_word_to_int[d.Convert(i)],trg_word_to_int[d.Convert(it->first)]));
        if (i >= 2) {
          word_pairs.push_back(make_pair(i,it->first));
          //cerr<<"Added pair "<<output_word_to_int[d.Convert(i)]<<" and "<<input_word_to_int[d.Convert(it->first)]<<endl;
          probs.push_back(it->second);
          if (it->second > 1.){
            greater_one++;
          }
          joint_prob_sum += it->second;
        }
      }
      if (!tot) tot = 1;
      if ( i ==1 ) {
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second /= tot;
      }
      //for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
      //  it->second /= tot;
    }
    counts.clear();
    //Now to renormalize the joint probs and sample
    double sum =0.;
    for (unsigned i=0; i<probs.size(); i++){
      probs[i] /= joint_prob_sum;
      sum += probs[i];
      //cerr<<"prob if item "<<i<<"is "<<probs[i]<<endl;
    }

    cerr<<"The joint probs sum is "<<sum<<endl;
    cerr<<"the number of events were "<<probs.size()<<endl;
    cerr<<"greater than one was "<<greater_one<<endl;
    multinomial<int> joint_sampler;
    joint_sampler.setup(probs);
    //Sampling training data
    data_size_t training_data_size=num_training_samples;
    data_size_t validation_data_size = num_validation_samples;
    Matrix<int,Dynamic,Dynamic> training_data;
    training_data.resize(myParam.ngram_size,training_data_size);
    Matrix<int,Dynamic,Dynamic> validation_data;
    validation_data.resize(myParam.ngram_size,validation_data_size);
    //cerr<<"Size of training data is "<<training_data.cols()<<endl;
    //cerr<<"Size of validation data is "<<validation_data.size()<<endl;
    //unsigned seed = 1234; //for testing only
    //mt19937 rng(seed);
    //cerr<<"Size of word pairs is "<<word_pairs.size()<<endl;
    for (unsigned i=0; i<training_data_size+validation_data_size; i++){
      //cerr<<"Sample number is "<<i<<endl;
      data_size_t sample = joint_sampler.sample(rng);
      if (src_word_to_nn_input_tuple.find(word_pairs[sample].first) == 
            src_word_to_nn_input_tuple.end()) {
        cerr<<"the id "<<word_pairs[sample].first<<" was not found in src to input map"<<endl;
      }
      if (trg_word_to_nn_output_word.find(word_pairs[sample].second) == 
            trg_word_to_nn_output_word.end()) {
        cerr<<"the id "<<word_pairs[sample].second<<" was not found in trg to output map"<<endl;
      }
      if (i<training_data_size) {
        for (int index=0; 
            index<src_word_to_nn_input_tuple[word_pairs[sample].first].size();
            index++){
          training_data(index,i) = src_word_to_nn_input_tuple[word_pairs[sample].first][index];
        }
        training_data(myParam.ngram_size-1,i) = trg_word_to_nn_output_word[word_pairs[sample].second];
        //cerr<<"sample was "<<training_data.col(i)<<endl;
      } else {

        //cerr<<"Sample is ";
        //cerr<<"validation data sample "<<i-training_data_size<<endl;
        for (int index=0;
            index<src_word_to_nn_input_tuple[word_pairs[sample].first].size();
            index++) {
          validation_data(index,i-training_data_size) = 
            src_word_to_nn_input_tuple[word_pairs[sample].first][index];
         //cerr<<src_word_to_nn_input_tuple[word_pairs[sample].first][index]<<" ";
        }
        //cerr<<endl;
        validation_data(myParam.ngram_size-1,i-training_data_size) = 
          trg_word_to_nn_output_word[word_pairs[sample].second];
        //cerr<<"Sample was "<<validation_data.col(i-training_data_size)<<endl;
      }

    }
    trainer.trainNN(myParam,
        training_data,
        validation_data,
        rng,
        outer_iteration);
    //GETTING THE PROBABILITIES FROM THE TRAINED NN
    //Matrix<double,Dynamic,Dynamic> scores;
    //scores.setZero(myParam.output_vocab_size,src_word_to_nn_input_tuple.size());
    trainer.getContextScores(src_word_to_nn_input_tuple,
        myParam,
        ttable,
        trg_word_to_nn_output_word,
        d);
    /*
    //NOW TO POPULATE THE T-table WITH THE SCORES
    populateTtableWithNNScores(scores,
      src_word_to_nn_input_tuple,
      trg_word_to_nn_output_word,
      d);
    */
  }

  template<typename DerivedA> 
  void populateTtableWithNNScores(const MatrixBase<DerivedA>& scores,
      int_to_vector_map& src_word_to_nn_input_tuple,
      int_to_int_map &trg_word_to_nn_output_word,
      Dict &d) {
    int_to_vector_map::iterator itr;
    int counter = 0;
    for (itr=src_word_to_nn_input_tuple.begin();
        itr != src_word_to_nn_input_tuple.end();
        itr++){
      int source_context_id = itr->first;
      Word2Double& cpd = ttable[source_context_id];
      double tot = 0;
      for (Word2Double::iterator cpd_it = cpd.begin(); cpd_it != cpd.end(); ++cpd_it){
          //std::cerr<<"The conditioned word is "<<d.Convert(it->first)<<std::endl;
          //word_pairs.push_back(make_pair(src_word_to_int[d.Convert(i)],trg_word_to_int[d.Convert(it->first)]));
          int index_in_col = trg_word_to_nn_output_word[cpd_it->first];
          //cerr<<"The index in col was "<<index_in_col<<endl;
          double approx_norm_prob = exp(scores(index_in_col,counter));
          cpd_it->second = approx_norm_prob;
          tot += approx_norm_prob;
      }
      //cerr<<"Total was "<<tot<<endl;
      //for (Word2Double::iterator cpd_it = cpd.begin(); cpd_it != cpd.end(); ++cpd_it){
      //  cpd_it->second /= tot;
      //}

      counter++;
    }


 
  }

  /*  
  //NORMALIZE WITH SMOOTHED L0
  void NormalizeL0() {
    ttable.swap(counts);
    for (unsigned i = 0; i < ttable.size(); ++i) {

      vector<double>  fractional_counts = vector<double>();
      vector<double>  probabilities = vector<double>();
      vector<double>  probabilities_optimized = vector<double> ();
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it) {
        fractional_counts.push_back(it->second);

        tot += it->second;
      }
      if (!tot) tot = 1;
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second /= tot;
    }
    counts.clear();
  }
  */
  // adds counts from another TTable - probabilities remain unchanged
  TTable& operator+=(const TTable& rhs) {
    if (rhs.counts.size() > counts.size()) counts.resize(rhs.counts.size());
    for (unsigned i = 0; i < rhs.counts.size(); ++i) {
      const Word2Double& cpd = rhs.counts[i];
      Word2Double& tgt = counts[i];
      for (Word2Double::const_iterator j = cpd.begin(); j != cpd.end(); ++j) {
        tgt[j->first] += j->second;
      }
    }
    return *this;
  }

  void ExportToFile(const char* filename, Dict& d) {
    std::ofstream file(filename);
    for (unsigned i = 0; i < ttable.size(); ++i) {
      //file<<"number is "<<i<<std::endl;
      const std::string& a = d.Convert(i);
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it) {
        const std::string& b = d.Convert(it->first);
        double c = log(it->second);
        file << a << '\t' << b << '\t' << c << std::endl;
        //file << i << '\t' << it->first << '\t' << c << std::endl;
      }
    }
    file.close();
  }
 public:
  Word2Word2Double ttable;
  Word2Word2Double counts;
};

#endif
