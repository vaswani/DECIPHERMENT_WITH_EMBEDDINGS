/* 
 * File:   main.cpp
 * Author: Qing
 *
 * Created on January 15, 2014, 3:02 PM
 */
#pragma once
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include "lm.h"
#include <math.h>
#include <float.h>
#include <limits.h>
#include <queue> 
#include <boost/random.hpp>

using namespace std;


class Bigram {
public:
    unsigned int tokens[2];
    unsigned int sample_tokens[2];
    long count;
    Bigram(unsigned int tok1, unsigned int tok2, long c) {
        tokens[0] = tok1;
        tokens[1] = tok2;
        count = c;
    }
};

class CountList{
  public:
    boost::unordered_map<unsigned int,unsigned int> members; // to quickly check if a given word is in the list
    vector<unsigned int> member_list;
};

class Candidate{
  public:
  unsigned int token;
  float prob;
  Candidate(){
  }
  Candidate(int t, float p) {
    token = t;
    prob = p;
  }
  bool operator<(const Candidate& c) const {
    return (prob > c.prob);
  }
};

class Decipherment {
   
  public:
  LM &lm;
  float** slice_list;
  int slice_list_size;
  vector<Bigram> cipher_bigrams;
  boost::unordered_map<long,long> acc_counts; // accumulated cooc count for last 1000 iterations
  boost::unordered_map<long,long> counts; // the cache
  boost::unordered_map<long,long> o_counts; // counts of observed cipher tokens
  boost::unordered_map<unsigned int,CountList> channel_list;
  boost::unordered_map<unsigned int, vector<Candidate> > seed_table;
  boost::unordered_map<unsigned int, unsigned int> observed_v;
  boost::unordered_map<long, float> base; // base distribution
  int max_iter;
  float uniform_base;
  float alpha;
  double optr_count;
  float token_count; 
  double corpus_prob;
  bool is_viterbi;
  unsigned int seed;
  int last_iterations;
  
  boost::mt19937 int_gen,flt_gen;
  boost::uniform_int<int>* int_vocab_distribution;
  boost::uniform_real<double>* real_distribution; 
 
  Decipherment(LM &dep_lm, unsigned int s):
    lm(dep_lm),
    seed(s){
    uniform_base = 0.0;
    alpha = 1.0;
    corpus_prob = 0;
    slice_list_size = 2000;
    last_iterations = 1000;
    int_gen.seed(seed);
    flt_gen.seed(seed + 100);
    int_vocab_distribution = new boost::uniform_int<int>(0, lm.vocab_size - 1);
    real_distribution = new boost::uniform_real<double>(0,1);
  }

  long concatNumbers(unsigned int w1, unsigned int w2){
    return (((long)w1) << 30 | ((long)w2));
  }  

  void loadBaseFromFile(const char* file) {
      ifstream ifs(file);
      string line;
      if(!ifs.is_open()) {
          cout << file << "not found" << endl;
          return;
      }
      boost::unordered_map<unsigned int, pair<unsigned int, float> > bestTrans;
      while(getline(ifs,line)) {
          vector<string> entries;
          LM::split(line, entries, " ");
          unsigned int eid = atoi(entries[0].c_str());
          unsigned int fid = atoi(entries[1].c_str());
          float score = atof(entries[2].c_str());
          if(bestTrans.count(eid) == 0) {
              bestTrans[eid] = pair<unsigned int, float>(fid, score);  
          } else if(score > bestTrans[eid].second) {
              bestTrans[eid].first = fid;
              bestTrans[eid].second = score;
          }
      }
      // now construct the base distribution matrix;
      for(int i = 0; i < lm.vocab_size; i++) {
          unsigned int eid = lm.hidden_vocab[i];
          bool evenBase = true;
          unsigned int bestID = 0;
          float bestScore = 0.0;
          if(bestTrans.count(eid) != 0) {
              bestID = bestTrans[eid].first;
              if(observed_v.count(bestID) != 0 && bestTrans[eid].second >= 0.1) {
                  evenBase = false;
                  bestScore = bestTrans[eid].second > 0.9? 0.9:bestTrans[eid].second;
                  //cout << "best" << " " << eid << " " << bestID << " " << bestScore << endl;
              }
          }
          for(boost::unordered_map<unsigned int, unsigned int>::iterator itr = observed_v.begin();
              itr != observed_v.end(); itr++) {
              unsigned int fid = itr->first;
              if(evenBase) {
                  base[(long)eid << 30 | fid] = uniform_base;
              } else {
                  if(fid == bestID) {
                      base[(long)eid << 30 | fid] = bestScore;
                      //cout << "found" << " " << eid << " " << fid << " " << bestTrans[eid].second << endl;
                  } else {
                      base[(long)eid << 30 | fid] = (1.0 - bestScore) / (float)(observed_v.size() - 1);
                  }
              }
              if(base[(long)eid << 30 | fid] > uniform_base) {
                  cout << eid << " " << fid << " " << bestScore << " " << base[(long)eid << 30 | fid] << endl;
                  if(channel_list[fid].members.count(eid) == 0) {
                      channel_list[fid].members[eid] = 1;
                      channel_list[fid].member_list.push_back(eid);
                  } 
              } 
          }
      }
      cout << "base size: " << base.size() << endl; 
  }

  void shuffle() {
    unsigned int n = cipher_bigrams.size();
    boost::mt19937 rd_gen(time(0) + seed);
    for (int i = 0; i < n; i++) {
      boost::uniform_int<int> int_distribution(0, n - i - 1);
      int change = i + int_distribution(rd_gen);
      Bigram buffer = cipher_bigrams[i];
      cipher_bigrams[i] = cipher_bigrams[change];
      cipher_bigrams[change] = buffer;
    }
  }

  void loadSeedTable(const char* file) {
       ifstream ifs(file);
       string line;
       if(!ifs.is_open()) {
         cout << "seed table file not found" << endl;
         return;
       }
       while(getline(ifs,line)){
          vector<string> entries;
          LM::split(line, entries, " ||| ");
          vector<string> scores;
          LM::split(entries[2], scores, " ");
          //cerr << entries.size() << " " << entries[0] << " " << entries[1] << " " << entries[2] << endl;
          unsigned int plain = atoi(entries[1].c_str());
          unsigned int cipher = atoi(entries[0].c_str());
          float score = atof(scores[1].c_str());
          if(seed_table.count(cipher) == 0) {
            seed_table[cipher] = vector<Candidate>();
          }
          seed_table[cipher].push_back(Candidate(plain, score));          
       }
       for(boost::unordered_map<unsigned int, vector<Candidate> >::iterator itr = seed_table.begin();
           itr != seed_table.end(); itr++) {
           stable_sort(itr->second.begin(), itr->second.end());
           //for(int i = 0; i < itr->second.size(); i++) {
           //  cout << itr->first << " " << (itr->second)[i].token << " " << (itr->second)[i].prob << endl;
           //}
       }
       cout << "seed table loaded" << endl;
  }

  void loadSlice(const char* file) {
    ifstream ifs(file);
    string line;
    if(!ifs.is_open()) {
      cout << "file not found" << endl;
      exit(0);
    }
    unsigned int rank = 0;
    unsigned int context = 0;
    slice_list = new float*[2 * lm.pseu_end];
    for(int i = 0; i < 2 * lm.pseu_end; i++) {
      slice_list[i] = 0;
    }
    while(getline(ifs,line)) {
      vector<string> entries;
      LM::split(line, entries, " ");
      if(entries.size() == 1) {
        cout << "slice list format err" << endl;
        exit(0);
        //slice_list = atoi(entries[0].c_str());
        //continue;
      } 
      vector<string> tokens;
      LM::split(entries[0], tokens, "|");
      context = lm.get_token_id(tokens[0]) + lm.get_token_id(tokens[1]);
      float prob = pow(10.0, 
                       lm.get_ngram_prob(lm.get_token_id(tokens[0]), lm.get_token_id(entries[1])) 
                       + lm.get_ngram_prob(lm.get_token_id(entries[1]), lm.get_token_id(tokens[1])));
      if(slice_list[context] == 0) {
        slice_list[context] = new float[2 * slice_list_size];
      }

      slice_list[context][rank] = lm.get_token_id(entries[1]);
      slice_list[context][rank + 1] = prob;
      
      //cout << context <<  " " << prob << endl;
      rank += 2;
      if(rank == 2 * slice_list_size) {
        rank = 0;
      }
    }
    cout << "slice table loaded" << endl;
  }

  void getBestSequence(const Bigram& f, unsigned int* e){
    if(seed_table.count(f.tokens[0]) == 0 
       || seed_table.count(f.tokens[1]) == 0){
      for(int i = 0; i < 2; i++) {
        unsigned int hidden = 0;
        if(seed_table.count(f.tokens[i]) > 0){
          hidden = seed_table[f.tokens[i]][0].token;
          if(hidden >= lm.pseu_end + 1 || lm.uni_gram_prob[hidden][0] == -10.0){
            hidden = lm.hidden_vocab[(*int_vocab_distribution)(int_gen)];
          } 
        }else{
          hidden = lm.hidden_vocab[(*int_vocab_distribution)(int_gen)];
        }
        e[i] = hidden;
      } 
      is_viterbi = false;
    }else{
      Candidate lattice[20][2];
      // populate lattice with candidates for first token
      vector<Candidate> cands = seed_table[f.tokens[0]];
      int i;
      for(i = 0; i < cands.size() && i < 20; i++){
        lattice[i][0].token = cands[i].token;
        lattice[i][0].prob = log10(cands[i].prob) + 
                             lm.get_ngram_prob(lm.pseu_start, cands[i].token);
      }
      cands = seed_table[f.tokens[1]];
      float best_prob = -FLT_MAX;
      for(int j = 0; j < cands.size() && j < 20; j++){
        lattice[j][1].token = cands[j].token;
        lattice[j][1].prob = -FLT_MAX;
        // compute viterbi sequence
        for(int k = 0; k < i; k++) {
          float prob = lattice[k][0].prob +
                       log10(cands[j].prob) +
                       lm.get_ngram_prob(lattice[k][0].token, cands[j].token) +
                       lm.get_ngram_prob(cands[j].token, lm.pseu_end);
          if(prob > best_prob) {
            best_prob = prob;
            e[0] = lattice[k][0].token;
            e[1] = lattice[j][1].token; 
          }
                       
        }
      }
      is_viterbi = true;
      //cout << e[0] << " " << e[1] << endl; 
    }    
  }
  
  float getChannelProb(unsigned int hidden, unsigned int observe){
    int joint_count = 0;
    int condition_count = 0;
    long joint = concatNumbers(hidden, observe);
    long condition = hidden;
    
    if(counts.count(joint) !=0 ) {
      joint_count = counts[joint];
    }
    if(counts.count(condition) != 0) {
      condition_count = counts[condition];
    }
    if(base.count(joint) == 0) {
        cout << " base 0 exception " << hidden << " " << observe << endl;
        exit(0);
    }
    return (alpha * base[joint] + joint_count)/(alpha + condition_count);
  }



  int getCandidates(float* cand_score, boost::unordered_map<unsigned int,unsigned int>& members, 
                    float threshold, float raw_channel, vector<unsigned int>& candidates) {
    int location = 2;
    while(location <= 2 * slice_list_size && cand_score[location - 1] * raw_channel >= threshold) {
      location += 4;
    }
    if(location > (2 * slice_list_size)) {
      return slice_list_size;
    }else {
      return (location - 2) >> 1;
    }  
  }

  void loadCipherBigrams(const char* file) {
       ifstream ifs(file);
       string line;
       if(!ifs.is_open()) {
           cout << "file not found" << endl;
           exit(0);
       }
       while(getline(ifs,line)){
          vector<string> entries;
          LM::split(line, entries, "\t");
          vector<string> s_tokens;
          LM::split(entries[1], s_tokens, " ");
          Bigram tmp(atoi(s_tokens[0].c_str()), atoi(s_tokens[1].c_str()), atoi(entries[0].c_str()));
          cipher_bigrams.push_back(tmp);
       }
       cout << cipher_bigrams.size() << " bigrams loaded" << endl;
  }

  void initSamples() {
    unsigned int sample[2];
    for(int i = 0; i < cipher_bigrams.size(); i++){
      Bigram& cipher = cipher_bigrams[i];
      getBestSequence(cipher, sample);
      for(int j = 0; j < 2; j++) {
        observed_v[cipher.tokens[j]] = 1;
        if(o_counts.count(cipher.tokens[j]) == 0) {
          o_counts[cipher.tokens[j]] = cipher.count;
        }else {
          o_counts[cipher.tokens[j]] += cipher.count; 
        }
        // update sample and  counts
        if(sample[j] >= lm.pseu_end + 1 || lm.uni_gram_prob[sample[j]][0] == -10.0){
          sample[j] = lm.hidden_vocab[(*int_vocab_distribution)(int_gen)];
        }
        cipher.sample_tokens[j] = sample[j];
        long plain_token = sample[j];
        if(counts.count(plain_token) == 0) {
          counts[plain_token] = cipher.count;
        }else {
          counts[plain_token] += cipher.count;
        }
       
        long plain_cipher_pair = concatNumbers(sample[j], cipher.tokens[j]);
        if(counts.count(plain_cipher_pair) == 0) {
          counts[plain_cipher_pair] = cipher.count;
        }else {
          counts[plain_cipher_pair] += cipher.count;
        }
        // update countlist
        if(channel_list.count(cipher.tokens[j]) == 0) {
          channel_list[cipher.tokens[j]] = CountList();
        }
        CountList& count_list = channel_list[cipher.tokens[j]];
        if(count_list.members.count(sample[j]) == 0) {
          count_list.members[sample[j]] = 1;
          count_list.member_list.push_back(sample[j]);
        }
        
      }
    }
    uniform_base = 1.0 / (float)observed_v.size(); 
    cout << "Observe Vocab: " << observed_v.size() << endl;
    //cout << counts.size() << " " << o_counts.size() << " " << channel_list.size() << endl;
  }

  unsigned int drawSample(int pos, Bigram& cipher, unsigned int hidden, unsigned int observed) {
    unsigned int pre_hidden, post_hidden;
    if(pos == 0) {
      pre_hidden = lm.pseu_start;
      post_hidden = cipher.sample_tokens[1];
    }else {
      pre_hidden = cipher.sample_tokens[0];
      post_hidden = lm.pseu_end;
    }

    float ngram_prob = pow(10, lm.get_ngram_prob(pre_hidden, hidden) +
                                       lm.get_ngram_prob(hidden, post_hidden));
    float channel_prob = getChannelProb(hidden, observed);
    double random_num = (*real_distribution)(flt_gen);
    float threshold = ngram_prob * channel_prob * random_num;

    unsigned int context = pre_hidden + post_hidden;
    token_count++;

    CountList& exist_trans = channel_list[observed];
    float* slice_cand_list = slice_list[context]; 
    int slice_cand_size = slice_list_size * 2;
    float score = 1.0;
    float raw_channel = (alpha * uniform_base + 0) / (alpha + 0);
    if(slice_cand_list[slice_cand_size - 1] * raw_channel < threshold) {
      vector<unsigned int> candidates;
      int range1 = getCandidates(slice_cand_list, exist_trans.members, threshold, raw_channel, candidates);
      int range2 = exist_trans.member_list.size();
      int range = range1 + range2;
      int cand_index = 0;
      //boost::unordered_set<unsigned int> cand_to_remove;
      while(true) {
        optr_count++;
        boost::uniform_int<int> int_cand_distribution(0, range - 1);
        cand_index = int_cand_distribution(int_gen);
        if(cand_index < range1) { // drop samples from set A: P(trigram)*prior > T
          int location = cand_index << 1;
          unsigned int new_hidden = slice_cand_list[location];
          if(exist_trans.members.count(new_hidden) != 0) {
            continue;
          }
          float ngram_prob = slice_cand_list[location + 1];
          float channel_prob = getChannelProb(new_hidden, observed);
          if(ngram_prob * channel_prob >= threshold) {
            exist_trans.members[new_hidden] = 1;
            exist_trans.member_list.push_back(new_hidden);
            return new_hidden;
          }
        }else { // draw samples from set B: count(e,f) > 0
          cand_index -= range1;
          unsigned int new_hidden = exist_trans.member_list[cand_index];
          if(new_hidden == hidden) {
            /*for(boost::unordered_set<unsigned int>::iterator itr = cand_to_remove.begin();
                itr != cand_to_remove.end(); itr++) {
              exist_trans.members.erase(*itr);
            }
            exist_trans.member_list.clear();
            for(boost::unordered_map<unsigned int,unsigned int>::iterator itr = exist_trans.members.begin();
                itr != exist_trans.members.end(); itr++) {
              exist_trans.member_list.push_back(itr->first);
            }*/
            return new_hidden;
          } 
          float channel_prob = getChannelProb(new_hidden, observed);
          score = pow(10, lm.get_ngram_prob(pre_hidden, new_hidden) +
                        lm.get_ngram_prob(new_hidden, post_hidden)) * channel_prob;
          if(score >= threshold) {
            /*for(boost::unordered_set<unsigned int>::iterator itr = cand_to_remove.begin();
                itr != cand_to_remove.end(); itr++) {
              exist_trans.members.erase(*itr);
            }
            exist_trans.member_list.clear();
            for(boost::unordered_map<unsigned int,unsigned int>::iterator itr = exist_trans.members.begin();
                itr != exist_trans.members.end(); itr++) {
              exist_trans.member_list.push_back(itr->first);
            }*/
            return new_hidden;
          } 
          // remove obligated items
          long cand_pair = (long)new_hidden << 30 | observed;
          if(counts.count(cand_pair) == 0 && base[cand_pair] <= uniform_base) {
            exist_trans.members.erase(new_hidden);
            exist_trans.member_list.erase(exist_trans.member_list.begin() + cand_index);
            --range;
          }                 
        }
      }
    }else { // back off to slow mode when P(k)*prior > threshold
      while(true) { // while loop
        optr_count++;
        unsigned int new_hidden = lm.hidden_vocab[(*int_vocab_distribution)(int_gen)];
        if(new_hidden == hidden) {
          return new_hidden;
        }
        float channel_prob = getChannelProb(new_hidden, observed);
        score = pow(10, lm.get_ngram_prob(pre_hidden, new_hidden) +
                        lm.get_ngram_prob(new_hidden, post_hidden)) * channel_prob;
        if(score >= threshold) {
          if(exist_trans.members.count(new_hidden) == 0) {
            exist_trans.members[new_hidden] = 1;
            exist_trans.member_list.push_back(new_hidden);
          }
          /*if(counts.count(concatNumbers(hidden, observed)) == 0) {
            exist_trans.members.erase(hidden);
          }*/
          return new_hidden;
        }
      } // while loop
    }
    return 0;
  }

  void runSampling(int iteration) {
    shuffle();
    for(int i = 0; i < iteration; i++) {
      corpus_prob = 0;
      optr_count = 0;
      token_count = 0;
      for(int k = 0; k < cipher_bigrams.size(); k++) { // for each cipher bigram
        Bigram& cipher = cipher_bigrams[k];
        long count = cipher.count;
        unsigned int pre_hidden = lm.pseu_start;
        for(int j = 0; j < 2; j++) { // for each token in a bigram
          unsigned int old_hidden = cipher.sample_tokens[j];
          unsigned int observed = cipher.tokens[j];
          long plain_cipher_pair = concatNumbers(old_hidden, observed);
          unsigned int new_hidden = 0;
          // reduce old counts
          counts[plain_cipher_pair] -= count; 
          /*if(counts[plain_cipher_pair] < 0) { // check that counts never become smaller than zero
            cout << "fatal error" << endl;
            exit(0);
          }*/
          if(counts[plain_cipher_pair] == 0) {
            counts.erase(plain_cipher_pair);
          }

          counts[(long)old_hidden] -= count;
          /*if(counts[(long)old_hidden] < 0) {
            cout << "fatal error" << endl;
            exit(0);
          }*/
          if(counts[(long)old_hidden] == 0) {
            counts.erase((long)old_hidden);
          }
           
          new_hidden = drawSample(j, cipher, old_hidden, observed);
          cipher.sample_tokens[j] = new_hidden; // update sample
          if(i % 500 == 0) {
            corpus_prob += lm.get_ngram_prob(pre_hidden, old_hidden); 
            corpus_prob += log10(getChannelProb(old_hidden, observed));
          }
          pre_hidden = new_hidden;
          // accumulate count
          plain_cipher_pair = concatNumbers(new_hidden, observed);
          if(iteration - i <= 1000) {
              if(i % 100 == 0) {
                  acc_counts[plain_cipher_pair] += count;  
              }
          }
          // update count
          //if(counts.count(plain_cipher_pair) ==0 ) {
          //  counts[plain_cipher_pair] = count;
          //}else {
          counts[plain_cipher_pair] += count;
          //}
          //if(counts.count((long)new_hidden) ==0 ) {
          //  counts[(long)new_hidden] = count;
          //}else {
          counts[(long)new_hidden] += count;
          //}          
        }
        if(i % 500 == 0) {
          corpus_prob += lm.get_ngram_prob(pre_hidden, lm.pseu_end);
        }
      } 
      // output corpus probability
      if(i % 500 == 0) {
        cout << i << " " << corpus_prob << " avg optr:" << optr_count/token_count << endl;
      }
    }
  }

  void printTTable(const char* file) {
    ofstream ofs(file);
    if(!ofs.is_open()) {
      cout << "can't open file for writing" << endl;
      exit(0);
    }
    long mask = INT_MAX >> 1; 
    for(boost::unordered_map<long,long>::iterator itr = counts.begin();
           itr != counts.end(); itr++) {
      long key = itr->first;
      long token1 = key & mask;
      long token0 = (key >> 30) & mask;
      if(token0 != 0) {
        float prob = (float)counts[key] / (float)counts[token0];
        float reverse_prob = (float)counts[key] / (float)o_counts[token1];
        ofs << token1 << " ||| " << token0 << " ||| " << prob << " " << reverse_prob << endl;
      }  
    }      
    ofs.close();
  }

  void printAccCounts(const char* file) {
    ofstream ofs(file);
    if(!ofs.is_open()) {
      cout << "can't open file for writing" << endl;
      exit(0);
    }
    long mask = INT_MAX >> 1; 
    for(boost::unordered_map<long,long>::iterator itr = acc_counts.begin();
           itr != acc_counts.end(); itr++) {
      long key = itr->first;
      long token1 = key & mask;
      long token0 = (key >> 30) & mask;
      if(token0 != 0) {
        ofs << token1 << " ||| " << token0 << " ||| " << acc_counts[key] << endl;
      }  
    }      
    ofs.close();
  }
  
}; // end of Decipherment class
