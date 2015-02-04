#ifndef _TRAIN_NEURAL_NETWORK_H
#define _TRAIN_NEURAL_NETWORK_H

#include <cmath>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "param.h"
#include "propagator.h"
#include "model.h"
#include "SoftmaxLoss.h"


using namespace Eigen;
using namespace std;
using namespace boost;

namespace nplm {
typedef boost::unordered_map<string,unsigned> word_to_int_map;
typedef boost::unordered_map<unsigned,vector<unsigned> > int_to_vector_map;
typedef boost::unordered_map<unsigned,unsigned > int_to_int_map;

typedef long long int data_size_t; // training data can easily exceed 2G instances
typedef unordered_map<Matrix<int,Dynamic,1>, double> vector_map;
typedef boost::unordered_map<Matrix<int,Dynamic,1>, double> vector_map;
typedef long long int data_size_t; // training data can easily exceed 2G instances

typedef boost::unordered_map<unsigned, double> Word2Double;
typedef std::vector<Word2Double> Word2Word2Double;

class neuralNetworkTrainer {
  private:

    vector_map c_h, c_h_running_gradient;
    model nn;
    propagator prop,prop_validation,prop_test;
    loss_function_type loss_function;
    Matrix<int,Dynamic,Dynamic> test_data;
    //Matrix<int,Dynamic,Dynamic> scores;

  public :
  
    neuralNetworkTrainer(param &myParam,
        mt19937 &rng,
        string input_embeddings_file,
        string output_embeddings_file,
        string output_biases_file):
    c_h(vector_map()),
    c_h_running_gradient(vector_map()){ 
      cerr<<"Creating the network..."<<endl;
      /*
      this->nn = model(myParam.ngram_size,
          myParam.input_vocab_size,
          myParam.output_vocab_size,
          myParam.input_embedding_dimension,
          myParam.num_hidden,
          myParam.output_embedding_dimension,
          myParam.share_embeddings);
      */

    if (myParam.model_file != ""){
      nn.read(myParam.model_file);
      cerr<<"reading the model from "<<myParam.model_file<<endl;
    } else {
      cerr<<"Initializing the neural network from scratch"<<endl;
      nn.resize(myParam.ngram_size,
          myParam.input_vocab_size,
          myParam.output_vocab_size,
          myParam.input_embedding_dimension,
          myParam.num_hidden,
          myParam.output_embedding_dimension);

      nn.initialize(rng,
          myParam.init_normal,
          myParam.init_range,
          -log(myParam.output_vocab_size),
          myParam.parameter_update,
          myParam.adagrad_epsilon);
      nn.set_activation_function(string_to_activation_function(myParam.activation_function));

      //READ THE embeddings and biases if the files have been provided
      	
      if(input_embeddings_file != ""){
				cerr<<"Reading input embeddings file"<<endl;
				ifstream file(input_embeddings_file.c_str());
    		if (!file) throw runtime_error("Could not open file " + input_embeddings_file);
				nn.input_layer.read(file);
				file.close();
		}
		if(output_embeddings_file != ""){
			cerr<<"Reading output embeddings file"<<endl;
			ifstream file(output_embeddings_file.c_str());
			if (!file) throw runtime_error("Could not open file " + output_embeddings_file);
			nn.output_layer.read_weights(file);
			file.close();
		}
		
		if(output_biases_file != ""){
			cerr<<"Reading output biases file"<<endl;
			ifstream file(output_biases_file.c_str());
			if (!file) throw runtime_error("Could not open file " + output_biases_file);
			nn.output_layer.read_biases(file);
			file.close();
		}

    }
      /*
      nn.resize(myParam.ngram_size,
        myParam.input_vocab_size,
        myParam.output_vocab_size,
        myParam.input_embedding_dimension,
        myParam.num_hidden,
        myParam.output_embedding_dimension);
      cerr<<"Created the model"<<endl;
      this->nn.initialize(rng,
          myParam.init_normal,
          myParam.init_range,
          -log(myParam.output_vocab_size),
          myParam.parameter_update,
          myParam.adagrad_epsilon);
      //cerr<<"the nn is "<<&nn<<endl;
      //getchar();
      nn.set_activation_function(string_to_activation_function(myParam.activation_function));
      */

      loss_function = string_to_loss_function(myParam.loss_function);
      cerr<<"Initialized the model"<<endl;
      cerr<<"Mini-batch97 size is "<<myParam.minibatch_size<<endl;
      cerr<<"Validation minibatch size "<<myParam.validation_minibatch_size<<endl;
      this->prop = propagator(this->nn, myParam.minibatch_size);
      cerr<<"Created the propagator"<<endl;
      this->prop_validation = propagator(this->nn, myParam.validation_minibatch_size);
      cerr<<"Created the validation propagator"<<endl;
      this->prop_test = propagator(this->nn,myParam.test_minibatch_size);
      cerr<<"Created the test propagator"<<endl;
      //SoftmaxNCELoss<multinomial<data_size_t> > softmax_loss(unigram);
      // normalization parameters
      //vector_map c_h, c_h_running_gradient;

      loss_function = string_to_loss_function(myParam.loss_function);
      cerr<<"Finished creating the network..."<<endl;
    }
    
	/*
    void createTestData(int_to_vector_map& src_word_to_nn_input_tuple,
        param& myParam) {
      cerr<<"Preparing the contexts for populating the probability table"<<endl;
      int test_data_size = src_word_to_nn_input_tuple.size();
      this->test_data.setZero(myParam.ngram_size,test_data_size);
      int_to_vector_map::iterator itr;
      int counter = 0;
      for (itr=src_word_to_nn_input_tuple.begin();
          itr != src_word_to_nn_input_tuple.end();
          itr++){
        for (int i=0; i<myParam.ngram_size-1; i++){
          this->test_data(i,counter) = itr->second[i];
        }
        counter++;
      }
    }
	*/
	
	/*
    //COMPUTING THE PROBABILITIES FROM THE NEURAL NETWORK
    //AND THEN POPULATING THE T-table
    void getContextScores(
        int_to_vector_map& src_word_to_nn_input_tuple,
        param &myParam,
        Word2Word2Double& ttable,
        int_to_int_map& trg_word_to_nn_output_word,
        Dict &d){
      //UNCONST(DerivedA,const_scores,scores);
      //FIRST SET THE SCORES TO ZERO
      int test_data_size = test_data.cols();
      //scores.setZero(myParam.output_vocab_size,test_data_size);

      //scores.setZero();
      //cerr<<"The test data size is "<<test_data_size<<endl;
      int num_batches = (test_data_size-1)/myParam.test_minibatch_size + 1;
      //cerr<<"Number of test minibatches: "<<num_batches<<endl;
      //cerr<<"the test data is "<<test_data<<endl;
      //Matrix<double,Dynamic,Dynamic> scores(nn.output_vocab_size, myParam.test_minibatch_size);
      //Matrix<double,Dynamic,Dynamic> output_probs(nn.output_vocab_size, myParam.test_minibatch_size);
      int counter = 0;
      int_to_vector_map::iterator itr = src_word_to_nn_input_tuple.begin();
      for (int batch = 0; batch < num_batches; batch++)
      {
        //cerr<<"Batch number is "<<batch<<endl;
        int minibatch_start_index = myParam.test_minibatch_size * batch;
        int current_minibatch_size = min(myParam.test_minibatch_size,
                 test_data_size - minibatch_start_index);

        Matrix<int,Dynamic,Dynamic> minibatch = test_data.middleCols(minibatch_start_index, current_minibatch_size);
        prop_test.fProp(minibatch.topRows(myParam.ngram_size-1));
        //NOW GOING OVER ALL THE TARGET WORDS FOR THE 
        //PARTICULAR SOURCE WORD AND COMPUTING THE 
        //SCORES 
        for (int batch_index=0; 
            batch_index<current_minibatch_size; 
            batch_index++){
          int src_word = itr->first;
          Word2Double& cpd = ttable[src_word];
          //cerr<<"The source word is "<<d.Convert(src_word)<<endl;
          double tot;
          for (Word2Double::iterator cpd_it = cpd.begin(); cpd_it != cpd.end(); ++cpd_it){
              //std::cerr<<"The conditioned word is "<<d.Convert(cpd_it->first)<<std::endl;
              //word_pairs.push_back(make_pair(src_word_to_int[d.Convert(i)],trg_word_to_int[d.Convert(it->first)]));
              int output_word_index = trg_word_to_nn_output_word[cpd_it->first];
              //cerr<<"The index in col was "<<index_in_col<<endl;
              //GETTING THE SCORE FROM THE NN
							#ifdef TRIPLE
              double score =  prop_test.output_layer_node.param->fProp(
                  prop_test.third_hidden_activation_node.fProp_matrix,
                  output_word_index,
                  batch_index);
							#endif

              #ifdef SINGLE
              //cerr<<"SINGLE score computation"<<endl;
              double score =  prop_test.output_layer_node.param->fProp(
                  prop_test.first_hidden_activation_node.fProp_matrix,
                  output_word_index,
                  batch_index);

              #endif
							#ifdef DOUBLE
              double score =  prop_test.output_layer_node.param->fProp(
                  prop_test.second_hidden_activation_node.fProp_matrix,
                  output_word_index,
                  batch_index);
              #endif
              //cerr<<"the score is "<<cpd_it->second<<endl;
							cpd_it->second = exp(score);
              tot += cpd_it->second;
          }
          itr++;
          //if (itr > src_word_to_nn_input_tuple.end()){
          //  cerr<<"We went out of bounds on the src word to input tuple"<<endl;
          //}
        }
        //cerr<<"the size of scores is "<<scores.rows()<<" "<<scores.cols()<<endl;
        //cerr<<"I fpropped till the last layer "<<endl;
        //cerr<<"The fprop matrix dimensions are "<<prop_test.second_hidden_activation_node.fProp_matrix.rows()<<" "
        //  <<prop_test.second_hidden_activation_node.fProp_matrix.cols()<<endl;
        //cerr<<"The scores matrix dimensions are "<<scores.rows()<<" "<<scores.cols()<<endl;
        //cerr<<"The scores matrix dimensions are "<<scores.middleCols(minibatch_start_index, current_minibatch_size).rows()<<" "
        //  <<scores.middleCols(minibatch_start_index, current_minibatch_size).cols()<<endl;
        //cerr<<"fprop matrix dimensions are "<<prop_test.second_hidden_activation_node.fProp_matrix.leftCols(current_minibatch_size).rows()<<" "
        //  <<prop_test.second_hidden_activation_node.fProp_matrix.leftCols(current_minibatch_size).cols()<<endl;
        // Do full forward prop through output word embedding layer
        
		
		
        //prop_test.output_layer_node.param->fProp(
        //    prop_test.second_hidden_activation_node.fProp_matrix.leftCols(current_minibatch_size), 
        //    scores.middleCols(minibatch_start_index, current_minibatch_size));
		
      }	

    }
	*/
	
	//To compute the base disctribution, all we have to do is create a matrix where the number of rows is equal to the input vocab size
	//The first column will contain the plaintext id's from 0 to input vocab size (which is also the plaintext vocab size). The second column
	//is just 0 because it doesn't matter what the target word is. We are interested int the distribution from the softmax. 
	template <typename DerivedA>
    void getBaseDistribution(param &myParam,
		const MatrixBase<DerivedA> &const_base_distribution){			
          double log_likelihood = 0.0;
		  UNCONST(DerivedA, const_base_distribution, base_distribution);
		  Matrix<int, Dynamic,Dynamic> plain_word_and_dummy_cipher_word;
		  plain_word_and_dummy_cipher_word.setZero(myParam.ngram_size,myParam.input_vocab_size);
		  for (int plain_id=0; plain_id<myParam.input_vocab_size; plain_id++){
		  	plain_word_and_dummy_cipher_word(0,plain_id) =plain_id;
			plain_word_and_dummy_cipher_word(1,plain_id) =0;
		  }
		  int validation_minibatch_size = myParam.validation_minibatch_size;
		  //myParam.validation_minibatch_size = validation_minibatch_size;
		  data_size_t validation_data_size = myParam.input_vocab_size;
          Matrix<double,Dynamic,Dynamic> scores(myParam.output_vocab_size, validation_minibatch_size);
          //Matrix<double,Dynamic,Dynamic> output_probs(output_vocab_size, validation_minibatch_size);
          Matrix<int,Dynamic,Dynamic> minibatch(myParam.ngram_size, validation_minibatch_size);
		  int num_validation_batches = (myParam.input_vocab_size-1)/myParam.validation_minibatch_size+1;
		  
          for (int validation_batch =0;validation_batch < num_validation_batches;validation_batch++)
          {
                int validation_minibatch_start_index = validation_minibatch_size * validation_batch;
		        int current_minibatch_size = std::min(static_cast<data_size_t>(validation_minibatch_size),
		                 validation_data_size - validation_minibatch_start_index);
		        minibatch.leftCols(current_minibatch_size) = plain_word_and_dummy_cipher_word.middleCols(validation_minibatch_start_index, 
		                          current_minibatch_size);
		        prop_validation.fProp(minibatch.topRows(myParam.ngram_size-1));

		        // Do full forward prop through output word embedding layer
		        start_timer(4);
		        #ifdef SINGLE
		        prop_validation.output_layer_node.param->fProp(prop_validation.first_hidden_activation_node.fProp_matrix, scores);
		        #endif
				#ifdef DOUBLE
		        prop_validation.output_layer_node.param->fProp(prop_validation.second_hidden_activation_node.fProp_matrix, scores);
		        #endif
				#ifdef TRIPLE
		        prop_validation.output_layer_node.param->fProp(prop_validation.third_hidden_activation_node.fProp_matrix, scores);
				#endif

		        stop_timer(4);

		        // And softmax and loss. Be careful of short minibatch
		        double minibatch_log_likelihood;
		        start_timer(5);
		        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
		                   minibatch.row(myParam.ngram_size-1),
						   base_distribution.middleCols(validation_minibatch_start_index, 
						   	                          current_minibatch_size),
		                   minibatch_log_likelihood);
		        stop_timer(5);
				//We don't need the log likelihood computation part
				/*
		        log_likelihood += minibatch_log_likelihood;
		          }

		                cerr << "Validation log-likelihood: "<< log_likelihood << endl;
		                cerr << "           perplexity:     "<< exp(-log_likelihood/validation_data_size) << endl;

		          // If the validation perplexity decreases, halve the learning rate.
		          if (epoch > 0 && log_likelihood < current_validation_ll && myParam.parameter_update != "ADA")
		          { 
		              current_learning_rate /= 2;
		          }
		          current_validation_ll = log_likelihood;			
	        	*/
			}
		}
	
    template <typename DerivedA>
    void trainNN(param &myParam,
        const MatrixBase<DerivedA> &training_data,
        const MatrixBase<DerivedA> &validation_data,
        mt19937 &rng,
        int outer_iteration) {
        //cerr<<"The nn is is "<<&nn<<endl;
        //cerr<<"The learning rate is "<<myParam.learning_rate<<endl;
        //getchar();
        cerr<<"Starting to train the neural network"<<endl;
        data_size_t training_data_size = training_data.cols();
        data_size_t validation_data_size = validation_data.cols();
        ///// Construct unigram model and sampler that will be used for NCE
        //cerr<<"The training data is "<<training_data<<endl;
        //cerr<<"The validation data is "<<validation_data<<endl;
        //getchar();
        cerr<<"Creating unigram counts"<<endl;
        vector<data_size_t> unigram_counts(myParam.output_vocab_size);
        for (data_size_t train_id=0; train_id < training_data_size; train_id++)
        {
          int output_word = training_data(myParam.ngram_size-1, train_id);
          unigram_counts[output_word] += 1;
        }
        multinomial<data_size_t> unigram (unigram_counts);
        cerr<<"Created unigram counts"<<endl;
        ////CREATING THE OUTPUT SOFTMAX LAYER
        SoftmaxNCELoss<multinomial<data_size_t> > softmax_loss(unigram);
        ///////////////////////TRAINING THE NEURAL NETWORK////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////

        data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;
        cerr<<"Number of training minibatches: "<<num_batches<<endl;

        int num_validation_batches = 0;
        if (validation_data_size > 0)
        {
          num_validation_batches = (validation_data_size-1)/myParam.validation_minibatch_size+1;
          cerr<<"Number of validation minibatches: "<<num_validation_batches<<endl;
        } 

        double current_momentum = myParam.initial_momentum;
        double momentum_delta = (myParam.final_momentum - myParam.initial_momentum)/(myParam.num_epochs-1);
        double current_learning_rate = myParam.learning_rate;
        double current_validation_ll = 0.0;

        int ngram_size = myParam.ngram_size;
        int input_vocab_size = myParam.input_vocab_size;
        int output_vocab_size = myParam.output_vocab_size;
        int minibatch_size = myParam.minibatch_size;
        int validation_minibatch_size = myParam.validation_minibatch_size;
        int num_noise_samples = myParam.num_noise_samples;

        if (myParam.normalization)
        {
          for (data_size_t i=0;i<training_data_size;i++)
          {
              Matrix<int,Dynamic,1> context = training_data.block(0,i,ngram_size-1,1);
              if (c_h.find(context) == c_h.end())
              {
                  c_h[context] = -myParam.normalization_init;
              }
          }
        }

        for (int epoch=0; epoch<myParam.num_epochs; epoch++)
        { 
            cerr << "Epoch " << epoch+1 << endl;
            cerr << "Current learning rate: " << current_learning_rate << endl;

            if (myParam.use_momentum) {
              cerr << "Current momentum: " << current_momentum << endl;
            } else {
              current_momentum = -1;
            }

        cerr << "Training minibatches: ";

        double log_likelihood = 0.0;

        int num_samples = 0;
        if (loss_function == LogLoss)
            num_samples = output_vocab_size;
        else if (loss_function == NCELoss)
            num_samples = 1+num_noise_samples;

        Matrix<double,Dynamic,Dynamic> minibatch_weights(num_samples, minibatch_size);
        Matrix<int,Dynamic,Dynamic> minibatch_samples(num_samples, minibatch_size);
        Matrix<double,Dynamic,Dynamic> scores(num_samples, minibatch_size);
        Matrix<double,Dynamic,Dynamic> probs(num_samples, minibatch_size);

            for(data_size_t batch=0;batch<num_batches;batch++)
            {
                if (batch > 0 && batch % 10000 == 0)
                {
              cerr << batch <<"...";
                } 

                data_size_t minibatch_start_index = minibatch_size * batch;
                int current_minibatch_size = min(static_cast<data_size_t>(minibatch_size), training_data_size - minibatch_start_index);
          Matrix<int,Dynamic,Dynamic> minibatch = training_data.middleCols(minibatch_start_index, current_minibatch_size);

                double adjusted_learning_rate = current_learning_rate/current_minibatch_size;
                //cerr<<"Adjusted learning rate: "<<adjusted_learning_rate<<endl;

                ///// Forward propagation

                prop.fProp(minibatch.topRows(ngram_size-1));

          if (loss_function == NCELoss)
          {
              ///// Noise-contrastive estimation

              // Generate noise samples. Gather positive and negative samples into matrix.

              start_timer(3);

        minibatch_samples.block(0, 0, 1, current_minibatch_size) = minibatch.bottomRows(1);
        
        for (int sample_id = 1; sample_id < num_noise_samples+1; sample_id++)
            for (int train_id = 0; train_id < current_minibatch_size; train_id++)
                minibatch_samples(sample_id, train_id) = unigram.sample(rng);
          
        stop_timer(3);

        // Final forward propagation step (sparse)
        start_timer(4);
        #ifdef SINGLE
        prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix,
                    minibatch_samples, scores);
        #endif
				#ifdef DOUBLE
        prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix,
                    minibatch_samples, scores);
        #endif 
				#ifdef TRIPLE
        prop.output_layer_node.param->fProp(prop.third_hidden_activation_node.fProp_matrix,
                    minibatch_samples, scores);
        #endif 

        stop_timer(4);

        // Apply normalization parameters
        if (myParam.normalization)
        {
            for (int train_id = 0;train_id < current_minibatch_size;train_id++)
            {
          Matrix<int,Dynamic,1> context = minibatch.block(0, train_id, ngram_size-1, 1);
          scores.col(train_id).array() += c_h[context];
            }
        }

        double minibatch_log_likelihood;
        start_timer(5);
        softmax_loss.fProp(scores.leftCols(current_minibatch_size), 
               minibatch_samples,
               probs, minibatch_log_likelihood);
        stop_timer(5);
        log_likelihood += minibatch_log_likelihood;

        ///// Backward propagation

        start_timer(6);
        softmax_loss.bProp(probs, minibatch_weights);
        stop_timer(6);
        
        // Update the normalization parameters
        
        if (myParam.normalization)
        {
            for (int train_id = 0;train_id < current_minibatch_size;train_id++)
            {
          Matrix<int,Dynamic,1> context = minibatch.block(0, train_id, ngram_size-1, 1);
          c_h[context] += adjusted_learning_rate * minibatch_weights.col(train_id).sum();
            }
        }

        // Be careful of short minibatch
        prop.bProp(minibatch.topRows(ngram_size-1),
             minibatch_samples.leftCols(current_minibatch_size), 
             minibatch_weights.leftCols(current_minibatch_size),
             adjusted_learning_rate, 
             current_momentum,
             myParam.L2_reg,
             myParam.parameter_update,
             myParam.conditioning_constant,
             myParam.decay);
          }
          else if (loss_function == LogLoss)
          {
              ///// Standard log-likelihood
              start_timer(4);
        #ifdef SINGLE
        prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix, scores);
				#endif
        #ifdef DOUBLE
        prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);
        #endif 
				#ifdef TRIPLE
        prop.output_layer_node.param->fProp(prop.third_hidden_activation_node.fProp_matrix, scores);
				#endif
        stop_timer(4);
    

        double minibatch_log_likelihood;
        start_timer(5);
        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
                   minibatch.row(ngram_size-1), 
                   probs, 
                   minibatch_log_likelihood);
        stop_timer(5);
        log_likelihood += minibatch_log_likelihood;

        ///// Backward propagation
        
        start_timer(6);
        SoftmaxLogLoss().bProp(minibatch.row(ngram_size-1).leftCols(current_minibatch_size), 
                   probs.leftCols(current_minibatch_size), 
                   minibatch_weights);
        stop_timer(6);
        
        prop.bProp(minibatch.topRows(ngram_size-1).leftCols(current_minibatch_size),
             minibatch_weights,
             adjusted_learning_rate,
             current_momentum,
             myParam.L2_reg,
             myParam.parameter_update,
             myParam.conditioning_constant,
             myParam.decay);
          }
            }
      cerr << "done." << endl;

      if (loss_function == LogLoss)
      {
          cerr << "Training log-likelihood: " << log_likelihood << endl;
                cerr << "         perplexity:     "<< exp(-log_likelihood/training_data_size) << endl;
      }
      else if (loss_function == NCELoss)
          cerr << "Training NCE log-likelihood: " << log_likelihood << endl;

            current_momentum += momentum_delta;

      #ifdef USE_CHRONO
      cerr << "Propagation times:";
      for (int i=0; i<timer.size(); i++)
        cerr << " " << timer.get(i);
      cerr << endl;
      #endif
      
      if (myParam.model_prefix != "")
      {
          cerr << "Writing model" << endl;
          //if (myParam.input_words_file != "")
          //    nn.write(myParam.model_prefix + "." + lexical_cast<string>(epoch+1), input_words, output_words);
          //else
          //nn.write(myParam.model_prefix + "." + lexical_cast<string>(epoch+1)+"."+lexical_cast<string>(outer_iteration));
      }
      //WE WANT TO GET THE MAPPING MATRIX AND THE BIASES. 
	  /*
        if (epoch % 1 == 0 && validation_data_size > 0)
        {
            //////COMPUTING VALIDATION SET PERPLEXITY///////////////////////
            ////////////////////////////////////////////////////////////////

            double log_likelihood = 0.0;

          Matrix<double,Dynamic,Dynamic> scores(output_vocab_size, validation_minibatch_size);
          Matrix<double,Dynamic,Dynamic> output_probs(output_vocab_size, validation_minibatch_size);
          Matrix<int,Dynamic,Dynamic> minibatch(ngram_size, validation_minibatch_size);

	        for (int validation_batch =0;validation_batch < num_validation_batches;validation_batch++)
	        {
                    int validation_minibatch_start_index = validation_minibatch_size * validation_batch;
		        int current_minibatch_size = std::min(static_cast<data_size_t>(validation_minibatch_size),
		                 validation_data_size - validation_minibatch_start_index);
		        minibatch.leftCols(current_minibatch_size) = validation_data.middleCols(validation_minibatch_start_index, 
		                          current_minibatch_size);
		        prop_validation.fProp(minibatch.topRows(ngram_size-1));

		        // Do full forward prop through output word embedding layer
		        start_timer(4);
		        #ifdef SINGLE
		        prop_validation.output_layer_node.param->fProp(prop_validation.first_hidden_activation_node.fProp_matrix, scores);
		        #endif
						#ifdef DOUBLE
		        prop_validation.output_layer_node.param->fProp(prop_validation.second_hidden_activation_node.fProp_matrix, scores);
		        #endif
						#ifdef TRIPLE
		        prop_validation.output_layer_node.param->fProp(prop_validation.third_hidden_activation_node.fProp_matrix, scores);
						#endif

		        stop_timer(4);

		        // And softmax and loss. Be careful of short minibatch
		        double minibatch_log_likelihood;
		        start_timer(5);
		        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
		                   minibatch.row(ngram_size-1),
		                   output_probs,
		                   minibatch_log_likelihood);
		        stop_timer(5);
		        log_likelihood += minibatch_log_likelihood;
          }

          cerr << "Validation log-likelihood: "<< log_likelihood << endl;
          cerr << "           perplexity:     "<< exp(-log_likelihood/validation_data_size) << endl;

          // If the validation perplexity decreases, halve the learning rate.
          if (epoch > 0 && log_likelihood < current_validation_ll && myParam.parameter_update != "ADA")
          { 
              current_learning_rate /= 2;
          }
          current_validation_ll = log_likelihood;
        }
	  */
      }
    }
	
	//template <typename DerivedA>

	void set_input_output_embeddings( Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> &input_embedding_matrix,
					 Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> &output_embedding_matrix){
		nn.set_input_output_embeddings(input_embedding_matrix,output_embedding_matrix);
	}

};

} //namespace nplm

#endif
