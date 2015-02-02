namespace nplm {

  void prepareData(vocab) {
    // Read in training data and validation data
    vector<vector<string> > train_data;
    readSentFile(train_text, train_data);
    for (int i=0; i<train_data.size(); i++) {
        // if data is already ngramized, set/check ngram_size
        if (!ngramize) {
            if (ngram_size > 0) {
                if (ngram_size != train_data[i].size()) {
                    cerr << "Error: size of training ngrams does not match specified value of --ngram_size!" << endl;
                }
            }
            // else if --ngram_size has not been specified, set it now
            else {
                ngram_size=train_data[i].size();
            }
        }
    }
    
    vector<vector<string> > validation_data;
    if (validation_text != "") {
        readSentFile(validation_text, validation_data);
        for (int i=0; i<validation_data.size(); i++) {
	    // if data is already ngramized, set/check ngram_size
            if (!ngramize) {
                // if --ngram_size has been specified, check that it does not conflict with --ngram_size
                if (ngram_size > 0) {
                    if (ngram_size != validation_data[i].size()) {
                        cerr << "Error: size of validation ngrams does not match specified value of --ngram_size!" << endl;
                    }
                }
                // else if --ngram_size has not been specified, set it now
                else {
                    ngram_size=validation_data[i].size();
                }
            }
        }
    }
    else if (validation_size > 0)
    {
        // Create validation data
        if (validation_size > train_data.size())
	{
	    cerr << "error: requested validation size is greater than training data size" << endl;
	    exit(1);
	}
	validation_data.insert(validation_data.end(), train_data.end()-validation_size, train_data.end());
	train_data.resize(train_data.size() - validation_size);
    }

    // Construct vocabulary
    vocabulary vocab;
    int start, stop;

    // read vocabulary from file
    if (words_file != "") {
        vector<string> words;
        readWordsFile(words_file,words);
        for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
            vocab.insert_word(*it);
        }

        // was vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
        if (vocab_size > 0) {
            if (vocab.size() != vocab_size) {
                cerr << "Error: size of vocabulary file " << vocab.size() << " != --vocab_size " << vocab_size << endl;
            }
        }
        // else, set it to the size of vocabulary read from file
        else {
            vocab_size = vocab.size();
        }

    }

    // construct vocabulary to contain top <vocab_size> most frequent words; all other words replaced by <unk>
    else {
        vocab.insert_word("<s>");
	vocab.insert_word("</s>");
	vocab.insert_word("<null>");

        // warn user that if --numberize is not set, there will be no vocabulary!
        if (!numberize) {
            cerr << "Warning: with --numberize 0 and --words_file == "", there will be no vocabulary!" << endl;
        }
        unordered_map<string,int> count;
        for (int i=0; i<train_data.size(); i++) {
            for (int j=0; j<train_data[i].size(); j++) {
                count[train_data[i][j]] += 1; 
            }
        }

        vocab.insert_most_frequent(count, vocab_size);
        if (vocab.size() < vocab_size) {
            cerr << "warning: fewer than " << vocab_size << " types in training data; the unknown word will not be learned" << endl;
        }
    }
  }
}
