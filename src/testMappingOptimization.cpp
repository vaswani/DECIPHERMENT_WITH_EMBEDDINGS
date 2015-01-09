#include <Eigen/Dense>
#include <Eigen/Core>

#include "util.h"
#include "reestimateMapping.h"

using namespace Eigen;
using namespace std;
//using namespace nplm;
using namespace boost::random;

template<typename Derived>
void readWeights(string &embeddings_filename, Eigen::MatrixBase<Derived> &const_embeddings_matrix){
    ifstream embeddings_file(embeddings_filename.c_str());
    if (!embeddings_file) throw runtime_error("Could not open file " + embeddings_filename);
	readMatrix(embeddings_file,const_embeddings_matrix);	
}

int main (int argc, char *argv[]) 
{
	Matrix<double,Dynamic,Dynamic> source_embeddings,target_embeddings;
	source_embeddings.setZero(5001,25);
	target_embeddings.setZero(5001,25);
	string source_embeddings_filename="vectors.s25.es.ordered";
	string target_embeddings_filename="vectors.s25.en.ordered";
	//Reading source and target embeddings
	readWeights(source_embeddings_filename,source_embeddings);
	cerr<<"Read source embeddings "<<endl;
	readWeights(target_embeddings_filename,target_embeddings);
	cerr<<"Read target embeddings "<<endl;
	source_embeddings /= 10;
	target_embeddings /= 10;
	//Creating and reading the counts matrix
	Matrix<double,Dynamic,Dynamic> counts_matrix,base_distribution;
	counts_matrix.setZero(5001,5001);
	base_distribution.setZero(5001,5001);
	string counts_file = "ptable.es.1m-0.counts.final.ordered.array.es-en";
	readWeights(counts_file,counts_matrix);
	cerr<<"Read counts "<<endl;
	//Getting the source counts
	Matrix<double,Dynamic,1> source_counts = counts_matrix.rowwise().sum();
	//initialize the mapping Matrix
	Matrix<double,Dynamic,Dynamic> M;
	M.setZero(25,25);
    //unsigned seed = std::time(0);
    unsigned seed = 1234; //for testing only
    mt19937 rng(seed);
	initMatrix(rng, M, 1, 0.01);
	Matrix<double,Dynamic,1> alphas;
	alphas.setZero(5001);
	
	reestimateMapping(base_distribution, 
	counts_matrix,
	source_counts,
	alphas,
	M,
	source_embeddings,
	target_embeddings,
	1.,
	5);
}
