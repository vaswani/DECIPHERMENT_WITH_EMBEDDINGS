#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

template<typename Derived>
readWeights(string &embeddings_filename, Eigen::MatrixBase<Derived> &const_embeddings_matrix){
    ifstream embeddings_file(embeddings_filename.c_str());
    if (!embeddings_file) throw runtime_error("Could not open file " + embeddings_filename);
	readMatrix(embeddings_file,const_embeddings_matrix);	
}

int main (int argc, char *argv[]) 
{
	Matrix<double,Dynamic,Dynamic> source_embeddings,target_embeddings;
	source_embeddings.setZero(5000,25);
	target_embeddings.setZero(5000,25);
	source_embeddings_filename = "";
	
	
}