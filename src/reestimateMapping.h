#include <algorithm>
#include <fstream>
#include <cmath>

//#include <boost/algorithm/string/join.hpp>
//#include <boost/thread/thread.hpp>
//#include <tclap/CmdLine.h>

#include <Eigen/Core>
#include <Eigen/Dense>

//#include "param.h"

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "util.h"

using namespace std;
using namespace Eigen;
using namespace boost;
//using namespace nplm;

//typedef boost::math::digamma boost::math::digamma;

// Functions that take non-const matrices as arguments
// are supposed to declare them const and then use this
// to cast away constness.
#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD, typename DerivedE>
void reestimateMapping(const MatrixBase<DerivedA> &base_distribution, 
const MatrixBase<DerivedB> &source_target_counts,
const MatrixBase<DerivedC> &source_counts,
const MatrixBase<DerivedC> &alphas,
const MatrixBase<DerivedD> &const_M,
const MatrixBase<DerivedE> &source_embeddings,
const MatrixBase<DerivedE> &target_embeddings,
double reg_lambda,
int epochs) {

	UNCONST(DerivedD,const_M,M);
	int num_source_words = source_embeddings.rows();
	int num_target_words = target_embeddings.rows();
	double learning_rate = 0.01;
	for (int epoch=0; epoch<epochs; epoch++){
		Matrix<double,Dynamic,Dynamic> M_gradient;
		M_gradient.setZero(const_M.rows(),const_M.cols());
		//cerr<<"digamma of 2 is "<<boost::math::digamma(2)<<endl;
		double objective_function_value = 0.;
		for (int source_index=0; source_index<num_source_words; source_index++){
			//GET THE EXP SUMS
			double exp_sum = 0.;
			Matrix<double,Dynamic,1> affinity_scores;
			affinity_scores.setZero(num_target_words);
			for (int target_index=0; target_index<num_target_words; target_index++) {
				affinity_scores(target_index) = exp((source_embeddings.row(source_index)*M).dot(target_embeddings.row(target_index)));
				exp_sum += affinity_scores(target_index);
			}
			double sum_digamma_diff = boost::math::digamma(exp_sum) - boost::math::digamma(exp_sum-source_counts(source_index));
			objective_function_value += boost::math::lgamma(exp_sum) - boost::math::digamma(exp_sum-source_counts(source_index));
			for (int target_index=0; target_index<num_target_words; target_index++){
				Matrix<double,Dynamic,Dynamic> outer_product = source_embeddings.row(source_index).transpose()*target_embeddings.row(target_index);
				//double weight = exp((source_embeddings(source_index)*M).dot(target_embeddings(target_index)));
				//current_gradient_term = sum_digamma_diff;
				double weight = sum_digamma_diff;
				if (source_target_counts(source_index,target_index) != 0.) {
					weight += boost::math::digamma(affinity_scores(target_index)+source_target_counts(source_index,target_index))-
						boost::math::digamma(affinity_scores(target_index));
					objective_function_value += boost::math::lgamma(affinity_scores(target_index)+source_target_counts(source_index,target_index))-
						boost::math::lgamma(affinity_scores(target_index));
				}
				weight *= affinity_scores(target_index);
				M_gradient += outer_product*weight;
			}
		}
		
		//Now to update the weights. 
		Matrix<double,Dynamic,Dynamic> reg_gradient = reg_lambda*(M.array().square()).matrix();
		M += learning_rate*(M_gradient - reg_lambda*(M.array().square()).matrix());
		objective_function_value -= reg_gradient.sum();
		
		cerr<<"Objective function value in epoch "<<epoch<<" was "<<objective_function_value<<endl;
		learning_rate = learning_rate*(epoch+1)/(epoch+2);
		cerr<<"Learning rate is "<<learning_rate<<endl;
	}
	
}
							
