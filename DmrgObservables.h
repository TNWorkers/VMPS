#include <memory>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "MpsQ.h"
#include "MpoQ.h"
#include "MpKondoModel.h"
#include "DmrgLinearAlgebraQ.h"

/**Calculating observables for given sets of lattice sites. 
\describe_Nq
\describe_Scalar*/
template <size_t Nq, typename Scalar, typename Hamiltonian> class Observables
{
public:

	Observables<Nq,Scalar,Hamiltonian > (MpsQ<Nq,Scalar> * stateInput, Hamiltonian * Hinput);
	
	Eigen::MatrixXd ImpSubCorr (vector<size_t> impLocs, vector<size_t> subLocs);
	Eigen::MatrixXd ImpCorr (vector<size_t> imp1Locs, vector<size_t> imp2Locs);
	Eigen::MatrixXd SubCorr (vector<size_t> sub1Locs, vector<size_t> sub2Locs);
	Eigen::MatrixXd SizVal (vector<size_t> locs);
	Eigen::MatrixXd SixVal (vector<size_t> locs);
	Eigen::MatrixXd DoubleOcc (vector<size_t> locs);
	Eigen::MatrixXd localPksCorr (vector<size_t> locs1, vector<size_t> locs2);
	Eigen::MatrixXd localPksCorrMF (vector<size_t> locs1, vector<size_t> locs2, Eigen::MatrixXd * ImpSubCorr);
	Eigen::Tensor<double,3> PksCorr ();
	Eigen::Tensor<double,3> PksCorrMF (Eigen::MatrixXd * ImpSubCorr, Eigen::MatrixXd * ImpCorr);
	Eigen::MatrixXd Hopping (vector<size_t> sub1Locs, vector<size_t> sub2Locs, SPIN_INDEX sigma);

private:
	string info;
	MpsQ<Nq,Scalar> * state;
	Hamiltonian * H;   
};

template <size_t Nq, typename Scalar, typename Hamiltonian> Observables<Nq,Scalar,Hamiltonian> ::
Observables (MpsQ<Nq,Scalar> * stateInput, Hamiltonian * Hinput)
{
	state = stateInput;
	H = Hinput;
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
ImpSubCorr (vector<size_t> impLocs, vector<size_t> subLocs)
{
	Eigen::Matrix<Scalar,Dynamic,Dynamic> ImpSubCorr(impLocs.size(),impLocs.size());
	ImpSubCorr.setZero();
	
	for (size_t i=0; i<impLocs.size(); ++i)
	{
		for (size_t j=0; j<subLocs.size(); ++j)
		{
			ImpSubCorr(i,j) = avg(*state, H->SimpSsub(impLocs[i],SZ,subLocs[j],SZ), *state) +
				0.5*(avg(*state , H->SimpSsub(impLocs[i],SP,subLocs[j],SM), *state)
					 + avg(*state , H->SimpSsub(impLocs[i],SM,subLocs[j],SP), *state));
		}
	}
	return ImpSubCorr.real();
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
ImpCorr (vector<size_t> imp1Locs, vector<size_t> imp2Locs)
{
	Eigen::Matrix<Scalar,Dynamic,Dynamic> ImpCorr(imp1Locs.size(),imp2Locs.size()); ImpCorr.setZero();

	for (size_t i=0; i<imp1Locs.size(); ++i)
	{
		for (size_t j=0; j<imp2Locs.size(); ++j)
		{
			ImpCorr(i,j) = avg(*state, H->SimpSimp(imp1Locs[i],SZ,imp2Locs[j],SZ), *state) +
				0.5*(avg(*state , H->SimpSimp(imp1Locs[i],SP,imp2Locs[j],SM), *state)
					 + avg(*state , H->SimpSimp(imp1Locs[i],SM,imp2Locs[j],SP), *state));
		}
	}
	return ImpCorr.real();
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
SubCorr (vector<size_t> sub1Locs, vector<size_t> sub2Locs)
{
        Eigen::Matrix<Scalar,Dynamic,Dynamic> SubCorr(sub1Locs.size(),sub2Locs.size());
        SubCorr.setZero();

        for (size_t i=0; i<sub1Locs.size(); ++i)
        {
                for (size_t j=0; j<sub2Locs.size(); ++j)
                {
                        SubCorr(i,j) = avg(*state, H->SsubSsub(sub1Locs[i],SZ,sub2Locs[j],SZ), *state) +
                                0.5*(avg(*state , H->SsubSsub(sub1Locs[i],SP,sub2Locs[j],SM), *state)
                                         + avg(*state , H->SsubSsub(sub1Locs[i],SM,sub2Locs[j],SP), *state));
                }
        }
        return SubCorr.real();
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
SizVal (vector<size_t> locs)
{
	Eigen::Matrix<Scalar,Dynamic,1> SizVal(locs.size(),1);
	SizVal.setZero();
	for (size_t i=0; i<locs.size(); i++)
	{
		SizVal(i) = avg(*state , H->Simp(locs[i],SZ) , *state);
	}

	return SizVal.real();
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
SixVal (vector<size_t> locs)
{
	Eigen::Matrix<Scalar,Dynamic,1> SixVal(locs.size(),1);
	SixVal.setZero();
	for (size_t i=0; i<locs.size(); i++)
	{
		SixVal(i) = avg(*state , H->Simp(locs[i],SX) , *state);
	}

	return SixVal.real();
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
DoubleOcc (vector<size_t> locs)
{
	Eigen::Matrix<Scalar,Dynamic,1> DoubleOcc(locs.size(),1);
	DoubleOcc.setZero();
	for (size_t i=0; i<locs.size(); i++)
	{
		DoubleOcc(i) = avg(*state , H->d(locs[i],0) , *state);
	}

	return DoubleOcc.real();
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
Hopping (vector<size_t> subLocs1, vector<size_t> subLocs2, SPIN_INDEX sigma)
{
	Eigen::Matrix<Scalar,Dynamic,Dynamic> Hopping(subLocs1.size(),subLocs2.size());
	Hopping.setZero();
	
	for (size_t i=0; i<subLocs1.size(); ++i)
	{
		for (size_t j=0; j<subLocs2.size(); ++j)
		{
			Hopping(i,j) = avg(*state, H->cdagc(sigma,i,j), *state);
		}
	}
	return Hopping.real();
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
localPksCorr (vector<size_t> locs1, vector<size_t> locs2)
{
	Eigen::MatrixXd localPksCorr(locs1.size(),locs2.size());
	localPksCorr.setZero();
	
	for (size_t i=0; i<locs1.size(); i++)
	{
		for (size_t j=i+1; j<locs2.size(); j++)
		{
			localPksCorr(i,j) = avg(*state, H->SimpSsubSimpSsub(locs1[i],SZ,locs1[i],SZ,locs2[j],SZ,locs2[j],SZ), *state)
				+ 0.5*avg(*state, H->SimpSsubSimpSsub(locs1[i],SZ,locs1[i],SZ,locs2[j],SP,locs2[j],SM), *state)
				+ 0.5*avg(*state, H->SimpSsubSimpSsub(locs1[i],SZ,locs1[i],SZ,locs2[j],SM,locs2[j],SP), *state)
				+ 0.5*avg(*state, H->SimpSsubSimpSsub(locs1[i],SP,locs1[i],SM,locs2[j],SZ,locs2[j],SZ), *state)
				+ 0.25*avg(*state, H->SimpSsubSimpSsub(locs1[i],SP,locs1[i],SM,locs2[j],SP,locs2[j],SM), *state)
				+ 0.25*avg(*state, H->SimpSsubSimpSsub(locs1[i],SP,locs1[i],SM,locs2[j],SM,locs2[j],SP), *state)
				+ 0.5*avg(*state, H->SimpSsubSimpSsub(locs1[i],SM,locs1[i],SP,locs2[j],SZ,locs2[j],SZ), *state)
				+ 0.25*avg(*state, H->SimpSsubSimpSsub(locs1[i],SM,locs1[i],SP,locs2[j],SP,locs2[j],SM), *state)
				+ 0.25*avg(*state, H->SimpSsubSimpSsub(locs1[i],SM,locs1[i],SP,locs2[j],SM,locs2[j],SP), *state);
		}
	}
	return localPksCorr;
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::MatrixXd Observables<Nq,Scalar,Hamiltonian>::
localPksCorrMF (vector<size_t> locs1, vector<size_t> locs2, Eigen::MatrixXd * ImpSubCorr)
{
	Eigen::MatrixXd localPksCorrMF(locs1.size(),locs2.size());
	localPksCorrMF.setZero();
	
	for (size_t i=0; i<locs1.size(); i++)
	{
		for (size_t j=i+1; j<locs2.size(); j++)
		{
			localPksCorrMF(i,j) = (*ImpSubCorr)(i,i) * (*ImpSubCorr)(j,j);
		}
	}
	return localPksCorrMF;	
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::Tensor<double,3> Observables<Nq,Scalar,Hamiltonian>::
PksCorr ()
{
	size_t L = H->length();
	Eigen::Tensor<double,3> PksCorr(L,L,L);
	PksCorr.setZero();

	size_t triangle[] = {0,1,2};
	for (size_t i=0; i<(H->length() -2); i++)
	{
		size_t j=i+1;
		size_t k=j+1;
		triangle[0] = i;
		triangle[1] = j;
		triangle[2] = k;
		do {
			PksCorr(triangle[0],triangle[1],triangle[2]) = avg(*state, H->SimpSsubSimpSimp(triangle[0],SZ,triangle[0],SZ,triangle[1],SZ,triangle[2],SZ), *state)
			+ 0.5*avg(*state, H->SimpSsubSimpSimp(triangle[0],SZ,triangle[0],SZ,triangle[1],SP,triangle[2],SM), *state)
			+ 0.5*avg(*state, H->SimpSsubSimpSimp(triangle[0],SZ,triangle[0],SZ,triangle[1],SM,triangle[2],SP), *state)
			+ 0.5*avg(*state, H->SimpSsubSimpSimp(triangle[0],SP,triangle[0],SM,triangle[1],SZ,triangle[2],SZ), *state)
			+ 0.25*avg(*state, H->SimpSsubSimpSimp(triangle[0],SP,triangle[0],SM,triangle[1],SP,triangle[2],SM), *state)
			+ 0.25*avg(*state, H->SimpSsubSimpSimp(triangle[0],SP,triangle[0],SM,triangle[1],SM,triangle[2],SP), *state)
			+ 0.5*avg(*state, H->SimpSsubSimpSimp(triangle[0],SM,triangle[0],SP,triangle[1],SZ,triangle[2],SZ), *state)
			+ 0.25*avg(*state, H->SimpSsubSimpSimp(triangle[0],SM,triangle[0],SP,triangle[1],SP,triangle[2],SM), *state)
			+ 0.25*avg(*state, H->SimpSsubSimpSimp(triangle[0],SM,triangle[0],SP,triangle[1],SM,triangle[2],SP), *state);
		} while ( std::next_permutation(triangle,triangle+3) );
	}
	return PksCorr;
}

template <size_t Nq, typename Scalar, typename Hamiltonian>
Eigen::Tensor<double,3> Observables<Nq,Scalar,Hamiltonian>::
PksCorrMF (Eigen::MatrixXd * ImpSubCorr, Eigen::MatrixXd * ImpCorr)
{
	size_t L = H->length();	
	Eigen::Tensor<double,3> PksCorrMF(L,L,L);
	PksCorrMF.setZero();
	
	size_t triangle[] = {0,1,2};
	for (size_t i=0; i<(H->length() -2); i++)
	{
		size_t j=i+1;
		size_t k=j+1;
		triangle[0] = i;
		triangle[1] = j;
		triangle[2] = k;
		do {
			PksCorrMF(triangle[0],triangle[1],triangle[2]) = (*ImpSubCorr)(triangle[0],triangle[0])*(*ImpCorr)(triangle[1],triangle[2]);
		} while ( std::next_permutation(triangle,triangle+3) );
	}
	return PksCorrMF;
}

