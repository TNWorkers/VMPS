#ifndef SUN_H_
#define SUN_H_

#include <array>
#include <cstddef>

#include <unsupported/Eigen/CXX11/Tensor>

#include "ClebschGordan.h" //Library by Arne Alex for SU(N)-CGCs. See: https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/
                           //Seems to be not precise enough, maybe there is a better one.

#include <gsl/gsl_sf_coupling.h> //temporary necessary because class SiteOperator ist implemented properly for Symmetries with CGCs..

#include <boost/rational.hpp>

namespace Sym{

/** \class SUN
 * \ingroup Symmetry
 *
 * Class for handling a SU(N) symmetry of a Hamiltonian with explicitly storage the Clebsch-Gordon coefficients.
 *
 * \describe_Scalar
 * \tparam N : the \p N in SU(N) 
 * \warning Use the the external CGC-Library: https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/
 * \todo1 To use general SU(N) symmetries, several adjustions are necessary in the code, concerning the innermultiplicity.
 */
template<std::size_t N, typename Scalar>
class SUN // : SymmetryBase<SUN<N,Scalar> >
{
	typedef Eigen::Index Index;
	template<Index Rank> using TensorType = Eigen::Tensor<Scalar,Rank,Eigen::ColMajor,Index>;
public:
	typedef std::array<int,1> qType;

	SUN() {};

	static constexpr bool HAS_CGC = true;
	static constexpr bool SPECIAL = false;
	static constexpr std::size_t Nq=1;
	
	static qType qvacuum() { return {1}; }

	static std::string name() { std::stringstream ss; ss << "SU(" << N << ")"; return ss.str(); }

	inline static qType flip( const qType& q ) { return q; }
	inline static int degeneracy( const qType& q ) { return q[0]; }

	static TensorType<3> reduce(qType ql, qType qr, qType Q);
	
	static std::vector<qType> reduceSilent(qType ql, qType qr);

	static std::vector<qType> reduceSilent(std::vector<qType> ql, qType qr);

	static Eigen::Tensor<Scalar,2> calcCupTensor( qType q );
	
	static Eigen::Tensor<Scalar,2> calcCapTensor( qType q );

	static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);

	static Scalar coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
							  const qType& q4, const qType& q5, const qType& q6);

	static Scalar coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
							   const qType& q4, const qType& q5, const qType& q6,
							   const qType& q7, const qType& q8, const qType& q9);
	
	template<std::size_t M>
	static bool compare ( std::array<qType,M> q1, std::array<qType,M> q2 );

	template<std::size_t M>
	static bool validate( std::array<qType,M> qs );
};

template<std::size_t N, typename Scalar> bool operator== (const typename SUN<N,Scalar>::qType& lhs, const typename SUN<N,Scalar>::qType& rhs) {return lhs == rhs;}

template<std::size_t N, typename Scalar>
Eigen::Tensor<Scalar,3,Eigen::ColMajor,Eigen::Index> SUN<N,Scalar>::
reduce(typename SUN<N,Scalar>::qType ql, typename SUN<N,Scalar>::qType qr, typename SUN<N,Scalar>::qType Q)
{
	clebsch::weight wl(N,ql[0]-1), wr(N,qr[0]-1), W(N,Q[0]-1);
	clebsch::coefficients cgc(W,wl,wr);
	Index dimwl = wl.dimension(),
		dimwr = wr.dimension(),
		dimW = W.dimension();
	TensorType<3> T(dimwl,dimW,dimwr); T.setZero();
	for (Index i = 0; i < dimW; ++i) {
		for (Index j = 0; j < dimwl; ++j) {
			for (Index k = 0; k < dimwr; ++k)
			{
				double x = Scalar(cgc(j, k, 0, i)); //The 0 is the innermultiplicity. For N>2 you need to adjust here something.
				if (std::abs(x) > clebsch::EPS)
				{
					T(j,i,k) = x;
				}
			}
		}
	}
	return T;
}

template<std::size_t N, typename Scalar>
std::vector<typename SUN<N,Scalar>::qType> SUN<N,Scalar>::
reduceSilent( qType ql, qType qr )
{
	std::vector<typename SUN<N,Scalar>::qType> vout;
	int qmin = std::abs(ql[0]-qr[0]) +1;
	int qmax = std::abs(ql[0]+qr[0]) -1;
	for ( int i=qmin; i<=qmax; i+=2 ) { vout.push_back({i}); }
	return vout;
}

template<std::size_t N, typename Scalar>
std::vector<typename SUN<N,Scalar>::qType> SUN<N,Scalar>::
reduceSilent( std::vector<qType> ql, qType qr )
{
	std::vector<typename SUN<N,Scalar>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		int qmin = std::abs(ql[q][0]-qr[0]) +1;
		int qmax = std::abs(ql[q][0]+qr[0]) -1;
		for ( int i=qmin; i<=qmax; i+=2 ) { vout.push_back({i}); }
	}
	return vout;
}

template<std::size_t N, typename Scalar>
template<std::size_t M>
bool SUN<N,Scalar>::
compare ( std::array<qType,M> q1, std::array<qType,M> q2 )
{
	for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][0] < q2[m][0]) { return false; }
		else if (q1[m][0] > q2[m][0]) {return true; }
	}
	return false;
}

template<std::size_t N, typename Scalar>
template<std::size_t M>
bool SUN<N,Scalar>::
validate ( std::array<SUN<N,Scalar>::qType,M> qs )
{
	if constexpr( M > 1 )
				{
					std::vector<SUN<N,Scalar>::qType> decomp = SUN::reduceSilent(qs[0],qs[1]);
					for (std::size_t i=2; i<M; i++)
					{
						decomp = SUN::reduceSilent(decomp,qs[i]);
					}
					for (std::size_t i=0; i<decomp.size(); i++)
					{
						if ( decomp[i] == SUN::qvacuum() ) { return true; }
					}
					return false;
				}
	else { return true; }
}

template<std::size_t N, typename Scalar>
Eigen::Tensor<Scalar,2,Eigen::ColMajor,Eigen::Index> SUN<N,Scalar>::
calcCapTensor ( SUN<N,Scalar>::qType q )
{
	Index dim = q[0];
	TensorType<2> Tout(dim,dim); Tout.setZero();
	for (Index mInt=0; mInt<dim; mInt++)
	{
	    Tout(mInt,dim-1-mInt) = std::pow(Scalar(-1),mInt)*pow(Scalar(-1),Scalar(dim-1));
	}		
	return Tout;

}

template<std::size_t N, typename Scalar>
Eigen::Tensor<Scalar,2,Eigen::ColMajor,Eigen::Index> SUN<N,Scalar>::
calcCupTensor ( SUN<N,Scalar>::qType q )
{
	Index dim = q[0];
	std::cout << dim << std::endl;
	TensorType<2> Tout(dim,dim); Tout.setZero();
	for (Index mInt=0; mInt<dim; mInt++)
	{
		Tout(mInt,dim-1-mInt) = std::pow(Scalar(-1),mInt);
	}
	return Tout;
}

template<std::size_t N, typename Scalar>
Scalar SUN<N,Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::pow(static_cast<Scalar>(q1[0]),Scalar(0.5)) * std::pow(static_cast<Scalar>(q2[0]),Scalar(-0.5))*
		std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q3[0]+q1[0]-q2[0]-1));
	return out;
}

template<std::size_t N, typename Scalar>
Scalar SUN<N,Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1)*
		std::pow(static_cast<Scalar>(q3[0]*q6[0]),Scalar(0.5))*
		std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q1[0]+q5[0]+q6[0]-3));
	return out;
}

template<std::size_t N, typename Scalar>
Scalar SUN<N,Scalar>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::pow(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]),Scalar(0.5));
	return out;
}

} //end namespace Sym

std::ostream& operator<< (std::ostream& os, const typename Sym::SUN<2,double>::qType &q)
{
	boost::rational<int> s = boost::rational<int>(q[0]-1,2);
	if      (s.numerator()   == 0) {os << " " << 0 << " ";}
	else if (s.denominator() == 1) {os << " " << s.numerator() << " ";}
	else {os << s;}
	return os;
}

#endif
