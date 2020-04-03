#ifndef SITEOPERATORQ_H_
#define SITEOPERATORQ_H_

/// \cond
#include <unsupported/Eigen/KroneckerProduct>
/// \endcond

#include "tensors/Qbasis.h"
#include "tensors/Biped.h"
#include "numeric_limits.h"

//Forward declaration
template<typename Symmetry, typename Scalar> struct SiteOperator;

template<typename Operator>
class EDSolver
{
	typedef typename Operator::qType qType;
	typedef typename Operator::MatrixType MatrixType;
public:
	EDSolver(){};
	EDSolver(const Operator &Op_in, const std::vector<qType> &blocks_in={}, Eigen::DecompositionOptions opt_in=Eigen::DecompositionOptions::EigenvaluesOnly)
		{compute(Op_in,blocks_in,opt_in);}

	void compute(const Operator &Op, const std::vector<qType> &blocks={}, Eigen::DecompositionOptions opt=Eigen::DecompositionOptions::EigenvaluesOnly);

	const Operator& eigenvalues() const {return eigvals_;}
	const Operator& eigenvectors() const {return eigvecs_;}

	Operator groundstate( qType Q ) const;
	
private:
	Operator eigvals_;
	Operator eigvecs_;
	bool COMPUTED=false;
};

template<typename Operator>
void EDSolver<Operator>::
compute(const Operator &Op, const std::vector<qType> &blocks, Eigen::DecompositionOptions opt)
{
	eigvals_.data().clear(); eigvecs_.data().clear();
	eigvals_.Q() = Op.Q();	eigvecs_.Q() = Op.Q();
	eigvals_.basis() = Op.basis(); eigvecs_.basis() = Op.basis();
	MatrixType Mtmp,eigva,eigve;
	for(std::size_t nu=0; nu<Op.data().size(); nu++)
	{
		if(blocks.size()>0) { if(auto it = std::find(blocks.begin(),blocks.end(),Op.data().in[nu]); it == blocks.end()) {continue;} }
		Mtmp = Op.data().block[nu];
		Eigen::SelfAdjointEigenSolver<MatrixType> John(Mtmp,opt);
		eigva = John.eigenvalues();//.asDiagonal();
		eigvals_.data().push_back(Op.data().in[nu],Op.data().out[nu],eigva);
		if(opt == Eigen::DecompositionOptions::ComputeEigenvectors)
		{
			eigve = John.eigenvectors();
			eigvecs_.data().push_back(Op.data().in[nu],Op.data().out[nu],eigve);
		}
	}
	COMPUTED = true;
	return;
}

template<typename Operator>
Operator EDSolver<Operator>::
groundstate( qType Q ) const
{
	assert(COMPUTED and "First diagonlize the Operator before accessing the groundstate!");
	auto all_gs = eigenvectors();
	Operator out(all_gs.Q(),all_gs.basis());
	auto it = all_gs.data().dict.find({Q,Q});
	assert(it != all_gs.data().dict.end() and "The groundstate to the given Q is not present.");
	out.data().push_back(all_gs.data().in[it->second],all_gs.data().out[it->second],all_gs.data().block[it->second]);
	return out;
}


/** 
 * \class SiteOperatorQ
 *
 * \ingroup Tensors
 *
 * This class is the type for local operators and defines the relevant operations: adjoint(), prod(), outerprod(),...
 *
 * \describe_Symmetry
 * \describe_Scalar
 *
 */
template<typename Symmetry, typename MatrixType_>
class SiteOperatorQ // : public Biped<Symmetry,MatrixType_>
{
private:
	typedef Eigen::Index Index;
	typedef Biped<Symmetry,MatrixType_> base;
	
public:
	typedef typename Symmetry::qType qType;
	typedef MatrixType_ MatrixType;
	typedef typename MatrixType::Scalar Scalar;

	/**Does nothing.*/
	SiteOperatorQ() {};

	SiteOperatorQ( const qType& Q_in, const Qbasis<Symmetry>& basis_in, std::string label_in="" )
		: Q_(Q_in),basis_(basis_in),label_(label_in) {};

	SiteOperatorQ( const qType& Q_in, const Qbasis<Symmetry>& basis_in, const base& data_in )
		:Q_(Q_in),basis_(basis_in),data_(data_in) {};
	
	base& data() {return data_;}
	const base& data() const {return data_;}

	qType& Q() {return Q_;}
	const qType& Q() const {return Q_;}

	Qbasis<Symmetry>& basis() {return basis_;}
	const Qbasis<Symmetry>& basis() const {return basis_;}

	std::string& label() {return label_;}
	const std::string& label() const {return label_;}
	
	MatrixType operator() ( const qType& bra, const qType& ket ) const;
	MatrixType& operator() ( const qType& bra, const qType& ket );

	Scalar operator() ( const std::string& bra, const std::string& ket ) const;
	Scalar& operator() ( const std::string& bra, const std::string& ket );

	SiteOperatorQ<Symmetry,MatrixType_>& operator+= ( const SiteOperatorQ<Symmetry,MatrixType_>& Op );
	SiteOperatorQ<Symmetry,MatrixType_>& operator-= ( const SiteOperatorQ<Symmetry,MatrixType_>& Op );

	SiteOperatorQ<Symmetry,MatrixType_> adjoint() const;

	SiteOperatorQ<Symmetry,MatrixType_> hermitian_conj() const;

	void setZero();
	void setIdentity();
	
	static SiteOperatorQ<Symmetry,MatrixType_> prod( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2,
													 const qType& target );
	static SiteOperatorQ<Symmetry,MatrixType_> outerprod( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2,
														  const qType& target );
	static SiteOperatorQ<Symmetry,MatrixType_> outerprod( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2)
		{
			auto target = Symmetry::reduceSilent(O1.Q(),O2.Q());
			assert(target.size() == 1 and "Use other outerprod!");
			return SiteOperatorQ<Symmetry,MatrixType_>::outerprod(O1,O2,target[0]);
		}
			
	SiteOperatorQ<Symmetry,MatrixType_> diagonalize(const std::vector<qType> &blocks={}, Eigen::DecompositionOptions opt=Eigen::DecompositionOptions::EigenvaluesOnly) const;

	typename MatrixType_::Scalar norm() const;

	template<typename Scalar>
	SiteOperator<Symmetry,Scalar> plain() const;
	
	// template<typename Scalar>
	// SiteOperator<Symmetry,Scalar> fullPlain(int m) const;

	/**Prints the operator.*/
	std::string print(bool PRINT_BASIS=false) const;

	template<typename OtherScalar>
	SiteOperatorQ<Symmetry,Eigen::Matrix<OtherScalar, -1, -1 > > cast() const
	{
		SiteOperatorQ<Symmetry,Eigen::Matrix<OtherScalar, -1, -1 > > Oout;
		Oout.Q() = Q();
		Oout.basis() = basis();
		Oout.data() = data().template cast<Eigen::Matrix<OtherScalar, -1, -1> >();
		Oout.label() = label();
		return Oout;
	}
	
private:
	base data_;
	qType Q_;
	Qbasis<Symmetry> basis_;

	std::string label_="";
};

template<typename Symmetry, typename MatrixType_>
MatrixType_& SiteOperatorQ<Symmetry,MatrixType_>::
operator() ( const qType& bra, const qType& ket )
{
	std::array<qType,2> index = {bra,ket};
	auto it = data_.dict.find(index);
	if ( it != data_.dict.end() ) { return data_.block[it->second]; }
	else
	{
		Eigen::Index dim1 = basis_.inner_dim(bra);
		Eigen::Index dim2 = basis_.inner_dim(ket);
		MatrixType A(dim1,dim2); A.setZero();
		data_.push_back(index,A);
		return data_.block[data_.size()-1];
	}
}

template<typename Symmetry, typename MatrixType_>
MatrixType_ SiteOperatorQ<Symmetry,MatrixType_>::
operator() ( const qType& bra, const qType& ket ) const
{
	std::array<qType,2> index = {bra,ket};
	auto it = data_.dict.find(index);
	assert( it != data_.dict.end() and "The element does not exist in the SiteOperatorQ." );
	return data_.block[it->second];
}

template<typename Symmetry, typename MatrixType_>
typename MatrixType_::Scalar& SiteOperatorQ<Symmetry,MatrixType_>::
operator() ( const std::string& bra, const std::string& ket )
{
	qType bra_ = basis_.find(bra);
	qType ket_ = basis_.find(ket);
	std::array<qType,2> index = {bra_,ket_};
	auto it = data_.dict.find(index);
	if ( it != data_.dict.end() )
	{
		Eigen::Index i = basis_.location(bra);
		Eigen::Index j = basis_.location(ket);
		if constexpr ( std::is_same<MatrixType_,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> >::value )
			{
				return data_.block[it->second](i,j);
			}
				else if constexpr ( std::is_same<MatrixType_,Eigen::SparseMatrix<Scalar> >::value )
			{
				return data_.block[it->second].coeffRef(i,j);
			}
	}
	else
	{
		Eigen::Index dim1 = basis_.inner_dim(bra_);
		Eigen::Index dim2 = basis_.inner_dim(ket_);
		MatrixType A(dim1,dim2); A.setZero();
		Eigen::Index i = basis_.location(bra);
		Eigen::Index j = basis_.location(ket);
		data_.push_back(index,A);
		if constexpr ( std::is_same<MatrixType_,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> >::value )
			{
				return data_.block[data_.size()-1](i,j);
			}
		else if constexpr ( std::is_same<MatrixType_,Eigen::SparseMatrix<Scalar> >::value )
			{
				return data_.block[data_.size()-1].coeffRef(i,j);
			}
	}
}

template<typename Symmetry, typename MatrixType_>
typename MatrixType_::Scalar SiteOperatorQ<Symmetry,MatrixType_>::
operator() ( const std::string& bra, const std::string& ket ) const
{
	qType bra_ = basis_.find(bra);
	qType ket_ = basis_.find(ket);
	std::array<qType,2> index = {bra_,ket_};
	auto it = data_.dict.find(index);
	assert( it != data_.dict.end() and "The element does not exist in the SiteOperatorQ." );
	Eigen::Index i = basis_.location(bra);
	Eigen::Index j = basis_.location(ket);		
	return data_.block[it->second](i,j);
}

template<typename Symmetry, typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> SiteOperatorQ<Symmetry,MatrixType_>::
diagonalize (const std::vector<qType> &blocks, Eigen::DecompositionOptions opt) const
{
	assert(this->Q() == Symmetry::qvacuum() and "Only a singlet operator can get diagonalized.");
	SiteOperatorQ<Symmetry,MatrixType_> out( this->Q(), this->basis() );

	MatrixType Mtmp,res;
	for( std::size_t nu=0; nu<this->data().size(); nu++ )
	{
		if(blocks.size()>0) { if(auto it = std::find(blocks.begin(),blocks.end(),this->data().in[nu]); it == blocks.end()) {continue;} }
		Mtmp = this->data().block[nu];
		Eigen::SelfAdjointEigenSolver<MatrixType> John(Mtmp);
		res = John.eigenvalues().asDiagonal();
		out.data().push_back(this->data().in[nu],this->data().out[nu],res);
	}
	return out;
}

template<typename Symmetry, typename MatrixType_>
typename MatrixType_::Scalar SiteOperatorQ<Symmetry,MatrixType_>::
norm () const
{
	auto tmp = SiteOperatorQ<Symmetry,MatrixType_>::prod(*this, this->adjoint(), Symmetry::qvacuum());
	return tmp.data().trace();
}

template<typename Symmetry, typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> SiteOperatorQ<Symmetry,MatrixType_>::
adjoint () const
{
	SiteOperatorQ<Symmetry,MatrixType_> out( Symmetry::flip(this->Q()), this->basis() );

	for( std::size_t nu=0; nu<this->data().size(); nu++ )
	{
		std::array<qType,2> index = {this->data().out[nu],this->data().in[nu]};
		MatrixType A = this->data().block[nu].adjoint();
		A *= Symmetry::coeff_adjoint(this->data().in[nu],this->data().out[nu],this->Q());
		out.data().push_back(index,A);
	}
	out.label() = this->label() + "†";
	return out;
}

template<typename Symmetry, typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> SiteOperatorQ<Symmetry,MatrixType_>::
hermitian_conj () const
{
	SiteOperatorQ<Symmetry,MatrixType_> out( Symmetry::flip(this->Q()), this->basis() );

	for( std::size_t nu=0; nu<this->data().size(); nu++ )
	{
		std::array<qType,2> index = {this->data().out[nu],this->data().in[nu]};
		MatrixType A = this->data().block[nu].adjoint();
		// A *= Symmetry::coeff_adjoint(this->data().in[nu],this->data().out[nu],this->Q());
		out.data().push_back(index,A);
	}
	return out;
}

template<typename Symmetry, typename MatrixType_>
template<typename Scalar>
SiteOperator<Symmetry,Scalar> SiteOperatorQ<Symmetry,MatrixType_>::
plain() const
{
	SiteOperator<Symmetry,Scalar> out;
	MatrixType_ Mtmp(basis().size(), basis().size()); Mtmp.setZero();
	for( auto itQ = basis().cbegin(); itQ != basis().cend(); itQ++ )
	{
		auto [qVal,qNum,qPlain] = *itQ;
		for( auto itP = basis().cbegin(); itP != basis().cend(); itP++ )
		{
			auto [pVal,pNum,pPlain] = *itP;
			if( auto it = data().dict.find({{qVal,pVal}}); it != data().dict.end() )
			{
				Mtmp.block(qNum,pNum,qPlain.size(),pPlain.size()) = data().block[it->second];
			}
		}
	}
//	if constexpr ( std::is_same<OtherMatrixType,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> >::value )
//		{
//			out.data = Mtmp;
//		}
//	else if constexpr ( std::is_same<OtherMatrixType,Eigen::SparseMatrix<Scalar> >::value )
//		{
//			out.data = Mtmp.sparseView();
//		}
	out.data = Mtmp.sparseView();
	out.Q = this->Q();
	out.label = this->label();
	return out;
}

// template<typename Symmetry, typename MatrixType_>
// template<typename Scalar>
// SiteOperator<Symmetry,Scalar> SiteOperatorQ<Symmetry,MatrixType_>::
// fullPlain(int m) const
// {
// 	SiteOperator<Symmetry,Scalar> out;
// 	MatrixType_ Mtmp(basis().fullM(), basis().fullM()); Mtmp.setZero();
// 	for( auto itQ = basis().cbegin(); itQ != basis().cend(); itQ++ )
// 	{
// 		auto [qVal,qNum,qPlain] = *itQ;
// 		for( auto itP = basis().cbegin(); itP != basis().cend(); itP++ )
// 		{
// 			auto [pVal,pNum,pPlain] = *itP;
// 			if( auto it = data().dict.find({{qVal,pVal}}); it != data().dict.end() )
// 			{
// 				for (int qm=0; qm<Symmetry::degeneracy(qVal); qm++)
// 				for (int pm=0; pm<Symmetry::degeneracy(pVal); pm++)
// 				Mtmp.block(qNum,pNum,qPlain.size(),pPlain.size()) = data().block[it->second];
// 			}
// 		}
// 	}
// 	if constexpr ( std::is_same<OtherMatrixType,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> >::value )
// 		{
// 			out.data = Mtmp;
// 		}
// 	else if constexpr ( std::is_same<OtherMatrixType,Eigen::SparseMatrix<Scalar> >::value )
// 		{
// 			out.data = Mtmp.sparseView();
// 		}
// 	out.data = Mtmp.sparseView();
// 	out.Q = this->Q();
// 	out.label = this->label();
// 	return out;
// }

template<typename Symmetry, typename MatrixType_>
void SiteOperatorQ<Symmetry,MatrixType_>::
setZero()
{
	for(const auto& q : basis().qloc())
	{
		(*this)(q,q).setZero();
	}
}

template<typename Symmetry, typename MatrixType_>
void SiteOperatorQ<Symmetry,MatrixType_>::
setIdentity()
{
	for(const auto& q : basis().qloc())
	{
		(*this)(q,q).setIdentity();
	}
}

template<typename Symmetry, typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> SiteOperatorQ<Symmetry,MatrixType_>::
prod( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2, const qType& target )
{
	std::array<qType,3> checkIndex = {O1.Q(),O2.Q(),target};
	assert( Symmetry::validate(checkIndex) and "Operators O1 and O2 cannot get multplied to a quantum number target operator" );
	assert( O1.basis() == O2.basis() and "For a prod() operation the two operators need to have the same basis" );
	
	SiteOperatorQ out( target, O1.basis() );

	std::array<qType,2> totIndex, index;
	MatrixType A;
	Scalar factor_cgc;
	for ( std::size_t nu=0; nu<O1.data().size(); nu++ )
	{
		auto qvec = Symmetry::reduceSilent(O1.data().out[nu],Symmetry::flip(O2.Q()));
		for (const auto& q : qvec)
		{
			index = {O1.data().out[nu],q};
			auto it = O2.data().dict.find(index);
			if (it == O2.data().dict.end()) {continue;}
			std::size_t mu = it->second;
			factor_cgc = Symmetry::coeff_prod( O1.data().in[nu], O1.Q(), O1.data().out[nu],
											   O2.Q(), O2.data().out[mu], target);
			// factor_cgc = Symmetry::coeff_Apair( O2.data().out[mu], O2.Q(), O1.data().out[nu],
			// 									O1.Q(), O1.data().in[nu], target);
			if ( std::abs(factor_cgc) < std::abs(::mynumeric_limits<Scalar>::epsilon()) ) { continue; }
			totIndex = { O1.data().in[nu], O2.data().out[mu] };
			A = O1.data().block[nu] * O2.data().block[mu] * factor_cgc;
			// A = O2.data().block[mu] * O1.data().block[nu] * factor_cgc;
			auto check = out.data().dict.find(totIndex);
			if ( check == out.data().dict.end() )
			{
				out.data().push_back(totIndex,A);
			}
			else { out.data().block[check->second] += A; }
		}
	}
	// for ( std::size_t nu=0; nu<O1.data().size(); nu++ )
	// {
	// 	// auto qvec = Symmetry::reduceSilent(O1.data().in[nu],Symmetry::flip(O2.Q()));
	// 	auto qvec = Symmetry::reduceSilent(O1.data().in[nu],O2.Q());

	// 	for (const auto& q : qvec)
	// 	{
	// 		index = {q,O1.data().in[nu]};
	// 		auto it = O2.data().dict.find(index);
	// 		if (it == O2.data().dict.end()) {continue;}
	// 		std::size_t mu = it->second;
	// 		// factor_cgc = Symmetry::coeff_Apair( O1.data().out[nu], O1.Q(), O1.data().in[nu],
	// 		// 									O2.Q(), O2.data().in[mu], target);
	// 		factor_cgc = Symmetry::coeff_Apair( O2.data().in[mu], O2.Q(), O1.data().in[nu],
	// 											O1.Q(), O1.data().out[nu], target);
	// 		if ( std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon() ) { continue; }
	// 		totIndex = { O1.data().in[nu], O2.data().out[mu] };
	// 		A = O1.data().block[nu] * O2.data().block[mu] * factor_cgc;
	// 		// A = O2.data().block[mu] * O1.data().block[nu] * factor_cgc;
	// 		auto check = out.data().dict.find(totIndex);
	// 		if ( check == out.data().dict.end() )
	// 		{
	// 			out.data().push_back(totIndex,A);
	// 		}
	// 		else { out.data().block[check->second] += A; }
	// 	}
	// }
	stringstream ss;
	if (O1.label() == O2.label()) { ss << O1.label() << "²"; }
	else {ss << O1.label() << "*" << O2.label();}
	
	out.label() = ss.str();

	return out;
}

template<typename Symmetry, typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> SiteOperatorQ<Symmetry,MatrixType_>::
outerprod( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2, const qType& target )
{
	std::array<qType,3> checkIndex = {O1.Q(),O2.Q(),target};
	assert(Symmetry::validate(checkIndex) and "Operators O1 and O2 cannot get multiplied to an operator with quantum number = target");
	
	auto TensorBasis = O1.basis().combine(O2.basis());
	
	SiteOperatorQ out(target,TensorBasis);
	
	std::array<qType,2> totIndex;
	MatrixType Atmp,A;
	Index rows,cols;
	Scalar factor_cgc;
	
	for (std::size_t nu=0; nu<O1.data().size(); nu++)
	for (std::size_t mu=0; mu<O2.data().size(); mu++)
	{
		auto reduce1 = Symmetry::reduceSilent(O1.data().in[nu],  O2.data().in[mu]);
		auto reduce2 = Symmetry::reduceSilent(O1.data().out[nu], O2.data().out[mu]);
		for (const auto& q1:reduce1)
		for (const auto& q2:reduce2)
		{
			factor_cgc = Symmetry::coeff_tensorProd(O1.data().out[nu], O2.data().out[mu], q2,
			                                        O1.Q(), O2.Q(), target,
			                                        O1.data().in[nu], O2.data().in[mu], q1);
			if (std::abs(factor_cgc) < ::mynumeric_limits<Scalar>::epsilon()) {continue;}
			totIndex = { q1, q2 };
			rows = O1.data().block[nu].rows() * O2.data().block[mu].rows();
			cols = O1.data().block[nu].cols() * O2.data().block[mu].cols();
			Atmp.resize(rows,cols);
			
			Atmp = kroneckerProduct(O1.data().block[nu], O2.data().block[mu]);
			Index left1  = TensorBasis.leftAmount (q1,{O1.data().in[nu],  O2.data().in[mu]});
			Index right1 = TensorBasis.rightAmount(q1,{O1.data().in[nu],  O2.data().in[mu]});
			Index left2  = TensorBasis.leftAmount (q2,{O1.data().out[nu], O2.data().out[mu]});
			Index right2 = TensorBasis.rightAmount(q2,{O1.data().out[nu], O2.data().out[mu]});
			A.resize(rows+left1+right1,cols+left2+right2); A.setZero();
			A.block(left1,left2,rows,cols) = Atmp;
			
			auto it = out.data().dict.find(totIndex);
			if (it == out.data().dict.end())
			{
				out.data().push_back(totIndex, factor_cgc*A);
			}
			else
			{
				out.data().block[it->second] += factor_cgc * A;
			}
		}
	}
	stringstream ss;
	ss << O1.label() << "⊗" << O2.label();
	out.label() = ss.str();
	return out;
}

template<typename Symmetry,typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_>& SiteOperatorQ<Symmetry,MatrixType_>::operator+= ( const SiteOperatorQ<Symmetry,MatrixType_>& Op )
{
	*this = *this + Op;
	return *this;
}

template<typename Symmetry,typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_>& SiteOperatorQ<Symmetry,MatrixType_>::operator-= ( const SiteOperatorQ<Symmetry,MatrixType_>& Op )
{
	*this = *this - Op;
	return *this;
}

template<typename Symmetry, typename MatrixType_>
std::string SiteOperatorQ<Symmetry,MatrixType_>::
print(bool PRINT_BASIS) const
{
	std::stringstream out;
	out << "Operator " << label() << endl;
	if (PRINT_BASIS) {out << basis() << endl;}
	out << data().print(true) << endl;
	return out.str();
}

template<typename Symmetry,typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> operator* ( const typename MatrixType_::Scalar& s, const SiteOperatorQ<Symmetry,MatrixType_>& op )
{
	SiteOperatorQ<Symmetry,MatrixType_> out = op;
	out.data() = s*op.data();
	return out;
}

template<typename Symmetry,typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> operator* (const SiteOperatorQ<Symmetry,MatrixType_> &O1, const SiteOperatorQ<Symmetry,MatrixType_> &O2)
{
	auto Qtots = Symmetry::reduceSilent(O1.Q(), O2.Q());
	assert(Qtots.size() == 1 and "The operator * for SiteOperatorQ can only be used uf the target quantumnumber is unique. Use SiteOperatorQ::prod() insteat.");
	auto Oout = SiteOperatorQ<Symmetry,MatrixType_>::prod(O1, O2, Qtots[0]);
	return Oout;
}

template<typename Symmetry,typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> operator+ ( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2 )
{
	assert(O1.basis() == O2.basis() and "For addition of SiteOperatorQs the basis needs to be the same.");
	assert(O1.Q() == O2.Q() and "For addition of SiteOperatorQs the operator quantum number needs to be the same.");
	SiteOperatorQ<Symmetry,MatrixType_> out(O1.Q(),O1.basis());
	out.data() = O1.data() + O2.data();
	stringstream ss;
	ss << "(" << O1.label() << "+" << O2.label() << ")";
	out.label() = ss.str();

	return out;
}

template<typename Symmetry,typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> operator- ( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2 )
{
	assert(O1.basis() == O2.basis() and "For subtraction of SiteOperatorQs the basis needs to be the same.");
	assert(O1.Q() == O2.Q() and "For subtraction of SiteOperatorQs the operator quantum number needs to be the same.");
	SiteOperatorQ<Symmetry,MatrixType_> out(O1.Q(),O1.basis());
	out.data() = O1.data() - O2.data();
	stringstream ss;
	ss << "(" << O1.label() << "-" << O2.label() << ")";
	out.label() = ss.str();
	return out;
}

template<typename Symmetry,typename MatrixType_>
SiteOperatorQ<Symmetry,MatrixType_> kroneckerProduct( const SiteOperatorQ<Symmetry,MatrixType_>& O1, const SiteOperatorQ<Symmetry,MatrixType_>& O2)
{
	return SiteOperatorQ<Symmetry,MatrixType_>::outerprod(O1,O2);
}

template<typename Symmetry,typename MatrixType_>
std::ostream& operator<<(std::ostream& os, const SiteOperatorQ<Symmetry,MatrixType_> &Op)
{
	os << Op.print(false);
	return os;
}

#endif
