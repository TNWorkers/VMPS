#include <unsupported/Eigen/CXX11/Tensor>
#include "NestedLoopIterator.h"
#include "numeric_limits.h"

template<typename Scalar, Eigen::Index Rank>
Scalar sumAbs (const Eigen::Tensor<Scalar,Rank> &T)
{
	typedef Eigen::Index Index;
	Scalar out=0.;
	std::array<Index,Rank> dims;
	for (std::size_t i=0; i<Rank; i++) { dims[i] = T.dimension(i); }
	
	std::vector<std::size_t> dimsSizet;
	for (std::size_t i=0; i<dims.size(); i++) { dimsSizet.push_back(static_cast<std::size_t>(dims[i])); }
	NestedLoopIterator Nelly(static_cast<std::size_t>(Rank),dimsSizet);
	size_t numElem=0;
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		Nelly.make_tensorIndex();
		std::array<Index,Rank> index;
		for (std::size_t i=0;i<Rank;i++) {index[i] = static_cast<Index>(Nelly.tensor_index[i]);}
		out += std::abs(T(index));
		numElem++;
	}
	out = out*std::pow(numElem,Scalar(-1));
	return out;
}

template<>
double sumAbs<double,0> (const Eigen::Tensor<double,0> &T)
{
	double out=T();
	return fabs(out);
}

template<typename Scalar, Eigen::Index Rank>
void ridOfNoise( Eigen::Tensor<Scalar,Rank> &Tin )
{
	typedef Eigen::Index Index;
	std::array<Index,Rank> dims;
	for (std::size_t i=0; i<Rank; i++) {dims[i]=Tin.dimension(i);}
	
	std::vector<std::size_t> dimsSizet;
	for (std::size_t i=0; i<dims.size(); i++) {dimsSizet.push_back(static_cast<std::size_t>(dims[i]));}
	NestedLoopIterator Nelly(static_cast<std::size_t>(Rank),dimsSizet);

	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		Nelly.make_reverseTensorIndex();
		std::array<Index,Rank> index;
		for (size_t i=0;i<Rank;i++) {index[i]=static_cast<Index>(Nelly.tensor_index[i]);}
		if ( Scalar(0.) < std::abs(Tin(index)) and std::abs(Tin(index)) < ::numeric_limits<Scalar>::epsilon() ) { Tin(index) = Scalar(0.); }
	}
}

template<>
void ridOfNoise<double,0>( Eigen::Tensor<double,0> &Tin)
{
	if ( std::abs(Tin()) > 0. and std::abs(Tin()) < std::numeric_limits<double>::epsilon() ) { Tin() = 0.; }
}

template<Eigen::Index Rank, typename Scalar>
Scalar prop_to (const Eigen::Tensor<Scalar,Rank,Eigen::ColMajor,Eigen::Index> &M1, const Eigen::Tensor<Scalar,Rank,Eigen::ColMajor,Eigen::Index> &M2 )
{
	typedef Eigen::Index Index;
	Scalar check,out ;
	bool GATE=true;
	std::array<Index,Rank> dims;
	for (std::size_t i=0; i<Rank; i++) { assert(M1.dimension(i) == M2.dimension(i) and "Can't compare tensors with different dimensions"); }
	for (std::size_t i=0; i<Rank; i++) { dims[i]=M1.dimension(i); }
	
	std::vector<std::size_t> dimsSizet;
	for (std::size_t i=0; i<dims.size(); i++) { dimsSizet.push_back(static_cast<std::size_t>(dims[i])); }
	NestedLoopIterator Nelly(static_cast<std::size_t>(Rank),dimsSizet);

	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		Nelly.make_reverseTensorIndex();
		std::array<Index,Rank> index;
		for (std::size_t i=0;i<Rank;i++) {index[i] = static_cast<Index>(Nelly.tensor_index[i]);}
		if ( std::abs(M1(index)) > ::numeric_limits<Scalar>::epsilon() and std::abs(M2(index)) > ::numeric_limits<Scalar>::epsilon() )
		{
			if (GATE) { check = M1(index)*std::pow(M2(index),Scalar(-1)); out = M1(index); GATE = false; }
			if ( std::abs(M1(index)*pow(M2(index),Scalar(-1)) - check) > ::numeric_limits<Scalar>::epsilon() ) { return std::numeric_limits<Scalar>::infinity(); }
		}
		else if ( std::abs(M1(index)) < ::numeric_limits<Scalar>::epsilon() and std::abs(M2(index)) > ::numeric_limits<Scalar>::epsilon() )
		{ return std::numeric_limits<Scalar>::infinity(); }
		else if ( std::abs(M1(index)) > ::numeric_limits<Scalar>::epsilon() and std::abs(M2(index)) < ::numeric_limits<Scalar>::epsilon() )
		{ return std::numeric_limits<Scalar>::infinity(); }
		else {}
	}
	return check;
}

template<>
double prop_to<0,double> (const Eigen::Tensor<double,0> &M1, const Eigen::Tensor<double,0> &M2 )
{
	if (std::abs(M2()) < ::numeric_limits<double>::epsilon() and std::abs(M1()) > ::numeric_limits<double>::epsilon()) { return std::numeric_limits<double>::infinity(); }
	else if ( std::abs(M2()) < ::numeric_limits<double>::epsilon() and std::abs(M1()) < ::numeric_limits<double>::epsilon() ) 
	{ assert( 1 == 1 and "undefined behaviour because of 0/0"); }
	double out=M1()*pow(M2(),-1);
	return out;
}

template<Eigen::Index Rank, typename Scalar>
bool validate( const Eigen::Tensor<Scalar,Rank> &M )
{
	typedef Eigen::Index Index;
	bool out=true;
	std::array<Index,Rank> dims;
	for (size_t i=0; i<Rank; i++) {dims[i]=M.dimension(i);}
	
	std::vector<std::size_t> dimsSizet;
	for (std::size_t i=0; i<dims.size(); i++) {dimsSizet.push_back(static_cast<std::size_t>(dims[i]));}
	NestedLoopIterator Nelly(static_cast<std::size_t>(Rank),dimsSizet);

	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		Nelly.make_reverseTensorIndex();
		std::array<Index,Rank> index;
		for (size_t i=0;i<Rank;i++) {index[i]=static_cast<Index>(Nelly.tensor_index[i]);}
		if( std::abs(M(index)) > ::numeric_limits<Scalar>::infinity() ) { out=false; return out; }
	}
	return out;
}
//template<>
//bool validate( const Eigen::Tensor<double,0> &M )
//{
//	bool out=true;
//	if( std::abs(M()) > ::numeric_limits<double>::infinity() ) { out=false; return out; }
//	return out;
//}

// template<int Dim1, int Dim2>
// Eigen::Tensor<double,Dim1+Dim2> product (const Eigen::Tensor<double,Dim1> &T1, const Eigen::Tensor<double,Dim2> &T2)
// {
// 	std::vector<int> dims;
// 	for (size_t i=0; i<Dim1; i++) {int dim = T1.dimension(i);dims.push_back(dim);}

// 	for (size_t i=0; i<Dim2; i++) {int dim = T2.dimension(i);dims.push_back(dim);}

// 	Eigen::Tensor<double,Dim1+Dim2> Mout;
// 	Mout.resize(dims);
	
// 	std::vector<size_t> dimsSizet;
// 	for (size_t i=0; i<dims.size(); i++) {dimsSizet.push_back(static_cast<size_t>(dims[i]));}
// 	NestedLoopIterator Nelly(static_cast<size_t>(Dim1+Dim2),dimsSizet);

// 	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
// 	{
// 		Nelly.make_tensorIndex();
// 		std::vector<size_t> index1;
// 		for (size_t i=0;i<Dim1;i++) {index1.push_back(Nelly.tensor_index[i]);}
// 		std::vector<size_t> index2;
// 		for (size_t i=Dim1;i<Dim1+Dim2;i++) {index2.push_back(Nelly.tensor_index[i]);}
		
// 		Mout(Nelly.tensor_index) = T1(index1)*T2(index2);
// 	}

// 	return Mout;
// }

// template<size_t Trank>
// Eigen::Tensor<size_t,Trank> TmemAna( int dim )
// {
// 	std::vector<int> dims;
// 	for (size_t i=0; i<Trank; i++) {dims.push_back(dim);}

// 	Eigen::Tensor<size_t,Trank> T;
// 	T.resize(dims);
// 	std::vector<size_t> dimsSizet;
// 	for (size_t i=0; i<dims.size(); i++) {dimsSizet.push_back(static_cast<size_t>(dims[i]));}
// 	NestedLoopIterator Nelly(static_cast<size_t>(Trank),dimsSizet);
// 	size_t count = 0;
// 	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
// 	{
// 		Nelly.make_reverseTensorIndex();
// 		T(Nelly.tensor_index) = count;
// 		count++;
// 	}
// 	return T;		
// }
