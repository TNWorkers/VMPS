#ifndef STRAWBERRY_MULTIPEDE
#define STRAWBERRY_MULTIPEDE

#define LEGLIMIT 2

#include <unordered_map>

#include "boost/multi_array.hpp"
//using namespace boost;

#include "qarray.h"
#include "DmrgExternalQ.h"
#include "MemCalc.h"

/**
General tensor with quantum number blocks.
\tparam Nlegs : Amount of tensor legs, must be >=2. A two-legged tensor is coded separately as Biped.
\describe_Symmetry
\describe_MatrixType
\note special typedefs not caught by Doxygen: \n 
\p Tripod<Symmetry,MatrixType> (\p Nlegs=3, e.g. the transfer matrix in \f$\left<\Psi|H|\Psi\right>\f$) \n
\p Quadruped<Symmetry,MatrixType> (\p Nlegs=4, e.g. the transfer matrix in \f$\left<\Psi|H^2|\Psi\right>\f$).*/
template<size_t Nlegs, typename Symmetry, typename MatrixType>
struct Multipede
{
typedef typename Symmetry::qType qType;
	
	Multipede(){dim=0;}

	/**Const reference to the number of legs \p Nlegs */
	inline constexpr size_t rank() const {return Nlegs;}

	///@{
	/**Convenience access to the amount of blocks.
	Equal to either of the following: \p index.size(), \p block.size()*/
	size_t dim;

	/**Vector of all incoming quantum numbers.
	The entries are arrays of size \p Nlegs. The sorting is according to the following convention:
	1. incoming 2. outgoing 3. middle (\p Nlegs=3)
	1. incoming 2. outgoing 3. bottom 4. top (\p Nlegs=4)
	The middle, bottom and top quantum numbers are always outgoing for the left transfer matrix and incoming for the right transfer matrix in matrix element calculations.*/
	vector<std::array<qType,Nlegs> > index;
	
	/**Vector of quantum number blocks.
	The matrix \p block[q] is characterized by the quantum number array \p index[q]. Since the handling of template-sized arrays is no fun at all, the implementation is done in the following way: \p block[q] is always a boost \p multi_array of dimension \p LEGLIMIT which is set = 2. Tripods need only one dimension (two are already absorbed into \p MatrixType), therefore the rest needs to be set to 1 and the access goes by \p T.block[q][a][0]. \p LEGLIMIT can be increased and some code adjustment can be made (filling in dummy "[0]") if bigger tensors are required.*/
	vector<boost::multi_array<MatrixType,LEGLIMIT> > block;
	
	/**Dictionary allowing one to find the index of \p block for a given array of \p Nlegs quantum numbers in \f$O(1)\f$ operations without looping over the blocks.
	For the ordering convention see Multipede::index.*/
	unordered_map<std::array<qType,Nlegs>,size_t> dict; // key format: {qin,qout,qmid}
	///@}
	
	///@{
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=MB) const;
	
	/**Prints Multipede<Nlegs,Symmetry,MatrixType>::dict into a string.*/
	string dict_info() const;

	string print(const bool &SHOW_MATRICES=false, const size_t &precision=3) const;
//	void rebuild_dict();
	///@}
	
	///@{
	/**Deletes the contents of \p index, \p block, \p dict.*/
	void clear();
	
	/**Sets all matrices in Multipede<Nlegs,Symmetry,MatrixType>::block to zero, preserving the rows and columns.*/
	void setZero();
	
	/**Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to the vacuum (all of them).
	Needed in for the transfer matrix to the first site in matrix element calculations.*/
	void setVacuum();
	
	/**Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers according to the input \p Q.
	Needed in for the transfer matrix from the last site in matrix element calculations.*/
	void setTarget (std::array<qType,Nlegs> Q);
	
	/***/
	void setIdentity (size_t Drows, size_t Dcols, size_t amax=1, size_t bmax=1);
	///@}
	
	///@{
	/**Convenience function to return a quantum number of the block \p q to preserve the sanity of the programmer.
	For the naming convention see Multipede::index.*/
	inline qType in  (size_t q) const {return index[q][0];}
	inline qType out (size_t q) const {return index[q][1];}
	inline qType mid (size_t q) const {return index[q][2];}
	inline qType bot (size_t q) const {return index[q][2];}
	inline qType top (size_t q) const {return index[q][3];}
	///@}
	
	///@{
	/**Adds a new block to the tensor specified by the incoming quantum numbers \p quple.
	\warning Does not check whether the block for these quantum numbers already exists.*/
	void push_back (std::array<qType,Nlegs> quple, const boost::multi_array<MatrixType,LEGLIMIT> &M);
	
	/**Adds a new block to the tensor specified by the initializer list \p qlist (must be of size \p Nlegs).
	For the ordering convention see Multipede::index.
	\warning Does not check whether the block for these quantum numbers already exists.*/
	void push_back (std::initializer_list<qType> qlist, const boost::multi_array<MatrixType,LEGLIMIT> &M);
	///@}
};

template<typename Symmetry, typename MatrixType> using Tripod    = Multipede<3,Symmetry,MatrixType>;
template<typename Symmetry, typename MatrixType> using Quadruped = Multipede<4,Symmetry,MatrixType>;

template<size_t Nlegs, typename Symmetry, typename MatrixType>
string Multipede<Nlegs,Symmetry,MatrixType>::
dict_info() const
{
	stringstream ss;
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		if (Nlegs==2) {ss << "in: " << it->first[0] << ", out: " << it->first[1];}
		if (Nlegs==3) {ss << "in: " << it->first[0] << ", out: " << it->first[1] << ", mid:" << it->first[2];}
		if (Nlegs==4) {ss << "in: " << it->first[0] << ", out: " << it->first[1] << ", bot:" << it->first[2] << ", top:" << it->first[3];}
		ss << "\t→\t" << it->second << endl;
	}
	return ss.str();
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
double Multipede<Nlegs,Symmetry,MatrixType>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t q=0; q<dim; ++q)
	for (auto B=block[q].data(); B!=block[q].data()+block[q].num_elements(); ++B)
	{
		res += calc_memory(*B, memunit);
	}
	return res;
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
double Multipede<Nlegs,Symmetry,MatrixType>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
	res += 2. * Nlegs * Symmetry::Nq * calc_memory<int>(dim, memunit); // in,out,mid; dict.keys
	res += Symmetry::Nq * calc_memory<size_t>(dim, memunit); // dict.vals
	return res;
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
setZero()
{
	for (size_t q=0; q<dim; ++q)
	for (auto B=block[q].data(); B!=block[q].data()+block[q].num_elements(); ++B)
	{
		B->setZero();
	}
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
clear()
{
//	for (size_t leg=0; leg<Nlegs; ++leg)
//	{
//		index[leg].clear();
//	}
	index.clear();
	block.clear();
	dict.clear();
	dim = 0;
}

//template<size_t Nlegs, typename Symmetry, typename MatrixType>
//void Multipede<Nlegs,Symmetry,MatrixType>::
//rebuild_dict()
//{
//	dict.clear();
//	for (size_t q=0; q<dim; ++q)
//	{
//		std::array<qType,Nlegs> quple;
//		for (size_t leg=0; leg<Nlegs; ++leg)
//		{
//			quple[leg][q] = index[leg][q];
//		}
//		dict.insert({quple,q});
//	}
//}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
push_back (std::array<qType,Nlegs> quple, const boost::multi_array<MatrixType,LEGLIMIT> &M)
{
//	for (size_t leg=0; leg<Nlegs; ++leg)
//	{
//		index[leg].push_back(quple[leg]);
//	}
	index.push_back(quple);
	block.push_back(M);
	dict.insert({quple, dim});
	++dim;
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
push_back (std::initializer_list<qType> qlist, const boost::multi_array<MatrixType,LEGLIMIT> &M)
{
	assert(qlist.size() == Nlegs);
	std::array<qType,Nlegs> quple;
	copy(qlist.begin(), qlist.end(), quple.data());
	push_back(quple,M);
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
setVacuum()
{
	MatrixType Mtmp(1,1); Mtmp << 1.;
	boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[1][1]);
	Mtmparray[0][0] = Mtmp;
	
	std::array<qType,Nlegs> quple;
	for (size_t leg=0; leg<Nlegs; ++leg)
	{
		quple[leg] = Symmetry::qvacuum();
	}
	
	push_back(quple, Mtmparray);
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
setTarget (std::array<qType,Nlegs> Q)
{
	MatrixType Mtmp(1,1); Mtmp << Symmetry::coeff_dot(Q[0]);
	boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[1][1]);
	Mtmparray[0][0] = Mtmp;
	
	push_back(Q,Mtmparray);
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
setIdentity (size_t Drows, size_t Dcols, size_t amax, size_t bmax)
{
	boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[amax][bmax]);
	for (size_t a=0; a<amax; ++a)
	for (size_t b=0; b<bmax; ++b)
	{
		MatrixType Mtmp(Drows,Dcols);
		Mtmp.setIdentity();
		Mtmparray[a][b] = Mtmp;
	}
	
	std::array<qType,Nlegs> quple;
	for (size_t leg=0; leg<Nlegs; ++leg)
	{
		quple[leg] = Symmetry::qvacuum();
	}
	push_back(quple, Mtmparray);
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
std::string Multipede<Nlegs,Symmetry,MatrixType>::
print(const bool &SHOW_MATRICES, const std::size_t &precision) const
{
#ifndef HELPERS_IO_TABLE
	std::stringstream out;
	out << "Texttable library is missing. -> no output" << std::endl;
	return out.str();
#else //Use TextTable library for nicer output.
	std::stringstream out;

	TextTable t( '-', '|', '+' );
	t.add("ν");
	t.add("Q_ν");
	t.add("A_ν");
	t.endOfRow();
	for (std::size_t nu=0; nu<dim; nu++)
	{
		std::stringstream ss,tt,uu,vv;
		ss << nu;
		tt << "(";
		for (std::size_t q=0; q<index[nu].size(); q++)
		{
			tt << index[nu][q];
			if(q==index[nu].size()-1) {tt << ")";} else {tt << ",";}
		}
		uu << block[nu].shape()[0] << ":[";
		// uu << block[nu][0][0].cols() << "x" << block[nu][0][0].rows() << "x" << block[nu].shape()[0];
		for (std::size_t q=0; q<block[nu].shape()[0]; q++)
		{
			uu << "(" << block[nu][q][0].cols() << "x" << block[nu][q][0].rows() << ")";
			if(q==block[nu].shape()[0]-1) {uu << "";} else {uu << ",";}
		}
		uu << "]";
		t.add(ss.str());
		t.add(tt.str());
		t.add(uu.str());
		t.endOfRow();
	}
	t.setAlignment( 0, TextTable::Alignment::RIGHT );
	out << t;

	if (SHOW_MATRICES)
	{
		out << TCOLOR(BLUE) << "\e[4mA-tensors:\e[0m" << std::endl;
		for (std::size_t nu=0; nu<dim; nu++)
		{
			out << TCOLOR(BLUE) << "ν=" << nu << std::endl;
			for (std::size_t q=0; q<block[nu].shape()[0]; q++)
			{
				out << TCOLOR(GREEN) << "q=" << q << "\t" << std::setprecision(precision) << std::fixed << block[nu][q][0] << std::endl;
			}
		}
		out << TCOLOR(BLACK) << std::endl;
	}

	return out.str();
#endif
}
#endif
