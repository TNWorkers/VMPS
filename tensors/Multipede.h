#ifndef STRAWBERRY_MULTIPEDE
#define STRAWBERRY_MULTIPEDE

#define LEGLIMIT 2

/// \cond
#include <unordered_map>

#include "boost/multi_array.hpp"
/// \endcond


#include "MemCalc.h" // from TOOLS
#include "symmetry/functions.h"
#include "HDF5Interface.h"

//include "symmetry/qarray.h"
//include "DmrgExternal.h"

/**
 * \ingroup Tensors
 *
 * General tensor with quantum number blocks.
 * \tparam Nlegs : Amount of tensor legs, must be >=2. A two-legged tensor is coded separately as Biped.
 * \describe_Symmetry
 * \describe_MatrixType
 * \note special typedefs not caught by Doxygen: \n 
 * \p Tripod<Symmetry,MatrixType> (\p Nlegs=3, e.g. the transfer matrix in \f$\left<\Psi|H|\Psi\right>\f$) \n
 * \p Quadruped<Symmetry,MatrixType> (\p Nlegs=4, e.g. the transfer matrix in \f$\left<\Psi|H^2|\Psi\right>\f$).
 */
template<size_t Nlegs, typename Symmetry, typename MatrixType>
struct Multipede
{
typedef typename Symmetry::qType qType;
typedef typename MatrixType::Scalar Scalar;
	
	Multipede(){}
	
	/**
	 * Converts a Biped to a Tripod, adding a trivial middle leg equal to the vacuum.
	 * Made explicit to prohibit automatic conversion between Biped and Multipede.
	 */
	explicit Multipede (const Biped<Symmetry,MatrixType> &B, qType Q = Symmetry::qvacuum())
	{
		assert(Nlegs == 3);
		for (size_t q=0; q<B.dim; ++q)
		{
			// assert(B.in[q] == B.out[q]);
			if (Symmetry::triangle(qarray3<Symmetry::Nq>{B.in[q], Q, B.out[q]}))
			{
				boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[1][1]);
				Mtmpvec[0][0] = B.block[q];
				push_back(qarray3<Symmetry::Nq>{B.in[q], B.out[q], Q}, Mtmpvec);
			}
		}
	}
	
	/**Const reference to the number of legs \p Nlegs */
	inline constexpr size_t rank() const {return Nlegs;}
	
	///@{
	/**
	 * Convenience access to the amount of blocks.
	 * Equal to either of the following: \p index.size(), \p block.size()
	 */
	size_t dim = 0;
	inline std::size_t size() const {return dim;}
	
	/**
	 * Vector of all incoming quantum numbers.
	 * The entries are arrays of size \p Nlegs. The sorting is according to the following convention:
	 * 1. incoming 2. outgoing 3. middle (\p Nlegs=3)
	 * 1. incoming 2. outgoing 3. bottom 4. top (\p Nlegs=4)
	 * The middle, bottom and top quantum numbers are always outgoing for the left transfer matrix 
	 * and incoming for the right transfer matrix in matrix element calculations.
	*/
	vector<std::array<qType,Nlegs> > index;
	
	/**
	 * Vector of quantum number blocks.
	 * The matrix \p block[q] is characterized by the quantum number array \p index[q]. 
	 * Since the handling of template-sized arrays is no fun at all, 
	 * the implementation is done in the following way: \p block[q] is always a boost \p multi_array of dimension \p LEGLIMIT 
	 * which is set = 2. Tripods need only one dimension (two are already absorbed into \p MatrixType), 
	 * therefore the rest needs to be set to 1 and the access goes by \p T.block[q][a][0]. 
	 * \p LEGLIMIT can be increased and some code adjustment can be made (filling in dummy "[0]") if bigger tensors are required.
	 */
	vector<boost::multi_array<MatrixType,LEGLIMIT> > block;
	
	/**
	 * Dictionary allowing one to find the index of \p block for a given array of \p Nlegs quantum numbers in \f$O(1)\f$ operations 
	 * without looping over the blocks.
	 * For the ordering convention see Multipede::index.
	 */
	unordered_map<std::array<qType,Nlegs>,size_t> dict; // key format: {qin,qout,qmid}
	///@}
	
	///@{
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=MB) const;
	
	/**Prints Multipede<Nlegs,Symmetry,MatrixType>::dict into a string.*/
	string dict_info() const;
	
	string print (const bool &SHOW_MATRICES=false, const size_t &precision=3) const;
//	void rebuild_dict();
	///@}
	
	///@{
	/**Deletes the contents of \p index, \p block, \p dict.*/
	void clear();
	
	/**Sets all matrices in Multipede<Nlegs,Symmetry,MatrixType>::block to zero, preserving the rows and columns.*/
	void setZero();
	
	/**
	 * Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to the vacuum (all of them).
	 * Needed in for the transfer matrix to the first site in matrix element calculations.
	 */
	void setVacuum();
	
	/**
	 * Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers according to the input \p Q.
	 * Needed for the transfer matrix from the last site in matrix element calculations.
	 */
	void setTarget (std::array<qType,Nlegs> Q);
	
	void setTarget (vector<std::array<qType,Nlegs> > Q);
	
	/***/
	void setIdentity (size_t Drows, size_t Dcols, size_t amax=1, size_t bmax=1);
	
	void setIdentity (size_t amax, size_t bmax, const Qbasis<Symmetry> &base, const qarray<Symmetry::Nq> &Q=Symmetry::qvacuum());
	
	void addScale (const Scalar &factor, const Multipede<Nlegs,Symmetry,MatrixType> &Mrhs);
	///@}
	
	///@{
	void save (string filename, bool PRINT=false) const;
	
	void load (string filename, bool PRINT=false);
	///@}
	
	///@{
	/**
	 * Convenience function to return a quantum number of the block \p q to preserve the sanity of the programmer.
	 * For the naming convention see Multipede::index.
	 */
	inline qType in  (size_t q) const {return index[q][0];}
	inline qType out (size_t q) const {return index[q][1];}
	inline qType mid (size_t q) const {return index[q][2];}
	
	inline qType bot (size_t q) const {return index[q][2];}
	inline qType top (size_t q) const {return index[q][3];}
	///@}
	
	///@{
	/**
	 * Adds a new block to the tensor specified by the incoming quantum numbers \p quple.
	 * \warning Does not check whether the block for these quantum numbers already exists.
	 */
	void push_back (std::array<qType,Nlegs> quple, const boost::multi_array<MatrixType,LEGLIMIT> &M);
	
	/**
	 * Adds a new block to the tensor specified by the initializer list \p qlist (must be of size \p Nlegs).
	 * For the ordering convention see Multipede::index.
	 * \warning Does not check whether the block for these quantum numbers already exists.
	 */
	void push_back (std::initializer_list<qType> qlist, const boost::multi_array<MatrixType,LEGLIMIT> &M);
	
	void insert (std::pair<qType,size_t> ab, const Multipede<Nlegs,Symmetry,MatrixType> &Trhs);
	///@}
	
	///@{
	/** Casts tensors to \p OtherMatrixType, i.e. usually from real to complex.*/
	template<typename OtherMatrixType>
	Multipede<Nlegs,Symmetry,OtherMatrixType> cast() const
	{
		Multipede<Nlegs,Symmetry,OtherMatrixType> Vout;
		
		Vout.dict = dict;
		Vout.block.clear();
		Vout.block.resize(block.size());
		Vout.index = index;
		Vout.dim = dim;
		
		for (size_t q=0; q<dim; ++q)
		{
			Vout.block[q].resize(boost::extents[block[q].shape()[0]][block[q].shape()[1]]);
			for (size_t a=0; a<block[q].shape()[0]; ++a)
			for (size_t b=0; b<block[q].shape()[1]; ++b)
			{
				Vout.block[q][a][b] = block[q][a][b].template cast<typename OtherMatrixType::Scalar>();
			}
		}
		
		return Vout;
	}
	
	// deprecated, can be deleted later:
	/** Shifts \p qin and \p qout by \p Q, \p qmid is unchanged*/
//	void shift_Qin (const qarray<Symmetry::Nq> &Q);
	/** Shifts \p qmid and \p qout by \p Q, \p qin is unchanged*/
//	void shift_Qmid (const qarray<Symmetry::Nq> &Q);
	
	Scalar compare (const Multipede<Nlegs,Symmetry,MatrixType> &Mrhs) const;
	
	/** Takes Biped-slice from a Tripod over the middle quantum number \p qslice.*/
	Biped<Symmetry,MatrixType> BipedSliceQmid (qType qslice = Symmetry::qvacuum()) const;
	///@}
	
	// Needs to be implemented explicitly because multi_array doesn't resize when assigning A=B.
	Multipede<Nlegs,Symmetry,MatrixType>& operator= (const Multipede<Nlegs,Symmetry,MatrixType> &Vin)
	{
		dict = Vin.dict;
		block.clear();
		block.resize(Vin.block.size());
		index = Vin.index;
		dim = Vin.dim;
		
		for (size_t q=0; q<dim; ++q)
		{
			block[q].resize(boost::extents[Vin.block[q].shape()[0]][Vin.block[q].shape()[1]]);
			for (size_t a=0; a<block[q].shape()[0]; ++a)
			for (size_t b=0; b<block[q].shape()[1]; ++b)
			{
				block[q][a][b] = Vin.block[q][a][b];
			}
		}
		
		return *this;
	}
};

template<size_t Nlegs, typename Symmetry, typename MatrixType>
Multipede<Nlegs,Symmetry,MatrixType> operator- (const Multipede<Nlegs,Symmetry,MatrixType> &M1, const Multipede<Nlegs,Symmetry,MatrixType> &M2)
{
	Multipede<Nlegs,Symmetry,MatrixType> Mout;
	for (size_t q=0; q<M1.dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {M1.in(q), M1.out(q), M1.mid(q)};
		auto it = M2.dict.find(quple);
		boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[M1.block[q].shape()[0]][1]);
		for (size_t a=0; a<M1.block[q].shape()[0]; ++a)
		{
			Mtmpvec[a][0] = M1.block[q][a][0]-M2.block[it->second][a][0];
		}
		Mout.push_back(quple, Mtmpvec);
	}
	return Mout;
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
addScale (const Scalar &factor, const Multipede<Nlegs,Symmetry,MatrixType> &Mrhs)
{
	if (abs(factor) < 1.e-14) {return;}
	vector<qarray3<Symmetry::Nq> > matching_blocks;
	Multipede<Nlegs,Symmetry,MatrixType> Mout;
	
	for (size_t q=0; q<dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {in(q), out(q), mid(q)};
		auto it = Mrhs.dict.find(quple);
		if (it != Mrhs.dict.end())
		{
			matching_blocks.push_back(quple);
		}
		
		for (size_t a=0; a<block[q].shape()[0]; ++a)
		{
			MatrixType Mtmp;
			if (it != Mrhs.dict.end())
			{
				assert(block[q].shape()[0] == Mrhs.block[it->second].shape()[0]);
				
				if (block[q][a][0].size() != 0 and Mrhs.block[it->second][a][0].size() != 0)
				{
//					cout << "M1+factor*Mrhs" << endl;
					Mtmp = block[q][a][0] + factor * Mrhs.block[it->second][a][0]; // M1+factor*Mrhs
				}
				else if (block[q][a][0].size() == 0 and Mrhs.block[it->second][a][0].size() != 0)
				{
//					cout << "0+factor*Mrhs" << endl;
					Mtmp = factor * Mrhs.block[it->second][a][0]; // 0+factor*Mrhs
				}
				else if (block[q].size() != 0 and Mrhs.block[it->second][a][0].size() == 0)
				{
//					cout << "M1+0" << endl;
					Mtmp = block[q][a][0]; // M1+0
				}
				// else: block[q].size() == 0 and Mrhs.block[it->second][a][0].size() == 0 -> do nothing -> Mtmp.size() = 0
			}
			else
			{
//				cout << "M1+0" << endl;
				Mtmp = block[q][a][0]; // M1+0
			}
			
			if (Mtmp.size() != 0)
			{
				auto ip = Mout.dict.find(quple);
				if (ip != Mout.dict.end())
				{
					assert(1==0 and "Error in Multipede::addScale, block already exists!");
				}
				else
				{
					boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[block[q].shape()[0]][1]);
					Mtmpvec[a][0] = Mtmp;
					Mout.push_back(quple, Mtmpvec);
				}
			}
		}
	}
	
	// Mrhs has additional blocks which are not in *this
	if (matching_blocks.size() != Mrhs.dim)
	{
		for (size_t qrhs=0; qrhs<Mrhs.size(); ++qrhs)
		{
			qarray3<Symmetry::Nq> quple = {Mrhs.in(qrhs), Mrhs.out(qrhs), Mrhs.mid(qrhs)};
			auto it = find(matching_blocks.begin(), matching_blocks.end(), quple);
			if (it == matching_blocks.end())
			{
				for (size_t a=0; a<Mrhs.block[qrhs].shape()[0]; ++a)
				{
					if (Mrhs.block[qrhs][a][0].size() != 0)
					{
						MatrixType Mtmp = factor * Mrhs.block[qrhs][a][0]; // 0+factor*Mrhs
						
						auto ip = Mout.dict.find(quple);
						if (ip != Mout.dict.end())
						{
							assert(1==0 and "Error in Multipede::addScale, block already exists!");
						}
						else
						{
							boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[Mrhs.block[qrhs].shape()[0]][1]);
							Mtmpvec[a][0] = Mtmp;
							Mout.push_back(quple, Mtmpvec);
						}
					}
				}
			}
		}
	}
	
	*this = Mout;
}

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
	MatrixType Mtmp(1,1);
	Mtmp << 1.;
	
	boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[1][1]);
	Mtmparray[0][0] = Mtmp;
	
	push_back(Q,Mtmparray);
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
setTarget (vector<std::array<qType,Nlegs> > Q)
{
	MatrixType Mtmp(1,1);
	Mtmp << 1.;
	
	boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[1][1]);
	Mtmparray[0][0] = Mtmp;
	
	for (const auto &Qval:Q)
	{
		push_back(Qval,Mtmparray);
	}
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
void Multipede<Nlegs,Symmetry,MatrixType>::
setIdentity (size_t amax, size_t bmax, const Qbasis<Symmetry> &base, const qarray<Symmetry::Nq> &Q)
{
	static_assert(Nlegs == 3);
	if (amax == 0 or bmax ==0) {return;}
	
	for (size_t q=0; q<base.Nq(); ++q)
	{
		qarray3<Symmetry::Nq> quple = {base[q], base[q], Q};
//		cout << "quple=" << quple[0] << ", " << quple[1] << ", " << quple[2] << endl;
//		qarray3<Symmetry::Nq> checkquple = {base[q], Q, base[q]};
//		if (!Symmetry::triangle(checkquple)) {continue;}
		
		boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[amax][bmax]);
		for (size_t a=0; a<amax; ++a)
		for (size_t b=0; b<bmax; ++b)
		{
			MatrixType Mtmp(base.inner_dim(base[q]), base.inner_dim(base[q]));
			Mtmp.setIdentity();
			Mtmparray[a][b] = Mtmp;
		}
		push_back(quple, Mtmparray);
	}
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
insert (std::pair<qType,size_t> ab, const Multipede<Nlegs,Symmetry,MatrixType> &Trhs)
{
	static_assert(Nlegs == 3);
	
	for (size_t q=0; q<Trhs.dim; ++q)
	{
		if (Trhs.mid(q) != ab.first) {continue;}
		
		if (Trhs.block[q][ab.second][0].size() == 0) {continue;}
		qarray3<Symmetry::Nq> quple = {Trhs.in(q), Trhs.out(q), Trhs.mid(q)};
		
		auto it = dict.find(quple);
		if (it != dict.end())
		{
			if (block[it->second][ab.second][0].rows() == Trhs.block[q][ab.second][0].rows() and
			    block[it->second][ab.second][0].cols() == Trhs.block[q][ab.second][0].cols())
			{
//				cout << termcolor::green << "operator+= in insert" << termcolor::reset << endl;
				block[it->second][ab.second][0] += Trhs.block[q][ab.second][0];
			}
			else
			{
//				cout << termcolor::blue << "operator= in insert" << termcolor::reset << "\t" << block[it->second][ab][0].rows() << "x" << block[it->second][ab][0].cols() << endl;
				block[it->second][ab.second][0] = Trhs.block[q][ab.second][0];
			}
		}
		else
		{
//			cout << termcolor::red << "push_back in insert" << termcolor::reset << endl;
			boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[Trhs.block[q].shape()[0]][1]);
			Mtmparray[ab.second][0] = Trhs.block[q][ab.second][0];
			push_back(quple, Mtmparray);
		}
	}
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
std::string Multipede<Nlegs,Symmetry,MatrixType>::
print (const bool &SHOW_MATRICES, const std::size_t &precision) const
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
			tt << Sym::format<Symmetry>(index[nu][q]);
			if (q==index[nu].size()-1) {tt << ")";} else {tt << ",";}
		}
		uu << block[nu].shape()[0] << ":[";
		// uu << block[nu][0][0].cols() << "x" << block[nu][0][0].rows() << "x" << block[nu].shape()[0];
		for (std::size_t q=0; q<block[nu].shape()[0]; q++)
		{
//			if(block[nu][q][0].cols() != 0 or block[nu][q][0].rows() != 0)
			{
				uu << "(" << block[nu][q][0].cols() << "x" << block[nu][q][0].rows() << ")";
				if(q==block[nu].shape()[0]-1) {uu << "";} else {uu << ",";}
			}
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
		out << termcolor::blue << termcolor::underline << "A-tensors:" << termcolor::reset << std::endl;
		for (std::size_t nu=0; nu<dim; nu++)
		{
			out << termcolor::blue << "ν=" << nu << std::endl;
			for (std::size_t q=0; q<block[nu].shape()[0]; q++)
			{
				if(block[nu][q][0].size() == 0) {continue;}
				out << termcolor::green << "q=" << q << endl << std::setprecision(precision) << std::fixed << block[nu][q][0] << std::endl;
			}
		}
	}
	
	return out.str();
#endif
}

//template<size_t Nlegs, typename Symmetry, typename MatrixType>
//void Multipede<Nlegs,Symmetry,MatrixType>::
//shift_Qin (const qarray<Symmetry::Nq> &Q)
//{
//	assert(Nlegs == 3);
//	
//	auto index_tmp = index;
//	auto block_tmp = block;
//	auto dim_tmp = dim;
//	
//	index.clear();
//	block.clear();
//	dict.clear();
//	dim = 0;
//	
//	for (size_t q=0; q<dim_tmp; ++q)
//	{
//		push_back({index_tmp[q][0]+Q, index_tmp[q][1]+Q, index_tmp[q][2]}, block_tmp[q]);
//	}
//}

//template<size_t Nlegs, typename Symmetry, typename MatrixType>
//void Multipede<Nlegs,Symmetry,MatrixType>::
//shift_Qmid (const qarray<Symmetry::Nq> &Q)
//{
//	assert(Nlegs == 3);
//	
//	auto index_tmp = index;
//	auto block_tmp = block;
//	auto dim_tmp = dim;
//	
//	index.clear();
//	block.clear();
//	dict.clear();
//	dim = 0;
//	
//	for (size_t q=0; q<dim_tmp; ++q)
//	{
//		push_back({index_tmp[q][0], index_tmp[q][1]+Q, index_tmp[q][2]+Q}, block_tmp[q]);
//	}
//}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
typename MatrixType::Scalar Multipede<Nlegs,Symmetry,MatrixType>::
compare (const Multipede<Nlegs,Symmetry,MatrixType> &Mrhs) const 
{
	double res = 0;
	for (size_t q=0; q<dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {in(q), out(q), mid(q)};
		auto it = Mrhs.dict.find(quple);
		if (it == Mrhs.dict.end()) {return std::numeric_limits<typename MatrixType::Scalar>::infinity();}
		for (size_t a=0; a<block[q].shape()[0]; ++a)
		{
			res += (block[q][a][0]-Mrhs.block[it->second][a][0]).norm();
		}
	}
	return res;
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
Biped<Symmetry,MatrixType> Multipede<Nlegs,Symmetry,MatrixType>::
BipedSliceQmid (qType qslice) const
{
	assert(Nlegs == 3);
	
	Biped<Symmetry,MatrixType> Bout;
	for (size_t q=0; q<dim; ++q)
	for (size_t a=0; a<block[q].shape()[0]; ++a)
	{
		if (mid(q) == qslice)
		{
			Bout.push_back(in(q), out(q), block[q][a][0]);
		}
	}
	return Bout;
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
save (string filename, bool PRINT) const 
{
	filename += ".h5";
	if (PRINT) lout << termcolor::green << "Saving Multipede to: " << filename << termcolor::reset << std::endl;
	remove(filename.c_str());
	HDF5Interface target(filename, WRITE);
	
	target.save_scalar<size_t>(dim, "dim", "");
	
	for (size_t q=0; q<dim; ++q)
	{
		target.save_scalar<size_t>(block[q].shape()[0], make_string("dima_q=",q), "");
		target.save_scalar<size_t>(block[q].shape()[1], make_string("dimb_q=",q), "");
		//cout << "q=" << q << ", L.block[q].shape()[0]=" << L.block[q].shape()[0] << ", L.block[q].shape()[1]=" << L.block[q].shape()[1] << endl;
	}
	
	for (size_t q=0; q<dim; ++q)
	for (size_t a=0; a<block[q].shape()[0]; ++a)
	for (size_t b=0; b<block[q].shape()[1]; ++b)
	{
		target.save_matrix(block[q][a][b], make_string("block_q=",q,"_a=",a,"_b=",b), "");
	}
	
	MatrixXi Min(dim,Symmetry::Nq);
	MatrixXi Mout(dim,Symmetry::Nq);
	MatrixXi Mmid(dim,Symmetry::Nq);
	
	for (int i=0; i<dim; ++i)
	for (int q=0; q<Symmetry::Nq; ++q)
	{
		Min(i,q) = in(i)[q];
		Mout(i,q) = out(i)[q];
		Mmid(i,q) = mid(i)[q];
	}
	
	target.save_matrix<int>(Min, "in", "");
	target.save_matrix<int>(Mout, "out", "");
	target.save_matrix<int>(Mmid, "mid", "");
	
	target.close();
	
	print();
}

template<size_t Nlegs, typename Symmetry, typename MatrixType>
void Multipede<Nlegs,Symmetry,MatrixType>::
load (string filename, bool PRINT)
{
	filename += ".h5";
	if (PRINT) lout << termcolor::green << "Loading Multipede from: " << filename << termcolor::reset << std::endl;
	HDF5Interface target(filename, READ);
	
	target.load_scalar<size_t>(dim, "dim", "");
	
	block.clear();
	block.resize(dim);
	
	for (size_t q=0; q<dim; ++q)
	{
		size_t dima, dimb;
		target.load_scalar<size_t>(dima, make_string("dima_q=",q), "");
		target.load_scalar<size_t>(dimb, make_string("dimb_q=",q), "");
		block[q].resize(boost::extents[dima][dimb]);
		//cout << "q=" << q << ", dima=" << dima << ", dimb=" << dimb << endl;
	}
	
	for (size_t q=0; q<dim; ++q)
	for (size_t a=0; a<block[q].shape()[0]; ++a)
	for (size_t b=0; b<block[q].shape()[1]; ++b)
	{
		target.load_matrix(block[q][a][b], make_string("block_q=",q,"_a=",a,"_b=",b), "");
	}
	
	MatrixXi Min, Mout, Mmid;
	target.load_matrix<int>(Min, "in", "");
	target.load_matrix<int>(Mout, "out", "");
	target.load_matrix<int>(Mmid, "mid", "");
	//cout << in.rows() << "\t" << out.rows() << "\t" << mid.rows() << "\t" << L.dim << endl;
	assert(Min.rows() != 0);
	
	index.clear();
	index.resize(dim);
	dict.clear();
	
	for (int i=0; i<dim; ++i)
	{
		std::array<qType,Nlegs> quple;
		for (int q=0; q<Symmetry::Nq; ++q)
		{
			quple[0][q] = Min(i,q);
			quple[1][q] = Mout(i,q);
			quple[2][q] = Mmid(i,q);
		}
		index[i] = quple;
		dict.insert({quple,i});
	}
	
	target.close();
	
	print();
}

#endif
