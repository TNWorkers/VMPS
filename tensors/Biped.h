#ifndef STRAWBERRY_BIPED
#define STRAWBERRY_BIPED

#include <unordered_map>

#include "macros.h"
#include "PolychromaticConsole.h" // from HELPERS
#include "MemCalc.h" // from HELPERS

#include "DmrgExternal.h"
#include "symmetry/functions.h"

namespace contract {
	enum MODE {UNITY,OORR,DOT};
}

// using namespace std;

/**
 * Tensor with two legs and quantum number blocks.
 * One could have used a general tensor, but the special case of two legs is hardcoded to preserve the sanity of the programmer. 
 * For the general tensor see Multipede.
 * @describe_Symmetry
 * @describe_MatrixType
 */
template<typename Symmetry, typename MatrixType_>
struct Biped
{
private:
	typedef typename Symmetry::qType qType;
	typedef Eigen::Index Index;
public:
	typedef MatrixType_ MatrixType;
private:
	typedef typename MatrixType::Scalar Scalar;
	
public:
	
	Biped(){dim=0;}
	
	///@{
	/**
	 * Convenience access to the amount of blocks.
	 * Equal to either of the following: \p in.size(), \p out.size(), \p block.size()
	 */
	std::size_t dim;
	inline std::size_t size() const {return dim;}
	inline void plusplus() {++dim;}
	
	/**Vector of all incoming quantum numbers.*/
	std::vector<qType> in;
	
	/**Vector of all outgoing quantum numbers.*/
	std::vector<qType> out;
	
	/**
	 * Vector of quantum number blocks.
	 * The matrix \p block[q] is characterized by the incoming quantum number \p in[q] and the outgoing quantum number \p out[q]
	 */
	std::vector<MatrixType_> block;
	///@}
	
	///@{
	/**
	 *Dictionary allowing one to find the index of \p block for a given array of two quantum numbers \p qin, \p qout in \f$O(1)\f$ 
	 * operations without looping over the blocks.
	 */
	std::unordered_map<std::array<qType,2>,std::size_t> dict; // key format: {qin,qout}
	
	///@{
	/**Returns an Eigen vector of size \p dim containing all Matrix rows for every block nu.*/
	Eigen::VectorXi rows() const;
	/**Returns an Eigen vector of size \p dim containing all Matrix cols for every block nu.*/
	Eigen::VectorXi cols() const;
	/**Returns an Eigen vector of size \p dim containing all Matrix norm for every block nu.*/
	Eigen::VectorXi norm() const;
	///@}
	
	/**Prints the whole tensor, formatting the quantum numbers */
	std::string formatted () const;
	
	/**
	 * Function to print the full Biped 
	 * \param SHOW_MATRICES : if true, all the block-matrices are printed.
	 * \param precision : precision for the tensor components
	 */
	std::string print (const bool SHOW_MATRICES=false , const std::size_t precision=3 ) const;
	
	/**Prints Biped<Symmetry,MatrixType>::dict into a string.*/
	std::string print_dict() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=MB) const;
	///@}
	
	///@{
	/**Deletes the contents of \p in, \p out, \p block, \p dict.*/
	void clear();
	
	/**Sets all matrices in Biped<Symmetry,MatrixType>::block to zero, preserving the rows and columns.*/
	void setZero();
	
	/**Sets all matrices in Biped<Symmetry,MatrixType>::block to random values, preserving the rows and columns.*/
	void setRandom();
	
	/**
	 * Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to the vacuum (both \p in & \p out).
	 * Needed in for the transfer matrix to the first site in overlap calculations.
	 */
	void setVacuum();
	
	/**
	 * Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to \p Qtot (both \p in & \p out).
	 * Needed in for the transfer matrix from the last site in overlap calculations.
	 */
	void setTarget (qType Qtot);
	///@}
	
	///@{
	/**
	 * Returns the adjoint tensor where all the block matrices are adjoint and the quantum number arrows are flipped: 
	 * \p in \f$\to\f$ \p out and vice versa.
	 */
	Biped<Symmetry,MatrixType_> adjoint() const;
	
	/**
	 * Adds another tensor to the current one. 
	 * If quantum numbers match, the block is updated (block rows and columns must match), otherwise a new block is created.
	 */
	Biped<Symmetry,MatrixType_>& operator+= (const Biped<Symmetry,MatrixType_> &Arhs);
	
	/**
	 * This functions perform a contraction of \p this and \p A, which is a standard Matrix multiplication in this case.
	 * \param A : other Biped which is contracted together with \p this.
	 * \param MODE
	 */
	Biped<Symmetry,MatrixType_> contract(const Biped<Symmetry,MatrixType_> &A, const contract::MODE MODE = contract::MODE::UNITY) const;
	///@}
	
	/**Takes the trace of the Biped. Only useful if this Biped is really a matrix from symmetry perspektive (q_in = q_out in all blocks).*/
	Scalar trace() const;
	
	///@{
	/**
	 * Adds a new block to the tensor specified by the incoming quantum number \p qin and the outgoing quantum number \p qout.
	 * \warning Does not check whether the block for these quantum numbers already exists.
	 */
	void push_back (qType qin, qType qout, const MatrixType_ &M);
	
	/**
	 * Adds a new block to the tensor specified by the 2-array of quantum numbers \p quple.
	 * The ordering convention is: \p in, \p out.
	 * \warning Does not check whether the block for these quantum numbers already exists.
	 */
	void push_back (std::array<qType,2> quple, const MatrixType_ &M);
	///@}
};

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
clear()
{
	in.clear();
	out.clear();
	block.clear();
	dict.clear();
	dim = 0;
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setZero()
{
	for (std::size_t q=0; q<dim; ++q) {block[q].setZero();}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setRandom()
{
	for (std::size_t q=0; q<dim; ++q) {block[q].setRandom();}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setVacuum()
{
	MatrixType_ Mtmp(1,1); Mtmp << 1.;
	push_back(Symmetry::qvacuum(), Symmetry::qvacuum(), Mtmp);
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setTarget (qType Qtot)
{
	MatrixType_ Mtmp(1,1); Mtmp << 1.;
	push_back(Qtot, Qtot, Mtmp);
}

template<typename Symmetry, typename MatrixType_>
Eigen::VectorXi Biped<Symmetry,MatrixType_>::
rows () const
{
	Eigen::VectorXi Vout(size());
	for (std::size_t nu=0; nu<size(); nu++) { Vout[nu] = block[nu].rows(); }
	return Vout;
}

template<typename Symmetry, typename MatrixType_>
Eigen::VectorXi Biped<Symmetry,MatrixType_>::
cols () const
{
	Eigen::VectorXi Vout(size());
	for (std::size_t nu=0; nu<size(); nu++) { Vout[nu] = block[nu].cols(); }
	return Vout;
}

template<typename Symmetry, typename MatrixType_>
Eigen::VectorXi Biped<Symmetry,MatrixType_>::
norm () const
{
	Eigen::VectorXi Vout(size());
	for (std::size_t nu=0; nu<size(); nu++) { Vout[nu] = block[nu].norm(); }
	return Vout;
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
push_back (qType qin, qType qout, const MatrixType_ &M)
{
	push_back({qin,qout},M);
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
push_back (std::array<qType,2> quple, const MatrixType_ &M)
{
	in.push_back(quple[0]);
	out.push_back(quple[1]);
	block.push_back(M);
	dict.insert({quple, dim});
	++dim;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
adjoint() const
{
	Biped<Symmetry,MatrixType_> Aout;
	Aout.dim = dim;
	Aout.in = out;
	Aout.out = in;
	
	// new dict with reversed keys {qin,qout}->{qout,qin}
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		auto qin  = get<0>(it->first);
		auto qout = get<1>(it->first);
		Aout.dict.insert({{qout,qin}, it->second});
	}
	
	Aout.block.resize(dim);
	for (std::size_t q=0; q<dim; ++q)
	{
		Aout.block[q] = block[q].adjoint();
	}
	
	return Aout;
}

template<typename Symmetry, typename MatrixType_>
typename MatrixType_::Scalar Biped<Symmetry,MatrixType_>::
trace() const
{
	typename MatrixType_::Scalar res=0.;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		assert(in[nu] == out[nu] and "A trace can only be taken from a matrix");
		res += block[nu].trace()*Symmetry::coeff_dot(in[nu]);
	}
	return res;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_>& Biped<Symmetry,MatrixType_>::operator+= (const Biped<Symmetry,MatrixType_> &Arhs)
{
	std::vector<std::size_t> addenda;
	
	for (std::size_t q=0; q<Arhs.dim; ++q)
	{
		std::array<qType,2> quple = {Arhs.in[q], Arhs.out[q]};
		auto it = dict.find(quple);
		
		if (it != dict.end())
		{
			block[it->second] += Arhs.block[q];
		}
		else
		{
			addenda.push_back(q);
		}
	}
	
	for (size_t q=0; q<addenda.size(); ++q)
	{
		push_back(Arhs.in[addenda[q]], Arhs.out[addenda[q]], Arhs.block[addenda[q]]);
	}
	
	return *this;
}
template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
contract(const Biped<Symmetry,MatrixType_> &A, const contract::MODE MODE) const
{
	Biped<Symmetry,MatrixType_> Ares;
	Scalar factor_cgc;
	for (std::size_t q1=0; q1<this->size(); ++q1)
		for (std::size_t q2=0; q2<A.size(); ++q2)
		{
			if (this->out[q1] == A.in[q2])
			{
				if (this->in[q1] == A.out[q2])
				{
					if (this->block[q1].rows() != 0 and A.block[q2].rows() != 0)
					{
						factor_cgc = Scalar(1);
						if ( MODE == contract::MODE::OORR )
						{
							factor_cgc = Symmetry::coeff_rightOrtho(this->out[q1],this->in[q2]);
						}
						else if ( MODE == contract::MODE::DOT )
						{
							factor_cgc = Symmetry::coeff_dot(this->out[q1]);
						}
						if ( auto it = Ares.dict.find({{this->in[q1],A.out[q2]}}); it == Ares.dict.end() )
						{
							Ares.push_back(this->in[q1], A.out[q2], factor_cgc*this->block[q1]*A.block[q2]);
						}
						else
						{
							Ares.block[it->second] += factor_cgc*this->block[q1]*A.block[q2];
						}
					}
				}
			}
		}
	return Ares;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> operator* (const Biped<Symmetry,MatrixType_> &A1, const Biped<Symmetry,MatrixType_> &A2)
{
	Biped<Symmetry,MatrixType_> Ares;
	for (std::size_t q1=0; q1<A1.dim; ++q1)
	for (std::size_t q2=0; q2<A2.dim; ++q2)
	{
		if (A1.out[q1] == A2.in[q2])
		{
			if (A1.block[q1].rows() != 0 and A2.block[q2].rows() != 0)
			{
				Ares.push_back(A1.in[q1], A2.out[q2], A1.block[q1]*A2.block[q2]);
			}
		}
	}
	return Ares;
}

template<typename Symmetry, typename MatrixType_, typename Scalar>
Biped<Symmetry,MatrixType_> operator* (const Scalar &alpha, const Biped<Symmetry,MatrixType_> &A)
{
	Biped<Symmetry,MatrixType_> Ares = A;
	for (std::size_t q=0; q<Ares.dim; ++q)
	{
		Ares.block[q] *= alpha;
	}
	return Ares;
}

// template<typename Symmetry, typename MatrixType_>
// Biped<Symmetry,MatrixType_> operator+ (const Biped<Symmetry,MatrixType_> &A1, const Biped<Symmetry,MatrixType_> &A2)
// {
// 	Biped<Symmetry,MatrixType_> Ares = A1;
// 	Ares += A2;
// 	return Ares;
// }

template<typename Symmetry, typename MatrixType_>
string Biped<Symmetry,MatrixType_>::
print_dict() const
{
	std::stringstream ss;
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		ss << "in:" << get<0>(it->first) << "\tout:" << get<1>(it->first) << "\t→\t" << it->second << endl;
	}
	return ss.str();
}

template<typename Symmetry, typename MatrixType_>
double Biped<Symmetry,MatrixType_>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (std::size_t q=0; q<dim; ++q)
	{
		res += calc_memory(block[q], memunit);
	}
	return res;
}

template<typename Symmetry, typename MatrixType_>
double Biped<Symmetry,MatrixType_>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
	res += 2. * 2. * Symmetry::Nq * calc_memory<int>(dim, memunit); // in,out; dict.keys
	res += Symmetry::Nq * calc_memory<std::size_t>(dim, memunit); // dict.vals
	return res;
}

template<typename Symmetry, typename MatrixType_>
std::string Biped<Symmetry,MatrixType_>::
formatted () const
{
	std::stringstream ss;
	ss << "•Biped(" << dim << "):" << endl;
	for (std::size_t q=0; q<dim; ++q)
	{
		ss << "  [" << q << "]: " << Sym::format<Symmetry>(in[q]) << "→" << Sym::format<Symmetry>(out[q]);
		ss << ":" << endl;
		ss << "   " << block[q];
		if (q!=dim-1) {ss << endl;}
	}
	return ss.str();
}

template<typename Symmetry, typename MatrixType_>
std::string Biped<Symmetry,MatrixType_>::
print ( const bool SHOW_MATRICES, const std::size_t precision ) const
{
#ifdef HELPERS_IO_TABLE
	std::stringstream out_string;

	TextTable t( '-', '|', '+' );
	t.add("ν");
	t.add("Q_ν");
	t.add("A_ν");
	t.endOfRow();
	for (std::size_t nu=0; nu<size(); nu++)
	{
		std::stringstream ss,tt,uu;
		ss << nu;
		tt << "(" << in[nu] << "," << out[nu] << ")";
		uu << block[nu].rows() << "x" << block[nu].cols();
		t.add(ss.str());
		t.add(tt.str());
		t.add(uu.str());
		t.endOfRow();
	}
	t.setAlignment( 0, TextTable::Alignment::RIGHT );
	out_string << t;

	if (SHOW_MATRICES)
	{
		out_string << TCOLOR(GREEN) << "\e[4mA-tensors:\e[0m" << std::endl;
		for (std::size_t nu=0; nu<dim; nu++)
		{
			out_string << TCOLOR(GREEN) << "ν=" << nu << std::endl << std::setprecision(precision) << std::fixed << block[nu] << std::endl;
		}
		out_string << TCOLOR(BLACK) << std::endl;
	}
	return out_string.str();
#else
	return "Can't print. Table Library is missing.";
#endif
}

/**Adds two Bipeds block- and coefficient-wise.*/
template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> operator+ (const Biped<Symmetry,MatrixType_> &M1, const Biped<Symmetry,MatrixType_> &M2)
{
	if (M1.size() < M2.size()) {return M2+M1;}
	std::vector<std::size_t> blocks_in_2nd_biped;
	
	Biped<Symmetry,MatrixType_> Mout;
	MatrixType_ Mtmp;
	for (std::size_t nu=0; nu<M1.size(); nu++)
	{
		auto it1 = M2.dict.find({{M1.in[nu],M1.out[nu]}});
		if ( it1 != M2.dict.end() )
		{
			blocks_in_2nd_biped.push_back(it1->second);
			Mtmp = M1.block[nu] + M2.block[it1->second];
		}
		else
		{
			Mtmp = M1.block[nu];
		}
		Mout.push_back({{M1.in[nu],M1.out[nu]}},Mtmp);
	}
	if (blocks_in_2nd_biped.size() != M2.size())
	{
		for(std::size_t nu=0; nu<M2.size(); nu++)
		{
			auto it = std::find(blocks_in_2nd_biped.begin(),blocks_in_2nd_biped.end(),nu);
			if(it == blocks_in_2nd_biped.end())
			{
				Mtmp = M2.block[nu];
				Mout.push_back({{M2.in[nu],M2.out[nu]}},Mtmp);
			}
		}
	}
	return Mout;
}

/**Subtracts two Bipeds block- and coefficient-wise.*/
template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> operator- (const Biped<Symmetry,MatrixType_> &M1, const Biped<Symmetry,MatrixType_> &M2)
{
	std::vector<std::size_t> blocks_in_2nd_biped;
	Biped<Symmetry,MatrixType_> Mout;
	MatrixType_ Mtmp;
	for (std::size_t nu=0; nu<M1.size(); nu++)
	{
		auto it1 = M2.dict.find({{M1.in[nu],M1.out[nu]}});
		if ( it1 != M2.dict.end() )
		{
			blocks_in_2nd_biped.push_back(it1->second);
			Mtmp = M1.block[nu] - M2.block[it1->second];
		}
		else
		{
			Mtmp = M1.block[nu];
		}
		Mout.push_back({{M1.in[nu],M1.out[nu]}},Mtmp);
	}
	if (blocks_in_2nd_biped.size() != M2.size())
	{
		for(std::size_t nu=0; nu<M2.size(); nu++)
		{
			auto it = std::find(blocks_in_2nd_biped.begin(),blocks_in_2nd_biped.end(),nu);
			if(it == blocks_in_2nd_biped.end())
			{
				Mtmp = -M2.block[nu];
				Mout.push_back({{M2.in[nu],M2.out[nu]}},Mtmp);
			}
		}
	}
	return Mout;
}

template<typename Symmetry, typename MatrixType_>
std::ostream& operator<< (std::ostream& os, const Biped<Symmetry,MatrixType_> &V)
{
	os << V.print(true,4);
	return os;
}

#endif
