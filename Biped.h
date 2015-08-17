#ifndef STRAWBERRY_BIPED
#define STRAWBERRY_BIPED

#include <unordered_map>

#include "qarray.h"
#include "DmrgExternalQ.h"
#include "MemCalc.h"

/**
Tensor with two legs and quantum number blocks.
One could have used a general tensor, but the special case of two legs is hardcoded to preserve the sanity of the programmer. For the general tensor see Multipede.
@describe_Nq
@describe_MatrixType*/
template<size_t Nq, typename MatrixType>
struct Biped
{
	Biped(){dim=0;}
	
	///@{
	/**Convenience access to the amount of blocks.
	Equal to either of the following: \p in.size(), \p out.size(), \p block.size()*/
	size_t dim;
	
	/**Vector of all incoming quantum numbers.*/
	vector<qarray<Nq> > in;
	
	/**Vector of all outgoing quantum numbers.*/
	vector<qarray<Nq> > out;
	
	/**Vector of quantum number blocks.
	The matrix \p block[q] is characterized by the incoming quantum number \p in[q] and the outgoing quantum number \p out[q]*/
	vector<MatrixType> block;
	///@}
	
	///@{
	/**Dictionary allowing one to find the index of \p block for a given array of two quantum numbers \p qin, \p qout in \f$O(1)\f$ operations without looping over the blocks.*/
	unordered_map<qarray2<Nq>,size_t> dict; // key format: {qin,qout}
	
	/**Prints the whole tensor, formatting the quantum numbers accoridng the function \p formatFunction.*/
	string formatted (string (*formatFunction)(qarray<Nq>)=noFormat) const;
	
	/**Prints Biped<Nq,MatrixType>::dict into a string.*/
	string print_dict() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=MB) const;
	///@}
	
	///@{
	/**Deletes the contents of \p in, \p out, \p block, \p dict.*/
	void clear();
	
	/**Sets all matrices in Biped<Nq,MatrixType>::block to zero, preserving the rows and columns.*/
	void setZero();
	
	/**Sets all matrices in Biped<Nq,MatrixType>::block to random values, preserving the rows and columns.*/
	void setRandom();
	
	/**Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to the vacuum (both \p in & \p out).
	Needed in for the transfer matrix to the first site in overlap calculations.*/
	
	void setVacuum();
	/**Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to \p Qtot (both \p in & \p out).
	Needed in for the transfer matrix from the last site in overlap calculations.*/
	
	void setTarget (qarray<Nq> Qtot);
	///@}
	
	///@{
	/**Returns the adjoint tensor where all the block matrices are adjoint and the quantum number arrows are flipped: \p in \f$\to\f$ \p out and vice versa.*/
	Biped<Nq,MatrixType> adjoint() const;
	
	/**Adds another tensor to the current one. If quantum numbers match, the block is updated (block rows and columns must match), otherwise a new block is created.*/
	Biped<Nq,MatrixType>& operator+= (const Biped<Nq,MatrixType> &Arhs);
	///@}
	
	///@{
	/**Adds a new block to the tensor specified by the incoming quantum number \p qin and the outgoing quantum number \p qout.
	\warning Does not check whether the block for these quantum numbers already exists.*/
	void push_back (qarray<Nq> qin, qarray<Nq> qout, const MatrixType &M);
	
	/**Adds a new block to the tensor specified by the 2-array of quantum numbers \p quple.
	The ordering convention is: \p in, \p out.
	\warning Does not check whether the block for these quantum numbers already exists.*/
	void push_back (qarray2<Nq> quple, const MatrixType &M);
	///@}
};

template<size_t Nq, typename MatrixType>
void Biped<Nq,MatrixType>::
clear()
{
	in.clear();
	out.clear();
	block.clear();
	dict.clear();
	dim = 0;
}

template<size_t Nq, typename MatrixType>
void Biped<Nq,MatrixType>::
setZero()
{
	for (size_t q=0; q<dim; ++q) {block[q].setZero();}
}

template<size_t Nq, typename MatrixType>
void Biped<Nq,MatrixType>::
setRandom()
{
	for (size_t q=0; q<dim; ++q) {block[q].setRandom();}
}

template<size_t Nq, typename MatrixType>
void Biped<Nq,MatrixType>::
setVacuum()
{
	MatrixType Mtmp(1,1); Mtmp << 1.;
	push_back(qvacuum<Nq>(), qvacuum<Nq>(), Mtmp);
}

template<size_t Nq, typename MatrixType>
void Biped<Nq,MatrixType>::
setTarget (qarray<Nq> Qtot)
{
	MatrixType Mtmp(1,1); Mtmp << 1.;
	push_back(Qtot, Qtot, Mtmp);
}

template<size_t Nq, typename MatrixType>
void Biped<Nq,MatrixType>::
push_back (qarray<Nq> qin, qarray<Nq> qout, const MatrixType &M)
{
	push_back(qarray2<Nq>{qin,qout},M);
}

template<size_t Nq, typename MatrixType>
void Biped<Nq,MatrixType>::
push_back (qarray2<Nq> quple, const MatrixType &M)
{
	in.push_back(quple[0]);
	out.push_back(quple[1]);
	block.push_back(M);
	dict.insert({quple, dim});
	++dim;
}

template<size_t Nq, typename MatrixType>
Biped<Nq,MatrixType> Biped<Nq,MatrixType>::
adjoint() const
{
	Biped<Nq,MatrixType> Aout;
	Aout.dim = dim;
	Aout.in = out;
	Aout.out = in;
	
	// new dict with reversed keys {qin,qout}->{qout,qin}
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		auto qin  = get<0>(it->first);
		auto qout = get<1>(it->first);
		Aout.dict.insert({qarray2<Nq>{qout,qin}, it->second});
	}
	
	Aout.block.resize(dim);
	for (size_t q=0; q<dim; ++q)
	{
		Aout.block[q] = block[q].adjoint();
	}
	
	return Aout;
}

template<size_t Nq, typename MatrixType>
Biped<Nq,MatrixType>& Biped<Nq,MatrixType>::operator+= (const Biped<Nq,MatrixType> &Arhs)
{
	vector<size_t> addenda;
	
	for (size_t q=0; q<Arhs.dim; ++q)
	{
		qarray2<Nq> quple = {Arhs.in[q], Arhs.out[q]};
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

template<size_t Nq, typename MatrixType>
Biped<Nq,MatrixType> operator* (const Biped<Nq,MatrixType> &A1, const Biped<Nq,MatrixType> &A2)
{
	Biped<Nq,MatrixType> Ares;
	for (size_t q1=0; q1<A1.dim; ++q1)
	for (size_t q2=0; q2<A2.dim; ++q2)
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

template<size_t Nq, typename MatrixType, typename Scalar>
Biped<Nq,MatrixType> operator* (const Scalar &alpha, const Biped<Nq,MatrixType> &A)
{
	Biped<Nq,MatrixType> Ares = A;
	for (size_t q=0; q<Ares.dim; ++q)
	{
		Ares.block[q] *= alpha;
	}
	return Ares;
}

template<size_t Nq, typename MatrixType>
Biped<Nq,MatrixType> operator+ (const Biped<Nq,MatrixType> &A1, const Biped<Nq,MatrixType> &A2)
{
	Biped<Nq,MatrixType> Ares = A1;
	Ares += A2;
	return Ares;
}

template<size_t Nq, typename MatrixType>
string Biped<Nq,MatrixType>::
print_dict() const
{
	stringstream ss;
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		ss << "in:" << get<0>(it->first) << "\tout:" << get<1>(it->first) << "\t→\t" << it->second << endl;
	}
	return ss.str();
}

template<size_t Nq, typename MatrixType>
double Biped<Nq,MatrixType>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t q=0; q<dim; ++q)
	{
		res += calc_memory(block[q], memunit);
	}
	return res;
}

template<size_t Nq, typename MatrixType>
double Biped<Nq,MatrixType>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
	res += 2. * 2. * Nq * calc_memory<int>(dim, memunit); // in,out; dict.keys
	res += Nq * calc_memory<size_t>(dim, memunit); // dict.vals
	return res;
}

template<size_t Nq, typename MatrixType>
string Biped<Nq,MatrixType>::
formatted (string (*formatFunction)(qarray<Nq>)) const
{
	stringstream ss;
	ss << "•Biped(" << dim << "):" << endl;
	for (size_t q=0; q<dim; ++q)
	{
		ss << "  [" << q << "]: " << formatFunction(in[q]) << "→" << formatFunction(out[q]);
		ss << ":" << endl;
		ss << "   " << block[q];
		if (q!=dim-1) {ss << endl;}
	}
	return ss.str();
}

template<size_t Nq, typename MatrixType>
ostream& operator<< (ostream& os, const Biped<Nq,MatrixType> &V)
{
	os << V.formatted(noFormat);
	return os;
}

#endif
