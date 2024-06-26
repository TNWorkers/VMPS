#ifndef QBASIS_H_
#define QBASIS_H_

/// \cond
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <numeric>

#include <Eigen/Eigen>
/// \endcond

#include "macros.h"

//include "DmrgTypedefs.h"
#include "tensors/Basis.h"
#include "symmetry/functions.h"
//include "symmetry/qarray.h"
// #include "tensors/Biped.h"

template<typename Symmetry, typename MatrixType_> struct Biped;

/** \class Qbasis
  *
  * \ingroup Tensors
  * \describe_Symmetry
  *
  * This class is a container like class for a basis of a Hilbert space in which global symmetries are present.
  * For each irreducible representation irrep of the global symmetry (for each quantum number), 
  * the states of the Hilbert states that transforms under that irrep are collected together in a plain Basis object.
  *
  * One central function is the combine() method, which combine two instances of Qbasis to the tensor product basis, 
  * already proper sorted into irreps.
  *
  * \note Optionally an ident string can be added to each basis state, which give a convinient access for SiteOperatorQ instances.
  */
template<typename Symmetry>
class Qbasis
{
	typedef typename Symmetry::qType qType;
	
public:
	
	/**Does nothing.*/
	Qbasis() {};
	
	/**
	 * Inserts all quantum numbers in the Container \p qins with constant dimension \p dim into the basis.
	 */
	template<typename Container>
	Qbasis (const Container &qins, const Eigen::Index &dim)
	{
		for (const auto &qin:qins)
		{
			push_back(qin,dim);
		}
	}
	
	/**
	 * Construct the basis from an object as used in Mpo, Mps for qloc.
	 */
	Qbasis (const vector<qarray<Symmetry::Nq> > &base_input)
	{
		map<qarray<Symmetry::Nq>,size_t> counter;
		for (const auto &b:base_input)
		{
			counter[b]++;
		}
		for (const auto &[val,count]:counter)
		{
			push_back(val,count);
		}
	}
	
	///\{
	/**Returns the number of (reduced) basis states.*/
	std::size_t size() const { std::size_t out = 0; for(const auto& [qVal,num,plain] : data_) { out+=plain.size();} return out; }
	
	/**Convinient name for the size() function, when the basis is used for the auxilary legs of an Mps tensor.*/
	std::size_t M() const { return size(); }
	
	/**
	 * Returns the full number of basis states.
	 * If irreps has internal states, the basis states transforming under this irreps are multiplied by this degeneracy.
	 */
	std::size_t fullM() const { std::size_t out = 0; for(const auto& [qVal,num,plain] : data_) { out+=plain.size()*Symmetry::degeneracy(qVal);} return out; }
	
	/**Returns the largest state sector.*/
	std::size_t Dmax() const;
	
	/**Returns the number of quantum numbers (irreps) contained in this basis.*/
	std::size_t Nq() const { return data_.size(); }
	///\}
	
	///\{
	/**Returns a vector of size size(), where for every entry of the vector the quantum number (irrep) is inserted to the vector.*/
	const std::vector<qType> qloc() const;
	
	/**Returns a vector containing all quantum numbers (irreps) contained in the basis. The size of the vector is Nq().*/
	const std::vector<qType> qs() const;
	
	/**Same as qs(), but the quantum numbers are inserted to an std::unordered_set.*/
	const std::unordered_set<qType> unordered_qs() const;
	///\}
	
	///\{
	/**Returns the quantum number which is located at \p index in the data_ member.*/
	qType operator[] ( const std::size_t index ) const { return std::get<0>(data_[index]); }
	qType& operator[] ( const std::size_t index ) { return std::get<0>(data_[index]); }
	///\}
	
	/**
	 * Returns the quantum number of the state with ident \p ident.
	 * \todo2 Bad name for this function...
	 */
	qType find( const std::string& ident ) const;
	
	/**
	 * Returns the quantum number of the state with number \p num.
	 * \todo2 Bad name for this function...
	 */
	qType find( const Eigen::Index& num ) const;
	/**Checks whether states with quantum number \p q are in the basis. Returns true if the state is present.*/
	bool find( const qType& q ) const;
	
	Eigen::Index inner_num( const Eigen::Index& outer_num ) const;
	Eigen::Index location( const std::string& ident ) const;
	
	Eigen::Index inner_dim( const Eigen::Index& num_in ) const;
	Eigen::Index inner_dim( const qType& q ) const;

	Eigen::Index outer_num( const qType& q ) const;
	
	Eigen::Index leftAmount( const qType& qnew, const std::array<qType,2>& qold ) const; 
	Eigen::Index leftOffset( const qType& qnew, const std::array<qType,2>& qold, const std::array<Eigen::Index,2>& plain_old ) const;
	Eigen::Index rightAmount( const qType& qnew, const std::array<qType,2>& qold ) const;
	// Eigen::Index outer_num( const qType& qnew, const std::array<qType,2>& qold ) const;
	
	///\{
	/**Insert the \p state into the basis.*/
	void push_back( const std::tuple<qType,Eigen::Index,std::vector<std::string> >& state );
	void push_back( const qType& q_number, const Eigen::Index& inner_dim);
	void push_back( const qType& q_number, const Eigen::Index& inner_dim, const std::vector<std::string>& idents);
	///\}
	
	/**Completely clear the basis.*/
	void clear() {data_.clear(); history.clear(); curr_dim=0;}
	
	/**
	 * Pulls the info from a given MPS site tensor \p A. 
	 * If \p leg=0 the basis from the incoming leg is pulled. If \p leg=1 from the outgoing.
	 */
	template<typename MatrixType>
	void pullData (const vector<Biped<Symmetry,MatrixType> >& A, const Eigen::Index& leg);
    
    template<typename MatrixType>
    void pullData (const vector<vector<vector<Biped<Symmetry,MatrixType>>> > &W, const Eigen::Index &leg);
	
	void pullData (const std::vector<std::array<qType,3> > &qvec, const std::size_t& leg, const Eigen::Index &inner_dim_in);

	void pullData (const std::vector<qarray<Symmetry::Nq> > &qs);
	
	/**
	 * Returns the tensor product basis, already properly sorted with respect to the resulting irreps.
	 * This function also saves the history of the combination process for later use. See leftAmount() and rightAmount().
	 */
	Qbasis<Symmetry> combine (const Qbasis<Symmetry>& other, bool FLIP=false) const;
	
	/**
	 * Sets the history of a Qbasis which only has one quantum number \p Qval with inner dimension 1 
	 * so that the quantum number is arised from combining \p Q1 and \p Q2 
	 */
	void setHistoryEntry (const qType& Qval, const qType &Q1, const qType &Q2, Eigen::Index dim);
	
	/**Adds to bases together.*/
	Qbasis<Symmetry> add (const Qbasis<Symmetry>& other) const;
	
	/**Prints the basis.*/
	std::string print() const;
	
	/**Prints the history.*/
	std::string printHistory() const;
	
	bool operator== (const Qbasis<Symmetry>& other) const;
	
	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::iterator begin() {return data_.begin();}
	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::iterator end()   {return data_.end();}
	
	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::const_iterator cbegin() const {return data_.cbegin();}
	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::const_iterator cend()   const {return data_.cend();}
	
	/**Swaps with another Qbasis.*/
	void swap (Qbasis<Symmetry> &other) { this->data.swap(other.data()); }

	void sort ();

//private:
	struct fuseData
	{
		Eigen::Index dim;
		std::array<qType,2> source;
		Basis::fuseData plainHistory;
	};

	//vector with entry: {Quantumnumber (QN), state number of the first plain state for this QN, all plain states for this QN in a Basis object.}
	//[{q1,0,plain_q1}, {q2,dim(plain_q1),plain_q2}, {q3,dim(plain_q1)+dim(plain_q2),plain_q3}, ..., {qi, sum_j^(i-1)dim(plain_qj), plain_qi}]
	std::vector<std::tuple<qType,Eigen::Index,Basis> > data_;
	Eigen::Index curr_dim=0;
	std::unordered_map<qType,std::vector<fuseData> > history;
};

template<typename Symmetry>
void Qbasis<Symmetry>::
push_back(const std::tuple<qType,Eigen::Index,std::vector<std::string> >& state)
{
	auto [ q_number, inner_dim, ident ] = state;
	push_back(q_number, inner_dim, ident);
}

template<typename Symmetry>
void Qbasis<Symmetry>::
push_back(const qType& q_number, const Eigen::Index& inner_dim)
{
	std::vector<std::string> dummy_idents(inner_dim,"");
	push_back(q_number, inner_dim, dummy_idents);
}

template<typename Symmetry>
void Qbasis<Symmetry>::
push_back(const qType& q_number, const Eigen::Index& inner_dim, const std::vector<std::string>& idents)
{
	// search if the quantum number is already in the Qbasis.
	auto it = std::find_if(data_.begin(), data_.end(), [&q_number] (const tuple<qType,Eigen::Index,Basis> &entry) {return std::get<0>(entry) == q_number;});
	// insert quantum number if it is not there
	if (it == data_.end()) // ATTENTION: workaround for ZN symmetries (e.g. Hubbard basis has parity 0,1,1,0 and 4 states)
	{
		Basis plain_basis(idents,inner_dim);
		auto entry = std::make_tuple(q_number,curr_dim,plain_basis);
		data_.push_back(entry);
	}
	else // append to quantum number if it is there
	{
		// I'm not sure this works...
		// Better to give the correct ident=1,2,3... from the Site class.
		// (Roman)
		std::get<2>(*it).push_back(idents);
		for (auto loop=it++; loop==data_.end(); loop++)
		{
			std::get<1>(*loop) += inner_dim;
		}
	}
	curr_dim += inner_dim;
}

template<typename Symmetry>
std::size_t Qbasis<Symmetry>::
Dmax() const
{
	std::size_t out = 0;
	for (const auto& entry : data_)
	{
		auto [qVal,num,plain] = entry;
		if (plain.size() > out) {out = plain.size();}
	}
	return out;
}

template<typename Symmetry>
const std::vector<typename Symmetry::qType> Qbasis<Symmetry>::
qloc() const
{
	std::vector<qType> out;
	for(const auto& [q,num,plain] : data_) { for(std::size_t c=0; c<plain.size(); c++) {out.push_back(q);} }
	return out;
}

template<typename Symmetry>
const std::vector<typename Symmetry::qType> Qbasis<Symmetry>::
qs() const
{
	std::vector<qType> out;
	for(const auto& [q,num,plain] : data_) { out.push_back(q); }
	return out;
}

template<typename Symmetry>
const std::unordered_set<typename Symmetry::qType> Qbasis<Symmetry>::
unordered_qs() const
{
	std::unordered_set<qType> out;
	for(const auto& [q,num,plain] : data_) { out.insert(q); }
	return out;
}

template<typename Symmetry>
typename Symmetry::qType Qbasis<Symmetry>::
find(const std::string& ident) const
{
	for (const auto& q : data_ )
	{
		auto [qVal,num,basis] = q;
		if(basis.find(ident)) {return qVal;}
	}
	assert(1!=1 and "The ident is not in the basis");
}

template<typename Symmetry>
typename Symmetry::qType Qbasis<Symmetry>::
find(const Eigen::Index& num_in ) const
{
	assert( num_in < size() and "The number is larger than the size of this basis." );
	Eigen::Index check = num_in;
	for (const auto& q : data_)
	{
		auto [qVal,num,basis] = q;
		if (check < num+basis.size()) { return qVal; }
	}
	assert(false and "Something went wrong in Qbasis::find(Eigen::Index num_in)");
}

template<typename Symmetry>
bool Qbasis<Symmetry>::
find (const qType& q ) const
{
	for (const auto& entry : data_)
	{
		auto [qVal,num,basis] = entry;
		if (qVal == q) {return true;}
	}
	return false;
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
inner_num(const Eigen::Index& outer_num) const
{
	assert( outer_num < size() and "The number larger than the size of this basis." );
	Eigen::Index check = outer_num;
	for (const auto& q : data_)
	{
		auto [qVal,num,plain] = q;
		for (const auto& elem : plain)
		{
			auto [ident,inner_num] = elem;
			if (check == num+inner_num) { return inner_num; }
		}
	}
	assert(false and "Something went wrong in Qbasis::inner_num");
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
location(const std::string& ident) const
{
	for(const auto& elem : data_)
	{
		auto [qVal,num,plain] = elem;
		if(plain.location(ident) != std::numeric_limits<Eigen::Index>::max()) {return plain.location(ident);}
	}
	assert( 1!=1 and "The ident is not in the basis" );
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
inner_dim(const qType& q) const
{
	for(const auto& elem : data_)
	{
		auto [qVal,num,plain] = elem;
		if (qVal == q) {return plain.size();}
	}
	return 0;
	// assert( 1!=1 and "The qType is not in the basis" );
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
outer_num(const qType& q) const
{
	for(const auto& elem : data_)
	{
		auto [qVal,num,plain] = elem;
		if (qVal == q) {return num;}
	}
	// return 0;
	assert( 1!=1 and "The qType is not in the basis" );
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
inner_dim(const Eigen::Index& num_in) const
{
	for(const auto& elem : data_)
	{
		auto [qVal,num,plain] = elem;
		if (num == num_in) {return plain.size();}
	}
	assert( 1!=1 and "This number is not in the basis" );
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
rightAmount(const qType& qnew, const std::array<qType,2>& qold) const
{
	assert( history.size() == data_.size() and "The history for this basis is not defined properly");
	auto it = history.find(qnew);
	assert( it != history.end() and "The history for this basis is not defined properly");

	Eigen::Index out=0;
	bool SCHALTER=false;
	for( const auto& i: it->second )
	{
		if(i.source != qold and SCHALTER==true) { out+=i.dim; }
		if(i.source == qold) { SCHALTER = true; }
	}
	return out;
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
leftAmount(const qType& qnew, const std::array<qType,2>& qold) const
{
	assert( history.size() == data_.size() and "The history for this basis is not defined properly");

	auto it = history.find(qnew);
	assert( it != history.end() and "The history for this basis is not defined properly");

	Eigen::Index out=0;
	bool SCHALTER=false;
	for( const auto& i: it->second )
	{
		if(i.source != qold and SCHALTER==false) { out+=i.dim; }
		if(i.source == qold) { break; }// SCHALTER = true;
	}
	return out;
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
leftOffset(const qType& qnew, const std::array<qType,2>& qold, const std::array<Eigen::Index,2>& plain_old) const
{
	assert( history.size() == data_.size() and "The history for this basis is not defined properly");

	auto it = history.find(qnew);
	assert( it != history.end() and "The history for this basis is not defined properly");

	Eigen::Index out=0;

	for( const auto& i: it->second )
	{
		if(i.source != qold) { out+=i.dim; }
		if(i.source == qold)
		{
			for (size_t j=0; j<i.plainHistory.dim1*i.plainHistory.dim2; j++)
			{
				if(i.plainHistory.source(j) != plain_old) { out+=1; }
				if(i.plainHistory.source(j) == plain_old) { break; }
			}
			break;
		}
	}
	return out;
}
// template<typename Symmetry>
// Eigen::Index Qbasis<Symmetry>::
// outer_num(const qType& qnew, const std::array<qType,2>& qold) const
// {
// 	assert( history.size() == data_.size() and "The history for this basis is not defined properly");

// 	auto it = history.find(qnew);
// 	assert( it != history.end() and "The history for this basis is not defined properly");

// 	Eigen::Index out=0;
// 	bool SCHALTER=false;
// 	for( const auto& i: it->second )
// 	{
// 		if(i.source != qold and SCHALTER==false) { out+=i.dim; }
// 		if(i.source == qold) { break; }// SCHALTER = true;
// 	}
// 	return out;
// }

template<typename Symmetry>
template<typename MatrixType>
void Qbasis<Symmetry>::
pullData (const vector<Biped<Symmetry,MatrixType> > &A, const Eigen::Index &leg)
{
	std::unordered_set<qType> unique_controller;
	for (std::size_t s=0; s<A.size(); s++)
	for (std::size_t q=0; q<A[s].size(); q++)
	{
		if(leg==0)
		{
			auto it = unique_controller.find(A[s].in[q]);
			if( it==unique_controller.end() )
			{
				qType q_number = A[s].in[q];
				Eigen::Index inner_dim = A[s].block[q].rows();
				push_back(q_number,inner_dim);
				unique_controller.insert(q_number);			
			}
		}
		else if(leg==1)
		{
			auto it = unique_controller.find(A[s].out[q]);
			if( it==unique_controller.end() )
			{
				qType q_number = A[s].out[q];
				Eigen::Index inner_dim = A[s].block[q].cols();
				push_back(q_number,inner_dim);
				unique_controller.insert(q_number);
			}
		}
	}
}

template<typename Symmetry>
template<typename MatrixType>
void Qbasis<Symmetry>::
pullData (const vector<vector<vector<Biped<Symmetry,MatrixType>>> > &W, const Eigen::Index &leg)
{
    std::unordered_set<qType> unique_controller;
    for (std::size_t s1=0; s1<W.size(); s1++)
    for(std::size_t s2=0; s2<W[s1].size(); s2++)
    for(std::size_t k=0; k<W[s1][s2].size(); k++)
    for (std::size_t q=0; q<W[s1][s2][k].size(); q++)
    {
        if(leg==0)
        {
            auto it = unique_controller.find(W[s1][s2][k].in[q]);
            if( it==unique_controller.end() )
            {
                qType q_number = W[s1][s2][k].in[q];
                Eigen::Index inner_dim = W[s1][s2][k].block[q].rows();
                push_back(q_number,inner_dim);
                unique_controller.insert(q_number);
            }
        }
        else if(leg==1)
        {
            auto it = unique_controller.find(W[s1][s2][k].out[q]);
            if( it==unique_controller.end() )
            {
                qType q_number = W[s1][s2][k].out[q];
                Eigen::Index inner_dim = W[s1][s2][k].block[q].cols();
                push_back(q_number,inner_dim);
                unique_controller.insert(q_number);
            }
        }
    }
}

template<typename Symmetry>
void Qbasis<Symmetry>::
pullData (const std::vector<std::array<qType,3> > &qvec, const std::size_t &leg, const Eigen::Index &inner_dim_in)
{
	std::unordered_set<qType> unique_controller;
	Eigen::Index inner_dim = inner_dim_in;
	for (std::size_t nu=0; nu<qvec.size(); nu++)
	{
		auto it = unique_controller.find(qvec[nu][leg]);
		if( it==unique_controller.end() )
		{
			qType q_number = qvec[nu][leg];
			push_back(q_number,inner_dim);
			unique_controller.insert(q_number);
		}
	}
}

template<typename Symmetry>
void Qbasis<Symmetry>::
pullData (const std::vector<qarray<Symmetry::Nq> > &qs)
{
//	if (ORDERED)
	{
		for (const auto & q : qs)
		{
			push_back(q,1);
		}
		return;
	}
//	std::unordered_map<qType,size_t> qmap;
//	for (std::size_t s=0; s<qs.size(); s++)
//	{
//		auto it = qmap.find(qs[s]);
//		if( it==qmap.end() )
//		{
//			qmap[qs[s]] = 1;
//		}
//		else
//		{
//			qmap[qs[s]]++;
//		}
//	}
//	for (const auto &[q,dim]:qmap) {push_back(q,dim);}
}

template<typename Symmetry>
void Qbasis<Symmetry>::
sort ()
{

	std::vector<std::size_t> index_sort(data_.size());
	std::iota(index_sort.begin(),index_sort.end(),0);
	std::sort (index_sort.begin(), index_sort.end(),
			   [&] (std::size_t n1, std::size_t n2)
				   {
					   qarray<Symmetry::Nq> q1 = std::get<0>(data_[n1]);
					   qarray<Symmetry::Nq> q2 = std::get<0>(data_[n2]);
					   return Symmetry::compare(std::array{q1},std::array{q2});
				   }
		);
	auto new_data_ = data_;
	for (std::size_t i=0; i<data_.size(); i++)
	{
		new_data_[i] = data_[index_sort[i]];
	}
	std::get<1>(new_data_[0]) = 0;
	for (std::size_t i=1; i<data_.size(); i++)
	{
		std::get<1>(new_data_[i]) = 0;
		for (std::size_t j=0; j<i; j++)
		{
			std::get<1>(new_data_[i]) += std::get<2>(new_data_[j]).size();
		}
	}
	data_ = new_data_;
}
	
template<typename Symmetry>
bool Qbasis<Symmetry>::
operator==( const Qbasis<Symmetry>& other ) const
{
	return (this->data_ == other.data_);
}

template<typename Symmetry>
Qbasis<Symmetry> Qbasis<Symmetry>::
add( const Qbasis<Symmetry>& other ) const
{
	// cout << termcolor::red << termcolor::bold << "Using function Qbasis::add" << termcolor::reset << endl;
	std::unordered_set<qType> uniqueController;
	Qbasis out;
	for(const auto& [q1,num1,plain1] : this->data_)
	{
		auto qtmp = q1;
		auto it_other = std::find_if(other.data_.begin(), other.data_.end(), [qtmp] (std::tuple<qType,Eigen::Index,Basis> entry) { return std::get<0>(entry) == qtmp; });
		if (it_other != other.data_.end())
		{
			out.push_back(q1,plain1.add(std::get<2>(*it_other)).size());
		}
		else
		{
			out.push_back(q1,plain1.size());
		}
	}
	
	for(const auto& [q2,num2,plain2] : other.data_)
	{
		auto qtmp = q2;
		auto it_this = std::find_if(data_.begin(), data_.end(), [qtmp] (std::tuple<qType,Eigen::Index,Basis> entry) { return std::get<0>(entry) == qtmp; });
		if (it_this == data_.end())
		{
			out.push_back(q2,plain2.size());
		}
	}
	out.sort();
	return out;
}

template<typename Symmetry>
void Qbasis<Symmetry>::
setHistoryEntry( const qType &Qval, const qType &Q1, const qType &Q2, Eigen::Index dim )
{
	vector<fuseData> history_;
	fuseData entry;
	entry.source = {Q1,Q2};
	entry.dim = dim;
	history_.push_back(entry);
	history.insert(std::make_pair(Qval,history_));
}

template<typename Symmetry>
Qbasis<Symmetry> Qbasis<Symmetry>::
combine (const Qbasis<Symmetry>& other, bool FLIP) const
{
	Qbasis out;
	//build the history of the combination. Data is relevant for MultipedeQ contractions which include a fuse of two leg.
	for(const auto& elem1 : this->data_)
	{
		auto [q1,num1,plain1] = elem1;
		for(const auto& elem2 : other.data_)
		{
			auto [q2,num2,plain2] = elem2;
			if (FLIP)
			{
				q2 = Symmetry::flip(q2);
			}
			auto plain = plain1.combine(plain2);
			auto qVec = Symmetry::reduceSilent(q1,q2);
			for (const auto& q: qVec)
			{
				auto it = out.history.find(q);
				if( it==out.history.end() )
				{
					std::vector<fuseData> history_;
					fuseData entry;
					entry.source = {q1,q2};
					entry.dim = plain1.size() * plain2.size();
					// std::vector<Basis::fuseData> plainHistory(entry.dim);
					// for (Eigen::Index i=0; i<plain1.size(); i++)
					// for (Eigen::Index j=0; j<plain2.size(); j++)
					// {
					// 	Basis::fuseData plain_entry = {{i,j}};
					// 	// plainHistory[i+plain1.size()*j] = plain_entry;
					// 	plainHistory[j+plain2.size()*i] = plain_entry;
					// }
					entry.plainHistory = plain.history;
					history_.push_back(entry);
					out.history.insert(std::make_pair(q,history_));
				}
				else
				{
					bool DONT_COUNT=false;
					for( const auto& entry: it->second )
					{
						std::array<qType,2> tmp = {q1,q2};
						if ( entry.source ==  tmp )
						{
							DONT_COUNT = true;
						}
					}
					if ( DONT_COUNT == false )
					{
						fuseData entry;
						entry.source = {q1,q2};
						entry.dim = plain1.size() * plain2.size();
						// std::vector<Basis::fuseData> plainHistory(entry.dim);
						// for (Eigen::Index i=0; i<plain1.size(); i++)
						// for (Eigen::Index j=0; j<plain2.size(); j++)
						// {
						// 	Basis::fuseData plain_entry = {{i,j}};
						// 	// plainHistory[i+plain1.size()*j] = plain_entry;
						// 	plainHistory[j+plain2.size()*i] = plain_entry;
						// }
						entry.plainHistory = plain.history;
						(it->second).push_back(entry);
					}
				}
			}
		}
	}
	
	//sort the history on the basis of Symmetry::compare()
	for ( auto it=out.history.begin(); it!=out.history.end(); it++ )
	{
		std::vector<std::size_t> index_sort((it->second).size());
		std::iota(index_sort.begin(),index_sort.end(),0);
		std::sort (index_sort.begin(), index_sort.end(),
				   [&] (std::size_t n1, std::size_t n2)
				   {
					   return Symmetry::compare((it->second)[n1].source,(it->second)[n2].source);
				   }
			);
		std::vector<fuseData> entry2 = it->second;
		for (std::size_t i=0; i<entry2.size(); i++)
		{
			(it->second)[i] = entry2[index_sort[i]];
		}
	}

	//build up the new basis
	for ( auto it=out.history.begin(); it!=out.history.end(); it++ )
	{
		Eigen::Index inner_dim = 0;
		std::vector<string> idents;
		for(const auto& i: it->second)
		{
			// qType q1 = i.source[0];
			// qType q2 = i.source[1];
			// for (Eigen::Index plain=0; plain<i.plainHistory.size(); plain++)
			// {
			// 	Eigen::Index plain1 = i.plainHistory[plain].source[0];
			// 	Eigen::Index plain2 = i.plainHistory[plain].source[1];
			// 	auto pos_1 = std::find_if(this->data_.begin(), this->data_.end(), [q1](auto t) {return std::get<0>(t) == q1;});
			// 	auto pos_2 = std::find_if(other.data_.begin(), other.data_.end(), [q2](auto t) {return std::get<0>(t) == q2;});
			// 	std::string label1 = std::get<0>(std::get<2>(*pos_1).data_[plain1]);
			// 	std::string label2 = std::get<0>(std::get<2>(*pos_2).data_[plain2]);
			// 	idents.push_back(label1+"⊗"+label2);
			// }
			inner_dim+=i.dim;
		}
		out.push_back(it->first,inner_dim);
	}
	out.sort();
	return out;
}

template<typename Symmetry>
std::string Qbasis<Symmetry>::
print() const
{
	std::stringstream out;
        #ifdef TOOLS_IO_TABLE
	TextTable t( '-', '|', '+' );
	t.add("Q");
	t.add("Dim(Q)");
	t.add("idents");
	t.endOfRow();
	for(const auto& entry : data_)
	{
		auto [q_Phys,curr_num,plain] = entry;
		std::stringstream ss, tt, uu;
		ss << Sym::format<Symmetry>(q_Phys);
		
		//ss << q_Phys;
		tt << plain.size();
		
		uu << curr_num;
		// uu << "(";
		// for (auto it = plain.cbegin(); it != plain.cend(); it++)
		// // for (const auto & [ident, num] : plain)
		// {
		// 	auto [ident,num] = *it;
		// 	if (std::distance(it,plain.cend()) == 1)
		// 		uu << ident << "(" << num << ")";
		// 	else
		// 		uu << ident << "(" << num << ")" << ", ";
		// }
		// uu << ")";
		t.add(ss.str());
		t.add(tt.str());
		t.add(uu.str());
		t.endOfRow();
	}
	out << t;
	#else
	out << "The stream operator for Qbasis needs the TextTable library.";
	#endif
	return out.str();
}

template<typename Symmetry>
std::string Qbasis<Symmetry>::
printHistory() const
{
	std::stringstream out;
	for(auto it=history.begin(); it!=history.end(); it++)
	{
		out << it->second.size() << " quantumnumber pair(s) merge to Q=" << it->first << std::endl;
		for(const auto& i: it->second)
		{
			out << i.source[0] << "," << i.source[1] << "\t→\t" << it->first << ": dim=" << i.dim << std::endl;
			assert(i.dim == i.plainHistory.dim1*i.plainHistory.dim2);
			for (std::size_t j=0; j<i.dim; j++)
			{
				out << "j=" << j << " results from(" << i.plainHistory.source(j)[0] << "," << i.plainHistory.source(j)[1] << ")" << std::endl;
			}
			out << std::endl;
		}
		out << std::endl;
	}
	return out.str();
}

template<typename Symmetry>
std::ostream& operator<<(std::ostream& os, const Qbasis<Symmetry> &basis)
{
	os << basis.print();
	return os;
}

#endif
