#ifndef QBASIS_H_
#define QBASIS_H_

#include <unordered_set>
#include <set>
#include <unordered_map>
#include <numeric>

#include <Eigen/Eigen>

#include "macros.h"

/** \class Basis
  *
  * Class for a plain basis without symmetries. A basis with symmetries is a collection of plain bases for each Symmetry entry
  *
  */
class Basis
{
public:
	Basis(std::vector<std::string> idents, Eigen::Index dim) {
		Eigen::Index count = 0;
		for(const auto& ident : idents) {
			auto entry = std::make_tuple(ident,count);
			data_.push_back(entry);
			count++;
		}
	};

	Basis(Eigen::Index dim) {
		std::vector<std::string> idents(dim,"");
		Basis(idents,dim);
	};

	Eigen::Index size() const {return data_.size();}
	void push_back( const std::string& ident) {
		curr_dim++;
		data_.push_back(std::make_tuple(ident,curr_dim));
	};

	bool find( std::string ident_in ) {
		for(const auto& elem : data_) {auto [ident,num] = elem; if (ident == ident_in) {return true;}}
		return false;
	}
	Eigen::Index location( std::string ident_in ) {
		for(const auto& elem : data_) {auto [ident,num] = elem; if (ident == ident_in) {return num;}}
		return std::numeric_limits<Eigen::Index>::max();
	}
	
	std::vector<std::tuple<std::string,Eigen::Index> >::iterator begin() { return data_.begin(); }
	std::vector<std::tuple<std::string,Eigen::Index> >::iterator end() { return data_.end(); }

	std::vector<std::tuple<std::string,Eigen::Index> >::const_iterator cbegin() const { return data_.cbegin(); }
	std::vector<std::tuple<std::string,Eigen::Index> >::const_iterator cend() const { return data_.cend(); }

	bool operator==( const Basis& other ) const {return (this->data_ == other.data_);}
private:
	std::vector<std::tuple<std::string,Eigen::Index> > data_;
	Eigen::Index curr_dim = 0;
};

/** \class Qbasis
  *
  * Class for ... 
  *
  * \describe_Symmetry
  */
template<typename Symmetry>
class Qbasis
{
	typedef typename Symmetry::qType qType;
public:
	/**Does nothing.*/
	Qbasis() {};
	
	template<typename Container>
	Qbasis (const Container &qins, const Eigen::Index &dim)
	{
		for (const auto &qin:qins)
		{
			push_back(qin,dim);
		}
	}
	
	std::size_t Mmax() const {
		std::size_t out = 0; for(const auto& entry : data_) {auto [qVal,num,plain] = entry; out+=plain.size();} return out;
	}
	std::size_t fullMmax() const {
		std::size_t out = 0; for(const auto& entry : data_) {auto [qVal,num,plain] = entry; out+=plain.size()*Symmetry::degeneracy(qVal);} return out;
	}
	
	std::size_t Dmax() const
	{
		std::size_t out = 0;
		for (const auto& entry : data_)
		{
			auto [qVal,num,plain] = entry;
			if (plain.size() > out) {out = plain.size();}
		}
		return out;
	}
	
	std::size_t Nqmax() const
	{
		return data_.size();
	}
	
	const std::vector<qType> qloc() const {
		std::vector<qType> out;
		for(const auto& elem : data_) {
			auto [q,num,plain] = elem; for(std::size_t c=0; c<plain.size(); c++) {out.push_back(q);} }
		return out;}
	const std::vector<Eigen::Index> qlocDeg() const {
		std::vector<Eigen::Index> out; for(const auto& elem : data_) {auto [q,num,plain] = elem; out.push_back(plain.size());} return out;
	}
	const std::vector<qType> qs() const {
		std::vector<qType> out;
		for(const auto& elem : data_) {
			auto [q,num,plain] = elem; out.push_back(q); }
		return out;}

	const std::unordered_set<qType> unordered_qs() const {
		std::unordered_set<qType> out;
		for(const auto& elem : data_) {
			auto [q,num,plain] = elem; out.insert(q); }
		return out;}

	qType find( const std::string& ident ) const;
	qType find( const Eigen::Index& num ) const;

	Eigen::Index inner_num( const Eigen::Index& outer_num ) const;
	Eigen::Index location( const std::string& ident ) const;

	Eigen::Index inner_dim( const Eigen::Index& num_in ) const;
	Eigen::Index inner_dim( const qType& q ) const;
	Eigen::Index leftAmount( const qType& qnew, const std::array<qType,2>& qold ) const;
	Eigen::Index rightAmount( const qType& qnew, const std::array<qType,2>& qold ) const;

	void push_back( const std::tuple<qType,Eigen::Index,std::vector<std::string> >& state );
	void push_back( const qType& q_number, const Eigen::Index& inner_dim);
	void push_back( const qType& q_number, const Eigen::Index& inner_dim, const std::vector<std::string>& idents);
	
	void clear() {data_.clear(); history.clear();}

	// template<Eigen::Index Nlegs, typename Scalar, Eigen::Index Nextra>
	// void pullData( const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra>& M, const Eigen::Index& leg );

	void pullData( const std::vector<std::array<qType,3> > &qvec, const std::size_t& leg, const Eigen::Index &inner_dim_in );

	Qbasis<Symmetry> combine( const Qbasis<Symmetry>& other ) const;

	Qbasis<Symmetry> add( const Qbasis<Symmetry>& other ) const;

	std::string print() const;
	std::string printHistory() const;
	
	bool operator==( const Qbasis<Symmetry>& other ) const;

	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::iterator begin() { return data_.begin(); }
	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::iterator end() { return data_.end(); }

	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::const_iterator cbegin() const { return data_.cbegin(); }
	typename std::vector<std::tuple<qType,Eigen::Index,Basis> >::const_iterator cend() const { return data_.cend(); }

private:
	struct fuseData
	{
		Eigen::Index dim;
		std::array<qType,2> source;
	};

	std::vector<std::tuple<qType,Eigen::Index,Basis> > data_;
	Eigen::Index curr_dim=0;
	std::unordered_map<qType,std::vector<fuseData> > history;
};

template<typename Symmetry>
void Qbasis<Symmetry>::
push_back(const std::tuple<qType,Eigen::Index,std::vector<std::string> >& state)
{
	auto [ q_number, inner_dim, ident ] = state;
	// qType q_number;
	// Eigen::Index inner_dim;
	// std::vector<std::string> ident;
	// std::tie(q_number, inner_dim, ident) = state;
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
	Basis plain_basis(idents,inner_dim);
	auto entry = std::make_tuple(q_number,curr_dim,plain_basis);
	data_.push_back(entry);
	curr_dim += inner_dim;
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
	assert( 1!=1 and "The ident is not in the basis" );
}

template<typename Symmetry>
typename Symmetry::qType Qbasis<Symmetry>::
find(const Eigen::Index& num_in ) const
{
	assert( num_in < Mmax() and "The number larger than the size of this basis." );
	Eigen::Index check = num_in;
	for (const auto& q : data_)
	{
		auto [qVal,num,basis] = q;
		if (check < num+basis.size()) { return qVal; }
	}
}

template<typename Symmetry>
Eigen::Index Qbasis<Symmetry>::
inner_num(const Eigen::Index& outer_num) const
{
	assert( outer_num < Mmax() and "The number larger than the size of this basis." );
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

// template<typename Symmetry>
// template<Eigen::Index Nlegs, typename Scalar, Eigen::Index Nextra> 
// void Qbasis<Symmetry>::
// pullData(const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra>& M, const Eigen::Index& leg)
// {
// 	std::unordered_set<qType> unique_controller;
// 	for (std::size_t nu=0; nu<M.size(); nu++)
// 	{
// 		auto it = unique_controller.find(M.index[nu][leg]);
// 		if( it==unique_controller.end() )
// 		{
// 			qType q_number = M.index[nu][leg];
// 			Eigen::Index inner_dim = M.block[nu].dimension(leg);
// 			push_back(q_number,inner_dim);
// 			unique_controller.insert(q_number);			
// 		}
// 	}
// }

template<typename Symmetry>
void Qbasis<Symmetry>::
pullData(const std::vector<std::array<qType,3> > &qvec, const std::size_t& leg, const Eigen::Index &inner_dim_in)
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
bool Qbasis<Symmetry>::
operator==( const Qbasis<Symmetry>& other ) const
{
	return (this->data_ == other.data_);
}

template<typename Symmetry>
Qbasis<Symmetry> Qbasis<Symmetry>::
add( const Qbasis<Symmetry>& other ) const
{
	std::unordered_set<qType> uniqueController;
	Qbasis out;
	for(const auto& elem1 : this->data_)
	{
		auto [q1,num1,plain1] = elem1;
		for(const auto& elem2 : other.data_)
		{
			auto [q2,num2,plain2] = elem2;
			auto qs = Symmetry::reduceSilent(q1,q2);
			for (const auto& q: qs)
			{
				if(auto it=uniqueController.find(q); it==uniqueController.end())
				{
					uniqueController.insert(q);
					out.push_back(q,plain1.size());
				}
			}
		}
	}
	return out;
}

template<typename Symmetry>
Qbasis<Symmetry> Qbasis<Symmetry>::
combine( const Qbasis<Symmetry>& other ) const
{
	Qbasis out;
	//build the history of the combination. Data is relevant for MultipedeQ contractions which includ a fuse of two leg.
	for(const auto& elem1 : this->data_)
	{
		auto [q1,num1,plain1] = elem1;
		for(const auto& elem2 : other.data_)
		{
			auto [q2,num2,plain2] = elem2;
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
		for(const auto& i: it->second) { inner_dim+=i.dim; }
		out.push_back(it->first,inner_dim);
	}
	return out;
}

template<typename Symmetry>
std::string Qbasis<Symmetry>::
print() const
{
	std::stringstream out;
#ifdef HELPERS_IO_TABLE
	TextTable t( '-', '|', '+' );
	t.add("Q");
	t.add("Dim(Q)");
	// t.add("Idents");
	t.endOfRow();
	for(const auto& entry : data_)
	{
		auto [q_Phys,curr_num,plain] = entry;
		std::stringstream ss, tt, uu;
		ss << q_Phys;
		tt << plain.size();
		// uu << "(";
		// if( idents.size() > 0 )
		// {
		// 	for(const auto& inner : plain) { (j<idents[i].size()-1) ? uu << idents[i][j] << ", " : uu << idents[i][j]; }
		// }
		// uu << ")";
		t.add(ss.str());
		t.add(tt.str());
		// t.add(uu.str());
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
