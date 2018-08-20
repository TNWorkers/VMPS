#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

// #include "DmrgTypedefs.h"
#include "DmrgExternal.h"

namespace Sym
{
	// Crazy that this enum needs to be here, because it is also in DmrgTypedefs.h. But without this, it doesn't compile...
	#ifndef KIND_ENUM
	#define KIND_ENUM
	enum KIND {S,T,N,M,Nup,Ndn};
	#endif
	
	/**
	 * Returns a formatted string for \p qnum.
	 * \describe_Symmetry
	 * \param qnum : quantum number for formatting.
	 * \note Uses the kind() function provided from Symmetry for deducing the correct format function.
	 */
	template<typename Symmetry>
	std::string format (qarray<Symmetry::Nq> qnum)
	{
		std::stringstream ss;
		for (int q=0; q<Symmetry::Nq; ++q)
		{
			if (Symmetry::kind()[q] == KIND::S or Symmetry::kind()[q] == KIND::T)
			{
				ss << print_frac_nice(boost::rational<int>(qnum[q]-1,2));
			}
			else if (Symmetry::kind()[q] == KIND::M)
			{
				ss << print_frac_nice(boost::rational<int>(qnum[q],2));
			}
			else
			{
				ss << qnum[q];
			}
			if (q!=Symmetry::Nq-1) {ss << ",";}
		}
		return ss.str();
	}
	
	template<typename Scalar>
	Scalar phase(int q)
	{
		if (q % 2) {return Scalar(-1.);}
		return Scalar(1.);
	}
	
	/** 
	 * Splits the quantum number \p Q into pairs q1,q2 with \f$Q \in q1 \otimes q2\f$.
	 * q1 and q2 can take all values from the given parameters \p ql and \p qr, respectively.
	 * \note : Without specifying \p ql and \p qr, there exist infinity solutions.
	 */
	template<typename Symmetry>
	std::vector<std::pair<typename Symmetry::qType,typename Symmetry::qType> >
	split(const typename Symmetry::qType Q,
		  const std::vector<typename Symmetry::qType>& ql,
		  const std::vector<typename Symmetry::qType> qr)
	{
		std::vector<std::pair<typename Symmetry::qType,typename Symmetry::qType> > vout;
		for (std::size_t q1=0; q1<ql.size(); q1++)
			for (std::size_t q2=0; q2<qr.size(); q2++)
			{
				auto Qs = Symmetry::reduceSilent(ql[q1],qr[q2]);
				if(auto it = std::find(Qs.begin(),Qs.end(),Q) != Qs.end()) {vout.push_back({ql[q1],qr[q2]});}
			}
		return vout;
	}
	
	template<typename Symmetry>
	std::vector<std::pair<std::size_t,std::size_t> >
	split(const typename Symmetry::qType Q,
		  const std::vector<typename Symmetry::qType>& ql,
		  const std::vector<typename Symmetry::qType> qr,
		  bool INDEX)
	{
		std::vector<std::pair<std::size_t,std::size_t> > vout;
		for (std::size_t q1=0; q1<ql.size(); q1++)
			for (std::size_t q2=0; q2<qr.size(); q2++)
			{
				auto Qs = Symmetry::reduceSilent(ql[q1],qr[q2]);
				if(auto it = std::find(Qs.begin(),Qs.end(),Q) != Qs.end()) {vout.push_back({q1,q2});}
			}
		return vout;
	}

} //end namespace Sym

template<typename Symmetry>
void transform_base (vector<vector<qarray<Symmetry::Nq> > > &qloc, qarray<Symmetry::Nq> Qtot, bool PRINT = false)
{
	if (Qtot != Symmetry::qvacuum())
	{
		for (size_t l=0; l<qloc.size(); ++l)
		for (size_t i=0; i<qloc[l].size(); ++i)
		for (size_t q=0; q<Symmetry::Nq; ++q)
		{
			if (Symmetry::kind()[q] != Sym::KIND::S and Symmetry::kind()[q] != Sym::KIND::T) //Do not transform the base for non Abelian symmetries
			{
				qloc[l][i][q] = qloc[l][i][q] * static_cast<int>(qloc.size()) - Qtot[q];
			}
		}
		
		if (PRINT)
		{
			lout << "transformed base:" << endl;
			for (size_t l=0; l<qloc.size(); ++l)
			{
				lout << "l=" << l << endl;
				for (size_t i=0; i<qloc[l].size(); ++i)
				{
					cout << "qloc: " << qloc[l][i] << endl;
				}
			}
		}
	}
};

template<typename Symmetry>
qarray<Symmetry::Nq> adjustQN(const qarray<Symmetry::Nq> &qin, const size_t number_cells)
{
	qarray<Symmetry::Nq> out;
	for (size_t q=0; q<Symmetry::Nq; ++q)
	{
		if (Symmetry::kind()[q] != Sym::KIND::S and Symmetry::kind()[q] != Sym::KIND::T) //Do not transform the base for non Abelian symmetries
		{
			out[q] = qin[q] * number_cells;
		}
		else
		{
			out[q] = qin[q];
		}
	}
	return out;
};
#endif
