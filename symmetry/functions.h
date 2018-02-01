#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include "DmrgExternal.h"

namespace Sym{
	
	template<typename Symmetry>
	string format (qarray<Symmetry::Nq> qnum)
	{
		stringstream ss;
		for (int q=0; q<Symmetry::Nq; ++q)
		{
			if (Symmetry::kind()[q] == KIND::S or Symmetry::kind()[q] == KIND::T) {ss << SfromD(qarray<1>{qnum[q]});}
			else if (Symmetry::kind()[q] == KIND::M)                              {ss << halve(qarray<1>{qnum[q]});}
			else                                                                  {ss << noFormat(qarray<1>{qnum[q]});}
			if (q!=Symmetry::Nq-1)                                                {ss << ",";}
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
		Splits the quantum number \p Q into pairs q1,q2 with \f$Q \in q1 \otimes q2\f$.
		q1 and q2 can take all values from the given parameters \p ql and \p qr, respectively.
		\note : Without specifying \p ql and \p qr, there exist infinity solutions.
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

#endif
