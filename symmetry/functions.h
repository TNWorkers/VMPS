#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

// #include "DmrgTypedefs.h"
#include "SU2Wrappers.h"
#include "DmrgExternal.h"

namespace Sym
{
	// Crazy that this enum needs to be here, because it is also in DmrgTypedefs.h. But without this, it doesn't compile...
	#ifndef KIND_ENUM
	#define KIND_ENUM
	enum KIND {S,Salt,T,N,M,Nup,Ndn,Z2};
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
			if (Symmetry::kind()[q] == KIND::S or Symmetry::kind()[q] == KIND::Salt or Symmetry::kind()[q] == KIND::T)
			{
				ss << print_frac_nice(boost::rational<int>(qnum[q]-1,2));
			}
			else if (Symmetry::kind()[q] == KIND::M)
			{
				ss << print_frac_nice(boost::rational<int>(qnum[q],2));
			}
			else if (Symmetry::kind()[q] == KIND::Z2)
			{
				string parity = (posmod<2>(qnum[q])==0)? "evn":"odd";
				ss << parity;
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

	/** 
	 * This routine initializes the relevant objects for the calculation of \f$3nj\f$-symbols.
	 * The specific code varies from library to library:
	 * 1. GSL: Nothing to to do.
	 * 2. WIGXJPF: The tables for the prime factorization need to get build. The parameter \p maxJ is the maximum angular momentum.
	 * It should be chosen high enough. The required memory for the prime factorization table is negligible.
	 * 3. FASTWIGXJ: Same initialization as WIGXJPF for fallback to symbols which are not precomputed (\p maxJ).
	 * Additionaly the filenames to the precalculated symbols are required. \p f_3j for \f$3j\f$-symbol, \p f_6j for \f$6j\f$-symbol
	 * and \p f_9j for \f$9j\f$-symbol. For the creation of the precomputed values see manual http://fy.chalmers.se/subatom/fastwigxj/README.
	 * The precomputed symbols (especially 9j) can be quite large in memory. Therefore, this library should only be used
	 * when performance gains are clearly present.
	 *
	 * \warning The current initialize() method is for a single thread. WIGXJPF and FASTWIGXJ can both be used in multi-tread applications.
	 * The initialize function however needs to get adapted accordingly. The details can be found on the websites above.
	 */
	void initialize(int maxJ=1, std::string f_3j="", std::string f_6j="", std::string f_9j="")
	{
#ifdef USE_WIG_SU2_COEFFS
		wig_table_init(2*maxJ,9);
		wig_temp_init(2*maxJ);
#endif

#ifdef USE_FAST_WIG_SU2_COEFFS
		fastwigxj_load(f_3j.c_str(), 3, NULL);
		fastwigxj_load(f_6j.c_str(), 6, NULL);
		fastwigxj_load(f_9j.c_str(), 9, NULL);
		
		wig_table_init(2*maxJ,9);
		wig_temp_init(2*maxJ);
#endif
	}

	void finalize(bool PRINT_STATS=false)
	{
#ifdef USE_WIG_SU2_COEFFS
		wig_temp_free();
		wig_table_free();
#endif

#ifdef USE_FAST_WIG_SU2_COEFFS
		if (PRINT_STATS) {std::cout << std::endl; fastwigxj_print_stats();}
		
		fastwigxj_unload(3);
		fastwigxj_unload(6);
		fastwigxj_unload(9);

		wig_temp_free();
		wig_table_free();
#endif
	}

} //end namespace Sym

template<typename Symmetry>
void transform_base (vector<vector<qarray<Symmetry::Nq> > > &qloc, qarray<Symmetry::Nq> Qtot, bool PRINT=false, bool BACK=false, int L=-1)
{
	int length = (L==-1)? static_cast<int>(qloc.size()):L;
	
	if (Qtot != Symmetry::qvacuum())
	{
		if (PRINT) lout << "•old base:" << endl;
		for (size_t l=0; l<qloc.size(); ++l)
		{
			if (PRINT) lout << "l=" << l << endl;
			for (size_t i=0; i<qloc[l].size(); ++i)
			{
				if (PRINT) lout << "qloc: " << qloc[l][i] << endl;
				for (size_t q=0; q<Symmetry::Nq; ++q)
				{
					if (Symmetry::kind()[q] != Sym::KIND::S and Symmetry::kind()[q] != Sym::KIND::Salt and Symmetry::kind()[q] != Sym::KIND::T) //Do not transform the base for non Abelian symmetries
					{
						if (BACK) // back transform
						{
							qloc[l][i][q] = (qloc[l][i][q] + Qtot[q]) / length;
						}
						else // forward transform
						{
							qloc[l][i][q] = qloc[l][i][q] * length - Qtot[q];
						}
					}
				}
			}
		}
		
		if (PRINT)
		{
			lout << "•transformed base:" << endl;
			for (size_t l=0; l<qloc.size(); ++l)
			{
				lout << "l=" << l << endl;
				for (size_t i=0; i<qloc[l].size(); ++i)
				{
					lout << "qloc: " << qloc[l][i] << endl;
				}
			}
		}
	}
};

template<typename Symmetry>
qarray<Symmetry::Nq> adjustQN (const qarray<Symmetry::Nq> &qin, const size_t number_cells, bool BACK=false)
{
	qarray<Symmetry::Nq> out;
	for (size_t q=0; q<Symmetry::Nq; ++q)
	{
		if (Symmetry::kind()[q] != Sym::KIND::S and Symmetry::kind()[q] != Sym::KIND::Salt and Symmetry::kind()[q] != Sym::KIND::T) //Do not transform the base for non-Abelian symmetries
		{
			if (BACK)
			{
				out[q] = qin[q] / number_cells;
			}
			else
			{
				out[q] = qin[q] * number_cells;
			}
		}
		else
		{
			out[q] = qin[q];
		}
	}
	return out;
};
#endif
