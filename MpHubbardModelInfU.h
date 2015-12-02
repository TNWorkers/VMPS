#ifndef STRAWBERRY_HUBBARDMODELINFU
#define STRAWBERRY_HUBBARDMODELINFU

#include "MpHubbardModel.h"

namespace VMPS
{

class HubbardModelInfU : public HubbardModel
{
public:
	
	HubbardModelInfU (size_t Lx_input, double V_input=0., double tPrime_input=0., size_t Ly_input=1, bool CALC_SQUARE=true);
	
	class qarrayIterator;
	
	typedef MpsQ<2,double>                  StateXd;
	typedef MpsQ<2,complex<double> >        StateXcd;
	typedef DmrgSolverQ<2,HubbardModelInfU> Solver;
};

HubbardModelInfU::
HubbardModelInfU (size_t Lx_input, double V_input, double tPrime_input, size_t Ly_input, bool CALC_SQUARE)
:HubbardModel(Lx_input, numeric_limits<double>::infinity(), V_input, tPrime_input, Ly_input)
{}

class HubbardModelInfU::qarrayIterator
{
public:
	
	/**
	\param qloc_input : vector of local bases
	\param l_frst : first site
	\param l_last : last site
	\param N_legs : dimension in y-direction
	*/
	qarrayIterator (const vector<vector<qarray<2> > > &qloc_input, int l_frst, int l_last, size_t N_legs=1)
	{
		if (l_last<0 or l_frst>=qloc_input.size())
		{
			N_sites = 0;
		}
		else
		{
			N_sites = l_last-l_frst+1;
		}
		
		for (int N=0; N<=N_sites*static_cast<int>(N_legs); ++N)
		for (int Nup=0; Nup<=N; ++Nup)
		{
			qarray<2> q = {Nup,N-Nup};
			qarraySet.insert(q);
		}
		
		it = qarraySet.begin();
	};
	
	qarray<2> operator*() {return value;}
	
	qarrayIterator& operator= (const qarray<2> a) {value=a;}
	bool operator!=           (const qarray<2> a) {return value!=a;}
	bool operator<=           (const qarray<2> a) {return value<=a;}
	bool operator<            (const qarray<2> a) {return value< a;}
	
	qarray<2> begin()
	{
		return *(qarraySet.begin());
	}
	
	qarray<2> end()
	{
		return *(qarraySet.end());
	}
	
	void operator++()
	{
		++it;
		value = *it;
	}
	
private:
	
	qarray<2> value;
	
	set<qarray<2> > qarraySet;
	set<qarray<2> >::iterator it;
	
	int N_sites;
};

}

#endif
