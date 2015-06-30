#ifndef DMRGJANITOR
#define DMRGJANITOR

#include "DmrgPivotStuffQ.h"

/**\brief Flips the sweep direction when the edge is reached.*/
void bring_her_about (int pivot, size_t L, DMRG::DIRECTION::OPTION &DIR)
{
	if (pivot == L-1 and 
	    DIR == DMRG::DIRECTION::RIGHT)
	{
		DIR = DMRG::DIRECTION::LEFT;
	}
	if (pivot == 0 and 
	    DIR == DMRG::DIRECTION::LEFT)
	{
		DIR = DMRG::DIRECTION::RIGHT;
	}
}

/**\class DmrgJanitor
\brief Base class for all the sweeping stuff.
Needs to know \p PivotMatrixType because sweeps using DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD involve non-local information, i.e.\ knowledge of the transfer matrices and the Hamiltonian (given by PivotMatrixQ).*/
template<typename PivotMatrixType>
class DmrgJanitor
{
public:
	
	DmrgJanitor();
	DmrgJanitor(size_t L_input);
	
	///@{
	/**Sweeps from \p pivot to a specific site. Mainly useful for testing purposes.
	\param loc : pivot will go here
	\param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	\param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.*/
	void sweep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL);
	/**Makes a full sweep to the left or right. Mainly useful for testing purposes.
	Asserts that the \p pivot is =-1 or at the opposite edge.
	\param DIR : If DMRG::DIRECTION::LEFT, sweeps to left. If DMRG::DIRECTION::RIGHT, sweeps to right.
	\param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	\param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.*/
	void skim (DMRG::DIRECTION::OPTION DIR, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL);
	/**Makes a full sweep to the opposite edge. Mainly useful for testing purposes.
	Asserts that the \p pivot is = 0 or = \p N_sites-1.
	\param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	\param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.*/
	void skim (DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL);
	///@}
	
	///@{
	/**Calls the next sweep step from site \p loc according to the direction \p DIR.
	Switches from DMRG::BROOM::RDM to DMRG::BROOM::SVD if DmrgJanitor::alpha_noise < 1e-15 or DmrgJanitor::alpha_rsvd < 1e-15.*/
	void sweepStep (DMRG::DIRECTION::OPTION DIR, size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL);
	/**Core function for a sweep to the right. Just a virtual placeholder in DmrgJanitor, overwritten by MpsQ with the real code.
	\param loc : Sweeps to the right from the site \p loc, updating the A-matrices at \p loc and \p loc+1, shifting the pivot to \p loc+1.
	\param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	\param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.*/
	virtual void rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL){};
	/**Core function for a sweep to the left. Just a virtual placeholder in DmrgJanitor, overwritten by MpsQ with the real code.
	\param loc : Sweeps to the left from the site \p loc, updating the A-matrices at \p loc and \p loc-1, shifting the pivot to \p loc-1.
	\param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	\param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.*/
	virtual void leftSweepStep  (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL){};
	///@}
	
	///@{
	/**Returns the length of the chain.*/
	inline size_t length() const {return N_sites;}
	///@}
	
	///@{
	/**Cutoff criterion for DMRG::BROOM::OPTION.*/
	double alpha_noise, eps_rdm, eps_svd, alpha_rsvd;
	size_t N_sv, Dlimit;
	size_t N_mow;
	///@}
	
	void set_defaultCutoffs();
	
protected:
	
	///@{
	/**\describe_pivot*/
	int pivot = -1;
	/**Length of the chain.*/
	size_t N_sites;
	///@}
};

template<typename PivotMatrixType>
DmrgJanitor<PivotMatrixType>::
DmrgJanitor()
{
	set_defaultCutoffs();
}

template<typename PivotMatrixType>
DmrgJanitor<PivotMatrixType>::
DmrgJanitor (size_t L_input)
:N_sites(L_input)
{
	set_defaultCutoffs();
}

template<typename PivotMatrixType>
void DmrgJanitor<PivotMatrixType>::
set_defaultCutoffs()
{
	eps_svd = 1e-7;
	eps_rdm = 1e-14;
	alpha_noise = 1e-10;
	alpha_rsvd = 1e-2;
	N_sv = 500;
	N_mow = 0;
}

template<typename PivotMatrixType>
void DmrgJanitor<PivotMatrixType>::
skim (DMRG::DIRECTION::OPTION DIR, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H)
{
	if (DIR == DMRG::DIRECTION::LEFT)
	{
		assert(pivot == N_sites-1 or pivot == -1);
		for (size_t l=N_sites-1; l>0; --l) {sweepStep(DMRG::DIRECTION::LEFT,l,TOOL,H);}
	}
	else
	{
		assert(pivot == 0 or pivot == -1);
		for (size_t l=0; l<N_sites-1; ++l) {sweepStep(DMRG::DIRECTION::RIGHT,l,TOOL,H);}
	}
}

template<typename PivotMatrixType>
void DmrgJanitor<PivotMatrixType>::
skim (DMRG::BROOM::OPTION TOOL, PivotMatrixType *H)
{
	assert(pivot == 0 or pivot == N_sites-1);
	if (pivot == 0)
	{
		skim(DMRG::DIRECTION::RIGHT,TOOL,H);
	}
	else
	{
		skim(DMRG::DIRECTION::LEFT,TOOL,H);
	}
}

template<typename PivotMatrixType>
void DmrgJanitor<PivotMatrixType>::
sweep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H)
{
	assert(loc<N_sites);
	if (pivot == -1)
	{
		skim(DMRG::DIRECTION::LEFT,TOOL,H);
		for (size_t l=0; l<loc; ++l) {sweepStep(DMRG::DIRECTION::RIGHT,l,TOOL,H);}
	}
	else if (pivot!=-1 and pivot<loc)
	{
		for (size_t l=pivot; l<loc; ++l) {sweepStep(DMRG::DIRECTION::RIGHT,l,TOOL,H);}
	}
	else if (pivot!=-1 and pivot>loc)
	{
		for (size_t l=pivot; l>loc; --l) {sweepStep(DMRG::DIRECTION::LEFT,l,TOOL,H);}
	}
}

template<typename PivotMatrixType>
inline void DmrgJanitor<PivotMatrixType>::
sweepStep (DMRG::DIRECTION::OPTION DIR, size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H)
{
	DMRG::BROOM::OPTION NEW_TOOL = TOOL;
	if ((TOOL == DMRG::BROOM::RDM and alpha_noise < 1e-15) or
	    (TOOL == DMRG::BROOM::RICH_SVD and alpha_rsvd < 1e-15))
	{
		NEW_TOOL = DMRG::BROOM::SVD;
	}
	(DIR==DMRG::DIRECTION::LEFT)? leftSweepStep(loc,NEW_TOOL,H) : rightSweepStep(loc,NEW_TOOL,H);
}

#endif
