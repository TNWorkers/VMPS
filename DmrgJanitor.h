#ifndef DMRGJANITOR
#define DMRGJANITOR

#include "DmrgTypedefs.h"

/**\brief Flips the sweep direction when the edge is reached.*/
void turnaround (int pivot, size_t L, DMRG::DIRECTION::OPTION &DIR)
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

/**
 * \class DmrgJanitor
 * \brief Base class for all the sweeping stuff.
 * Needs to know \p PivotMatrixType because sweeps using DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD involve non-local information, 
 * i.e.\ knowledge of the transfer matrices and the Hamiltonian (given by PivotMatrix1).
 */
template<typename PivotMatrixType>
class DmrgJanitor
{
public:
	
	DmrgJanitor();
	DmrgJanitor(size_t L_input);
	
	///@{
	/**
	 * Sweeps from \p pivot to a specific site. Mainly useful for testing purposes.
	 * \param loc : pivot will go here
	 * \param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	 * \param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.
	 */
	void sweep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL);
	
	/**
	 * Makes a full sweep to the left or right. Mainly useful for testing purposes.
	 * Asserts that the \p pivot is =-1 or at the opposite edge.
	 * \param DIR : If DMRG::DIRECTION::LEFT, sweeps to left. If DMRG::DIRECTION::RIGHT, sweeps to right.
	 * \param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	 * \param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.
	 */
	void skim (DMRG::DIRECTION::OPTION DIR, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL);
	
	/**
	 * Makes a full sweep to the opposite edge. Mainly useful for testing purposes.
	 * Asserts that the \p pivot is = 0 or = \p N_sites-1.
	 * \param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	 * \param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.
	 */
	void skim (DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL);
	///@}
	
	///@{
	/**
	 * Calls the next sweep step from site \p loc according to the direction \p DIR.
	 */
	void sweepStep (DMRG::DIRECTION::OPTION DIR, size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL, bool DISCARD_U_or_V=false);
	
	/**
	 * Core function for a sweep to the right. Just a virtual placeholder in DmrgJanitor, overwritten by Mps with the real code.
	 * \param loc : Sweeps to the right from the site \p loc, updating the A-matrices at \p loc and \p loc+1, shifting the pivot to \p loc+1.
	 * \param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	 * \param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.
	 * \param DISCARD_V: If \p true, don't multiply the V-matrix onto the next site
	 */
	virtual void rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL, bool DISCARD_V=false){};
	
	/**
	 * Core function for a sweep to the left. Just a virtual placeholder in DmrgJanitor, overwritten by Mps with the real code.
	 * \param loc : Sweeps to the left from the site \p loc, updating the A-matrices at \p loc and \p loc-1, shifting the pivot to \p loc-1.
	 * \param TOOL : with which broom to sweep, see DMRG::BROOM::OPTION
	 * \param H : Non-local information for DMRG::BROOM::RDM and DMRG::BROOM::RICH_SVD enters thorugh here.
	 * \param DISCARD_U: If \p true, don't multiply the U-matrix onto the next site
	 */
	virtual void leftSweepStep  (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H = NULL, bool DISCARD_U=false){};
	///@}
	
	///@{
	/**Returns the length of the chain.*/
	inline size_t length() const {return N_sites;}
	///@}
	
	///@{
	/**Cutoff criterion for DMRG::BROOM::OPTION.*/
	double eps_svd, alpha_rsvd;
	size_t max_Nsv, min_Nsv;
	int max_Nrich;
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
	eps_svd    = DMRG::CONTROL::DEFAULT::eps_svd(0);
	alpha_rsvd = DMRG::CONTROL::DEFAULT::max_alpha_rsvd(0);
	min_Nsv    = DMRG::CONTROL::DEFAULT::min_Nsv(0);
	max_Nsv    = DMRG::CONTROL::DEFAULT::Dlimit;
	max_Nrich  = DMRG::CONTROL::DEFAULT::max_Nrich(0);
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
sweepStep (DMRG::DIRECTION::OPTION DIR, size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixType *H, bool DISCARD_U_or_V)
{
	(DIR==DMRG::DIRECTION::LEFT)? leftSweepStep(loc,TOOL,H,DISCARD_U_or_V) : rightSweepStep(loc,TOOL,H,DISCARD_U_or_V);
}

#endif
