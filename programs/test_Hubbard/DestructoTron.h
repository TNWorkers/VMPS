#ifndef DESTRUCTOTRON
#define DESTRUCTOTRON

#include <Eigen/Dense>
#include <Eigen/SparseCore>
using namespace Eigen;

#include "OperatorFloor.h"
#include "SiteIterator.h"
#include "HoppingParticles.h"

enum DESTR_TARGET {DT_SUM=-1, DT_IDLE=-2};

class DestructoTron : public OperatorFloor
{
//friend class Photo;
//friend class Auger;
//template<typename,typename> friend class PolynomialBath;
public:
	
	DestructoTron (std::shared_ptr<OccNumBasis> BraBasis_input, std::shared_ptr<OccNumBasis> KetBasis_input, int isite=-1, 
	               DECONSTRUCTION DTYPE=ANNIHILATE, SPIN_STATISTICS SPINSTAT=FERMIONS);
//	DestructoTron (std::shared_ptr<OccNumBasis> BraBasis_input, std::shared_ptr<OccNumBasis> KetBasis_input, DESTR_TARGET DT_input, 
//	               DECONSTRUCTION DTYPE=ANNIHILATE, SPIN_STATISTICS SPINSTAT=FERMIONS);
	
	void add_site (int isite, DECONSTRUCTION DTYPE, SPIN_STATISTICS SPINSTAT, SparseMatrixXd &Mout, double scaling=1.);
	
private:
	
	std::shared_ptr<OccNumBasis> BraBasis;
	std::shared_ptr<OccNumBasis> KetBasis;
};

// note:
// isite=-1: sum over all sites
// isite=-2: do nothing
DestructoTron::
DestructoTron (std::shared_ptr<OccNumBasis> BraBasis_input, std::shared_ptr<OccNumBasis> KetBasis_input, int isite, DECONSTRUCTION DTYPE, SPIN_STATISTICS SPINSTAT)
:OperatorFloor(),
BraBasis(BraBasis_input), KetBasis(KetBasis_input)
{
	assert
	(
		((DTYPE==ANNIHILATE and  (*BraBasis)[0].count() == (*KetBasis)[0].count()-1)  or
		( DTYPE==CREATE      and (*BraBasis)[0].count() == (*KetBasis)[0].count()+1)) and
		"Mismatched Bra and Ket spaces in DestructoTron!"
	);
	
	storedOperator.resize(BraBasis->size(), KetBasis->size());
	
	if (isite == -1)
	{
		for (int iL=0; iL<(*KetBasis)[0].size()-1; ++iL)
		{
			add_site(iL, DTYPE,SPINSTAT, storedOperator);
		}
	}
	else if (isite >= 0)
	{
		add_site(isite, DTYPE,SPINSTAT, storedOperator);
	}
	else
	{
		// do nothing for isite = -2
	}
	
	storedOperator.makeCompressed();
}

//DestructoTron::
//DestructoTron (std::shared_ptr<OccNumBasis> BraBasis_input, std::shared_ptr<OccNumBasis> KetBasis_input, DESTR_TARGET DT_input, DECONSTRUCTION DTYPE, SPIN_STATISTICS SPINSTAT)
//:DestructoTron (BraBasis_input, KetBasis_input, static_cast<int>(DT_input), DTYPE, SPINSTAT)
//{}

void DestructoTron::
add_site (int isite, DECONSTRUCTION DTYPE, SPIN_STATISTICS SPINSTAT, SparseMatrixXd &Mout, double scaling)
{
	SiteIterator ir(KetBasis,isite,static_cast<PARTICLE_TYPE>(DTYPE)); // note: annihilate particles = create holes
	
	for (ir=ir.begin(); ir<ir.end(); ++ir)
	{
		short int comm_sign = HoppingParticles::parity((*KetBasis)[*ir], -1,isite, SPINSTAT);
		OccNumVector annihilated = (*KetBasis)[*ir];
		annihilated[isite].flip();
		size_t il = HoppingParticles::get_stateNr(*BraBasis, annihilated, *ir+1);
		Mout.insert(il,*ir) = scaling * comm_sign;
	}
}

#endif
