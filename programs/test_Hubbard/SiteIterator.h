#ifndef SITEITERATOR
#define SITEITERATOR

// Goes through all states which have the occupation number = PT_input at given site curr_index

#include "IteratorFloor.h"
#include "HilbertTypedefs.h"

class SiteIterator : public IteratorFloor
{
public:

	SiteIterator (const std::shared_ptr<OccNumBasis> &basis_ptr_input, int isite_input, PARTICLE_TYPE PT_input);
	
	size_t begin();
	size_t end();
	void operator++();
	
	SiteIterator& operator= (const size_t &other_index) {curr_index = static_cast<size_t>(other_index); return *this;}

private:

	std::shared_ptr<OccNumBasis> basis;
	
	size_t N_states;
	int isite;
	PARTICLE_TYPE PTYPE;
};

SiteIterator::
SiteIterator (const std::shared_ptr<OccNumBasis> &basis_ptr_input, int isite_input, PARTICLE_TYPE PT_input)
:IteratorFloor(), basis(basis_ptr_input), isite(isite_input), PTYPE(PT_input)
{
	N_states = basis->size();
}

size_t SiteIterator::
begin()
{
	if (N_states == 1) {return 0;}
	else
	{
		size_t first = 0;
		while ((*basis)[first][isite] != PTYPE && first<N_states-1)
		{
			++first;
		}
		return first;
	}
}

size_t SiteIterator::
end()
{
	if      (N_states == 1 and (*basis)[0][isite] == PTYPE) {return 1;}
	else if (N_states == 1 and (*basis)[0][isite] != PTYPE) {return 0;}
	else
	{
		size_t last = N_states-1;
		while ((*basis)[last][isite] != PTYPE && last>0)
		{
			--last;
		}
		return last+1;
	}
}

void SiteIterator::
operator++()
{
	++curr_index;
	if (curr_index<end())
	{
		while ((*basis)[curr_index][isite] != PTYPE)
		{
			++curr_index;
		}
	}
}

#endif
