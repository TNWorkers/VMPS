#ifndef SU2WRAPPERS_H_
#define SU2WRAPPERS_H_

// For now, only the gsl library is supported so it is defined per default.
// If other libraries are present, this definition can be set in the main cpp file.
#define USE_GSL_SU2_COEFFS 1

/// \cond
#ifdef USE_GSL_SU2_COEFFS
#include <gsl/gsl_sf_coupling.h>
#endif
/// \endcond

#include "DmrgExternal.h"

/*
 * First, we have the wrappers to the implementation of the used external libraries, called coupl_Xj_base.
 * All of these functions take values in the format q=2S+1 which is always of type integer.
 * Internally these values are converted to the respective convention of the used library.
 * For example gsl uses the convention q_gsl=2S, so that we have to use q_gsl = q-1.
 * Second, we have a naive hash implementation which uses the coupl_Xj_base-functions but also handles a std::unordered_map for hashing the symbols.
 * Note that this implmentation is not thread-safe!
 * This is turned off per default but can be used by definng OWN_HASH_CGC.
 */

#ifdef USE_GSL_SU2_COEFFS
inline double coupl_9j_base(const int q1, const int q2, const int q3, 
							const int q4, const int q5, const int q6, 
							const int q7, const int q8, const int q9)
{
	return gsl_sf_coupling_9j(q1-1, q2-1, q3-1,
							  q4-1, q5-1, q6-1,
							  q7-1, q8-1, q9-1);
}

inline double coupl_6j_base (const int q1, const int q2, const int q3, 
							 const int q4, const int q5, const int q6)
{
	return gsl_sf_coupling_6j(q1-1, q2-1, q3-1,
							  q4-1, q5-1, q6-1);
}

inline double coupl_3j_base (const int q1  , const int q2  , const int q3,
							 const int q1_z, const int q2_z, const int q3_z)
{
	return gsl_sf_coupling_3j(q1-1, q2-1, q3-1,
							  q1_z, q2_z, q3_z);
}
#endif

#ifdef OWN_HASH_CGC
//assert(OMP_NUM_THREADS == 1 and "Hashing the cgcs is not treadsafe!");
std::unordered_map<std::array<int,9>,double > Table9j;
std::unordered_map<std::array<int,6>,double > Table6j;

double coupling_9j (const int q1, const int q2, const int q3, 
                    const int q4, const int q5, const int q6, 
                    const int q7, const int q8, const int q9)
{
	auto it = Table9j.find(std::array<int,9>{q1,q2,q3,q4,q5,q6,q7,q8,q9});
	
	if (it != Table9j.end())
	{
		return Table9j[std::array<int,9>{q1,q2,q3,q4,q5,q6,q7,q8,q9}];
	}
	else
	{
		double out = coupl_9j_base(q1,q2,q3,
								   q4,q5,q6,
								   q7,q8,q9);
		Table9j[std::array<int,9>{q1,q2,q3,q4,q5,q6,q7,q8,q9}] = out;
		return out;
	}
}

double coupling_6j (const int q1, const int q2, const int q3, 
                    const int q4, const int q5, const int q6)
{
	auto it = Table6j.find(std::array<int,6>{q1,q2,q3,q4,q5,q6});	
	if (it != Table6j.end())
	{
		return Table6j[std::array<int,6>{q1,q2,q3,q4,q5,q6}];
	}
	else
	{
		double out = coupl_6j_base(q1,q2,q3,
								   q4,q5,q6);
		Table6j[std::array<int,6>{q1,q2,q3,q4,q5,q6}] = out;
		return out;
	}
}
#else
inline double coupling_9j (const int q1, const int q2, const int q3, 
						   const int q4, const int q5, const int q6, 
						   const int q7, const int q8, const int q9)
{
	return coupl_9j_base(q1,q2,q3,
						 q4,q5,q6,
						 q7,q8,q9);
}

inline double coupling_6j (const int q1, const int q2, const int q3, 
						   const int q4, const int q5, const int q6)
{
	return coupl_6j_base(q1,q2,q3,
						 q4,q5,q6);
}

inline double coupling_3j (const int q1  , const int q2  , const int q3,
						   const int q1_z, const int q2_z, const int q3_z)
{
	return coupl_3j_base(q1  ,q2  ,q3,
						 q1_z,q2_z,q3_z);
}

#endif

#endif
