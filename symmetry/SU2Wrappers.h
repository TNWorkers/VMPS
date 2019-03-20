#ifndef SU2WRAPPERS_H_
#define SU2WRAPPERS_H_

// As the default libraries for the 3nj-symbols, we use the GSL (Gnu scientific library):
#if !defined USE_WIG_SU2_COEFFS && !defined USE_WIG_SU2_COEFFS && !defined USE_FAST_WIG_SU2_COEFFS
#define USE_GSL_SU2_COEFFS 1
#endif

/// \cond
#ifdef USE_GSL_SU2_COEFFS
#include <gsl/gsl_sf_coupling.h>
#pragma message("Using GSL library for 3nj-symbols.")
#endif

#ifdef USE_WIG_SU2_COEFFS
#include "wigxjpf.h"
#pragma message("Using WIGXJPF library for 3nj-symbols.")
#endif

#ifdef USE_FAST_WIG_SU2_COEFFS
#include "fastwigxj.h"
#include "wigxjpf.h"
#pragma message("Using FASTWIGXJ library for 3nj-symbols.")
#endif
/// \endcond

#include "DmrgExternal.h"

/** @file
 * In this file, the wrappers are defined to include different external libraries which compute \f$3nj\f$-symbols for \f$SU(2)\f$. The wrappers are called coupl_Xj_base.
 * All of these functions take values in the format q=2S+1 and q_z=2M which is always of type integer.
 * Internally these values are converted to the respective convention of the used library.
 * For example gsl uses the convention q_gsl=2S and q_z_gsl=2M, so that we have to use q_gsl = q-1 and q_gsl_z = q_z.
 * Currently, one can use three different libraries for the \f$3nj\f$-symbols:
 *
 * 1. GSL (Default library):
 * The Gnu Scientific libraries (GSL) provides methods for 3j-, 6j- and 9j-symbols.
 * See https://www.gnu.org/software/gsl/manual/html_node/Coupling-Coefficients.html for reference.
 *   - Link parameter: -lgsl
 *   - macro: USE_GSL_SU2_COEFFS
 *
 * 2. WIGXJPF:
 * Specific implementation for 3j, 6j, and 9j-symbols using prime factorization and multiword integer arithmetic.
 * Provides the coefficients for arbitrary angular momenta to machine precision.
 * See http://fy.chalmers.se/subatom/wigxjpf for reference and download.
 * The library needs to be compiled first.
 *   - Link parameter: -lwigxjpf (with /path_to_wig/lib in the library path -L/path_to_wig/lib)
 *   - macro: USE_WIG_SU2_COEFFS
 * \note When extensively using this library you may cite the corresponding publication (See website above for details)
 *
 * 3. FASTWIGXJ:
 * A implementation wich uses hash-tables with precomputed symbols.
 * This library builds up on top of WIGXJPF.
 * See http://fy.chalmers.se/subatom/fastwigxj for reference and download.
 * The library needs to be compiled first.
 *   - Link parameter: -lwigxjpf -lfastwigxj -lwigxjpf_quadmath -lquadmath (Note, the order is important! 
 *     with /path_to_fastwig/lib in the library path -L/path_to_fastwig/lib)
 *   - macro: USE_FAST_WIG_SU2_COEFFS
 * \note The hash-tables need to be precomputed by the user and passed to Sym::initialize(). See manual http://fy.chalmers.se/subatom/fastwigxj/README for instructions.
 * The precomputed tables are commonly named as table_Y.Xj, where X is {3,6,9} and Y the maximal value of J.
 * The tables are commonly stored in the folder cgc_hash within the root-folder of the project.
 * \note For 9j-symbols which are not pre-computed, the library uses a fallback and compute the 9j-symbol from 6j-symbol.
 * To do so, the library needs 128bit floats (quadmath). This enforces the compiler gcc, since clang does not support quadmath.
 * \note When extensively using this library you may cite the corresponding publication (See website above for details)
 * 4. General library + own hash function:
 * A naive hash implementation which uses the coupl_Xj_base-functions but also handles a std::unordered_map for hashing the symbols.
 * Note that this implmentation is not thread-safe!
 * This is turned off per default but can be used by defining OWN_HASH_CGC.
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
#endif //USE_GSL_SU2_COEFFS

#ifdef USE_WIG_SU2_COEFFS
inline double coupl_9j_base(const int q1, const int q2, const int q3, 
							const int q4, const int q5, const int q6, 
							const int q7, const int q8, const int q9)
{
	return wig9jj(q1-1, q2-1, q3-1,
				  q4-1, q5-1, q6-1,
				  q7-1, q8-1, q9-1);
}

inline double coupl_6j_base (const int q1, const int q2, const int q3, 
							 const int q4, const int q5, const int q6)
{
	return wig6jj(q1-1, q2-1, q3-1,
				  q4-1, q5-1, q6-1);
}

inline double coupl_3j_base (const int q1  , const int q2  , const int q3,
							 const int q1_z, const int q2_z, const int q3_z)
{
	return wig3jj(q1-1, q2-1, q3-1,
				  q1_z, q2_z, q3_z);
}
#endif //USE_WIG_SU2_COEFFS

/** 
 */
#ifdef USE_FAST_WIG_SU2_COEFFS
inline double coupl_9j_base(const int q1, const int q2, const int q3, 
							const int q4, const int q5, const int q6, 
							const int q7, const int q8, const int q9)
{
	return fw9jja(q1-1, q2-1, q3-1,
				  q4-1, q5-1, q6-1,
				  q7-1, q8-1, q9-1);
}

inline double coupl_6j_base (const int q1, const int q2, const int q3, 
							 const int q4, const int q5, const int q6)
{
	return fw6jja(q1-1, q2-1, q3-1,
				  q4-1, q5-1, q6-1);
}

inline double coupl_3j_base (const int q1  , const int q2  , const int q3,
							 const int q1_z, const int q2_z, const int q3_z)
{
	return fw3jja(q1-1, q2-1, q3-1,
				  q1_z, q2_z);
}
#endif //USE_FAST_WIG_SU2_COEFFS

#ifdef OWN_HASH_CGC
#ifdef _OPENMP
assert(omp_get_max_threads() == 1 and "Hashing the cgcs is not treadsafe!");
#endif

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

inline double coupling_3j (const int q1  , const int q2  , const int q3,
						   const int q1_z, const int q2_z, const int q3_z)
{
	return coupl_3j_base(q1  ,q2  ,q3,
						 q1_z,q2_z,q3_z);
}

#else //no OWN_HASH_CGC
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

#endif //OWN_HASH_CGC

#endif
