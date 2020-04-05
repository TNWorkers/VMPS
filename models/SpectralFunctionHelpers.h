#ifndef SPECTRALFUNCTIONHELPERS
#define SPECTRALFUNCTIONHELPERS

namespace VMPS
{

template<typename MODEL>
typename MODEL::Operator get_Op (const MODEL &H, size_t loc, std::string spec)
{
	typename MODEL::Operator Res;
	
	// spin structure factor
	if (spec == "SSF")
	{
		#if defined(USE_SPIN_SU2)
		{
			Res = H.S(loc,0,1.);
		}
		#elif defined(USE_SPIN_ABELIAN)
		{
			Res = H.Scomp(SP,loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	else if (spec == "SSZ")
	{
		#if defined(USE_SPIN_ABELIAN)
		{
			Res = H.Scomp(SZ,loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	// photemission
	else if (spec == "PES")
	{
		cout << "spec=" << spec << endl;
		#if defined(USE_SPIN_SU2) || defined(USE_SPINLESS)
		{
			Res = H.c(loc,0,1.);
		}
		#elif defined(USE_SPIN_ABELIAN)
		{
			Res = H.c<UP>(loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	// inverse photoemission
	else if (spec == "IPE")
	{
		#if defined(USE_SPIN_SU2) || defined(USE_SPINLESS)
		{
			Res = H.cdag(loc,0,1.);
		}
		#elif defined(USE_SPIN_ABELIAN)
		{
			Res = H.cdag<UP>(loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	// charge structure factor
	else if (spec == "CSF")
	{
		#if defined(USE_CHARGE_ABELIAN) || defined(USE_SPINLESS)
		{
			Res = H.n(loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	// Auger electron spectroscopy
	else if (spec == "AES")
	{
		#if defined(USE_CHARGE_ABELIAN)
		{
			Res = H.cc(loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	// Appearance potential spectroscopy
	else if (spec == "APS")
	{
		#if defined(USE_CHARGE_ABELIAN)
		{
			Res = H.cdagcdag(loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	// pseudospin structure factor
	else if (spec == "PSF")
	{
		#if defined(USE_CHARGE_ABELIAN) && defined(USE_SPIN_ABELIAN)
		{
			Res = H.T(loc,0);
		}
		#elif defined(USE_CHARGE_ABELIAN)
		{
			Res = H.Tp(loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	// pseudospin structure factor: z-component
	else if (spec == "PSZ")
	{
		#if defined(USE_CHARGE_ABELIAN)
		{
			Res = H.Tz(loc,0);
		}
		#else
		{
			throw;
		}
		#endif
	}
	return Res;
}

bool TIME_DIR (std::string spec)
{
	return (spec=="PES" or spec=="AES")? false:true;
}

} // namespace VMPS

#endif
