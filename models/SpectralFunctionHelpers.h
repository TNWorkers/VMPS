#ifndef SPECTRALFUNCTIONHELPERS
#define SPECTRALFUNCTIONHELPERS

namespace VMPS
{

template<typename MODEL, typename Symmetry>
typename MODEL::Operator get_Op (const MODEL &H, size_t loc, std::string spec)
{
	typename MODEL::Operator Res;
	
	// spin structure factor
	if (spec == "SSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			Res = H.S(loc,0,1.);
		}
		else
		{
			Res = H.Scomp(SP,loc,0);
		}
	}
	else if (spec == "SSZ")
	{
		if constexpr (!Symmetry::IS_SPIN_SU2())
		{
			Res = H.Scomp(SZ,loc,0);
		}
		else
		{
			throw;
		}
	}
	// photemission
	else if (spec == "PES" or spec == "PESUP")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.c(loc,0,1.);
		}
		else
		{
			Res = H.template c<UP>(loc,0);
		}
	}
	else if (spec == "PESDN")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.c(loc,0,1.);
		}
		else
		{
			Res = H.template c<DN>(loc,0);
		}
	}
	// inverse photoemission
	else if (spec == "IPE" or spec == "IPEUP")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.cdag(loc,0,1.);
		}
		else
		{
			Res = H.template cdag<UP>(loc,0,1.);
		}
	}
	else if (spec == "IPEDN")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.cdag(loc,0,1.);
		}
		else
		{
			Res = H.template cdag<DN>(loc,0,1.);
		}
	}
	// charge structure factor
	else if (spec == "CSF")
	{
		if constexpr (!Symmetry::IS_CHARGE_SU2())
		{
			Res = H.n(loc,0);
		}
		else
		{
			throw;
		}
	}
	// Auger electron spectroscopy
	else if (spec == "AES")
	{
		if constexpr (!Symmetry::IS_CHARGE_SU2())
		{
			Res = H.cc(loc,0);
		}
		else
		{
			throw;
		}
	}
	// Appearance potential spectroscopy
	else if (spec == "APS")
	{
		if constexpr (!Symmetry::IS_CHARGE_SU2())
		{
			Res = H.cdagcdag(loc,0);
		}
		else
		{
			throw;
		}
	}
	// pseudospin structure factor
	else if (spec == "PSF")
	{
		if constexpr (Symmetry::IS_CHARGE_SU2())
		{
			Res = H.T(loc,0);
		}
		else
		{
			Res = H.Tp(loc,0);
		}
	}
	// pseudospin structure factor: z-component
	else if (spec == "PSZ")
	{
		if constexpr (!Symmetry::IS_CHARGE_SU2())
		{
			Res = H.Tz(loc,0);
		}
		else
		{
			throw;
		}
	}
	else
	{
		throw;
	}
	return Res;
}

bool TIME_DIR (std::string spec)
{
	return (spec=="PES" or spec=="PESUP" or spec=="PESDN" or spec=="AES")? false:true;
}

} // namespace VMPS

#endif
