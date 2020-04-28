#ifndef SPECTRALFUNCTIONHELPERS
#define SPECTRALFUNCTIONHELPERS

namespace VMPS
{

template<typename MODEL, typename Symmetry>
typename MODEL::Operator get_Op (const MODEL &H, size_t loc, std::string spec, double factor=1.)
{
	typename MODEL::Operator Res;
	
	// spin structure factor
	if (spec == "SSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			Res = H.S(loc,0,factor);
		}
		else
		{
			Res = H.Scomp(SP,loc,0);
		}
	}
	else if (spec == "SDAGSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			Res = H.Sdag(loc,0,factor);
		}
		else
		{
			Res = H.Scomp(SM,loc,0);
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
			Res = H.c(loc,0,factor);
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
			Res = H.c(loc,0,factor);
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
			Res = H.cdag(loc,0,factor);
		}
		else
		{
			Res = H.template cdag<UP>(loc,0,factor);
		}
	}
	else if (spec == "IPEDN")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.cdag(loc,0,factor);
		}
		else
		{
			Res = H.template cdag<DN>(loc,0,factor);
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
	// pseudospin structure factor
	else if (spec == "PDAGSF")
	{
		if constexpr (Symmetry::IS_CHARGE_SU2())
		{
			Res = H.Tdag(loc,0);
		}
		else
		{
			Res = H.Tm(loc,0);
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
	// true=forwards in time
	// false=backwards in time
	return (spec=="PES" or spec=="PESUP" or spec=="PESDN" or spec=="AES")? false:true;
}

string DAG (std::string spec)
{
	string res;
	if (spec == "PES")        res = "IPE";
	if (spec == "PESUP")      res = "IPEUP";
	if (spec == "PESDN")      res = "IPEDN";
	else if (spec == "SSF")   res = "SDAGSF";
	else if (spec == "SSZ")   res = "SSZ";
	else if (spec == "IPE")   res = "PES";
	else if (spec == "IPEUP") res = "PESUP";
	else if (spec == "IPEDN") res = "PESDN";
	else if (spec == "AES")   res = "APS";
	else if (spec == "APS")   res = "AES";
	else if (spec == "CSF")   res = "CSF";
	else if (spec == "PSZ")   res = "PSZ";
	else if (spec == "PSF")   res = "PDAGSF";
	return res;
}

} // namespace VMPS

#endif
