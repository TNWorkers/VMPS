#ifndef SPECTRALFUNCTIONHELPERS
#define SPECTRALFUNCTIONHELPERS

namespace VMPS
{

template<typename MODEL, typename Symmetry, typename Scalar=double>
Mpo<Symmetry,Scalar> get_Op (const MODEL &H, size_t loc, std::string spec, double factor=1., size_t locy=0)
{
	Mpo<Symmetry,Scalar> Res;
	
	// spin structure factor
	if (spec == "SSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			Res = H.S(loc,locy,factor);
		}
		else
		{
			Res = H.Scomp(SP,loc,locy);
		}
	}
	else if (spec == "SDAGSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			Res = H.Sdag(loc,locy,factor);
		}
		else
		{
			Res = H.Scomp(SM,loc,locy);
		}
	}
	else if (spec == "SSZ")
	{
		if constexpr (!Symmetry::IS_SPIN_SU2())
		{
			Res = H.Scomp(SZ,loc,locy);
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
			Res = H.c(loc,locy,factor);
		}
		else
		{
			Res = H.template c<UP>(loc,locy);
		}
	}
	else if (spec == "PESDN")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.c(loc,locy,factor);
		}
		else
		{
			Res = H.template c<DN>(loc,locy);
		}
	}
	// inverse photoemission
	else if (spec == "IPE" or spec == "IPEUP")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.cdag(loc,locy,factor);
		}
		else
		{
			Res = H.template cdag<UP>(loc,locy,factor);
		}
	}
	else if (spec == "IPEDN")
	{
		if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
		{
			Res = H.cdag(loc,locy,factor);
		}
		else
		{
			Res = H.template cdag<DN>(loc,locy,factor);
		}
	}
	// charge structure factor
	else if (spec == "CSF")
	{
		if constexpr (!Symmetry::IS_CHARGE_SU2())
		{
			Res = H.n(loc,locy);
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
			Res = H.cc(loc,locy);
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
			Res = H.cdagcdag(loc,locy);
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
			Res = H.T(loc,locy);
		}
		else
		{
			Res = H.Tp(loc,locy);
		}
	}
	// pseudospin structure factor
	else if (spec == "PDAGSF")
	{
		if constexpr (Symmetry::IS_CHARGE_SU2())
		{
			Res = H.Tdag(loc,locy);
		}
		else
		{
			Res = H.Tm(loc,locy);
		}
	}
	// pseudospin structure factor: z-component
	else if (spec == "PSZ" or spec == "IPZ")
	{
		if constexpr (!Symmetry::IS_CHARGE_SU2())
		{
			Res = H.Tz(loc,locy);
		}
		else
		{
			throw;
		}
	}
	// hybridization structure factor
	else if (spec == "HSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			if (loc<H.length()-1)
			{
				Res = H.cdagc(loc,loc+1,0,0);
			}
			else
			{
				lout << termcolor::yellow << "HSF operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
		}
		else
		{
			throw;
		}
	}
	// inverse hybridization structure factor
	else if (spec == "IHSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			if (loc<H.length()-1)
			{
				Res = H.cdagc(loc+1,loc,0,0);
			}
			else
			{
				lout << termcolor::yellow << "IHSF operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
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
	
	Res.set_locality(loc);
	return Res;
}

bool TIME_DIR (std::string spec)
{
	// true=forwards in time
	// false=backwards in time
	return (spec=="PES" or spec=="PESUP" or spec=="PESDN" or spec=="AES" or spec=="IPZ" or spec=="SDAGSF" or spec=="PDAGSF")? false:true;
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
	else if (spec == "IPZ")   res = "IPZ";
	else if (spec == "PSF")   res = "PDAGSF";
	else if (spec == "HSF")   res = "IHSF";
	else if (spec == "IHSF")  res = "HSF";
	return res;
}

} // namespace VMPS

#endif
