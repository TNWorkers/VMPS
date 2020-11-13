#ifndef SPECTRAL_MANAGER
#define SPECTRAL_MANAGER

#include "GreenPropagator.h"
#include "DmrgLinearAlgebra.h"
#include "RootFinder.h" // from ALGS

template<typename Hamiltonian>
class SpectralManager
{
public:
	
	typedef typename Hamiltonian::Mpo::Scalar_ Scalar;
	typedef typename Hamiltonian::Symmetry Symmetry;
	
	SpectralManager(){};
	
	SpectralManager (const vector<string> &specs_input, const Hamiltonian &H, const vector<Param> &params, VUMPS::CONTROL::GLOB GlobSweepParams, qarray<Symmetry::Nq> Q,
	                 int Ncells_input, const vector<Param> &params_hetero, 
	                 string gs_label="gs", bool LOAD_GS=false, bool SAVE_GS=false,
	                 DMRG::VERBOSITY::OPTION VERB=DMRG::VERBOSITY::HALFSWEEPWISE);
	
	void compute (string wd, string label, int Ns, double tmax, double dt=0.2, 
	              double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA, 
	              size_t Mlim=500ul, double tol_DeltaS=1e-2, double tol_compr=1e-4);
	
	void reload (string wd, const vector<string> &specs_input, string label, int L, int Ncells, int Ns, double tmax, 
	             double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA);
	
	const Umps<Symmetry,Scalar> &ground() const {return g.state;};
	const double &energy() const {return g.energy;};
	
	void make_A1P (GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double> > &Gfull, string wd, string label, int Ns, 
	               double tmax, double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA, 
	               bool SAVE_N_MU=true);
	
	Mpo<Symmetry,Scalar> get_Op (const Hamiltonian &H, size_t loc, std::string spec, double factor=1., size_t locy=0);
	
	static bool TIME_DIR (std::string spec)
	{
		// true=forwards in time
		// false=backwards in time
		return (spec=="PES" or spec=="PESUP" or spec=="PESDN" or spec=="AES" or spec=="IPSZ" or spec=="ICSF" or spec=="SDAGSF" or spec=="PDAGSF")? false:true;
	}
	
	static string DAG (std::string spec)
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
		else if (spec == "ICSF")  res = "ICSF";
		else if (spec == "PSZ")   res = "PSZ";
		else if (spec == "IPSZ")  res = "IPSZ";
		else if (spec == "PSF")   res = "PDAGSF";
		else if (spec == "PDAGSF")res = "PSF";
		else if (spec == "HSF")   res = "IHSF";
		else if (spec == "IHSF")  res = "HSF";
		else if (spec == "IHS")   res = "HSF";
		else if (spec == "HTS")   res = "IHTS";
		else if (spec == "IHTS")  res = "HTS";
		return res;
	}
	
	static bool CHECK_SPEC (string spec)
	{
		std::array<string,19> possible_specs = {"PES","PESUP","PESDN", //3,3
		                                        "SSF","SSZ", //2,5
		                                        "IPE","IPEUP","IPEDN", //3,8
		                                        "AES","APS", //2,10
		                                        "CSF","ICSF","PSZ","IPSZ","PSF", //5,15
		                                        "HSF","IHSF", //2,17
		                                        "HTS","IHTS"}; //17,19
		return find(possible_specs.begin(), possible_specs.end(), spec) != possible_specs.end();
	}
	
private:
	
	size_t L, Lhetero, Ncells;
	size_t x0;
	size_t Nspec;
	vector<string> specs;
	
	Hamiltonian H_hetero;
	Mps<Symmetry,Scalar> Phi;
	double Eg;
	vector<vector<Mps<Symmetry,complex<double>>>> OxPhiCell;
	
	Eigenstate<Umps<Symmetry,Scalar>> g;
	vector<GreenPropagator<Hamiltonian,typename Hamiltonian::Symmetry,Scalar,complex<double>>> Green;
};

template<typename Hamiltonian>
SpectralManager<Hamiltonian>::
SpectralManager (const vector<string> &specs_input, const Hamiltonian &H, const vector<Param> &params, VUMPS::CONTROL::GLOB GlobSweepParams, qarray<Symmetry::Nq> Q,
	             int Ncells_input, const vector<Param> &params_hetero, 
	             string gs_label, bool LOAD_GS, bool SAVE_GS,
	             DMRG::VERBOSITY::OPTION VERB)
:specs(specs_input), Ncells(Ncells_input)
{
	for (const auto &spec:specs)
	{
		assert(CHECK_SPEC(spec) and "Wrong spectral abbreviation!");
	}
	L = H.length();
	Nspec = specs.size();
	Lhetero = L*Ncells;
	
	typename Hamiltonian::uSolver uDMRG(VERB);
	
	if (LOAD_GS)
	{
		g.state.load(gs_label, g.energy);
		lout << "loaded: " << g.state.info() << endl;
	}
	else
	{
		uDMRG.userSetGlobParam();
		uDMRG.GlobParam = GlobSweepParams;
		uDMRG.edgeState(H,g,Q);
		if (SAVE_GS)
		{
			lout << "saving groundstate..." << endl;
			g.state.save(gs_label, "groundstate", g.energy);
		}
	}
	
	lout << setprecision(16) << "g.energy=" << g.energy << endl;
	
	H_hetero = Hamiltonian(Lhetero,params_hetero,BC::INFINITE);
	lout << "H_hetero: " << H_hetero.info() << endl;
	H_hetero.transform_base(Q,false,L); // PRINT=false
	H_hetero.precalc_TwoSiteData(true); // FORCE=true
	
	// Phi
	Phi = uDMRG.create_Mps(Ncells, g, H, Lhetero/2); // ground state as heterogenic MPS
	
	// shift values O→O-<O>
	vector<vector<Scalar>> Oshift(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		Oshift[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			if (specs[z] == "HSF")
			{
				Hamiltonian Haux(2*L, {{"maxPower",1ul}}, BC::INFINITE, DMRG::VERBOSITY::SILENT);
				Oshift[z][l] = avg(g.state, get_Op(Haux,l,specs[z]), g.state);
			}
			else
			{
				Oshift[z][l] = avg(g.state, get_Op(H,l,specs[z]), g.state);
			}
			lout << "spec=" << specs[z] << ", l=" << l << ", shift=" << Oshift[z][l] << endl;
		}
	}
	
	// O and OxPhiCell for counterpropagate
	vector<vector<Mpo<Symmetry,Scalar>>> O(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		O[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			O[z][l] = get_Op(H_hetero, Lhetero/2+l, specs[z]);
			O[z][l].scale(1.,-Oshift[z][l]);
			O[z][l].transform_base(Q,false,L); // PRINT=false
			cout << O[z][l].info() << endl;
			cout << "spec=" << specs[z] << ", <O[z][l]>=" << avg(g.state, O[z][l], g.state) << endl;
		}
	}
	
	OxPhiCell.resize(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		OxPhiCell[z].resize(L);
		auto tmp = uDMRG.create_Mps(Ncells, g, H, O[z][0], O[z]); // O[z][0] for boundaries, O[z] is multiplied (vector in cell size)
		for (int l=0; l<L; ++l)
		{
			OxPhiCell[z][l] = tmp[l].template cast<complex<double>>();
		}
	}
	
	Eg = isReal(avg_hetero(Phi, H_hetero, Phi, true)); // USE_BOUNDARY=true
	lout << setprecision(16) << "Eg=" << Eg << ", eg=" << g.energy << endl;
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
compute (string wd, string label, int Ns, double tmax, double dt, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT, size_t Mlim, double tol_DeltaS, double tol_compr)
{
	int Nt = static_cast<int>(tmax/dt);
	
	// GreenPropagator
	Green.resize(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z] = GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double>>
		           (wd+spec+"_"+label,tmax,Nt,Ns,wmin,wmax,wpoints,QR,qpoints,INT);
		Green[z].set_verbosity(DMRG::VERBOSITY::ON_EXIT);
	}
	Green[0].set_verbosity(DMRG::VERBOSITY::STEPWISE);
	
	// Propagation
	#pragma omp parallel for
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z].set_tol_DeltaS(tol_DeltaS);
		Green[z].set_lim_Nsv(Mlim);
		Green[z].set_tol_compr(tol_compr);
		Green[z].compute_cell(H_hetero, OxPhiCell[z], Eg, TIME_DIR(spec), true); // COUNTERPROPAGATE=true
		Green[z].save(false); // IGNORE_TX=false
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
make_A1P (GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double> > &Gfull, string wd, string label, int Ns, double tmax, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT, bool SAVE_N_MU)
{
	auto itPES = find(specs.begin(), specs.end(), "PES");
	auto itIPE = find(specs.begin(), specs.end(), "IPE");
	
	if (itPES != specs.end() and itIPE != specs.end())
	{
		int iPES = distance(specs.begin(), itPES);
		int iIPE = distance(specs.begin(), itIPE);
		
		// Add PES+IPE
		vector<vector<MatrixXcd>> GinA1P(L); for (int i=0; i<L; ++i) {GinA1P[i].resize(L);}
		for (int i=0; i<L; ++i) 
		for (int j=0; j<L; ++j)
		{
			GinA1P[i][j].resize(Green[iPES].get_GtxCell()[0][0].rows(),
				                Green[iPES].get_GtxCell()[0][0].cols());
			GinA1P[i][j].setZero();
		}
		
		for (int i=0; i<L; ++i)
		for (int j=0; j<L; ++j)
		{
			GinA1P[i][j] += Green[iPES].get_GtxCell()[i][j] + Green[iIPE].get_GtxCell()[i][j];
		}
		Gfull = GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double> >(wd+"A1P"+"_"+label,L,Ncells,Ns,tmax,GinA1P,QR,qpoints,INT);
		Gfull.recalc_FTwCell(wmin,wmax,wpoints);
		
		if (SAVE_N_MU)
		{
			IntervalIterator mu(wmin,wmax,501);
			for (mu=mu.begin(); mu!=mu.end(); ++mu)
			{
				mu << Gfull.integrate_Glocw_cell(*mu);
				mu.save(make_string(wd,"n(μ)_tmax=",tmax,"_",label,".dat"));
			}
		}
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
reload (string wd, const vector<string> &specs_input, string label, int L_input, int Ncells_input, int Ns, double tmax, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT)
{
	L = L_input;
	Ncells = Ncells_input;
	Lhetero = L*Ncells;
	x0 = Lhetero/2;
	specs = specs_input;
	Nspec = specs.size();
	
	for (const auto &spec:specs)
	{
		assert(CHECK_SPEC(spec) and "Wrong spectral abbreviation!");
	}
	
	Green.resize(Nspec);
	
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z] = GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double>> 
		          (wd+spec+"_"+label,L,Ncells,Ns,tmax,{wd+spec+"_"+label+make_string("_L=",L,"x",Ncells,"_tmax=",tmax,"_INT=",INT)},QR,qpoints,INT);
		Green[z].recalc_FTwCell(wmin,wmax,wpoints, TIME_DIR(spec));
		Green[z].save(true); // IGNORE_TX=true
	}
}

template<typename Hamiltonian>
Mpo<typename Hamiltonian::Symmetry,typename Hamiltonian::Mpo::Scalar_> SpectralManager<Hamiltonian>::
get_Op (const Hamiltonian &H, size_t loc, std::string spec, double factor, size_t locy)
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
	else if (spec == "CSF" or spec == "ICSF")
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
	else if (spec == "PSZ" or spec == "IPSZ")
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
			if (loc<H.length()-1)
			{
				Res = H.cdagc<UP,UP>(loc,loc+1,0,0);
			}
			else
			{
				lout << termcolor::yellow << "HSF operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
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
			if (loc<H.length()-1)
			{
				Res = H.cdagc<UP,UP>(loc+1,loc,0,0);
			}
			else
			{
				lout << termcolor::yellow << "IHSF operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
		}
	}
	// hybridization triplet structure factor
	else if (spec == "HTS")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			if (loc<H.length()-1)
			{
				Res = H.cdagc3(loc,loc+1,0,0);
			}
			else
			{
				lout << termcolor::yellow << "HTS operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
		}
		else
		{
			if (loc<H.length()-1)
			{
				Res = H.cdagc<UP,DN>(loc,loc+1,0,0);
			}
			else
			{
				lout << termcolor::yellow << "HTS operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
		}
	}
	// inverse hybridization triplet structure factor
	else if (spec == "IHTS")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			if (loc<H.length()-1)
			{
				Res = H.cdagc3(loc+1,loc,0,0);
			}
			else
			{
				lout << termcolor::yellow << "IHTS operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
		}
		else
		{
			if (loc<H.length()-1)
			{
				Res = H.cdagc<DN,UP>(loc+1,loc,0,0);
			}
			else
			{
				lout << termcolor::yellow << "IHTS operator hit right edge! Returning zero." << termcolor::reset << endl;
				Res = MODEL::Zero(H.qPhys);
			}
		}
	}
	else
	{
		throw;
	}
	
	Res.set_locality(loc);
	return Res;
}

#endif
