#ifndef SPECTRAL_MANAGER
#define SPECTRAL_MANAGER

#include "GreenPropagator.h"
#include "DmrgLinearAlgebra.h"
//#include "RootFinder.h" // from ALGS
#include "VUMPS/VumpsSolver.h"

template<typename Hamiltonian>
class SpectralManager
{
public:
	
	typedef typename Hamiltonian::Mpo::Scalar_ Scalar;
	typedef typename Hamiltonian::Symmetry Symmetry;
	
	SpectralManager(){};
	
	// IBC
	SpectralManager (const vector<string> &specs_input, const Hamiltonian &H, const vector<Param> &params, VUMPS::CONTROL::GLOB GlobSweepParams, qarray<Symmetry::Nq> Q,
	                 int Ncells_input, const vector<Param> &params_hetero, 
	                 string gs_label="gs", bool LOAD_GS=false, bool SAVE_GS=false,
	                 DMRG::VERBOSITY::OPTION VERB=DMRG::VERBOSITY::HALFSWEEPWISE,
	                 double tol_OxV=2., int locyBra=0, int locyKet=0);
	
	// OBC
	SpectralManager (const vector<string> &specs_input, const Hamiltonian &H, const vector<Param> &params, DMRG::CONTROL::GLOB GlobSweepParams, qarray<Symmetry::Nq> Q,
	                 string gs_label="gs", bool LOAD_GS=false, bool SAVE_GS=false,
	                 DMRG::VERBOSITY::OPTION VERB=DMRG::VERBOSITY::HALFSWEEPWISE,
	                 double tol_OxV=2., int locy=0);
	
	SpectralManager (const vector<string> &specs_input, const Hamiltonian &H, DMRG::VERBOSITY::OPTION VERB=DMRG::VERBOSITY::HALFSWEEPWISE)
	:specs(specs_input), Hwork(H), CHOSEN_VERB(VERB)
	{};
	
	template<typename HamiltonianThermal>
	void beta_propagation (const Hamiltonian &Hprop, const HamiltonianThermal &Htherm, int Lcell, int dLphys, 
	                       double betamax_input, double dbeta_input, double tol_compr_beta_input, size_t Mlim, qarray<Symmetry::Nq> Q, 
	                       double s_betainit, double betaswitch, 
	                       string wd, string th_label, bool LOAD_BETA=false, bool SAVE_BETA=true,
	                       DMRG::VERBOSITY::OPTION VERB=DMRG::VERBOSITY::HALFSWEEPWISE,
	                       vector<double> stateSavePoints={}, vector<string> stateSaveLabels={}, 
	                       int Ntaylor=0, bool CALC_C=true, bool CALC_CHI=true, bool USE_PHIT=false);
	
	void continue_beta_propagation (const Hamiltonian &Hprop, int Lcell, int dLphys, 
	                                double s_betainit, double betainit, double betamax, double dbeta, double tol_compr_beta, size_t Mlim, qarray<Hamiltonian::Symmetry::Nq> Q,
	                                double betaswitch, 
	                                string wd, string th_label, string LOAD_BETA, bool SAVE_BETA,
	                                DMRG::VERBOSITY::OPTION VERB, 
	                                vector<double> stateSavePoints, vector<string> stateSaveLabels, 
	                                bool CALC_C, bool CALC_CHI);
	
	void apply_operators_on_thermal_state (int Lcell, int dLphys, bool CHECK=true);
	
	void compute (string wd, string label, int Ns, double tmax, double dt=0.2, 
	              double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA, 
	              size_t Mlim=500ul, double tol_DeltaS=1e-2, double tol_compr=1e-4);
	
	void compute_finiteCell (int Lcell, int x0, string wd, string label, double tmax, double dt=0.1, 
	                         double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA, 
	                         size_t Mlim=500ul, double tol_DeltaS=1e-2, double tol_compr=1e-4);
	
	void compute_finite (size_t j0, string wd, string label, int Ns, double tmax, double dt=0.1, 
	                     double wmin=-10., double wmax=10., int wpoints=501, GREEN_INTEGRATION INT=OOURA, 
	                     size_t Mlim=500ul, double tol_DeltaS=1e-2, double tol_compr=1e-4);
	
	void compute_thermal (string wd, string label, int dLphys,
	                      double tmax, double dt=0.1, double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA, 
	                      size_t Mlim=500ul, double tol_DeltaS=1e-2, double tol_compr=1e-4);
	
	void reload (string wd, const vector<string> &specs_input, string label, int L, int Ncells, int Ns, double tmax, 
	             double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA);
	
	const Umps<Symmetry,Scalar> &ground() const {return g.state;};
	const double &energy() const {return g.energy;};
	const Mps<Symmetry,Scalar> &get_PhiT() const {return PhiT;};
	
	void make_A1P (GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double> > &Gfull, string wd, string label, int Ns, 
	               double tmax, double wmin=-10., double wmax=10., int wpoints=501, Q_RANGE QR=ZERO_2PI, int qpoints=501, GREEN_INTEGRATION INT=OOURA, 
	               bool SAVE_N_MU=true);
	
	void make_A1P_finite (GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double> > &Gfull, string wd, string label, 
	                      double tmax, double wmin=-10., double wmax=10., int wpoints=501, GREEN_INTEGRATION INT=OOURA);
	
	Mpo<Symmetry,Scalar> get_Op (const Hamiltonian &H, size_t loc, std::string spec, double factor=1., size_t locy=0, int dLphys=1);
	
	void set_measurement (int iz, string spec, double factor, int dLphys, qarray<Symmetry::Nq> Q, int Lcell, int measure_interval_input=10, string measure_name_input="M", string measure_subfolder_input=".", bool TRANSFORM=false);
	
	void resize_Green (string wd, string label, int Ns, double tmax, double dt, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT);
	
	void FTcell_xq();
	
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
		else if (spec == "SPM")   res = "SMP";
		else if (spec == "SMP")   res = "SPM";
		else if (spec == "QSF")   res = "QDAGSF";
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
		std::array<string,26> possible_specs = {"PES","PESUP","PESDN", //3,3
		                                        "SSF","SSZ", //2,5
		                                        "IPE","IPEUP","IPEDN", //3,8
		                                        "AES","APS", //2,10
		                                        "CSF","ICSF","PSZ","IPSZ","PSF", //5,15
		                                        "HSF","IHSF", //2,17
		                                        "HTS","IHTS", // 2,19
		                                        "JCJC","JEJE", "JCJE", "JEJC", // 4,23
		                                        "QSF", // 1,24
		                                        "SPM","SMP" // 2,26
		                                        }; 
		return find(possible_specs.begin(), possible_specs.end(), spec) != possible_specs.end();
	}
	
	vector<vector<MatrixXcd>> get_GwqCell (int z) const {return Green[z].get_GwqCell();};
	
	void set_Ncells (int Ncells_input) {Ncells=Ncells_input;};
	
	void set_PhiT (const Mps<Symmetry,Scalar> &PhiT_input)
	{
		PhiT = PhiT_input;
	}
	
private:
	
	size_t L, Lhetero, Ncells;
	size_t x0;
	size_t Nspec;
	vector<string> specs;
	
	Hamiltonian Hwork;
	Mps<Symmetry,Scalar> Phi;
	double Eg;
	vector<vector<Mps<Symmetry,complex<double>>>> OxPhiCellBra;
	vector<vector<Mps<Symmetry,complex<double>>>> OxPhiCellKet;
	
	Eigenstate<Umps<Symmetry,Scalar>> g;
	Eigenstate<Mps<Symmetry,Scalar>> gfinite;
	vector<GreenPropagator<Hamiltonian,typename Hamiltonian::Symmetry,Scalar,complex<double>>> Green;
	
	double betamax, dbeta, tol_compr_beta;
	Mps<Symmetry,Scalar> PhiT;
	Mps<Symmetry,complex<double>> PhiTt;
	vector<vector<Mps<Symmetry,complex<double>>>> OxPhiTt;
	vector<vector<Mpo<typename Hamiltonian::Symmetry,Scalar>>> Odag;
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERB;
	vector<vector<Scalar>> Oshift;
};

// non-thermal IBC
template<typename Hamiltonian>
SpectralManager<Hamiltonian>::
SpectralManager (const vector<string> &specs_input, const Hamiltonian &H, const vector<Param> &params, VUMPS::CONTROL::GLOB GlobSweepParams, qarray<Symmetry::Nq> Q,
	             int Ncells_input, const vector<Param> &params_hetero, 
	             string gs_label, bool LOAD_GS, bool SAVE_GS,
	             DMRG::VERBOSITY::OPTION VERB,
	             double tol_OxV, int locyBra, int locyKet)
:specs(specs_input), Ncells(Ncells_input), CHOSEN_VERB(VERB)
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
	
	Hwork = Hamiltonian(Lhetero,params_hetero,BC::INFINITE);
	lout << "H_hetero: " << Hwork.info() << endl;
	Hwork.transform_base(Q,false,L); // PRINT=false
	Hwork.precalc_TwoSiteData(true); // FORCE=true
	
	// Phi
	Phi = uDMRG.create_Mps(Ncells, g, H, Lhetero/2); // ground state as heterogenic MPS
	
	// shift values O→O-<O>
	vector<vector<Scalar>> OshiftKet(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		OshiftKet[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			if (specs[z] == "HSF")
			{
				Hamiltonian Haux(2*L, {{"maxPower",1ul}}, BC::INFINITE, DMRG::VERBOSITY::SILENT);
				OshiftKet[z][l] = avg(g.state, get_Op(Haux,l,specs[z],1.,locyKet), g.state);
			}
			else
			{
				OshiftKet[z][l] = avg(g.state, get_Op(H,l,specs[z],1.,locyKet), g.state);
			}
			lout << "spec=" << specs[z] << ", l=" << l << ", shift=" << OshiftKet[z][l] << endl;
		}
	}
	
	// O and OxPhiCell for counterpropagate
	vector<vector<Mpo<Symmetry,Scalar>>> Oket(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		Oket[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			Oket[z][l] = get_Op(Hwork, Lhetero/2+l, specs[z], 1., locyKet);
			if (abs(OshiftKet[z][l]) > 1e-6)
			{
				Oket[z][l].scale(1.,-OshiftKet[z][l]);
			}
			Oket[z][l].transform_base(Q,false,L); // PRINT=false
			lout << Oket[z][l].info() << endl;
		}
	}
	
	vector<vector<Mpo<Symmetry,Scalar>>> Obra(Nspec);
	if (locyBra == locyKet)
	{
		Obra = Oket;
	}
	else
	{
		vector<vector<Scalar>> OshiftBra(Nspec);
		for (int z=0; z<Nspec; ++z)
		{
			OshiftBra[z].resize(L);
			for (int l=0; l<L; ++l)
			{
				if (specs[z] == "HSF")
				{
					Hamiltonian Haux(2*L, {{"maxPower",1ul}}, BC::INFINITE, DMRG::VERBOSITY::SILENT);
					OshiftBra[z][l] = avg(g.state, get_Op(Haux,l,specs[z],1.,locyBra), g.state);
				}
				else
				{
					OshiftBra[z][l] = avg(g.state, get_Op(H,l,specs[z],1.,locyBra), g.state);
				}
				lout << "spec=" << specs[z] << ", l=" << l << ", shift=" << OshiftBra[z][l] << endl;
			}
		}
		
		for (int z=0; z<Nspec; ++z)
		{
			Obra[z].resize(L);
			for (int l=0; l<L; ++l)
			{
				Obra[z][l] = get_Op(Hwork, Lhetero/2+l, specs[z], 1., locyBra);
				if (abs(OshiftBra[z][l]) > 1e-6)
				{
					Obra[z][l].scale(1.,-OshiftBra[z][l]);
				}
				Obra[z][l].transform_base(Q,false,L); // PRINT=false
				lout << Obra[z][l].info() << endl;
			}
		}
	}
	
	OxPhiCellKet.resize(Nspec);
	OxPhiCellBra.resize(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		OxPhiCellKet[z].resize(L);
		OxPhiCellBra[z].resize(L);
		
		auto tmpKet = uDMRG.create_Mps(Ncells, g, H, Oket[z][0], Oket[z], tol_OxV); // O[z][0] for boundaries, O[z] is multiplied (vector in cell size)
		auto tmpBra = uDMRG.create_Mps(Ncells, g, H, Obra[z][0], Obra[z], tol_OxV);
		
		for (int l=0; l<L; ++l)
		{
			OxPhiCellKet[z][l] = tmpKet[l].template cast<complex<double>>();
			OxPhiCellBra[z][l] = tmpBra[l].template cast<complex<double>>();
		}
	}
	
	Eg = isReal(avg_hetero(Phi, Hwork, Phi, true)); // USE_BOUNDARY=true
	lout << setprecision(16) << "Eg=" << Eg << ", eg=" << g.energy << endl;
}

// non-thermal OBC
template<typename Hamiltonian>
SpectralManager<Hamiltonian>::
SpectralManager (const vector<string> &specs_input, const Hamiltonian &H, const vector<Param> &params, DMRG::CONTROL::GLOB GlobSweepParams, qarray<Symmetry::Nq> Q,
                 string gs_label, bool LOAD_GS, bool SAVE_GS, DMRG::VERBOSITY::OPTION VERB, double tol_OxV, int locy)
:specs(specs_input), CHOSEN_VERB(VERB)
{
	for (const auto &spec:specs)
	{
		assert(CHECK_SPEC(spec) and "Wrong spectral abbreviation!");
	}
	L = H.length();
	Ncells = 1;
	Nspec = specs.size();
	Lhetero = L*Ncells;
	
	typename Hamiltonian::Solver fDMRG(VERB);
	
	if (LOAD_GS)
	{
		gfinite.state.load(gs_label, gfinite.energy);
		lout << "loaded: " << gfinite.state.info() << endl;
	}
	else
	{
		fDMRG.userSetGlobParam();
		fDMRG.GlobParam = GlobSweepParams;
		fDMRG.edgeState(H,gfinite,Q);
		if (SAVE_GS)
		{
			lout << "saving groundstate..." << endl;
			gfinite.state.save(gs_label, "groundstate", gfinite.energy);
		}
	}
	
	lout << setprecision(16) << "gfinite.energy=" << gfinite.energy << endl;
	
	Hwork = Hamiltonian(Lhetero,params,BC::OPEN);
	lout << "H: " << Hwork.info() << endl;
	Hwork.precalc_TwoSiteData(true); // FORCE=true
	
	// Phi
	Phi = gfinite.state;
	
	// shift values O→O-<O>
	Oshift.resize(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		Oshift[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			if (specs[z] == "HSF")
			{
				Hamiltonian Haux(2*L, {{"maxPower",1ul}}, BC::OPEN, DMRG::VERBOSITY::SILENT);
				Oshift[z][l] = avg(gfinite.state, get_Op(Haux,l,specs[z],1.,locy), gfinite.state);
			}
			else
			{
				Oshift[z][l] = avg(gfinite.state, get_Op(H,l,specs[z],1.,locy), gfinite.state);
			}
			lout << "spec=" << specs[z] << ", l=" << l << ", shift=" << Oshift[z][l] << endl;
		}
	}
	
	// O and OxPhiCell
	vector<vector<Mpo<Symmetry,Scalar>>> O(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		O[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			O[z][l] = get_Op(Hwork, l, specs[z], 1., locy);
			if (O[z][l].Qtarget() == Symmetry::qvacuum())
			{
				O[z][l].scale(1.,-Oshift[z][l]);
			}
		}
	}
	
	OxPhiCellKet.resize(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		OxPhiCellKet[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			Mps<Symmetry,Scalar> tmp;
			Mpo<Symmetry,Scalar> Otmp = O[z][l];
			OxV_exact(Otmp, Phi, tmp, tol_OxV, DMRG::VERBOSITY::SILENT, 200, 1);
			OxPhiCellKet[z][l] = tmp.template cast<complex<double>>();
		}
	}
	
	Eg = gfinite.energy;
}

template<typename Hamiltonian>
template<typename HamiltonianThermal>
void SpectralManager<Hamiltonian>::
beta_propagation (const Hamiltonian &Hprop, const HamiltonianThermal &Htherm, int Lcell, int dLphys, 
                  double betamax, double dbeta, double tol_compr_beta, size_t Mlim, qarray<Hamiltonian::Symmetry::Nq> Q,
                  double s_betainit, double betaswitch, 
                  string wd, string th_label, bool LOAD_BETA, bool SAVE_BETA,
                  DMRG::VERBOSITY::OPTION VERB, 
                  vector<double> stateSavePoints, vector<string> stateSaveLabels, 
                  int Ntaylor, bool CALC_C, bool CALC_CHI, bool USE_PHIT)
{
	for (const auto &spec:specs)
	{
		assert(CHECK_SPEC(spec) and "Wrong spectral abbreviation!");
	}
	L = Hprop.length()/dLphys;
	Nspec = specs.size();
	
	if (LOAD_BETA)
	{
		lout << "loading beta result from: " << make_string(wd,"/betaRes_",th_label) << endl;
		PhiT.load(make_string(wd,"/betaRes_",th_label));
		lout << "loaded: " << PhiT.info() << endl;
	}
	else
	{
		if (!USE_PHIT)
		{
			lout << "Computing T=infinity state" << endl;
			typename HamiltonianThermal::Solver fDMRG(DMRG::VERBOSITY::ON_EXIT);
			Eigenstate<Mps<Symmetry,typename HamiltonianThermal::Mpo::Scalar_> > th;
			
			DMRG::CONTROL::GLOB GlobParam;
			fDMRG.GlobParam.CALC_S_ON_EXIT = false;
			DMRG::CONTROL::DYN DynParam;
			if (dLphys == 2)
			{
				GlobParam.Minit = 10ul;
				GlobParam.Qinit = 10ul;
				GlobParam.INITDIR = DMRG::DIRECTION::RIGHT; // 1=left->right, 0=right->left
				
				size_t start_2site = 0ul;
				size_t end_2site = 10ul;
				//int period_2site = 2ul;
				DynParam.iteration = [start_2site,end_2site] (size_t i) {return (i>=start_2site and i<=end_2site)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
				DynParam.max_alpha_rsvd = [] (size_t i) {return (i<20ul)? 1e8:0.;};
				fDMRG.DynParam = DynParam;
				fDMRG.userSetDynParam();
			}
			fDMRG.GlobParam = GlobParam;
			fDMRG.userSetGlobParam();
			th.state.eps_truncWeight = 0.;
			fDMRG.edgeState(Htherm, th, Q, LANCZOS::EDGE::GROUND, false);
			
			//lout << Htherm.info() << endl;
			th.state.entropy_skim();
			lout << th.state.entropy().transpose() << endl;
			
			bool ALL;
			
			if (dLphys==2)
			{
				vector<bool> ENTROPY_CHECK1;
				for (int l=0; l<dLphys*L-1; l+=dLphys) ENTROPY_CHECK1.push_back(abs(th.state.entropy()(l))>1e-10);
				
				vector<bool> ENTROPY_CHECK2;
				for (int l=1; l<dLphys*L-1; l+=dLphys) ENTROPY_CHECK2.push_back(abs(th.state.entropy()(l))<1e-10);
				
				ALL = all_of(ENTROPY_CHECK1.begin(), ENTROPY_CHECK1.end(), [](const bool v){return v;}) and 
					  all_of(ENTROPY_CHECK2.begin(), ENTROPY_CHECK2.end(), [](const bool v){return v;});
				
				//ALL = true;
			}
			else
			{
	//			vector<bool> ENTROPY_CHECK;
	//			for (int l=0; l<L-1; l+=1) ENTROPY_CHECK.push_back(abs(th.state.entropy()(l))<1e-10);
	//			
	//			ALL = all_of(ENTROPY_CHECK.begin(), ENTROPY_CHECK.end(), [](const bool v){return v;});
				
				ALL = true;
			}
			
			while (ALL == false)
			{
				lout << termcolor::yellow << "restarting..." << termcolor::reset << endl;
				
				typename HamiltonianThermal::Solver fDMRGnew(DMRG::VERBOSITY::ON_EXIT);
				
				DMRG::CONTROL::GLOB GlobParam;
				fDMRGnew.GlobParam.CALC_S_ON_EXIT = false;
				if (GlobParam.INITDIR == DMRG::DIRECTION::RIGHT)
				{
					GlobParam.INITDIR = DMRG::DIRECTION::LEFT;
				}
				else
				{
					GlobParam.INITDIR = DMRG::DIRECTION::RIGHT;
				}
				fDMRGnew.GlobParam = GlobParam;
				fDMRGnew.GlobParam.CALC_S_ON_EXIT = false;
				fDMRGnew.GlobParam.min_halfsweeps = 30ul;
				fDMRGnew.GlobParam.min_halfsweeps = 30ul;
				
				DMRG::CONTROL::DYN DynParam;
				if (dLphys == 2)
				{
					GlobParam.Minit = 10ul;
					GlobParam.Qinit = 10ul;
					
					size_t start_2site = 0ul;
					size_t end_2site = 10ul;
					//int period_2site = 2ul;
					DynParam.iteration = [start_2site,end_2site] (size_t i) {return (i>=start_2site and i<=end_2site)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
					DynParam.max_alpha_rsvd = [] (size_t i) {return (i<30ul)? 1e8:0.;};
					fDMRGnew.DynParam = DynParam;
				}
				
				fDMRGnew.userSetGlobParam();
				fDMRGnew.userSetDynParam();
				
				fDMRGnew.edgeState(Htherm, th, Q, LANCZOS::EDGE::GROUND, false);
				th.state.entropy_skim();
				
				if (dLphys==2)
				{
					lout << th.state.entropy().transpose() << endl;
					vector<bool> ENTROPY_CHECK1, ENTROPY_CHECK2;
					
					for (int l=0; l<2*L-1; l+=2) ENTROPY_CHECK1.push_back(abs(th.state.entropy()(l))>1e-10);
					for (int l=1; l<2*L-1; l+=2) ENTROPY_CHECK2.push_back(abs(th.state.entropy()(l))<1e-10);
					
					for (int l=1; l<2*L-1; l+=2)
					{
						bool TEST = abs(th.state.entropy()(l))<1e-10;
					}
					ALL = all_of(ENTROPY_CHECK1.begin(), ENTROPY_CHECK1.end(), [](const bool v){return v;}) and 
						  all_of(ENTROPY_CHECK2.begin(), ENTROPY_CHECK2.end(), [](const bool v){return v;});
				}
	//			else
	//			{
	//				vector<bool> ENTROPY_CHECK;
	//				for (int l=0; l<L-1; l+=1) ENTROPY_CHECK.push_back(abs(th.state.entropy()(l))<1e-10);
	//				
	//				ALL = all_of(ENTROPY_CHECK.begin(), ENTROPY_CHECK.end(), [](const bool v){return v;});
	//			}
			}
			
			for (int l=0; l<L-dLphys; l+=dLphys)
			{
				double val = 0.;
				if (dLphys==1)
				{
					val = avg(th.state, Htherm.SdagS(l,l,0,1), th.state);
				}
				else
				{
					val = avg(th.state, Htherm.SdagS(l,l+1,0,0), th.state);
				}
				cout << termcolor::yellow
					 << "l=" << l
					// << ", avg(th.state, Htherm.SdagS(NN), th.state)=" << avg(th.state, Htherm.SdagS(l,l+dLphys), th.state) 
					 << ", avg_loc=" << val
					 << termcolor::reset
					 << endl;
			}
			
			PhiT = th.state.template cast<typename Hamiltonian::Mpo::Scalar_>();
		}
		
		// continue with PhiT
		PhiT.eps_truncWeight = tol_compr_beta;
		PhiT.min_Nsv = 0ul;
		PhiT.max_Nsv = Mlim;
		TDVPPropagator<Hamiltonian,
			           typename Hamiltonian::Symmetry,
			           typename Hamiltonian::Mpo::Scalar_,
			           typename Hamiltonian::Mpo::Scalar_,
			           Mps<Symmetry,typename Hamiltonian::Mpo::Scalar_>
			           > TDVPT(Hprop,PhiT);
		
		vector<double> lnZvec;
		
		vector<double> betavals;
		vector<double> betasteps;
		betavals.push_back(0.01);
		betasteps.push_back(0.01);
//		double s_betainit = log((dLphys==2)?Hprop.locBasis(0).size():Hprop.locBasis(0).size()/2);
		
		for (int i=1; i<22; ++i)
		{
			double beta_last = betavals[betavals.size()-1];
			if (beta_last > betamax) break;
			if (i>=20)
			{
				betasteps.push_back(0.05);
				betavals.push_back(beta_last+0.05);
			}
			else
			{
				betasteps.push_back(0.01);
				betavals.push_back(beta_last+0.01);
			}
		}
		
		while (betavals[betavals.size()-1] < betamax)
		{
			betasteps.push_back(dbeta);
			double beta_last = betavals[betavals.size()-1];
			betavals.push_back(beta_last+dbeta);
			
		}
		if (betavals[betavals.size()-1] > betamax+0.005) // needs offset, otherwise random behaviour results from comparing floating point numbers
		{
	//		lout << "popping last value " << betavals[betavals.size()-1] << "\t" << betamax << endl;
			betavals.pop_back();
			betasteps.pop_back();
		}
	//	for (int i=0; i<betavals.size(); ++i)
	//	{
	//		lout << "betaval=" << betavals[i] << ", betastep=" << betasteps[i] << endl;
	//	}
		lout << endl;
		
		ofstream BetaFiler(make_string(wd,"/thermodyn_",th_label,".dat"));
		BetaFiler << "#T\tβ\tc\te\tchi\tchiz\tchic\ts";
		if constexpr (Hamiltonian::FAMILY == HUBBARD or Hamiltonian::FAMILY == KONDO) BetaFiler << "\tnphys";
		if constexpr (Hamiltonian::FAMILY == HEISENBERG) BetaFiler << "\tSpSm\tSzSz";
		BetaFiler << "truncWeightGlob\ttruncWeightLoc\tMmax";
		BetaFiler << endl;
		
		Mpo<Symmetry,typename Hamiltonian::Mpo::Scalar_> Hpropsq;
		if (Ntaylor>0)
		{
			lout << "computing H^2..." << endl;
			Hpropsq = prod(Hprop,Hprop);
			lout << "H^2 done!"  << endl;
		}
		
		Mpo<Symmetry,typename Hamiltonian::Mpo::Scalar_> U;
		
		for (int i=0; i<betasteps.size(); ++i)
		{
			Stopwatch<> betaStepper;
			double beta = betavals[i];
			
			if (i<Ntaylor)
			{
				if (i==0 or betasteps[i]!=betasteps[i-1])
				{
					Mpo<Symmetry,typename Hamiltonian::Mpo::Scalar_> Id = Hprop.Identity(PhiT.locBasis());
					
					Mpo<Symmetry,typename Hamiltonian::Mpo::Scalar_> Hmpo1 = Hprop;
					Hmpo1.scale(-0.5*betasteps[i]); // -1/2*dβ*H
					
					Mpo<Symmetry,typename Hamiltonian::Mpo::Scalar_> Hmpo2 = Hpropsq;
					Hmpo2.scale(0.125*betasteps[i]*betasteps[i]); // +1/8*(dβ*H)^2
					
					lout << "computing 1-1/2*dβ*H+1/8*(dβ*H)^2 for dβ=" << betasteps[i] << endl;
					U = sum(Id,sum(Hmpo1,Hmpo2));
				}
				
				//Mps<Symmetry,Scalar> PhiTmp;
				lout << "applying 1-1/2*dβ*H+1/8*(dβ*H)^2 and compressing for dβ=" << betasteps[i] << endl;
				//OxV_exact(U,PhiT,PhiTmp,1e-9,DMRG::VERBOSITY::ON_EXIT,200,1,Mlim,DMRG::BROOM::QR);
				//HxV(U,PhiT,PhiTmp,true,tol_compr_beta,50,Mlim);
				//OxV(U,U,PhiT,true,tol_compr_beta,50,Mlim,16,6,false);
				OxV(U,U,PhiT,true,tol_compr_beta,50,Mlim,16,6,false);
				//PhiT = PhiTmp;
				
				if (i==Ntaylor-1)
				{
					TDVPT = TDVPPropagator<Hamiltonian,
				           typename Hamiltonian::Symmetry,
				           typename Hamiltonian::Mpo::Scalar_,
				           typename Hamiltonian::Mpo::Scalar_,
				           Mps<Symmetry,typename Hamiltonian::Mpo::Scalar_>
				           >(Hprop,PhiT);
				}
			}
			else
			{
				if (beta>betaswitch)
				{
					TDVPT.t_step0(Hprop, PhiT, -0.5*betasteps[i], 1);
				}
				else
				{
					//if (i==0)
					//{
					//	PhiT.eps_truncWeight = 0.;
					//}
					//else
					//{
						PhiT.eps_truncWeight = tol_compr_beta;
					//}
					TDVPT.t_step(Hprop, PhiT, -0.5*betasteps[i], 1);
				}
			}
			double norm = isReal(dot(PhiT,PhiT));
			lnZvec.push_back(log(norm));
			PhiT /= sqrt(norm);
			lout << TDVPT.info() << endl;
			size_t standard_precision = cout.precision();
			lout << setprecision(16) << PhiT.info() << setprecision(standard_precision) << endl;
			
			bool INTERMEDIATE_SAVEPOINT = false;
			string saveLabel = "";
			for (int is=0; is<stateSavePoints.size(); ++is)
			{
				if (abs(beta-stateSavePoints[is]) < 1e-8)
				{
					INTERMEDIATE_SAVEPOINT = true;
					saveLabel = stateSaveLabels[is];
				}
			}
			if (SAVE_BETA and INTERMEDIATE_SAVEPOINT)
			{
				lout << termcolor::green << "saving the intermediate beta result to: " << saveLabel << " in directory=" << wd << termcolor::reset << endl;
				PhiT.save(make_string(wd,"/betaRes_",saveLabel));
			}
			
			double avg_H = isReal(avg(PhiT,Hprop,PhiT));
			
			double e = avg_H/L;
			
			double c = std::nan("0");
			if (CALC_C)
			{
				double avg_Hsq = (Hprop.maxPower()==1)? isReal(avg(PhiT,Hprop,Hprop,PhiT)):isReal(avg(PhiT,Hprop,PhiT,2));
				double avgH_sq = pow(avg_H,2);
				c = isReal(beta*beta*(avg_Hsq-avgH_sq))/L;
			}
			
			double chi = std::nan("0");
			double chiz = std::nan("0");
			double chic = 0;
			if (CALC_CHI)
			{
				#if defined(USING_SU2xU1) or defined(USING_SU2xSU2) or defined(USING_SU2)
				chi = isReal(beta*avg(PhiT, Hprop.Sdagtot(0,sqrt(3.),dLphys), Hprop.Stot(0,1.,dLphys), PhiT))/L;
				chiz = chi/3.;
				for (int l=0; l<dLphys*L; l+=dLphys)
				{
					chic += isReal(beta*avg(PhiT, Hprop.SdagS(l,dLphys*L/2), PhiT))/3.;
				}
				#else
				chi =  0.5*isReal(beta*avg(PhiT, Hprop.Scomptot(SP,0,1.,dLphys), Hprop.Scomptot(SM,0,1.,dLphys), PhiT))/L;
				chi += 0.5*isReal(beta*avg(PhiT, Hprop.Scomptot(SM,0,1.,dLphys), Hprop.Scomptot(SP,0,1.,dLphys), PhiT))/L;
				chiz = isReal(beta*avg(PhiT, Hprop.Scomptot(SZ,0,1.,dLphys), Hprop.Scomptot(SZ,0,1.,dLphys), PhiT))/L;
				chi += chiz;
				for (int l=0; l<dLphys*L; l+=dLphys)
				{
					chic += isReal(beta*avg(PhiT, Hprop.SzSz(l,dLphys*L/2), PhiT));
				}
				#endif
			}
			
//			double chi_ = beta*(2.*avg(PhiT,Hchi,PhiT)/L+0.75);
			
			VectorXd tmp = VectorXd::Map(lnZvec.data(), lnZvec.size());
			double s = s_betainit + tmp.sum()/L + beta*e;
//			cout << "s_betainit=" << s_betainit << ", tmp.sum()/L=" << tmp.sum()/L << ", beta*e=" << beta*e << ", total=" << s << endl;
//			svec.push_back(s);
			
			auto PhiTtmp = PhiT; PhiTtmp.entropy_skim();
			cout << "aa" << endl;
			lout << "S=" << PhiTtmp.entropy().transpose() << endl;
			
			VectorXd nphys(Lcell); nphys.setZero();
			//VectorXd SdagSphys(Lcell); for (int j=0; j<Lcell; ++j) SdagSphys(j) = 0.;
			double nancl = 0.;
			cout << "a" << endl;
			#if not defined(USING_SU2xSU2)
			if constexpr (Hamiltonian::FAMILY == HUBBARD or Hamiltonian::FAMILY == KONDO)
			{
				cout << "b" << endl;
				for (int j=0, icell=0; j<dLphys*L; j+=dLphys, icell+=1)
				{
					nphys(icell%Lcell) += isReal(avg(PhiT, Hprop.n(j,0), PhiT));
				}
				for (int j=dLphys-1; j<dLphys*L; j+=dLphys)
				{
					nancl += isReal(avg(PhiT, Hprop.n(j,dLphys%2), PhiT));
				}
			}
			#endif
			#if not defined(USING_SU2)
			if constexpr (Hamiltonian::FAMILY == HEISENBERG)
			{
				cout << "c" << endl;
				for (int j=0, icell=0; j<dLphys*L; j+=dLphys, icell+=1)
				{
					nphys(icell%Lcell) += isReal(avg(PhiT, Hprop.Sz(j,0), PhiT));
				}
				for (int j=dLphys-1; j<dLphys*L; j+=dLphys)
				{
					nancl += isReal(avg(PhiT, Hprop.Sz(j,dLphys%2), PhiT));
				}
//				for (int j=0, icell=0; j<dLphys*(L-1); j+=dLphys, icell+=1)
//				{
//					//cout << "j,j+dLphys=" << j << "," << j+dLphys << ", icell=" << icell << ", icellmodLcell=" << icell%Lcell << ", val=" <<  isReal(avg(PhiT, Hprop.SdagS(j,j+dLphys,0,0), PhiT)) << endl;
//					SdagSphys(icell%Lcell) += isReal(avg(PhiT, Hprop.SdagS(j,j+dLphys,0,0), PhiT));
//				}
//			}
//			double SzSz = 0.;
//			double SpSm = 0.;
//			if constexpr (Hamiltonian::FAMILY == HEISENBERG)
//			{
//				SpSm = isReal(avg(PhiT, Hprop.SpSm(L/4,3*L/4), PhiT));
//				SzSz = isReal(avg(PhiT, Hprop.SzSz(L/4,3*L/4), PhiT));
			}
			#endif
			cout << "d" << endl;
			
			nphys /= L;
			double nphystot = nphys.sum();
			nancl /= L;
			
			lout << termcolor::bold << "β=" << beta << ", T=" << 1./beta << ", c=" << c << ", e=" << e << ", chi=" << chi;
			#ifndef USING_SU2
			lout << ", chiz=" << chiz << "(3x=" << 3.*chiz << ")";
			#else
			lout << ", chic=" << chic;
			#endif
			lout << ", s=" << s;
			BetaFiler << setprecision(16) << 1./beta << "\t" << beta << "\t" << c << "\t" << e << "\t" << chi << "\t" << chiz << "\t" << chic << "\t" << s;
			if constexpr (Hamiltonian::FAMILY == HUBBARD or Hamiltonian::FAMILY == KONDO)
			{
				lout << ", nphys=" << nphys.transpose() << ", sum=" << nphystot << ", nancl=" << nancl;
				BetaFiler << "\t" << nphystot;
			}
			else if constexpr (Hamiltonian::FAMILY == HEISENBERG)
			{
				lout << ", Mphys=" << nphys.transpose() << ", sum=" << nphystot << ", Mancl=" << nancl;
				BetaFiler << "\t" << nphystot;
//				lout << ", SpSm(" << L/4 << "," << 3*L/4 << ")=" << SpSm << ", SzSz(" << L/4 << "," << 3*L/4 << ")=" << SzSz;
//				BetaFiler << "\t" << SpSm << "\t" << SzSz;
			}
			BetaFiler << "\t" << PhiT.get_truncWeight().sum() << "\t" << PhiT.get_truncWeight().maxCoeff() << "\t" << PhiT.calc_Mmax();
//			if constexpr (Hamiltonian::FAMILY == HEISENBERG)
//			{
//				double Nb = L-1;
//				lout << ", SdagSphys/Nb=" << SdagSphys.transpose()/Nb << ", sum/Nb=" << SdagSphys.sum()/Nb;
//				BetaFiler << "\t" << SdagSphys.transpose()/Nb;
//			}
			BetaFiler << endl;
			lout << termcolor::reset << endl;
			lout << betaStepper.info("βstep") << endl;
			lout << endl;
		}
		BetaFiler.close();
	}
	
	if (SAVE_BETA and !LOAD_BETA)
	{
		lout << termcolor::green << "saving the final beta result to: " << th_label << " in directory=" << wd << termcolor::reset << endl;
		PhiT.save(make_string(wd,"/betaRes_",th_label));
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
continue_beta_propagation (const Hamiltonian &Hprop, int Lcell, int dLphys, 
                           double s_betainit_input, double betainit, double betamax, double dbeta, double tol_compr_beta, size_t Mlim, qarray<Hamiltonian::Symmetry::Nq> Q, double betaswitch, 
                           string wd, string th_label, string LOAD_BETA, bool SAVE_BETA,
                           DMRG::VERBOSITY::OPTION VERB, 
                           vector<double> stateSavePoints, vector<string> stateSaveLabels, 
                           bool CALC_C, bool CALC_CHI)
{
	for (const auto &spec:specs)
	{
		assert(CHECK_SPEC(spec) and "Wrong spectral abbreviation!");
	}
	L = Hprop.length()/dLphys;
	Nspec = specs.size();
	
	lout << "loading beta result from: " << LOAD_BETA << endl;
	PhiT.load(LOAD_BETA);
	lout << "loaded: " << PhiT.info() << endl;
	
	PhiT.eps_truncWeight = tol_compr_beta;
	PhiT.min_Nsv = 0ul;
	PhiT.max_Nsv = Mlim;
	TDVPPropagator<Hamiltonian,
		           typename Hamiltonian::Symmetry,
		           typename Hamiltonian::Mpo::Scalar_,
		           typename Hamiltonian::Mpo::Scalar_,
		           Mps<Symmetry,typename Hamiltonian::Mpo::Scalar_>
		           > TDVPT(Hprop,PhiT);
	
	vector<double> lnZvec;
	vector<double> betavals;
	vector<double> betasteps;
	
	assert(betainit>=0.2);
	betavals.push_back(betainit+dbeta);
	betasteps.push_back(dbeta);
	lout << "betaval=" << betavals[betavals.size()-1] << ", betastep=" << betasteps[betasteps.size()-1] << endl;
	
	double e = isReal(avg(PhiT,Hprop,PhiT)/static_cast<double>(L));
	double s_betainit = s_betainit_input;
	s_betainit -= betainit*e;
	
	while (betavals[betavals.size()-1] < betamax)
	{
		betasteps.push_back(dbeta);
		double beta_last = betavals[betavals.size()-1];
		betavals.push_back(beta_last+dbeta);
		lout << "betaval=" << betavals[betavals.size()-1] << ", betastep=" << betasteps[betasteps.size()-1] << endl;
	}
	if (betavals[betavals.size()-1] > betamax+0.005) // needs offset, otherwise random behaviour results from comparing floating point numbers
	{
		betavals.pop_back();
		betasteps.pop_back();
	}
	lout << endl;
	
	ofstream BetaFiler(make_string(wd,"/thermodyn_",th_label,".dat"));
	BetaFiler << "#T\tβ\tc\te\tchi\tchiz\tchic\ts";
	if constexpr (Hamiltonian::FAMILY == HUBBARD or Hamiltonian::FAMILY == KONDO) BetaFiler << "\tnphys";
	if constexpr (Hamiltonian::FAMILY == HEISENBERG) BetaFiler << "\tSpSm\tSzSz";
	BetaFiler << "truncWeightGlob\ttruncWeightLoc\tMmax";
	BetaFiler << endl;
	
	for (int i=0; i<betasteps.size(); ++i)
	{
		Stopwatch<> betaStepper;
		double beta = betavals[i];
		
		if (beta>betaswitch)
		{
			PhiT.eps_truncWeight = 0.;
			TDVPT.t_step0(Hprop, PhiT, -0.5*betasteps[i], 1);
		}
		else
		{
			PhiT.eps_truncWeight = tol_compr_beta;
			TDVPT.t_step(Hprop, PhiT, -0.5*betasteps[i], 1);
		}
		double norm = isReal(dot(PhiT,PhiT));
		lnZvec.push_back(log(norm));
		PhiT /= sqrt(norm);
		lout << TDVPT.info() << endl;
		size_t standard_precision = cout.precision();
		lout << setprecision(16) << PhiT.info() << setprecision(standard_precision) << endl;
		
		bool INTERMEDIATE_SAVEPOINT = false;
		string saveLabel = "";
		for (int is=0; is<stateSavePoints.size(); ++is)
		{
			if (abs(beta-stateSavePoints[is]) < 1e-8)
			{
				INTERMEDIATE_SAVEPOINT = true;
				saveLabel = stateSaveLabels[is];
			}
		}
		if (SAVE_BETA and INTERMEDIATE_SAVEPOINT)
		{
			lout << termcolor::green << "saving the intermediate beta result to: " << saveLabel << " in directory=" << wd << termcolor::reset << endl;
			PhiT.save(make_string(wd,"/betaRes_",saveLabel));
		}
		
		double avg_H = isReal(avg(PhiT,Hprop,PhiT));
		
		e = avg_H/L;
		
		double c = std::nan("0");
		if (CALC_C)
		{
			double avg_Hsq = (Hprop.maxPower()==1)? isReal(avg(PhiT,Hprop,Hprop,PhiT)):isReal(avg(PhiT,Hprop,PhiT,2));
			double avgH_sq = pow(avg_H,2);
			c = isReal(beta*beta*(avg_Hsq-avgH_sq))/L;
		}
		
		double chi = std::nan("0");
		double chiz = std::nan("0");
		double chic = 0;
		if (CALC_CHI)
		{
			#ifdef USING_SU2
			chi = isReal(beta*avg(PhiT, Hprop.Sdagtot(0,sqrt(3.),dLphys), Hprop.Stot(0,1.,dLphys), PhiT))/L;
			chiz = chi/3.;
			for (int l=0; l<dLphys*L; l+=dLphys)
			{
				chic += isReal(beta*avg(PhiT, Hprop.SdagS(l,dLphys*L/2), PhiT))/3.;
			}
			#else
			chi =  0.5*isReal(beta*avg(PhiT, Hprop.Scomptot(SP,0,1.,dLphys), Hprop.Scomptot(SM,0,1.,dLphys), PhiT))/L;
			chi += 0.5*isReal(beta*avg(PhiT, Hprop.Scomptot(SM,0,1.,dLphys), Hprop.Scomptot(SP,0,1.,dLphys), PhiT))/L;
			chiz = isReal(beta*avg(PhiT, Hprop.Scomptot(SZ,0,1.,dLphys), Hprop.Scomptot(SZ,0,1.,dLphys), PhiT))/L;
			chi += chiz;
			for (int l=0; l<dLphys*L; l+=dLphys)
			{
				chic += isReal(beta*avg(PhiT, Hprop.SzSz(l,dLphys*L/2), PhiT));
			}
			#endif
		}
		
		VectorXd tmp = VectorXd::Map(lnZvec.data(), lnZvec.size());
		double s = s_betainit + tmp.sum()/L + beta*e;
		
		auto PhiTtmp = PhiT; PhiTtmp.entropy_skim();
		lout << "S=" << PhiTtmp.entropy().transpose() << endl;
		
		VectorXd nphys(Lcell); for (int j=0; j<Lcell; ++j) nphys(j) = 0.;
		double nancl = 0.;
		if constexpr (Hamiltonian::FAMILY == HUBBARD or Hamiltonian::FAMILY == KONDO)
		{
			for (int j=0, icell=0; j<dLphys*L; j+=dLphys, icell+=1)
			{
				nphys(icell%Lcell) += isReal(avg(PhiT, Hprop.n(j,0), PhiT));
			}
			for (int j=dLphys-1; j<dLphys*L; j+=dLphys)
			{
				nancl += isReal(avg(PhiT, Hprop.n(j,dLphys%2), PhiT));
			}
		}
		double SzSz = 0.;
		double SpSm = 0.;
		if constexpr (Hamiltonian::FAMILY == HEISENBERG)
		{
			SpSm = isReal(avg(PhiT, Hprop.SpSm(L/4,3*L/4), PhiT));
			SzSz = isReal(avg(PhiT, Hprop.SzSz(L/4,3*L/4), PhiT));
		}
		
		nphys /= L;
		double nphystot = nphys.sum();
		nancl /= L;
		
		lout << termcolor::bold << "β=" << beta << ", T=" << 1./beta << ", c=" << c << ", e=" << e << ", chi=" << chi;
		#ifndef USING_SU2
		lout << ", chiz=" << chiz << "(3x=" << 3.*chiz << ")";
		#else
		lout << ", chic=" << chic;
		#endif
		lout << ", s=" << s;
		BetaFiler << setprecision(16) << 1./beta << "\t" << beta << "\t" << c << "\t" << e << "\t" << chi << "\t" << chiz << "\t" << chic << "\t" << s;
		if constexpr (Hamiltonian::FAMILY == HUBBARD or Hamiltonian::FAMILY == KONDO)
		{
			lout << ", nphys=" << nphys.transpose() << ", sum=" << nphystot << ", nancl=" << nancl;
			BetaFiler << "\t" << nphystot;
		}
		if constexpr (Hamiltonian::FAMILY == HEISENBERG)
		{
			lout << ", SpSm(" << L/4 << "," << 3*L/4 << ")=" << SpSm << ", SzSz(" << L/4 << "," << 3*L/4 << ")=" << SzSz;
			BetaFiler << "\t" << SpSm << "\t" << SzSz;
		}
		BetaFiler << "\t" << PhiT.get_truncWeight().sum() << "\t" << PhiT.get_truncWeight().maxCoeff() << "\t" << PhiT.calc_Mmax();
		BetaFiler << endl;
		lout << termcolor::reset << endl;
		lout << betaStepper.info("βstep") << endl;
		lout << endl;
	}
	
	BetaFiler.close();
	
	if (SAVE_BETA)
	{
		lout << termcolor::green << "saving the final beta result to: " << th_label << " in directory=" << wd << termcolor::reset << endl;
		PhiT.save(make_string(wd,"/betaRes_",th_label));
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
apply_operators_on_thermal_state (int Lcell, int dLphys, bool CHECK)
{
	PhiTt = PhiT.template cast<complex<double>>();
//	PhiTt.eps_svd = tol_compr;
//	PhiTt.min_Nsv = 0ul;
//	PhiTt.max_Nsv = Mlim;
	
	// OxV for time propagation
	vector<vector<Mpo<typename Hamiltonian::Symmetry,Scalar>>> O(Nspec);
	Odag.resize(Nspec);
	for (int z=0; z<Nspec; ++z) O[z].resize(L);
	for (int z=0; z<Nspec; ++z) Odag[z].resize(L);
	
	for (int z=0; z<Nspec; ++z)
	for (int l=0; l<L; ++l)
	{
		O[z][l] = get_Op(Hwork, dLphys*l, specs[z], 1., 0, dLphys);
		double dagfactor;
		if (specs[z] == "SSF")
		{
			dagfactor = sqrt(3);
		}
		else if (specs[z] == "PES")
		{
			dagfactor = -sqrt(2);
		}
		else if (specs[z] == "IPE")
		{
			dagfactor = +sqrt(2);
		}
		else
		{
			dagfactor = 1.;
		}
		Odag[z][l] = get_Op(Hwork, dLphys*l, DAG(specs[z]), dagfactor, 0, dLphys);
	}
	
	// shift values O→O-<O>
	vector<vector<Scalar>> Oshift(Nspec);
	vector<vector<Scalar>> Odagshift(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		Oshift[z].resize(L);
		Odagshift[z].resize(L);
		for (int l=0; l<L; ++l)
		{
			Oshift[z][l] = avg(PhiT, O[z][l], PhiT);
			Odagshift[z][l] = avg(PhiT, Odag[z][l], PhiT);
//			lout << "spec=" << specs[z] << ", l=" << l << ", shift=" << Oshift[z][l] << endl;
		}
	}
	
	for (int z=0; z<Nspec; ++z)
	for (int l=0; l<L; ++l)
	{
		O[z][l].scale   (1.,-Oshift[z][l]);
		Odag[z][l].scale(1.,-Odagshift[z][l]);
	}
	
	//---------check---------
	if (CHECK)
	{
		lout << endl;
		for (int z=0; z<Nspec; ++z)
		{
			lout << "check z=" << z 
				 << ", spec=" << specs[z] << ", dag=" << DAG(specs[z]) 
				 << ", <O†[z][l]*O[z][l]>=" << avg(PhiTt, Odag[z][L/2], O[z][L/2], PhiTt) 
				 << ", <O†[z][l]>=" << avg(PhiTt, Odag[z][L/2], PhiTt) 
				 << ", <O[z][l]>=" << avg(PhiTt, O[z][L/2], PhiTt) 
//				 << ", g.state: " << avg(g.state, Odag[z][L/2], O[z][L/2], g.state) 
				 << endl;
		}
	}
	
	OxPhiTt.resize(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		OxPhiTt[z].resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		{
			OxV_exact(O[z][L/2+i], PhiTt, OxPhiTt[z][i], 2., DMRG::VERBOSITY::ON_EXIT, 200, 1);
		}
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
set_measurement (int iz, string spec, double factor, int dLphys, qarray<Symmetry::Nq> Q, int Lcell, int measure_interval, string measure_name, string measure_subfolder, bool TRANSFORM)
{
	assert(Green.size()>0);
	
	vector<Mpo<Symmetry,Scalar>> Measure(Hwork.length()/dLphys);
	for (int l=0, iphys=0; l<Hwork.length(); l+=dLphys, iphys+=1)
	{
		Measure[iphys] = get_Op(Hwork, l, spec, factor, 0, dLphys);
		if (TRANSFORM) Measure[l].transform_base(Q,false,L); // PRINT=false
	}
	Green[iz].set_measurement(Measure, Lcell, measure_interval, measure_name, measure_subfolder);
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
resize_Green (string wd, string label, int Ns, double tmax, double dt, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT)
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
	Green[0].set_verbosity(CHOSEN_VERB);
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
compute (string wd, string label, int Ns, double tmax, double dt, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT, size_t Mlim, double tol_DeltaS, double tol_compr)
{
	if (Green.size()==0)
	{
		resize_Green(wd,label,Ns,tmax,dt,wmin,wmax,wpoints,QR,qpoints,INT);
	}
	
	// Propagation
	#pragma omp parallel for
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z].set_tol_DeltaS(tol_DeltaS);
		Green[z].set_lim_Nsv(Mlim);
		Green[z].set_tol_compr(tol_compr);
		
		Green[z].compute_cell(Hwork, OxPhiCellBra[z], OxPhiCellKet[z], Eg, TIME_DIR(spec), true); // COUNTERPROPAGATE=true
		Green[z].save(false); // IGNORE_TX=false
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
compute_finite (size_t j0, string wd, string label, int Ns, double tmax, double dt, double wmin, double wmax, int wpoints, GREEN_INTEGRATION INT, size_t Mlim, double tol_DeltaS, double tol_compr)
{
	if (Green.size()==0)
	{
		resize_Green(wd,label,Ns,tmax,dt,wmin,wmax,wpoints,ZERO_2PI,501,INT);
	}
	
	// Propagation
	#pragma omp parallel for
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z].set_tol_DeltaS(tol_DeltaS);
		Green[z].set_lim_Nsv(Mlim);
		Green[z].set_tol_compr(tol_compr);
		
		Green[z].set_OxPhiFull(OxPhiCellKet[z]); 
		Green[z].compute_one(j0, Hwork, OxPhiCellKet[z], Eg, TIME_DIR(spec), false); // COUNTERPROPAGATE=false
		Green[z].save(false); // IGNORE_TX=false
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
compute_finiteCell (int Lcell, int x0, string wd, string label, double tmax, double dt, double wmin, double wmax, int wpoints,  Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT, size_t Mlim, double tol_DeltaS, double tol_compr)
{
	if (Green.size()==0)
	{
		resize_Green(wd,label,1,tmax,dt,wmin,wmax,wpoints,QR,qpoints,INT);
	}
	
	for (int z=0; z<Nspec; ++z)
	{
		Green[z].set_Lcell(Lcell);
	}
	
	// propagation
	#pragma omp parallel for
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z].set_tol_DeltaS(tol_DeltaS);
		Green[z].set_lim_Nsv(Mlim);
		Green[z].set_tol_compr(tol_compr);
		
		Green[z].set_OxPhiFull(OxPhiCellKet[z]);
		if (Oshift.size() != 0)
		{
			if (Oshift[z][0] != 0.)
			{
				Green[z].set_Oshift(Oshift[z]);
			}
		}
		Green[z].compute_cell(Hwork, OxPhiCellKet[z], Eg, TIME_DIR(spec), false, x0); // COUNTERPROPAGATE=false
		Green[z].save(false); // IGNORE_TX=false
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
compute_thermal (string wd, string label, int dLphys, double tmax, double dt, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT, size_t Mlim, double tol_DeltaS, double tol_compr)
{
	if (Green.size()==0)
	{
		resize_Green(wd,label,1,tmax,dt,wmin,wmax,wpoints,QR,qpoints,INT);
	}
	
	#pragma omp parallel for
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z].set_tol_DeltaS(tol_DeltaS);
		Green[z].set_lim_Nsv(Mlim);
		Green[z].set_tol_compr(tol_compr);
		
		Green[z].compute_thermal_cell(Hwork, Odag[z], PhiTt, OxPhiTt[z], TIME_DIR(spec));
//		Green[z].FT_allSites();
		Green[z].save(false); // IGNORE_TX=false
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
FTcell_xq()
{
	assert(Green.size() == Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		Green[z].FTcell_xq();
		Green[z].save_GtqCell();
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
make_A1P_finite (GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double> > &Gfull, string wd, string label, double tmax, double wmin, double wmax, int wpoints, GREEN_INTEGRATION INT)
{
	auto itPES = find(specs.begin(), specs.end(), "PES");
	auto itIPE = find(specs.begin(), specs.end(), "IPE");
	
	if (itPES != specs.end() and itIPE != specs.end())
	{
		int iPES = distance(specs.begin(), itPES);
		int iIPE = distance(specs.begin(), itIPE);
		
		// Add PES+IPE
		MatrixXcd GinA1P;
		GinA1P.resize(Green[iPES].get_Gtx().rows(), Green[iPES].get_Gtx().cols());
		GinA1P.setZero();
		GinA1P += Green[iPES].get_Gtx() + Green[iIPE].get_Gtx();
		
		Gfull = GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double> >(wd+"A1P"+"_"+label,L,tmax,GinA1P,INT);
		Gfull.recalc_FTw(wmin,wmax,wpoints);
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
		          (wd+spec+"_"+label,L,Ncells,Ns,tmax,{wd+spec+"_"+label+make_string("_L=",L,"x",Ncells,"_tmax=",tmax)},QR,qpoints,INT);
		Green[z].recalc_FTwCell(wmin,wmax,wpoints, TIME_DIR(spec));
		Green[z].save(true); // IGNORE_TX=true
	}
}

template<typename Hamiltonian>
Mpo<typename Hamiltonian::Symmetry,typename Hamiltonian::Mpo::Scalar_> SpectralManager<Hamiltonian>::
get_Op (const Hamiltonian &H, size_t loc, std::string spec, double factor, size_t locy, int dLphys)
{
	Mpo<Symmetry,Scalar> Res;
	bool OPERATOR_IS_SET = false;
	
	// spin structure factor
	if (spec == "SSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			Res = H.S(loc,locy,factor);
			OPERATOR_IS_SET = true;
		}
		else
		{
			Res = H.Scomp(SP,loc,locy);
			OPERATOR_IS_SET = true;
		}
	}
	else if (spec == "SDAGSF")
	{
		if constexpr (Symmetry::IS_SPIN_SU2())
		{
			Res = H.Sdag(loc,locy,factor);
			OPERATOR_IS_SET = true;
		}
		else
		{
			Res = H.Scomp(SM,loc,locy);
			OPERATOR_IS_SET = true;
		}
	}
	else if (spec == "SSZ")
	{
		if constexpr (!Symmetry::IS_SPIN_SU2())
		{
			Res = H.Scomp(SZ,loc,locy);
			OPERATOR_IS_SET = true;
		}
		else
		{
			throw;
		}
	}
	else if (spec == "SPM")
	{
		if constexpr (!Symmetry::IS_SPIN_SU2())
		{
			Res = H.Scomp(SM,loc,locy);
			OPERATOR_IS_SET = true;
		}
		else
		{
			throw;
		}
	}
	else if (spec == "SMP")
	{
		if constexpr (!Symmetry::IS_SPIN_SU2())
		{
			Res = H.Scomp(SP,loc,locy);
			OPERATOR_IS_SET = true;
		}
		else
		{
			throw;
		}
	}
	if constexpr (Hamiltonian::FAMILY == HEISENBERG)
	{
		if (spec == "QSF")
		{
			if constexpr (Symmetry::IS_SPIN_SU2())
			{
				Res = H.Q(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.Scomp(SP,loc,locy);
				OPERATOR_IS_SET = true;
			}
		}
		else if (spec == "QDAGSF")
		{
			if constexpr (Symmetry::IS_SPIN_SU2())
			{
				Res = H.Qdag(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.Scomp(SM,loc,locy);
				OPERATOR_IS_SET = true;
			}
		}
	}
	if constexpr (Hamiltonian::FAMILY == HUBBARD or Hamiltonian::FAMILY == KONDO)
	{
		// photemission
		if (spec == "PES" or spec == "PESUP")
		{
			if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
			{
				Res = H.c(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.template c<UP>(loc,locy);
				OPERATOR_IS_SET = true;
			}
		}
		else if (spec == "PESDN")
		{
			if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
			{
				Res = H.c(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.template c<DN>(loc,locy);
				OPERATOR_IS_SET = true;
			}
		}
		// inverse photoemission
		else if (spec == "IPE" or spec == "IPEUP")
		{
			if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
			{
				Res = H.cdag(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.template cdag<UP>(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
		}
		else if (spec == "IPEDN")
		{
			if constexpr (Symmetry::IS_SPIN_SU2()) // or spinless
			{
				Res = H.cdag(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.template cdag<DN>(loc,locy,factor);
				OPERATOR_IS_SET = true;
			}
		}
		// charge structure factor
		else if (spec == "CSF" or spec == "ICSF")
		{
			if constexpr (!Symmetry::IS_CHARGE_SU2())
			{
				Res = H.n(loc,locy);
				OPERATOR_IS_SET = true;
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
				OPERATOR_IS_SET = true;
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
				OPERATOR_IS_SET = true;
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
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.Tp(loc,locy);
				OPERATOR_IS_SET = true;
			}
		}
		// pseudospin structure factor
		else if (spec == "PDAGSF")
		{
			if constexpr (Symmetry::IS_CHARGE_SU2())
			{
				Res = H.Tdag(loc,locy);
				OPERATOR_IS_SET = true;
			}
			else
			{
				Res = H.Tm(loc,locy);
				OPERATOR_IS_SET = true;
			}
		}
		// pseudospin structure factor: z-component
		else if (spec == "PSZ" or spec == "IPSZ")
		{
			if constexpr (!Symmetry::IS_CHARGE_SU2())
			{
				Res = H.Tz(loc,locy);
				OPERATOR_IS_SET = true;
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
				if (loc<H.length()-dLphys)
				{
					Res = H.cdagc(loc,loc+dLphys,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "HSF operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
			else
			{
				if (loc<H.length()-dLphys)
				{
					Res = H.template cdagc<UP,UP>(loc,loc+dLphys,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "HSF operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
		}
		// inverse hybridization structure factor
		else if (spec == "IHSF")
		{
			if constexpr (Symmetry::IS_SPIN_SU2())
			{
				if (loc<H.length()-dLphys)
				{
					Res = H.cdagc(loc+dLphys,loc,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "IHSF operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
			else
			{
				if (loc<H.length()-dLphys)
				{
					Res = H.template cdagc<UP,UP>(loc+dLphys,loc,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "IHSF operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
		}
		// hybridization triplet structure factor
		else if (spec == "HTS")
		{
			if constexpr (Symmetry::IS_SPIN_SU2())
			{
				if (loc<H.length()-dLphys)
				{
					Res = H.cdagc3(loc,loc+dLphys,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "HTS operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
			else
			{
				if (loc<H.length()-dLphys)
				{
					Res = H.template cdagc<UP,DN>(loc,loc+dLphys,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "HTS operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
		}
		// inverse hybridization triplet structure factor
		else if (spec == "IHTS")
		{
			if constexpr (Symmetry::IS_SPIN_SU2())
			{
				if (loc<H.length()-dLphys)
				{
					Res = H.cdagc3(loc+dLphys,loc,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "IHTS operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
			else
			{
				if (loc<H.length()-dLphys)
				{
					Res = H.template cdagc<DN,UP>(loc+dLphys,loc,0,0);
					OPERATOR_IS_SET = true;
				}
				else
				{
					lout << termcolor::yellow << "IHTS operator hit right edge! Returning zero." << termcolor::reset << endl;
					Res = Hamiltonian::Zero(H.qPhys);
					OPERATOR_IS_SET = true;
				}
			}
		}
	}
	
	Res.set_locality(loc);
	assert(OPERATOR_IS_SET);
	return Res;
}

#endif
