#ifndef SPECTRAL_MANAGER
#define SPECTRAL_MANAGER

#include "GreenPropagator.h"
#include "models/SpectralFunctionHelpers.h"
#include "DmrgLinearAlgebra.h"

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
				Oshift[z][l] = avg(g.state, VMPS::get_Op<Hamiltonian,Symmetry,Scalar>(Haux,l,specs[z]), g.state);
			}
			else
			{
				Oshift[z][l] = avg(g.state, VMPS::get_Op<Hamiltonian,Symmetry,Scalar>(H,l,specs[z]), g.state);
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
			O[z][l] = VMPS::get_Op<Hamiltonian,Symmetry,Scalar>(H_hetero, Lhetero/2+l, specs[z]);
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
		Green[z].compute_cell(H_hetero, OxPhiCell[z], Eg, VMPS::TIME_DIR(spec), true); // COUNTERPROPAGATE=true
		Green[z].save(false); // IGNORE_TX=false
	}
}

template<typename Hamiltonian>
void SpectralManager<Hamiltonian>::
reload (string wd, const vector<string> &specs_input, string label, int L, int Ncells, int Ns, double tmax, double wmin, double wmax, int wpoints, Q_RANGE QR, int qpoints, GREEN_INTEGRATION INT)
{
	specs = specs_input;
	Nspec = specs.size();
	Green.resize(Nspec);
	int Lhetero = L*Ncells;
	
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z] = GreenPropagator<Hamiltonian,Symmetry,Scalar,complex<double>> 
		          (wd+spec+"_"+label,L,Ncells,Ns,tmax,{wd+spec+"_"+label+make_string("_L=",L,"x",Ncells,"_tmax=",tmax,"_INT=",INT)},QR,qpoints,INT);
		Green[z].recalc_FTwCell(wmin,wmax,wpoints, VMPS::TIME_DIR(spec));
		Green[z].save(true); // IGNORE_TX=true
	}
}

#endif
