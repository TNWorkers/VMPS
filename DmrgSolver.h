#ifndef VANILLA_DMRGSOLVER
#define VANILLA_DMRGSOLVER

#include "Mps.h"
#include "Mpo.h"
#include "DmrgPivotStuff.h"
#include "LanczosSolver.h"
#include "DmrgContractions.h"

template<size_t D, typename MpHamiltonian>
class DmrgSolver
{
public:
	
	DmrgSolver(DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CURRENT_VERBOSITY(VERBOSITY)
	{};
	
	string info() const;
	
	void edgeState (const MpHamiltonian &H, Eigenstate<Mps<D,double> > &Vout, size_t Dinit=D, 
	                LANCZOS::EDGE::OPTION EDGE=LANCZOS::EDGE::GROUND,
	                double tol_eigval_input=1e-6, double tol_state_input=1e-5);
	
//	double localAvg (const MatrixXd &O, const MPS &V, int loc);
	
private:
	
	size_t N_sites;
	size_t N_sweepsteps, N_halfsweeps;
	size_t Dmax, Mmax;
	double err_eigval, err_state;
	double tol_eigval, tol_state;
	double totalTruncWeight;
	double Eprev;
	
	void sweepStep (const MpHamiltonian &H, Eigenstate<Mps<D,double> > &Vout);
	void LanczosStep (const MpHamiltonian &H, Eigenstate<Mps<D,double> > &Vout, LANCZOS::EDGE::OPTION EDGE);
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	vector<PivotMatrix<D> > Heff;
	
	void build_R (const MpHamiltonian &H, const Eigenstate<Mps<D,double> > &Vout, size_t loc);
	void build_L (const MpHamiltonian &H, const Eigenstate<Mps<D,double> > &Vout, size_t loc);
	
	DMRG::VERBOSITY::OPTION CURRENT_VERBOSITY;
};

template<size_t D, typename MpHamiltonian>
string DmrgSolver<D,MpHamiltonian>::
info() const
{
	stringstream ss;
	ss << "DmrgSolver: ";
	ss << "L=" << N_sites << ", ";
	ss << "D=" << D << ", ";
	ss << "Mmax=" << Mmax << ", Dmax=" << Dmax << ", ";
	ss << "trunc_weight=" << totalTruncWeight << ", ";
	
	ss << "half-sweeps=";
	if ((N_sweepsteps-1)/(N_sites-1)>0)
	{
		ss << (N_sweepsteps-1)/(N_sites-1);
		if ((N_sweepsteps-1)%(N_sites-1)!=0) {ss << "+";}
	}
	if ((N_sweepsteps-1)%(N_sites-1)!=0) {ss << (N_sweepsteps-1)%(N_sites-1) << "/" << (N_sites-1);}
	ss << ", ";
	
	ss << "err_eigval=" << err_eigval << ", err_state=" << err_state << ", ";
	
	return ss.str();
}

template<size_t D, typename MpHamiltonian>
void DmrgSolver<D,MpHamiltonian>::
edgeState (const MpHamiltonian &H, Eigenstate<Mps<D,double> > &Vout, size_t Dinit, LANCZOS::EDGE::OPTION EDGE, double tol_eigval_input, double tol_state_input)
{
	tol_eigval = tol_eigval_input;
	tol_state  = tol_state_input;
	N_sites = H.length();
	N_sweepsteps = N_halfsweeps = 0;
	
	Vout.state.outerResize(H.length());
	Vout.state.innerResize(max(Dinit,D));
	Vout.state.setRandom();
	
	//---<set LR edges>---
	Heff.clear();
	Heff.resize(N_sites);
	MatrixXd Mtmp(1,1); Mtmp << 1.;
	
	Heff[0].L.resize(1);
	Heff[0].L[0] = Mtmp;
	
	Heff[N_sites-1].R.resize(1);
	Heff[N_sites-1].R[0] = Mtmp;
	
	for (size_t l=1; l<N_sites; ++l)
	{
		Heff[l].L.resize(H.auxdim());
		Heff[l-1].R.resize(H.auxdim());
	}
	//---</set LR edges>---
	
	//---<initial sweep>---
	for (size_t l=N_sites-1; l>0; --l)
	{
		Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
		build_R(H, Vout, l-1);
	}
	Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR);
	CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	pivot = 0;
	//---</initial sweep>---
	
	err_eigval = 1.;
	err_state  = 1.;
	
	Vout.state.eps_noise = 1e-8;
	Vout.state.eps_rdm   = 1e-10;
	
	double Eold = numeric_limits<double>::infinity();
	Mps<D,double> Vref = Vout.state;
	size_t halfSweepRange = N_sites;
	
	while (err_eigval>=tol_eigval or err_state>=tol_state or N_sweepsteps<2*N_sites)
	{
		Stopwatch Chronos;
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			LanczosStep(H, Vout, EDGE);
			// wait on the last sweep, could be end of algorithm
			if (j != halfSweepRange)
			{
				sweepStep(H,Vout);
			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1; // one extra step on 1st iteration
		++N_halfsweeps;
		
		err_eigval = fabs(Eold-Vout.energy);
		err_state = fabs(1.-fabs(Vout.state.dot(Vref)));
		
		Eold = Vout.energy;
		Vref = Vout.state;
		Mmax = Vout.state.calc_Mmax();
		Dmax = Vout.state.calc_Dmax();
		totalTruncWeight = Vout.state.truncWeight.sum();
		
		if (N_halfsweeps<2)
		{
			Vout.state.eps_noise = 1e-8;
			Vout.state.eps_rdm = 1e-10;
		}
		else if (N_halfsweeps>=2 and N_halfsweeps<4)
		{
			Vout.state.eps_noise = 1e-9;
			Vout.state.eps_rdm = 1e-12;
		}
		else if (N_halfsweeps>=4 and N_halfsweeps<6)
		{
			Vout.state.eps_noise = 1e-10;
			Vout.state.eps_rdm = 1e-14;
		}
		else if (N_halfsweeps>=6 and N_halfsweeps<8)
		{
			Vout.state.eps_noise = 1e-10;
			Vout.state.eps_rdm = 1e-14;
		}
		else
		{
			Vout.state.eps_noise = 0.;
			Vout.state.eps_rdm = 0.;
		}
		
		// sweep one last time if not at the end of algorithm
		if (err_eigval>=tol_eigval or err_state>=tol_state or N_sweepsteps<2*N_sites)
		{
			sweepStep(H,Vout);
		}
		
		if (CURRENT_VERBOSITY<=1)
		{
			if (EDGE == LANCZOS::EDGE::GROUND)
			{
				cout << "Emin=" << setprecision(13) << Vout.energy << setprecision(5) << " Emin/L=" << Vout.energy/N_sites << endl;
			}
			else
			{
				cout << "Emax=" << setprecision(13) << Vout.energy << setprecision(5) << " Emax/L=" << Vout.energy/N_sites << endl;
			}
			cout << info() << endl;
			cout << Chronos.info("half-sweep") << endl;
		}
	}
}

template<size_t D, typename MpHamiltonian>
void DmrgSolver<D,MpHamiltonian>::
sweepStep (const MpHamiltonian &H, Eigenstate<Mps<D,double> > &Vout)
{
	Vout.state.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::RICH_SVD, &Heff[pivot]);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++pivot) : build_R(H,Vout,--pivot);
	
//	PivotVector<D> Vtmp1;
//	PivotVector<D> Vtmp2;
//	Vtmp1.A = Vout.state.A[pivot];
//	HxV(Heff[pivot],Vtmp1,Vtmp2);
//	double DeltaE_O = Eprev-Vout.energy;
//	double DeltaE_T = dot(Vtmp1,Vtmp2)-Vout.energy;

////	cout << "DeltaE_O=" << DeltaE_O << ", DeltaE_T=" << DeltaE_T << endl;
//	if (DeltaE_T < -0.3*DeltaE_O)
//	{
//		Vout.state.eps_rsvd *= 1.2;
//	}
//	else
//	{
//		Vout.state.eps_rsvd *= 0.8;
//	}
////	cout << "Vout.state.eps_rsvd=" << Vout.state.eps_rsvd << endl << endl;
}

template<size_t D, typename MpHamiltonian>
void DmrgSolver<D,MpHamiltonian>::
LanczosStep (const MpHamiltonian &H, Eigenstate<Mps<D,double> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	if (Heff[pivot].dim == 0)
	{
		Heff[pivot].W = H.W[pivot];
	}
	
	Heff[pivot].dim = 0;
	for (size_t s=0; s<D; ++s)
	{
		Heff[pivot].dim += Vout.state.A[pivot][s].rows() * Vout.state.A[pivot][s].cols();
	}
	
	Eigenstate<PivotVector<D> > g;
	g.state.A = Vout.state.A[pivot];
	LanczosSolver<PivotMatrix<D>,PivotVector<D>,double> Lutz(LANCZOS::REORTHO::FULL);
	Lutz.set_dimK(min(30ul, Heff[pivot].dim));
	Lutz.edgeState(Heff[pivot],g, EDGE, tol_eigval,tol_state, false);
	
	if (CURRENT_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		cout << "loc=" << pivot << "\t" << Lutz.info() << endl;
	}
	
//	Vout.state.eps_rsvd = 0.3*(Vout.energy-g.energy);
//	cout << "Vout.state.eps_rsvd=" << Vout.state.eps_rsvd << endl;
	Eprev = Vout.energy;
	
	Vout.energy = g.energy;
	Vout.state.A[pivot] = g.state.A;
}

// building L and R
template<size_t D, typename MpHamiltonian>
void DmrgSolver<D,MpHamiltonian>::
build_L (const MpHamiltonian &H, const Eigenstate<Mps<D,double> > &Vout, size_t loc)
{
	contract_L(Heff[loc-1].L, Vout.state.A[loc-1], H.W[loc-1], Vout.state.A[loc-1], Heff[loc].L);
}

template<size_t D, typename MpHamiltonian>
void DmrgSolver<D,MpHamiltonian>::
build_R (const MpHamiltonian &H, const Eigenstate<Mps<D,double> > &Vout, size_t loc)
{
	contract_R(Heff[loc+1].R, Vout.state.A[loc+1], H.W[loc+1], Vout.state.A[loc+1], Heff[loc].R);
}

//double DmrgSolver<D,MpHamiltonian>::
//localAvg (const MatrixXd &O, const MPS &V, int loc)
//{
//	MPO OpMPO(N_sites,D);
//	OpMPO.setLocal(loc,O);
//	
//	MPS Vtmp(N_sites,D,Dmax);
//	HxV(OpMPO,V, Vtmp);
//	
//	return dot(V,Vtmp);
//}

#endif
