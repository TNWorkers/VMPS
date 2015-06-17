#ifndef STRAWBERRY_DMRGSOLVER_WITH_Q
#define STRAWBERRY_DMRGSOLVER_WITH_Q

#include "MpoQ.h"
#include "MpsQ.h"
#include "DmrgPivotStuffQ.h"
#include "DmrgIndexGymnastics.h"
#include "DmrgLinearAlgebraQ.h"
#include "LanczosSolver.h"

template<size_t Nq, typename MpHamiltonian>
class DmrgSolverQ
{
public:
	
	DmrgSolverQ (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	string info() const;
	string eigeninfo() const;
	double memory   (MEMUNIT memunit=GB) const;
	double overhead (MEMUNIT memunit=MB) const;
	
	void edgeState (const MpHamiltonian &H, Eigenstate<MpsQ<Nq,double> > &Vout, qarray<Nq> Qtot_input, size_t Dinit=5, 
	                LANCZOS::EDGE::OPTION EDGE=LANCZOS::EDGE::GROUND,
	                LANCZOS::CONVTEST::OPTION TEST = LANCZOS::CONVTEST::SQ_TEST,
	                double tol_eigval_input=1e-7, double tol_state_input=1e-5, size_t max_halfsweeps=42, size_t min_halfsweeps=6);
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
private:
	
	size_t N_sites;
	size_t N_sweepsteps, N_halfsweeps;
	size_t Dmax, Mmax, Nqmax;
	double err_eigval, err_state;
	double tol_eigval, tol_state;
	double totalTruncWeight;
	
	vector<PivotMatrixQ<Nq,double> > Heff;
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	void LanczosStep (const MpHamiltonian &H, Eigenstate<MpsQ<Nq,double> > &Vout, LANCZOS::EDGE::OPTION EDGE);
	void sweepStep (const MpHamiltonian &H, Eigenstate<MpsQ<Nq,double> > &Vout);
	
	/**Constructs the left transfer matrix at chain site \p loc (left environment of \p loc).*/
	void build_L (const MpHamiltonian &H, const Eigenstate<MpsQ<Nq,double> > &Vout, size_t loc);
	/**Constructs the right transfer matrix at chain site \p loc (right environment of \p loc).*/
	void build_R (const MpHamiltonian &H, const Eigenstate<MpsQ<Nq,double> > &Vout, size_t loc);
	
	DMRG::VERBOSITY::OPTION   CHOSEN_VERBOSITY;
	LANCZOS::CONVTEST::OPTION CHOSEN_CONVTEST;
};

template<size_t Nq, typename MpHamiltonian>
string DmrgSolverQ<Nq,MpHamiltonian>::
info() const
{
	stringstream ss;
	ss << "DmrgSolverQ: ";
	ss << "L=" << N_sites << ", ";
//	ss << "D=" << D << ", ";
	ss << "Mmax=" << Mmax << ", Dmax=" << Dmax << ", " << "Nqmax=" << Nqmax << ", ";
	ss << "trunc_weight=" << totalTruncWeight << ", ";
	ss << eigeninfo();
	return ss.str();
}

template<size_t Nq, typename MpHamiltonian>
string DmrgSolverQ<Nq,MpHamiltonian>::
eigeninfo() const
{
	stringstream ss;
	ss << "half-sweeps=";
	if ((N_sweepsteps-1)/(N_sites-1)>0)
	{
		ss << (N_sweepsteps-1)/(N_sites-1);
		if ((N_sweepsteps-1)%(N_sites-1)!=0) {ss << "+";}
	}
	if ((N_sweepsteps-1)%(N_sites-1)!=0) {ss << (N_sweepsteps-1)%(N_sites-1) << "/" << (N_sites-1);}
	ss << ", ";
	
	ss << "err_eigval=" << err_eigval << ", err_state=" << err_state << ", ";
	
	ss << "mem=" << round(memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	
	return ss.str();
}

template<size_t Nq, typename MpHamiltonian>
double DmrgSolverQ<Nq,MpHamiltonian>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<N_sites; ++l)
	{
		res += Heff[l].L.memory(memunit);
		res += Heff[l].R.memory(memunit);
		for (size_t s1=0; s1<Heff[l].W.size(); ++s1)
		for (size_t s2=0; s2<Heff[l].W[s1].size(); ++s2)
		{
			res += calc_memory(Heff[l].W[s1][s2],memunit);
		}
	}
	return res;
}

template<size_t Nq, typename MpHamiltonian>
double DmrgSolverQ<Nq,MpHamiltonian>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<N_sites; ++l)
	{
		res += Heff[l].L.overhead(memunit);
		res += Heff[l].R.overhead(memunit);
		res += 2. * calc_memory<size_t>(Heff[l].qlhs.size(),memunit);
		res += 4. * calc_memory<size_t>(Heff[l].qrhs.size(),memunit);
	}
	return res;
}

template<size_t Nq, typename MpHamiltonian>
void DmrgSolverQ<Nq,MpHamiltonian>::
edgeState (const MpHamiltonian &H, Eigenstate<MpsQ<Nq,double> > &Vout, qarray<Nq> Qtot_input, size_t Dinit, LANCZOS::EDGE::OPTION EDGE, LANCZOS::CONVTEST::OPTION TEST, double tol_eigval_input, double tol_state_input, size_t max_halfsweeps, size_t min_halfsweeps)
{
	tol_eigval = tol_eigval_input;
	tol_state  = tol_state_input;
	N_sites = H.length();
	N_sweepsteps = N_halfsweeps = 0;
	
	Stopwatch Aion;
	Vout.state = MpsQ<Nq,double>(H, Dinit, Qtot_input);
	Vout.state.setRandom();
	
	// set edges
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Nq>{Qtot_input, Qtot_input, qvacuum<Nq>()});
	
	// left-to-right:
	for (size_t l=N_sites-1; l>0; --l)
	{
		Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
		build_R(H,Vout,l-1);
	}
	Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
	CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	pivot = 0;
	
	// right-to-left:
//	for (size_t l=0; l<N_sites-1; ++l)
//	{
//		Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
//		build_L(H,Vout,l+1);
//	}
//	Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
//	CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
//	pivot = N_sites-1;
	
	if (CHOSEN_VERBOSITY>=2) {lout << Aion.info("initial state & sweep") << endl << endl;}
	
	err_eigval = 1.;
	err_state  = 1.;
	
	// lambda function to print tolerances
	auto print_eps = [this,&Vout] ()
	{
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << "ε_noise=" << Vout.state.eps_noise 
			     << ", ε_rdm=" << Vout.state.eps_rdm 
			     << ", ε_rsvd=" << Vout.state.eps_rsvd 
			     << endl;
		}
	};
	
	Vout.state.eps_noise = 1e-7;
	Vout.state.eps_rdm = 1e-11;
	Vout.state.eps_rsvd = 1e-2;
	print_eps();
	
	double Eold = numeric_limits<double>::quiet_NaN();
	MpsQ<Nq,double> Vref;
	if (TEST == LANCZOS::CONVTEST::NORM_TEST or
	    TEST == LANCZOS::CONVTEST::COEFFWISE)
	{
		Vref = Vout.state;
	}
	size_t halfSweepRange = N_sites;
	
	double err_state_before_end_of_noise;
	
	while (((err_eigval >= tol_eigval or err_state >= tol_state) and N_halfsweeps < max_halfsweeps) or 
	        N_halfsweeps < min_halfsweeps)
	{
		Stopwatch Chronos;
		
		// sweep
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			LanczosStep(H, Vout, EDGE);
			if (j != halfSweepRange) // wait on the last sweep, could be end of algorithm
			{
				sweepStep(H,Vout);
			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1; // one extra step on 1st iteration
		++N_halfsweeps;
		
		// calculate error
		err_eigval = fabs(Eold-Vout.energy)/this->N_sites;
		if (TEST == LANCZOS::CONVTEST::NORM_TEST or
		    TEST == LANCZOS::CONVTEST::COEFFWISE)
		{
			err_state = fabs(1.-fabs(dot(Vout.state,Vref)));
		}
		else if (TEST == LANCZOS::CONVTEST::SQ_TEST)
		{
			Stopwatch Aion;
			double avgHsq = (H.check_SQUARE()==true)? isReal(avg(Vout.state,H,Vout.state,true)) : isReal(avg(Vout.state,H,H,Vout.state)); 
			err_state = fabs(avgHsq-pow(Vout.energy,2));
			if (CHOSEN_VERBOSITY>=2)
			{
				lout << Aion.info("<H^2>") << endl;
			}
			if (N_halfsweeps == 24) {err_state_before_end_of_noise = err_state;}
		}
		
		Eold = Vout.energy;
		if (TEST == LANCZOS::CONVTEST::NORM_TEST or
		    TEST == LANCZOS::CONVTEST::COEFFWISE)
		{
			Vref = Vout.state;
		}
		
		// calculate stats
		Mmax = Vout.state.calc_Mmax();
		Dmax = Vout.state.calc_Dmax();
		Nqmax = Vout.state.calc_Nqmax();
		totalTruncWeight = Vout.state.truncWeight.sum();
		
		// sweep one last time if not at the end of algorithm
		if ((err_eigval >= tol_eigval or err_state >= tol_state) and 
		    N_halfsweeps < max_halfsweeps)
		{
			sweepStep(H,Vout);
		}
		
//		// adjust noise parameter
//		if (N_halfsweeps<6)
//		{
//			Vout.state.eps_noise = 1e-7;
//			Vout.state.eps_rdm = 1e-11;
//		}
//		else if (N_halfsweeps>=6 and N_halfsweeps<12)
//		{
//			Vout.state.eps_noise = 1e-8;
//			Vout.state.eps_rdm = 1e-12;
//		}
//		else if (N_halfsweeps>=12 and N_halfsweeps<18)
//		{
//			Vout.state.eps_noise = 1e-9;
//			Vout.state.eps_rdm = 1e-13;
//		}
//		else if (N_halfsweeps>=18 and N_halfsweeps<24)
//		{
//			Vout.state.eps_noise = 1e-10;
//			Vout.state.eps_rdm = 1e-14;
//		}
//		else if (N_halfsweeps>=24 and N_halfsweeps<30)
//		{
//			Vout.state.eps_noise = 0.;
//			Vout.state.eps_rdm = 0.;
//		}
//		else if (N_halfsweeps>=30 and N_halfsweeps<36)
//		{
//			// reshake if in local minimum
//			Vout.state.eps_noise = 1e-7;
//			Vout.state.eps_rdm = 1e-11;
//		}
//		else
//		{
//			Vout.state.eps_noise = 0.;
//			Vout.state.eps_rdm = 0.;
//		}
		
		// adjust noise parameter
		if (N_halfsweeps == 6)
		{
			Vout.state.eps_noise = 1e-8;
			Vout.state.eps_rdm = 1e-12;
			Vout.state.eps_rsvd = 1e-2;
			print_eps();
		}
		else if (N_halfsweeps == 12)
		{
			Vout.state.eps_noise = 1e-9;
			Vout.state.eps_rdm = 1e-13;
			Vout.state.eps_rsvd = 1e-2;
			print_eps();
		}
		else if (N_halfsweeps == 18)
		{
			Vout.state.eps_noise = 1e-10;
			Vout.state.eps_rdm = 1e-14;
			Vout.state.eps_rsvd = 1e-3;
			print_eps();
		}
		else if (N_halfsweeps == 24)
		{
			Vout.state.eps_noise = 0.;
			Vout.state.eps_rdm = 0.;
			Vout.state.eps_rsvd = 0.;
			print_eps();
		}
		else if (N_halfsweeps == 30)
		{
			// reshake if in local minimum
			if (err_state/err_state_before_end_of_noise >= 0.8)
			{
				Vout.state.eps_noise = 1e-7;
				Vout.state.eps_rdm = 1e-11;
				Vout.state.eps_rsvd = 1e-2;
				if (CHOSEN_VERBOSITY != DMRG::VERBOSITY::SILENT)
				{
					lout << "local minimum detected, reshaking!" << endl;
				}
				print_eps();
			}
			else
			{
				Vout.state.eps_noise = 0.;
				Vout.state.eps_rdm = 0.;
				Vout.state.eps_rsvd = 0.;
			}
		}
		else if (N_halfsweeps == 36)
		{
			Vout.state.eps_noise = 0.;
			Vout.state.eps_rdm = 0.;
			Vout.state.eps_rsvd = 0.;
			print_eps();
		}
		
		// print stuff
		if (CHOSEN_VERBOSITY >= 2)
		{
			size_t standard_precision = cout.precision();
			if (EDGE == LANCZOS::EDGE::GROUND)
			{
				lout << "Emin=" << setprecision(13) << Vout.energy << " Emin/L=" << Vout.energy/N_sites << setprecision(standard_precision) << endl;
			}
			else
			{
				lout << "Emax=" << setprecision(13) << Vout.energy << " Emax/L=" << Vout.energy/N_sites << setprecision(standard_precision) << endl;
			}
			lout << eigeninfo() << endl;
			lout << Vout.state.info() << endl;
			lout << Chronos.info("half-sweep") << endl;
			lout << endl;
		}
	}
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::ON_EXIT)
	{
		size_t standard_precision = cout.precision();
		if (EDGE == LANCZOS::EDGE::GROUND)
		{
			lout << "Emin=" << setprecision(13) << Vout.energy << " Emin/L=" << Vout.energy/N_sites << setprecision(standard_precision) << endl;
		}
		else
		{
			lout << "Emax=" << setprecision(13) << Vout.energy << " Emax/L=" << Vout.energy/N_sites << setprecision(standard_precision) << endl;
		}
		lout << "DmrgSolverQ: " << eigeninfo() << endl;
		lout << "Vout: " << Vout.state.info() << endl;
	}
}

template<size_t Nq, typename MpHamiltonian>
void DmrgSolverQ<Nq,MpHamiltonian>::
sweepStep (const MpHamiltonian &H, Eigenstate<MpsQ<Nq,double> > &Vout)
{
//	Vout.state.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::RDM, &Heff[pivot]);
	Vout.state.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::RICH_SVD, &Heff[pivot]);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++pivot) : build_R(H,Vout,--pivot);
}

template<size_t Nq, typename MpHamiltonian>
void DmrgSolverQ<Nq,MpHamiltonian>::
LanczosStep (const MpHamiltonian &H, Eigenstate<MpsQ<Nq,double> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	if (Heff[pivot].dim == 0)
	{
		Heff[pivot].W = H.W[pivot];
		precalc_blockStructure (Heff[pivot].L, Vout.state.A[pivot], Heff[pivot].W, Vout.state.A[pivot], Heff[pivot].R, 
		                        H.locBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs);
	}
	
	// reset dim
	Heff[pivot].dim = 0;
	for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
	for (size_t q=0; q<Vout.state.A[pivot][s].dim; ++q)
	{
		Heff[pivot].dim += Vout.state.A[pivot][s].block[q].rows() * Vout.state.A[pivot][s].block[q].cols();
	}
	
	Eigenstate<PivotVectorQ<Nq,double> > g;
	g.state.A = Vout.state.A[pivot];
	LanczosSolver<PivotMatrixQ<Nq,double>,PivotVectorQ<Nq,double>,double> Lutz(LANCZOS::REORTHO::FULL);
	
	Lutz.set_dimK(min(30ul, Heff[pivot].dim));
//	Lutz.edgeState(Heff[pivot],g, EDGE, tol_eigval,tol_state, false);
	Lutz.edgeState(Heff[pivot],g, EDGE, 1e-4,1e-3, false);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "loc=" << pivot << "\t" << Lutz.info() << endl;
//		lout << Vout.state.test_ortho() << endl;
	}
	
	Vout.energy = g.energy;
	Vout.state.A[pivot] = g.state.A;
}


template<size_t Nq, typename MpHamiltonian>
inline void DmrgSolverQ<Nq,MpHamiltonian>::
build_L (const MpHamiltonian &H, const Eigenstate<MpsQ<Nq,double> > &Vout, size_t loc)
{
	contract_L(Heff[loc-1].L, Vout.state.A[loc-1], H.W[loc-1], Vout.state.A[loc-1], H.locBasis(loc-1), Heff[loc].L);
}

template<size_t Nq, typename MpHamiltonian>
inline void DmrgSolverQ<Nq,MpHamiltonian>::
build_R (const MpHamiltonian &H, const Eigenstate<MpsQ<Nq,double> > &Vout, size_t loc)
{
	contract_R(Heff[loc+1].R, Vout.state.A[loc+1], H.W[loc+1], Vout.state.A[loc+1], H.locBasis(loc+1), Heff[loc].R);
}

#endif
