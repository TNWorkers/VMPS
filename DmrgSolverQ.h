#ifndef STRAWBERRY_DMRGSOLVER_WITH_Q
#define STRAWBERRY_DMRGSOLVER_WITH_Q

#include "MpoQ.h"
#include "MpsQ.h"
#include "DmrgPivotStuffQ.h"
#include "DmrgIndexGymnastics.h"
#include "DmrgLinearAlgebraQ.h"
#include "LanczosSolver.h"

template<typename Symmetry, typename MpHamiltonian, typename Scalar = double>
class DmrgSolverQ
{
	static constexpr size_t Nq = Symmetry::Nq;
	typedef typename Symmetry::qType qType;
public:
	
	DmrgSolverQ (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	string info() const;
	string eigeninfo() const;
	double memory   (MEMUNIT memunit=GB) const;
	double overhead (MEMUNIT memunit=MB) const;
	
	void edgeState (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, 
	                LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND,
	                LANCZOS::CONVTEST::OPTION TEST = LANCZOS::CONVTEST::SQ_TEST,
	                double tol_eigval_input=1e-7, double tol_state_input=1e-6, 
	                size_t Dinit=4, size_t Dlimit=500, 
	                size_t max_halfsweeps=50, size_t min_halfsweeps=6, 
                    double alpha_rsvd_input=1e-1, double eps_svd_input=1e-7, 
	                size_t savePeriod=0);
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	void prepare (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, bool useState=false, size_t Dinit=5,
	              double alpha_rsvd_input=10., double eps_svd_input=1e-7);
	void halfsweep (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, 
	                LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND, 
	                LANCZOS::CONVTEST::OPTION TEST = LANCZOS::CONVTEST::SQ_TEST);
	void cleanup (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, 
	              LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND);
	
	/**Returns the current error of the eigenvalue while the sweep process.*/
	inline double get_errEigval() const {return err_eigval;};
	
	/**Returns the current error of the state while the sweep process.*/
	inline double get_errState() const {return err_state;};
	
	void push_back(const MpsQ<Symmetry,Scalar> &Psi0_input)
	{
		Psi0.push_back(Psi0_input);
	};
	
private:
	
	size_t N_sites;
	size_t Dmax, Mmax, Nqmax;
	double tol_eigval, tol_state;
	double totalTruncWeight;
	size_t Dmax_old;
	size_t N_sweepsteps, N_halfsweeps;
	double err_eigval, err_state, err_state_before_end_of_noise;
	
	vector<PivotMatrixQ<Symmetry,Scalar,Scalar> > Heff; // Scalar = MpoScalar for ground state
	
	double Eold;
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	void LanczosStep (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE);
	void sweepStep (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout);
	
	/**Constructs the left transfer matrix at chain site \p loc (left environment of \p loc).*/
	void build_L (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc);
	
	/**Constructs the right transfer matrix at chain site \p loc (right environment of \p loc).*/
	void build_R (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc);
	
	void build_PL (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc);
	void build_PR (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc);
	
	/**Projected-out states to find the edge of the spectrum.*/
	vector<MpsQ<Symmetry,Scalar> > Psi0;
	
	/**Energy penalty for projected-out states.*/
	double Epenalty = 10.;
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	LANCZOS::CONVTEST::OPTION CHOSEN_CONVTEST;
};

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
string DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
info() const
{
	stringstream ss;
	ss << "DmrgSolverQ: ";
	ss << "L=" << N_sites << ", ";
	ss << "Mmax=" << Mmax << ", Dmax=" << Dmax << ", " << "Nqmax=" << Nqmax << ", ";
	ss << "trunc_weight=" << totalTruncWeight << ", ";
	ss << eigeninfo();
	return ss.str();
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
string DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
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

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
double DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
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

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
double DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
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

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
prepare (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, bool USE_STATE, size_t Dinit,
         double alpha_rsvd_input, double eps_svd_input)
{
	N_sites = H.length();
	N_sweepsteps = N_halfsweeps = 0;
	
	Stopwatch<> PrepTimer;
	
	if (!USE_STATE)
	{
		// resize Vout
		Vout.state = MpsQ<Symmetry,Scalar>(H, Dinit, Qtot_input);
		Vout.state.N_sv = Dinit;
		Vout.state.setRandom();
	}
	Dmax_old = Dinit;
	
	// set edges
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Nq>{Qtot_input, Qtot_input, Symmetry::qvacuum()});
	
	// initial sweep, right-to-left:
	for (size_t l=N_sites-1; l>0; --l)
	{
		Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
		build_R(H,Vout,l-1);
	}
	Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
	CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	pivot = 0;
	
//	// initial sweep, left-to-right:
//	for (size_t l=0; l<N_sites-1; ++l)
//	{
//		cout << "l=" << l << endl;
//		Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
//		build_L(H,Vout,l+1);
//	}
//	Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
//	CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
//	pivot = N_sites-1;
	
	// resize environments for projected-out states
	if (Psi0.size()>0)
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			Heff[l].Epenalty = Epenalty;
			Heff[l].PL.resize(Psi0.size());
			Heff[l].PR.resize(Psi0.size());
			Heff[l].A0.resize(Psi0.size());
			for (size_t n=0; n<Psi0.size(); ++n)
			{
				Heff[l].A0[n] = Psi0[n].A[l];
			}
		}
	}
	
	// build environments for projected-out states
	for (size_t n=0; n<Psi0.size(); ++n)
	{
		Heff[0].PL[n].setVacuum();
		for (size_t l=1; l<N_sites; ++l)
		{
			Heff[l].PL[n] = Vout.state.A[l-1][0].adjoint() * Heff[l-1].PL[n] * Psi0[n].A[l-1][0];
			for (size_t s=1; s<Vout.state.locBasis(l-1).size(); ++s)
			{
				Heff[l].PL[n] += Vout.state.A[l-1][s].adjoint() * Heff[l-1].PL[n] * Psi0[n].A[l-1][s];
			}
		}
		
		Heff[N_sites-1].PR[n].setTarget(Vout.state.Qtot);
		for (int l=N_sites-2; l>=0; --l)
		{
			Heff[l].PR[n] = Psi0[n].A[l+1][0] * Heff[l+1].PR[n] * Vout.state.A[l+1][0].adjoint();
			for (size_t s=1; s<Vout.state.locBasis(l+1).size(); ++s)
			{
				Heff[l].PR[n] += Psi0[n].A[l+1][s] * Heff[l+1].PR[n] * Vout.state.A[l+1][s].adjoint();
			}
		}
	}
	
	if (CHOSEN_VERBOSITY>=2) {lout << PrepTimer.info("initial state & sweep") << endl << endl;}
	
	// initial energy
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Rtmp;
	contract_R(Heff[0].R, Vout.state.A[0], H.W[0], Vout.state.A[0], H.locBasis(0), Rtmp);
	assert(Rtmp.dim == 1 and 
	       Rtmp.block[0][0][0].rows() == 1 and
	       Rtmp.block[0][0][0].cols() == 1 and
	       "Result of contraction <ψ|H|ψ> in DmrgSolverQ::prepare is not a scalar!");
	Eold = isReal(Rtmp.block[0][0][0](0,0));
	
	// initial cutoffs
	Vout.state.eps_svd = eps_svd_input;
	Vout.state.alpha_rsvd = alpha_rsvd_input;
	
	err_eigval = 1.;
	err_state  = 1.;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
halfsweep (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE, LANCZOS::CONVTEST::OPTION TEST)
{
	Stopwatch<> HalfsweepTimer;
	
	// save state for reference
	MpsQ<Symmetry,Scalar> Vref;
	if (TEST == LANCZOS::CONVTEST::NORM_TEST or
	    TEST == LANCZOS::CONVTEST::COEFFWISE)
	{
		Vref = Vout.state;
	}
	
	size_t halfsweepRange = (N_halfsweeps==0)? N_sites : N_sites-1; // one extra step on 1st iteration
	for (size_t j=1; j<=halfsweepRange; ++j)
	{
		bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
		LanczosStep(H, Vout, EDGE);
		sweepStep(H,Vout);
		++N_sweepsteps;
	}
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
		Stopwatch<> HsqTimer;
		double avgHsq = (H.check_SQUARE()==true)? isReal(avg(Vout.state,H,Vout.state,true)) : isReal(avg(Vout.state,H,H,Vout.state));
		err_state = fabs(avgHsq-pow(Vout.energy,2))/this->N_sites;
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << HsqTimer.info("<H^2>") << endl;
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
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
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
		lout << HalfsweepTimer.info("half-sweep") << endl; //", " << Saturn.info("total",false) << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
cleanup (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	if      (pivot==1)         {Vout.state.sweep(0,DMRG::BROOM::QR);}
	else if (pivot==N_sites-2) {Vout.state.sweep(N_sites-1,DMRG::BROOM::QR);}
	
	Vout.state.set_defaultCutoffs();
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		size_t standard_precision = cout.precision();
		string Eedge = (EDGE == LANCZOS::EDGE::GROUND)? "Emin" : "Emax";
		lout << Eedge << "=" << setprecision(13) << Vout.energy << ", " << Eedge << "/L=" << Vout.energy/N_sites << setprecision(standard_precision) << endl;
		lout << Vout.state.info() << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, LANCZOS::EDGE::OPTION EDGE, LANCZOS::CONVTEST::OPTION TEST, double tol_eigval_input, double tol_state_input, size_t Dinit, size_t Dlimit, size_t max_halfsweeps, size_t min_halfsweeps, double alpha_rsvd_input, double eps_svd_input, size_t savePeriod)
{
	tol_eigval = tol_eigval_input;
	tol_state  = tol_state_input;
	
	prepare(H, Vout, Qtot_input, false, Dinit, alpha_rsvd_input, eps_svd_input);
	
	Stopwatch<> Saturn;
	
	// lambda function to print tolerances
	auto print_alpha_eps = [this,&Vout] ()
	{
		if (CHOSEN_VERBOSITY>=2)
		{
			lout //<< "α_noise=" << Vout.state.alpha_noise << ", "
			     //<< "ε_rdm=" << Vout.state.eps_rdm << ", "
			     << "α_rsvd=" << Vout.state.alpha_rsvd << ", "
			     << "ε_svd=" << Vout.state.eps_svd 
			     << endl;
		}
	};
	
	print_alpha_eps();
	
 	// average local dimension for bond dimension increase
	size_t dimqlocAvg = 0;
	for (size_t l=0; l<H.length(); ++l)
	{
		dimqlocAvg += H.locBasis(l).size();
	}
	dimqlocAvg /= H.length();
	
	while (((err_eigval >= tol_eigval or err_state >= tol_state) and N_halfsweeps < max_halfsweeps) or 
		   N_halfsweeps < min_halfsweeps)
	{
		// sweep
		halfsweep(H,Vout,EDGE,TEST);
		
		// If truncated weight too large, increase upper limit per subspace by 10%, but at least by dimqlocAvg, overall never larger than Dlimit
		if (N_halfsweeps%2 == 0 and totalTruncWeight >= Vout.state.eps_svd)
		{
			Vout.state.N_sv = min(max(static_cast<size_t>(1.1*Vout.state.N_sv), Vout.state.N_sv+dimqlocAvg), Dlimit);
		}
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			if (Vout.state.N_sv != Dmax_old)
			{
				lout << "Dmax=" << Dmax_old << "→" << Vout.state.N_sv << endl;
				Dmax_old = Vout.state.N_sv;
			}
			lout << endl;
		}
		
		#ifdef USE_HDF5_STORAGE
		if (savePeriod != 0 and N_halfsweeps%savePeriod == 0)
		{
			Vout.state.save("mpsBackup");
		}
		#endif
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << Saturn.info("total runtime") << endl;
	}
	cleanup(H,Vout,EDGE);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
sweepStep (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout)
{
//	Vout.state.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::RDM, &Heff[pivot]); // obsolete
	Vout.state.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::RICH_SVD, &Heff[pivot]);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++pivot) : build_R(H,Vout,--pivot);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,pivot)  : build_PR(H,Vout,pivot);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
LanczosStep (const MpHamiltonian &H, Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	if (Heff[pivot].dim == 0)
	{
		Heff[pivot].W = H.W[pivot];
		precalc_blockStructure (Heff[pivot].L, Vout.state.A[pivot], Heff[pivot].W, Vout.state.A[pivot], Heff[pivot].R, 
		                        H.locBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs);
		Heff[pivot].qloc = H.locBasis(pivot);
	}
	
	// reset dim
	Heff[pivot].dim = 0;
	for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
	for (size_t q=0; q<Vout.state.A[pivot][s].dim; ++q)
	{
		Heff[pivot].dim += Vout.state.A[pivot][s].block[q].rows() * Vout.state.A[pivot][s].block[q].cols();
	}
	
	Eigenstate<PivotVectorQ<Nq,Scalar> > g;
	g.state.A = Vout.state.A[pivot];
	LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
	
	Lutz.set_dimK(min(30ul, Heff[pivot].dim));
	Lutz.edgeState(Heff[pivot],g, EDGE, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "loc=" << pivot << "\t" << Lutz.info() << endl;
		lout << Vout.state.test_ortho() << endl;
	}
	
	Vout.energy = g.energy;
	Vout.state.A[pivot] = g.state.A;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
build_L (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc)
{
	contract_L(Heff[loc-1].L, Vout.state.A[loc-1], H.W[loc-1], Vout.state.A[loc-1], H.locBasis(loc-1), Heff[loc].L);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
build_R (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc)
{
	contract_R(Heff[loc+1].R, Vout.state.A[loc+1], H.W[loc+1], Vout.state.A[loc+1], H.locBasis(loc+1), Heff[loc].R);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
build_PL (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc)
{
	for (size_t n=0; n<Psi0.size(); ++n)
	{
		Heff[loc].PL[n] = Vout.state.A[loc-1][0].adjoint() * Heff[loc-1].PL[n] * Psi0[n].A[loc-1][0];
		
		for (size_t s=1; s<Vout.state.locBasis(loc-1).size(); ++s)
		{
			Heff[loc].PL[n] += Vout.state.A[loc-1][s].adjoint() * Heff[loc-1].PL[n] * Psi0[n].A[loc-1][s];
		}
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolverQ<Symmetry,MpHamiltonian,Scalar>::
build_PR (const MpHamiltonian &H, const Eigenstate<MpsQ<Symmetry,Scalar> > &Vout, size_t loc)
{
	for (size_t n=0; n<Psi0.size(); ++n)
	{
		Heff[loc].PR[n] = Psi0[n].A[loc+1][0] * Heff[loc+1].PR[n] * Vout.state.A[loc+1][0].adjoint();
		
		for (size_t s=1; s<Vout.state.locBasis(loc+1).size(); ++s)
		{
			Heff[loc].PR[n] += Psi0[n].A[loc+1][s] * Heff[loc+1].PR[n] * Vout.state.A[loc+1][s].adjoint();
		}
	}
}

#endif
