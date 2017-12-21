#ifndef VANILLA_IDMRGSOLVER
#define VANILLA_IDMRGSOLVER

#include "MpoQ.h"
#include "UmpsQ.h"
#include "DmrgPivotStuffQ.h"
#include "DmrgPivotStuff2Q.h"
#include "DmrgIndexGymnastics.h"
#include "DmrgLinearAlgebraQ.h"
#include "LanczosSolver.h"

template<size_t Nq, typename MpHamiltonian, typename Scalar=double>
class iDmrgSolver
{
public:
	
	iDmrgSolver (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	string info() const;
	string eigeninfo() const;
	double memory   (MEMUNIT memunit=GB) const;
	double overhead (MEMUNIT memunit=MB) const;
	
	void edgeState (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, 
	                LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND,
	                LANCZOS::CONVTEST::OPTION TEST = LANCZOS::CONVTEST::NORM_TEST,
	                double tol_eigval_input=1e-7, double tol_state_input=1e-6, 
	                size_t Dlimit=500, 
	                size_t max_iterations=50, size_t min_iterations=6, 
                    double eps_svd_input=1e-7, 
	                size_t savePeriod=0);
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	void prepare (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, bool useState=false, double eps_svd_input=1e-7);
	void iteration (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, 
	                LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND, 
	                LANCZOS::CONVTEST::OPTION TEST = LANCZOS::CONVTEST::SQ_TEST,
	                bool DISCARD_SV=true);
	void cleanup (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, 
	              LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND);
	
	/**Returns the current error of the eigenvalue while the sweep process.*/
	inline double get_errEigval() const {return err_eigval;};
	/**Returns the current error of the state while the sweep process.*/
	inline double get_errState() const {return err_state;};
	
private:
	
	size_t N_sites;
	size_t Dmax, Mmax, Nqmax;
	double tol_eigval, tol_state;
	double totalTruncWeight;
	size_t Dmax_old;
	size_t N_iterations;
	double err_eigval, err_state, err_state_before_end_of_noise;
	
	PivotMatrix2Q<Nq,Scalar,Scalar> Heff2; // Scalar = MpoScalar for ground state
	
	double Eold, eold; // energy and energy density
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	void LanczosStep (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE);
	void sweepStep (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	/**Constructs the left transfer matrix at chain site \p loc (left environment of \p loc).*/
	void build_L (const MpHamiltonian &H, const Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	/**Constructs the right transfer matrix at chain site \p loc (right environment of \p loc).*/
	void build_R (const MpHamiltonian &H, const Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	LANCZOS::CONVTEST::OPTION CHOSEN_CONVTEST;
};

template<size_t Nq, typename MpHamiltonian, typename Scalar>
string iDmrgSolver<Nq,MpHamiltonian,Scalar>::
info() const
{
	stringstream ss;
	ss << "iDmrgSolver: ";
	ss << "L=" << N_sites << ", ";
	ss << "Mmax=" << Mmax << ", Dmax=" << Dmax << ", " << "Nqmax=" << Nqmax << ", ";
	ss << "trunc_weight=" << totalTruncWeight << ", ";
	ss << eigeninfo();
	return ss.str();
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
string iDmrgSolver<Nq,MpHamiltonian,Scalar>::
eigeninfo() const
{
	stringstream ss;
	ss << "half-sweeps=" << N_iterations << ", ";
	ss << "err_eigval=" << setprecision(13) << err_eigval << ", err_state=" << err_state << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	return ss.str();
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
double iDmrgSolver<Nq,MpHamiltonian,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	res += Heff2.L.memory(memunit);
	res += Heff2.R.memory(memunit);
	for (size_t s1=0; s1<Heff2.W12.size(); ++s1)
	for (size_t s2=0; s2<Heff2.W12[s1].size(); ++s2)
	{
		res += calc_memory(Heff2.W12[s1][s2],memunit);
	}
	for (size_t s3=0; s3<Heff2.W34.size(); ++s3)
	for (size_t s4=0; s4<Heff2.W34[s3].size(); ++s4)
	{
		res += calc_memory(Heff2.W34[s3][s4],memunit);
	}
	return res;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
double iDmrgSolver<Nq,MpHamiltonian,Scalar>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
	res += Heff2.L.overhead(memunit);
	res += Heff2.R.overhead(memunit);
	res += 2. * calc_memory<size_t>(Heff2.qloc12.size(),memunit);
	res += 4. * calc_memory<size_t>(Heff2.qloc34.size(),memunit);
	return res;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
prepare (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, bool USE_STATE, double eps_svd_input)
{
	N_sites = H.length();
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	if (!USE_STATE)
	{
		// resize Vout
		Vout.state = UmpsQ<Nq,Scalar>(H, 2, 1, Qtot_input);
		Vout.state.N_sv = 1;
		Vout.state.setRandom();
	}
	Dmax_old = 1;
	
	// set edges
	Heff2.L.setIdentity(1,1,H.auxdim());
	Heff2.R.setIdentity(1,1,H.auxdim());
	Heff2.W12 = H.W_at(0);
	Heff2.W34 = H.W_at(1);
	Heff2.qloc12 = H.locBasis(0);
	Heff2.qloc34 = H.locBasis(1);
	
	// initial sweep, right-to-left:
//	for (size_t l=N_sites-1; l>0; --l)
//	{
//		Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
////		build_R(H,Vout,l-1);
//	}
//	Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
//	CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
//	pivot = 0;
	
	// initial sweep, left-to-right:
//	for (size_t l=0; l<N_sites-1; ++l)
//	{
//		Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
//		build_L(H,Vout,l+1);
//	}
//	Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
//	CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
//	pivot = N_sites-1;
	
	if (CHOSEN_VERBOSITY>=2) {lout << PrepTimer.info("initial state & sweep") << endl << endl;}
	
	// initial energy
//	Eold = contract_LR(Heff[0].L,Vout.state.A[0],H.W[0],Vout.state.A[0],Heff[0].R,H.locBasis(0));
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Ltmp;
	contract_L(Heff2.L, Vout.state.A[GAUGE::C][0], H.W[0], Vout.state.A[GAUGE::C][0], H.locBasis(0), Ltmp);
	Eold = contract_LR(Ltmp,Vout.state.A[GAUGE::C][1],H.W[1],Vout.state.A[GAUGE::C][1],Heff2.R,H.locBasis(1));
	eold = Eold;
	
	// initial cutoffs
	Vout.state.eps_svd = eps_svd_input;
	
	err_eigval = 1.;
	err_state  = 1.;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
iteration (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE, LANCZOS::CONVTEST::OPTION TEST, bool DISCARD_SV)
{
	Stopwatch<> IterationTimer;
	
	// save state for reference
//	UmpsQ<Nq,Scalar> Vref;
//	if (TEST == LANCZOS::CONVTEST::NORM_TEST or
//	    TEST == LANCZOS::CONVTEST::COEFFWISE)
//	{
//		Vref = Vout.state;
//	}
	ArrayXd SigmaRef = Vout.state.Sigma[0][0].array();
	
//	pivot = 0;
//	CURRENT_DIRECTION = (N_iterations%2==0)? DMRG::DIRECTION::RIGHT : DMRG::DIRECTION::LEFT;
	
	if (N_iterations>0)
	{
		Vout.state.forcedResize(Vout.state.A[GAUGE::L][0][0].block[0].cols());
//		cout << "forced resize to: " << Vout.state.A[GAUGE::L][0][0].block[0].cols() << endl;
		Vout.state.setRandom();
	}
	
	LanczosStep(H, Vout, EDGE);
//	sweepStep(H,Vout);
	build_L(H,Vout);
	build_R(H,Vout);
	
//	Vout.state.forcedResize(Vout.state.A[0][0].block[0].cols());
//	Vout.state.setRandom();
//	cout << "setting random" << endl;
	
	++N_iterations;
	
	// calculate error
//	err_eigval = abs(Eold-Vout.energy)/N_sites;
	double enew = 0.5*(Vout.energy-Eold);
	err_eigval = abs(eold-enew);
	ArrayXd SigmaDiff(max(Vout.state.Sigma[0][0].rows(),SigmaRef.rows()));
	SigmaDiff.head(Vout.state.Sigma[0][0].rows()) = Vout.state.Sigma[0][0];
	SigmaDiff.head(SigmaRef.rows()) -= SigmaRef;
	err_state = abs(SigmaDiff.sum());
//	if (TEST == LANCZOS::CONVTEST::NORM_TEST or
//	    TEST == LANCZOS::CONVTEST::COEFFWISE)
//	{
//		err_state = abs(1.-abs(dot(Vout.state,Vref)));
//	}
//	else if (TEST == LANCZOS::CONVTEST::SQ_TEST)
//	{
//		Stopwatch<> HsqTimer;
//		double avgHsq = (H.check_SQUARE()==true)? isReal(avg(Vout.state,H,Vout.state,true)) : isReal(avg(Vout.state,H,H,Vout.state));
//		err_state = abs(avgHsq-pow(Vout.energy,2))/this->N_sites;
//		if (CHOSEN_VERBOSITY>=2)
//		{
//			lout << HsqTimer.info("<H^2>") << endl;
//		}
//		if (N_iterations == 24) {err_state_before_end_of_noise = err_state;}
//	}
	
	Eold = Vout.energy;
	eold = enew;
//	if (TEST == LANCZOS::CONVTEST::NORM_TEST or
//	    TEST == LANCZOS::CONVTEST::COEFFWISE)
//	{
//		Vref = Vout.state;
//	}
	
	// calculate stats
//	Mmax = Vout.state.calc_Mmax();
	Dmax = Vout.state.calc_Dmax();
//	Nqmax = Vout.state.calc_Nqmax();
	totalTruncWeight = Vout.state.truncWeight.sum();
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		if (EDGE == LANCZOS::EDGE::GROUND)
		{
			lout << "Emin/L=" << setprecision(13) << eold << setprecision(standard_precision) << endl;
		}
		else
		{
			lout << "Emax/L=" << setprecision(13) << eold << setprecision(standard_precision) << endl;
		}
		lout << eigeninfo() << endl;
		lout << Vout.state.info() << endl;
		lout << IterationTimer.info("half-sweep") << endl;
	}
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
cleanup (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
//	Vout.state.set_defaultCutoffs();
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		size_t standard_precision = cout.precision();
		string Eedge = (EDGE == LANCZOS::EDGE::GROUND)? "Emin/L" : "Emax/L";
		lout << Eedge << "=" << setprecision(13) << eold << setprecision(standard_precision) << endl;
		lout << Vout.state.info() << endl;
	}
	
//	halfsweep(H,Vout,EDGE,LANCZOS::CONVTEST::NORM_TEST,false);
	Vout.energy = eold;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, LANCZOS::EDGE::OPTION EDGE, LANCZOS::CONVTEST::OPTION TEST, double tol_eigval_input, double tol_state_input, size_t Dlimit, size_t max_halfsweeps, size_t min_halfsweeps, double eps_svd_input, size_t savePeriod)
{
	tol_eigval = tol_eigval_input;
	tol_state  = tol_state_input;
	
	prepare(H, Vout, Qtot_input, false, eps_svd_input);
	
	Stopwatch<> GlobalTimer;
	
	// lambda function to print tolerances
//	auto print_alpha_eps = [this,&Vout] ()
//	{
//		if (CHOSEN_VERBOSITY>=2)
//		{
//			lout //<< "α_noise=" << Vout.state.alpha_noise << ", "
//			     //<< "ε_rdm=" << Vout.state.eps_rdm << ", "
//			     << "ε_svd=" << Vout.state.eps_svd 
//			     << endl;
//		}
//	};
//	
//	print_alpha_eps();
	
	// average local dimension for bond dimension increase
	size_t dimqlocAvg = 0;
	for (size_t l=0; l<H.length(); ++l) {dimqlocAvg += H.locBasis(l).size();}
	dimqlocAvg /= H.length();
	
	while (((err_eigval >= tol_eigval or err_state >= tol_state) and N_iterations < max_halfsweeps) or 
	       N_iterations < min_halfsweeps)
	{
		// sweep
		iteration(H,Vout,EDGE,TEST);
		
		// If truncated weight too large, increase upper limit per subspace by 10%, but at least by dimqlocAvg, overall never larger than Dlimit
		if (N_iterations%2 == 0 and totalTruncWeight >= Vout.state.eps_svd)
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
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << GlobalTimer.info("total runtime") << endl;
	}
	cleanup(H,Vout,EDGE);
}

//template<size_t Nq, typename MpHamiltonian, typename Scalar>
//void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
//sweepStep (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
//{
////	Vout.state.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR, &Heff[pivot], true);
////	Vout.sweepStep2(CURRENT_DIRECTION, 0, Apair.A);
//	
////	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++pivot) : build_R(H,Vout,--pivot);
//	
//	build_L(H,Vout);
//	build_R(H,Vout);
//}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
LanczosStep (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	Eigenstate<PivotVector2Q<Nq,Scalar> > Apair;
	contract_AA(Vout.state.A[GAUGE::C][0], Vout.state.locBasis(0), Vout.state.A[GAUGE::C][1], Vout.state.locBasis(1), Apair.state.A);
	
	// reset dim
	Heff2.dim = 0;
	for (size_t s1=0; s1<H.locBasis(0).size(); ++s1)
	for (size_t s2=0; s2<H.locBasis(1).size(); ++s2)
	for (size_t q=0; q<Apair.state.A[s1][s2].dim; ++q)
	{
		Heff2.dim += Apair.state.A[s1][s2].block[q].rows() * Apair.state.A[s1][s2].block[q].cols();
	}
	
	LanczosSolver<PivotMatrix2Q<Nq,Scalar,Scalar>,PivotVector2Q<Nq,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
	Lutz.set_dimK(min(30ul, Heff2.dim));
	Lutz.edgeState(Heff2,Apair, EDGE, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << Lutz.info() << endl;
//		lout << Vout.state.test_ortho() << endl;
	}
	
	Vout.energy = Apair.energy;
	Vout.state.eps_svd = 1e-15;
	Vout.state.decompose(0,Apair.state.A); // Apair-> AL[0], AR[1], C[0]=C[1]
//	Vout.state.pivot = 1;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
inline void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
build_L (const MpHamiltonian &H, const Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Ltmp;
	contract_L(Heff2.L, Vout.state.A[GAUGE::L][0], H.W[0], Vout.state.A[GAUGE::L][0], H.locBasis(0), Ltmp);
	Heff2.L = Ltmp;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
inline void iDmrgSolver<Nq,MpHamiltonian,Scalar>::
build_R (const MpHamiltonian &H, const Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Rtmp;
	contract_R(Heff2.R, Vout.state.A[GAUGE::R][1], H.W[1], Vout.state.A[GAUGE::R][1], H.locBasis(1), Rtmp);
	Heff2.R = Rtmp;
}

#endif
