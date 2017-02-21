#ifndef VANILLA_VUMPSSOLVER
#define VANILLA_VUMPSSOLVER

#include "unsupported/Eigen/IterativeSolvers"

#include "MpoQ.h"
#include "UmpsQ.h"
#include "DmrgPivotStuffQ.h"
#include "DmrgPivotStuff2Q.h"
#include "DmrgIndexGymnastics.h"
#include "DmrgLinearAlgebraQ.h"
#include "LanczosSolver.h"
#include "VumpsContractions.h"

template<size_t Nq, typename MpHamiltonian, typename Scalar=double>
class VumpsSolver
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
typedef Matrix<Scalar,Dynamic,1>       VectorType;
typedef boost::multi_array<Scalar,4> TwoSiteHamiltonian;

public:
	
	VumpsSolver (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	string info() const;
	string eigeninfo() const;
	double memory   (MEMUNIT memunit=GB) const;
	double overhead (MEMUNIT memunit=MB) const;
	
	void edgeState (const TwoSiteHamiltonian &h2site, const vector<qarray<Nq> > &qloc_input, 
	                Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, 
	                double tol_eigval_input=1e-7, double tol_var_input=1e-6, 
	                size_t Dlimit=500, 
	                size_t max_iterations=50, size_t min_iterations=6);
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	void prepare (const TwoSiteHamiltonian &h2site, const vector<qarray<Nq> > &qloc_input,
	              Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t Dlimit, qarray<Nq> Qtot_input);
	void iteration (Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	void cleanup (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	/**Returns the current error of the eigenvalue while the sweep process.*/
	inline double get_errEigval() const {return err_eigval;};
	
	/**Returns the current error of the state while the sweep process.*/
	inline double get_errState() const {return err_var;};
	
private:
	
	size_t N_sites;
	double tol_eigval, tol_var;
	size_t N_iterations;
	double err_eigval, err_var;
	
	PivumpsMatrix<Nq,Scalar,Scalar> Heff;
	size_t D, M;
	
	double eL, eR, eoldR, eoldL;
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
};

template<size_t Nq, typename MpHamiltonian, typename Scalar>
string VumpsSolver<Nq,MpHamiltonian,Scalar>::
info() const
{
	stringstream ss;
	ss << "VumpsSolver: ";
	ss << "L=" << N_sites << ", ";
	ss << eigeninfo();
	return ss.str();
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
string VumpsSolver<Nq,MpHamiltonian,Scalar>::
eigeninfo() const
{
	stringstream ss;
	ss << "iterations=" << N_iterations << ", ";
	ss << "e0=" << min(eL,eR) << ", ";
	ss << "err_eigval=" << setprecision(13) << err_eigval << ", err_var=" << err_var << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	return ss.str();
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
double VumpsSolver<Nq,MpHamiltonian,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	res += calc_memory(Heff.L);
	res += calc_memory(Heff.R);
	for (size_t l=0; l<N_sites; ++l)
	{
		res += Heff.AL[l].memory(memunit);
		res += Heff.AR[l].memory(memunit);
	}
	return res;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
double VumpsSolver<Nq,MpHamiltonian,Scalar>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
//	res += Heff2.L.overhead(memunit);
//	res += Heff2.R.overhead(memunit);
//	res += 2. * calc_memory<size_t>(Heff2.qloc12.size(),memunit);
//	res += 4. * calc_memory<size_t>(Heff2.qloc34.size(),memunit);
	return res;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
prepare (const TwoSiteHamiltonian &h2site, const vector<qarray<Nq> > &qloc_input, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t M_input, qarray<Nq> Qtot_input)
{
	N_sites = 1;
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = h2site.shape()[0];
	M = M_input;
	Heff.h.resize(boost::extents[D][D][D][D]);
	Heff.h = h2site;
	Heff.qloc = qloc_input;
	
	// resize Vout
	Vout.state = UmpsQ<Nq,Scalar>(Heff.qloc, 1, M, Qtot_input);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	Vout.state.svdDecompose(0);
	
	// initial energy
	eoldL = energy_L(Heff.h, Vout.state.A[GAUGE::L][0], Vout.state.C[0], Heff.qloc);
	eoldR = energy_R(Heff.h, Vout.state.A[GAUGE::R][0], Vout.state.C[0], Heff.qloc);
	
	err_eigval = 1.;
	err_var  = 1.;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration (Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	MatrixType TL(M*M,M*M); TL.setZero();
	MatrixType TR(M*M,M*M); TR.setZero();
//	for (size_t s=0; s<D; ++s)
//	{
//		// only for real:
//		TL += kroneckerProduct(Vout.state.A[GAUGE::L][0][s].block[0], Vout.state.A[GAUGE::L][0][s].block[0]); 
//		TR += kroneckerProduct(Vout.state.A[GAUGE::R][0][s].block[0], Vout.state.A[GAUGE::R][0][s].block[0]);
//	}
	for (size_t s=0; s<D; ++s)
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	for (size_t k=0; k<M; ++k)
	for (size_t l=0; l<M; ++l)
	{
		size_t r = i + M*l; // note: rows of A & cols of A† (= rows of A*) become new rows of T
		size_t c = j + M*k; // note: cols of A & rows of A† (= cols of A*) become new cols of T
		TL(r,c) += Vout.state.A[GAUGE::L][0][s].block[0](i,j) * Vout.state.A[GAUGE::L][0][s].block[0].adjoint()(k,l);
		TR(r,c) += Vout.state.A[GAUGE::R][0][s].block[0](i,j) * Vout.state.A[GAUGE::R][0][s].block[0].adjoint()(k,l);
	}
	
	MatrixType Reigen = Vout.state.C[0].block[0] * Vout.state.C[0].block[0].adjoint();
	MatrixType Leigen = Vout.state.C[0].block[0].adjoint() * Vout.state.C[0].block[0];
	
	MatrixType hR = make_hR(Heff.h, Vout.state.A[GAUGE::R][0], Heff.qloc);
	MatrixType hL = make_hL(Heff.h, Vout.state.A[GAUGE::L][0], Heff.qloc);
	
	eL = (Leigen * hR).trace();
	eR = (hL * Reigen).trace();
	
	hR -= eL * MatrixType::Identity(M,M);
	hL -= eR * MatrixType::Identity(M,M);
	
	MatrixType UxL(M*M,M*M); UxL.setZero();
	MatrixType RxU(M*M,M*M); RxU.setZero();
	
	for (size_t i=0; i<M; ++i)
	for (size_t k=0; k<M; ++k)
	for (size_t l=0; l<M; ++l)
	{
		size_t r = i + M*i;
		size_t c = k + M*l;
		UxL(r,c) = Leigen(k,l);
	}
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	for (size_t k=0; k<M; ++k)
	{
		size_t r = i + M*j;
		size_t c = k + M*k;
		RxU(r,c) = Reigen(i,j);
	}
	
	VectorType bL(M*M), bR(M*M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		bR(r) = hR(i,j);
	}
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		bL(r) = hL(i,j);
	}
	
	VectorType xL, xR;
	
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			Stopwatch<> GmresTimerR;
			MatrixType LinearR = MatrixType::Identity(M*M,M*M)-TR+UxL;
			GMRES<MatrixType> Ronald(LinearR);
			xR = Ronald.solve(bR);
		//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		//	{
		//		lout << "GMRES(H_R): iterations=" << Jimmy.iterations() << ", error=" << Jimmy.error() << ", time" << GmresTimerR.info() << endl;
		//	}
		}
		#pragma omp section
		{
			Stopwatch<> GmresTimerL;
			MatrixType LinearL = (MatrixType::Identity(M*M,M*M)-TL+RxU).adjoint();
			GMRES<MatrixType> Leonard(LinearL);
			xL = Leonard.solve(bL);
		//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		//	{
		//		lout << "GMRES(H_L): iterations=" << Jimmy.iterations() << ", error=" << Jimmy.error() << ", time" << GmresTimerL.info() << endl;
		//	}
		}
	}
	
	MatrixType HR(M,M), HL(M,M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		HR(i,j) = xR(r);
	}
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		HL(i,j) = xL(r);
	}
	
	Heff.L = HL;
	Heff.R = HR;
	Heff.AL = Vout.state.A[GAUGE::L][0];
	Heff.AR = Vout.state.A[GAUGE::R][0];
	Heff.dim = Heff.qloc.size() * M * M;
	
	Heff.dim = Heff.qloc.size() * M * M;
	Eigenstate<PivotVectorQ<Nq,Scalar> > g1;
	g1.state.A = Vout.state.A[GAUGE::C][0];
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz1(LANCZOS::REORTHO::FULL);
	Lutz1.set_dimK(min(20ul, Heff.dim));
	Lutz1.edgeState(Heff,g1, LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << g1.energy << ", time" << LanczosTimer.info() << ", " << Lutz1.info() << endl;
	}
	
	Heff.dim = M*M;
	Eigenstate<PivumpsVector0<Nq,Scalar> > g0;
	g0.state.C = Vout.state.C[0];
	
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivumpsVector0<Nq,Scalar>,Scalar> Lutz0(LANCZOS::REORTHO::FULL);
	Lutz0.set_dimK(min(20ul, Heff.dim));
	Lutz0.edgeState(Heff,g0, LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << g0.energy << ", time" << LanczosTimer.info() << ", " << Lutz0.info() << endl;
	}
	
	Vout.state.A[GAUGE::C][0] = g1.state.A;
	Vout.state.C[0]           = g0.state.C;
	(err_var>0.1)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
	
	double epsL, epsR;
	Vout.state.calc_epsLR(0,epsL,epsR);
	err_var = max(epsL,epsR);
	
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << eigeninfo() << endl;
		lout << IterationTimer.info("full iteration") << endl;
		lout << endl;
	}
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
edgeState (const TwoSiteHamiltonian &h2site, const vector<qarray<Nq> > &qloc, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot, double tol_eigval_input, double tol_var_input, size_t M, size_t max_iterations, size_t min_iterations)
{
	tol_eigval = tol_eigval_input;
	tol_var = tol_var_input;
	
	prepare(h2site, qloc, Vout, M, Qtot);
	
	Stopwatch<> GlobalTimer;
	
	while (((err_eigval >= tol_eigval or err_var >= tol_var) and N_iterations < max_iterations) or N_iterations < min_iterations)
	{
		iteration(Vout);
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << GlobalTimer.info("total runtime") << endl;
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		size_t standard_precision = cout.precision();
		lout << "emin=" << setprecision(13) << Vout.energy << setprecision(standard_precision) << endl;
		lout << Vout.state.info() << endl;
		lout << endl;
	}
}

#endif
