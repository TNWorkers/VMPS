#ifndef VANILLA_VUMPSSOLVER
#define VANILLA_VUMPSSOLVER

#include "unsupported/Eigen/IterativeSolvers"

#include "MpoQ.h"
#include "UmpsQ.h"
#include "DmrgPivotStuffQ.h"
#include "DmrgPivotStuff2Q.h"
#include "DmrgPivotStuff0.h"
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
	
	void edgeState (const vector<TwoSiteHamiltonian> &h2site, const vector<qarray<Nq> > &qloc_input, 
	                Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, 
	                double tol_eigval_input=1e-7, double tol_var_input=1e-6, 
	                size_t Dlimit=500, 
	                size_t max_iterations=50, size_t min_iterations=6);
	
	void edgeState (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, 
	                double tol_eigval_input=1e-7, double tol_var_input=1e-6, 
	                size_t Dlimit=500, 
	                size_t max_iterations=50, size_t min_iterations=6);
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	void prepare (const vector<TwoSiteHamiltonian> &h2site, const vector<qarray<Nq> > &qloc_input,
	              Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t Dlimit, qarray<Nq> Qtot_input);
	void iteration1 (Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	void iteration2 (Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	void prepare (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t Dlimit, qarray<Nq> Qtot_input);
	void iteration1 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	void cleanup (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	/**Returns the current error of the eigenvalue while the sweep process.*/
	inline double get_errEigval() const {return err_eigval;};
	
	/**Returns the current error of the state while the sweep process.*/
	inline double get_errState() const {return err_var;};
	
	MatrixXd linearL (const MatrixType &hL, const MatrixType &TL, const MatrixType &Leigen, double offset=0);
	MatrixXd linearR (const MatrixType &hR, const MatrixType &TR, const MatrixType &Reigen, double offset=0);
	
private:
	
	size_t N_sites;
	double tol_eigval, tol_var;
	size_t N_iterations;
	double err_eigval, err_var;
	
	vector<PivumpsMatrix<Nq,Scalar,Scalar> > Heff;
	vector<PivotMatrixQ<Nq,Scalar,Scalar> > HeffMPO;
	vector<qarray<Nq> > qloc;
	std::array<boost::multi_array<Scalar,4>,2> h;
	size_t D, M;
	
	double eL, eR, eoldR, eoldL;
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	MatrixXd eigenvectorL (const MatrixType &TL);
	MatrixXd eigenvectorR (const MatrixType &TR);
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
//	res += calc_memory(Heff.L);
//	res += calc_memory(Heff.R);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		res += Heff.AL[l].memory(memunit);
//		res += Heff.AR[l].memory(memunit);
//	}
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
prepare (const vector<TwoSiteHamiltonian> &h2site, const vector<qarray<Nq> > &qloc_input, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t M_input, qarray<Nq> Qtot_input)
{
	N_sites = h2site.size();
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = h2site[0].shape()[0];
	M = M_input;
	Heff.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Heff[l].h[0].resize(boost::extents[D][D][D][D]);
		Heff[l].h[0] = h2site[0];
		Heff[l].h[1].resize(boost::extents[D][D][D][D]);
		Heff[l].h[1] = (N_sites==1)? h2site[0] : h2site[1];
		Heff[l].qloc = qloc_input;
	}
	
	h[0].resize(boost::extents[D][D][D][D]);
	h[0] = h2site[0];
	h[1].resize(boost::extents[D][D][D][D]);
	h[1] = (N_sites==1)? h2site[0] : h2site[1];
	
	// resize Vout
	Vout.state = UmpsQ<Nq,Scalar>(Heff[0].qloc, N_sites, M, Qtot_input);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
	}
	
	// initial energy
	eoldL = energy_L(Heff[0].h[0], Vout.state.A[GAUGE::L][0], Vout.state.C[0], Heff[0].qloc);
	eoldR = energy_R(Heff[0].h[0], Vout.state.A[GAUGE::R][0], Vout.state.C[0], Heff[0].qloc);
	
	err_eigval = 1.;
	err_var    = 1.;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration1 (Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
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
//	MatrixType Reigen = eigenvectorR(TR);
//	MatrixType Leigen = eigenvectorL(TL);
	
//	cout << "left:" << endl;
//	cout << eigenvectorL(TL) << endl << endl;
//	cout << Leigen << endl << endl;
//	
//	cout << "right:" << endl;
//	cout << eigenvectorR(TL) << endl << endl;
//	cout << Reigen << endl << endl;
	
//	cout << eigenvectorL(TR) << endl;
//	cout << endl;
//	cout << eigenvectorR(TL) << endl;
	
	MatrixType hR = make_hR(Heff[0].h[0], Vout.state.A[GAUGE::R][0], Heff[0].qloc);
	MatrixType hL = make_hL(Heff[0].h[0], Vout.state.A[GAUGE::L][0], Heff[0].qloc);
	
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
	
	Heff[0].L = HL;
	Heff[0].R = HR;
	Heff[0].AL = Vout.state.A[GAUGE::L][0];
	Heff[0].AR = Vout.state.A[GAUGE::R][0];
	Heff[0].dim = Heff[0].qloc.size() * M * M;
	
	Heff[0].dim = Heff[0].qloc.size() * M * M;
	Eigenstate<PivotVectorQ<Nq,Scalar> > g1;
	g1.state.A = Vout.state.A[GAUGE::C][0];
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz1(LANCZOS::REORTHO::FULL);
	Lutz1.set_dimK(min(20ul, Heff[0].dim));
	Lutz1.edgeState(Heff[0],g1, LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << g1.energy << ", time" << LanczosTimer.info() << ", " << Lutz1.info() << endl;
	}
	
	Heff[0].dim = M*M;
	Eigenstate<PivumpsVector0<Nq,Scalar> > g0;
	g0.state.C = Vout.state.C[0];
	
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivumpsVector0<Nq,Scalar>,Scalar> Lutz0(LANCZOS::REORTHO::FULL);
	Lutz0.set_dimK(min(20ul, Heff[0].dim));
	Lutz0.edgeState(Heff[0],g0, LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << g0.energy << ", time" << LanczosTimer.info() << ", " << Lutz0.info() << endl;
	}
	
	Vout.state.A[GAUGE::C][0] = g1.state.A;
	Vout.state.C[0]           = g0.state.C;
	(err_var>0.1)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
	
	double epsL, epsR;
	Vout.state.calc_epsLR(0,epsL,epsR);
	cout << "epsL=" << epsL << ", epsR=" << epsR << endl;
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
prepare (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t M_input, qarray<Nq> Qtot_input)
{
	N_sites = 1;
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = H.locBasis(0).size();
	M = M_input;
	HeffMPO.resize(1);
	HeffMPO[0].W = H.W[0];
	HeffMPO[0].qloc = H.locBasis(0);
	
	// resize Vout
	Vout.state = UmpsQ<Nq,Scalar>(HeffMPO[0].qloc, N_sites, M, Qtot_input);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
	}
	
	// initial energy
//	eoldL = energy_L(Heff[0].h[0], Vout.state.A[GAUGE::L][0], Vout.state.C[0], Heff[0].qloc);
//	eoldR = energy_R(Heff[0].h[0], Vout.state.A[GAUGE::R][0], Vout.state.C[0], Heff[0].qloc);
	eoldL = 1e3;
	eoldR = 1e3;
	
	err_eigval = 1.;
	err_var    = 1.;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration1 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	vector<vector<MatrixType> > TL(H.auxdim());
	vector<vector<MatrixType> > TR(H.auxdim());
	
	for (size_t a=0; a<H.auxdim(); ++a)
	{
		TL[a].resize(H.auxdim());
		TR[a].resize(H.auxdim());
	}
	
	for (size_t a=0; a<H.auxdim(); ++a)
	for (size_t b=0; b<H.auxdim(); ++b)
	{
		TL[a][b].resize(M*M,M*M); TL[a][b].setZero();
		TR[a][b].resize(M*M,M*M); TR[a][b].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<H.W[0][s1][s2].outerSize(); ++k)
	for (typename SparseMatrix<Scalar>::InnerIterator iW(H.W[0][s1][s2],k); iW; ++iW)
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	for (size_t k=0; k<M; ++k)
	for (size_t l=0; l<M; ++l)
	{
		size_t a = iW.row();
		size_t b = iW.col();
		
		size_t r = i + M*l; // note: rows of A & cols of A† (= rows of A*) become new rows of T
		size_t c = j + M*k; // note: cols of A & rows of A† (= cols of A*) become new cols of T
		
		TL[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::L][0][s2].block[0](i,j) * Vout.state.A[GAUGE::L][0][s1].block[0].adjoint()(k,l);
		TR[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::R][0][s2].block[0](i,j) * Vout.state.A[GAUGE::R][0][s1].block[0].adjoint()(k,l);
	}
	
	MatrixType Reigen = Vout.state.C[0].block[0] * Vout.state.C[0].block[0].adjoint();
	MatrixType Leigen = Vout.state.C[0].block[0].adjoint() * Vout.state.C[0].block[0];
	
	vector<MatrixType> YL(H.auxdim());
	vector<MatrixType> YR(H.auxdim());
	
	for (size_t a=0; a<H.auxdim(); ++a)
	{
		YL[a].resize(M,M); YL[a].setZero();
		YR[a].resize(M,M); YR[a].setZero();
	}
	
	boost::multi_array<MatrixType,LEGLIMIT> L(boost::extents[H.auxdim()][1]);
	boost::multi_array<MatrixType,LEGLIMIT> R(boost::extents[H.auxdim()][1]);
	
//	vector<MatrixType> L(H.auxdim());
	L[H.auxdim()-1][0].resize(M,M);
	L[H.auxdim()-1][0].setIdentity();
	
	for (int b=H.auxdim()-2; b>=0; --b)
	{
		YL[b] = make_YL(b, H.W[0], L, Vout.state.A[GAUGE::L][0], HeffMPO[0].qloc);
		
		if (TL[b][b].norm() == 0.)
		{
			L[b][0] = YL[b];
		}
		else
		{
			L[b][0] = linearL(YL[b], TL[b][b], Reigen, (YL[b] * Reigen).trace());
		}
	}
	
//	vector<MatrixType> R(H.auxdim());
	R[0][0].resize(M,M);
	R[0][0].setIdentity();
	
	for (int a=1; a<H.auxdim(); ++a)
	{
		YR[a] = make_YR(a, H.W[0], R, Vout.state.A[GAUGE::R][0], HeffMPO[0].qloc);
		
		if (TR[a][a].norm() == 0.)
		{
			R[a][0] = YR[a];
		}
		else
		{
			R[a][0] = linearR(YR[a], TR[a][a], Leigen, (Leigen * YR[a]).trace());
		}
	}
	
	HeffMPO[0].L.clear();
	HeffMPO[0].R.clear();
	HeffMPO[0].L.push_back(qloc3dummy, L);
	HeffMPO[0].R.push_back(qloc3dummy, R);
	
	if (HeffMPO[0].dim == 0)
	{
		precalc_blockStructure (HeffMPO[0].L, Vout.state.A[GAUGE::C][0], HeffMPO[0].W, Vout.state.A[GAUGE::C][0], HeffMPO[0].R, 
		                        H.locBasis(0), HeffMPO[0].qlhs, HeffMPO[0].qrhs);
	}
	
	// reset dim
	HeffMPO[0].dim = 0;
	for (size_t s=0; s<H.locBasis(0).size(); ++s)
	for (size_t q=0; q<Vout.state.A[GAUGE::C][0][s].dim; ++q)
	{
		HeffMPO[0].dim += Vout.state.A[GAUGE::C][0][s].block[q].rows() * Vout.state.A[GAUGE::C][0][s].block[q].cols();
	}
	
	Eigenstate<PivotVectorQ<Nq,Scalar> > gAC;
	gAC.state.A = Vout.state.A[GAUGE::C][0];
	
	LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
	Lutz.set_dimK(min(30ul, HeffMPO[0].dim));
	Lutz.edgeState(HeffMPO[0],gAC, LANCZOS::EDGE::GROUND, 1e-7,1e-4, true);
	cout << "eAC=" << gAC.energy << endl;
	cout << Lutz.info() << endl;
	
	Eigenstate<PivotVector0Q<Nq,Scalar> > gC;
	gC.state.A = Vout.state.C[0];
	
	HeffMPO[0].dim = 0;
	for (size_t q=0; q<Vout.state.C[0].dim; ++q)
	{
		HeffMPO[0].dim += Vout.state.C[0].block[q].rows() * Vout.state.C[0].block[q].cols();
	}
	
	LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVector0Q<Nq,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL);
	Lucy.set_dimK(min(30ul, HeffMPO[0].dim));
	Lucy.edgeState(HeffMPO[0],gC, LANCZOS::EDGE::GROUND, 1e-7,1e-4, true);
	cout << "eC=" << gC.energy << endl;
	cout << Lucy.info() << endl;
	
	eL = (YL[0] * Reigen).trace();
	eR = (Leigen * YR[H.auxdim()-1]).trace();
	cout << "eL=" << eL << ", eR=" << eR << endl;
	
	Vout.state.A[GAUGE::C][0] = gAC.state.A;
	Vout.state.C[0]           = gC.state.A;
	(err_var>0.1)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
	
	double epsL, epsR;
	Vout.state.calc_epsLR(0,epsL,epsR);
	cout << "epsL=" << epsL << ", epsR=" << epsR << endl;
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
edgeState (const vector<TwoSiteHamiltonian> &h2site, const vector<qarray<Nq> > &qloc, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot, double tol_eigval_input, double tol_var_input, size_t M, size_t max_iterations, size_t min_iterations)
{
	tol_eigval = tol_eigval_input;
	tol_var = tol_var_input;
	
	prepare(h2site, qloc, Vout, M, Qtot);
	
	Stopwatch<> GlobalTimer;
	
	while (((err_eigval >= tol_eigval or err_var >= tol_var) and N_iterations < max_iterations) or N_iterations < min_iterations)
	{
		(N_sites==1)? iteration1(Vout): iteration2(Vout);
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

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot, double tol_eigval_input, double tol_var_input, size_t M, size_t max_iterations, size_t min_iterations)
{
	tol_eigval = tol_eigval_input;
	tol_var = tol_var_input;
	
	prepare(H, Vout, M, Qtot);
	
	Stopwatch<> GlobalTimer;
	
	while (((err_eigval >= tol_eigval or err_var >= tol_var) and N_iterations < max_iterations) or N_iterations < min_iterations)
	{
//		(N_sites==1)? iteration1(H,Vout): iteration2(H,Vout);
		iteration1(H,Vout);
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

template<size_t Nq, typename MpHamiltonian, typename Scalar>
MatrixXd VumpsSolver<Nq,MpHamiltonian,Scalar>::
eigenvectorL (const MatrixType &TL)
{
	EigenSolver<MatrixType> Lutz(TL);
	int max_index;
	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
	
//	cout << "max eigenvalue abs=" << Lutz.eigenvalues()(max_index) << endl;
//	cout << Lutz.eigenvalues().transpose() << endl;
	
	MatrixType Mout(M,M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
	}
	
	return Mout;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
MatrixXd VumpsSolver<Nq,MpHamiltonian,Scalar>::
eigenvectorR (const MatrixType &TR)
{
	EigenSolver<MatrixType> Lutz(TR.adjoint());
	int max_index;
	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
	
//	cout << "max eigenvalue abs=" << Lutz.eigenvalues()(max_index) << endl;
//	cout << Lutz.eigenvalues().transpose() << endl;
	
	MatrixType Mout(M,M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
	}
	
	return Mout;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
MatrixXd VumpsSolver<Nq,MpHamiltonian,Scalar>::
linearL (const MatrixType &hL, const MatrixType &TL, const MatrixType &Reigen, double offset)
{
	MatrixType RxU(M*M,M*M); RxU.setZero();
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	for (size_t k=0; k<M; ++k)
	{
		size_t r = i + M*j;
		size_t c = k + M*k;
		RxU(r,c) = Reigen(i,j); // delta(k,l)
	}
	
	VectorType bL(M*M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		bL(r) = hL(i,j);
		
		if (i==j and offset!=0)
		{
			bL(r) -= offset;
		}
	}
	
	VectorType xL;
	
	Stopwatch<> GmresTimerL;
	MatrixType LinearL = (MatrixType::Identity(M*M,M*M)-TL+RxU).adjoint();
	GMRES<MatrixType> Leonard(LinearL);
	xL = Leonard.solve(bL);
	
	MatrixType HL(M,M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		HL(i,j) = xL(r);
	}
	
	return HL;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
MatrixXd VumpsSolver<Nq,MpHamiltonian,Scalar>::
linearR (const MatrixType &hR, const MatrixType &TR, const MatrixType &Leigen, double offset)
{
	MatrixType UxL(M*M,M*M); UxL.setZero();
	
	for (size_t i=0; i<M; ++i)
	for (size_t k=0; k<M; ++k)
	for (size_t l=0; l<M; ++l)
	{
		size_t r = i + M*i;
		size_t c = k + M*l;
		UxL(r,c) = Leigen(k,l); // delta(i,j)
	}
	
	VectorType bR(M*M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		bR(r) = hR(i,j);
		
		if (i==j and offset!=0)
		{
			bR(r) -= offset;
		}
	}
	
	VectorType xR;
	
	Stopwatch<> GmresTimerR;
	MatrixType LinearR = MatrixType::Identity(M*M,M*M)-TR+UxL;
	GMRES<MatrixType> Ronald(LinearR);
	xR = Ronald.solve(bR);
	
	MatrixType HR(M,M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		HR(i,j) = xR(r);
	}
	
	return HR;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration2 (Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	MatrixType TL01(M*M,M*M); TL01.setZero();
	MatrixType TL10(M*M,M*M); TL10.setZero();
	MatrixType TR01(M*M,M*M); TR01.setZero();
	MatrixType TR10(M*M,M*M); TR10.setZero();
//	for (size_t s=0; s<D; ++s)
//	{
//		// only for real:
//		TL += kroneckerProduct(Vout.state.A[GAUGE::L][0][s].block[0], Vout.state.A[GAUGE::L][0][s].block[0]); 
//		TR += kroneckerProduct(Vout.state.A[GAUGE::R][0][s].block[0], Vout.state.A[GAUGE::R][0][s].block[0]);
//	}
	
	vector<vector<MatrixType> > vTL01(D);
	vector<vector<MatrixType> > vTL10(D);
	vector<vector<MatrixType> > vTR01(D);
	vector<vector<MatrixType> > vTR10(D);
	
	for (size_t s1=0; s1<D; ++s1)
	{
		vTL01[s1].resize(D);
		vTL10[s1].resize(D);
		vTR01[s1].resize(D);
		vTR10[s1].resize(D);
		
		for (size_t s2=0; s2<D; ++s2)
		{
			vTL01[s1][s2].resize(M,M);
			vTL10[s1][s2].resize(M,M);
			vTR01[s1][s2].resize(M,M);
			vTR10[s1][s2].resize(M,M);
			
			vTL01[s1][s2].setZero();
			vTL10[s1][s2].setZero();
			vTR01[s1][s2].setZero();
			vTR10[s1][s2].setZero();
		}
	}
	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t q=0; q<Vout.state.A[GAUGE::L][0][s1].dim; ++q)
//	for (size_t s2=0; s2<D; ++s2)
//	{
//		qarray2<Nq> quple = {Vout.state.A[GAUGE::L][0][s1].out[q], 
//		                     Vout.state.A[GAUGE::L][0][s1].out[q] + Heff[0].qloc[s2]};
//		auto qA = Vout.state.A[GAUGE::L][1][s2].dict.find(quple);
//		
//		if (qA != Vout.state.A[GAUGE::L][1][s2].dict.end())
//		{
//			vTL[s1][s2] += Vout.state.A[GAUGE::L][0][s1].block[0] * Vout.state.A[GAUGE::L][1][s2].block[qA->second];
//		}
//	}
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t q=0; q<Vout.state.A[GAUGE::R][0][s1].dim; ++q)
//	for (size_t s2=0; s2<D; ++s2)
//	{
//		qarray2<Nq> quple = {Vout.state.A[GAUGE::R][0][s1].out[q], 
//		                     Vout.state.A[GAUGE::R][0][s1].out[q] + Heff[0].qloc[s2]};
//		auto qA = Vout.state.A[GAUGE::R][1][s2].dict.find(quple);
//		
//		if (qA != Vout.state.A[GAUGE::R][1][s2].dict.end())
//		{
//			vTR[s1][s2] += Vout.state.A[GAUGE::R][0][s1].block[0] * Vout.state.A[GAUGE::R][1][s2].block[0];
//		}
//	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	{
		vTL01[s1][s2] += Vout.state.A[GAUGE::L][0][s1].block[0] * Vout.state.A[GAUGE::L][1][s2].block[0];
		vTL10[s1][s2] += Vout.state.A[GAUGE::L][1][s1].block[0] * Vout.state.A[GAUGE::L][1][s2].block[0];
		vTR01[s1][s2] += Vout.state.A[GAUGE::R][0][s1].block[0] * Vout.state.A[GAUGE::R][1][s2].block[0];
		vTR10[s1][s2] += Vout.state.A[GAUGE::R][1][s1].block[0] * Vout.state.A[GAUGE::R][0][s2].block[0];
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	for (size_t k=0; k<M; ++k)
	for (size_t l=0; l<M; ++l)
	{
		size_t r = i + M*l; // note: rows of A & cols of A† (= rows of A*) become new rows of T
		size_t c = j + M*k; // note: cols of A & rows of A† (= cols of A*) become new cols of T
		TL01(r,c) += vTL01[s1][s2](i,j) * vTL01[s1][s2].adjoint()(k,l);
		TL10(r,c) += vTL10[s1][s2](i,j) * vTL10[s1][s2].adjoint()(k,l);
		TR01(r,c) += vTR01[s1][s2](i,j) * vTR01[s1][s2].adjoint()(k,l);
		TR10(r,c) += vTR10[s1][s2](i,j) * vTR10[s1][s2].adjoint()(k,l);
	}
	
//	for (size_t s=0; s<D; ++s)
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	for (size_t k=0; k<M; ++k)
//	for (size_t l=0; l<M; ++l)
//	{
//		size_t r = i + M*l; // note: rows of A & cols of A† (= rows of A*) become new rows of T
//		size_t c = j + M*k; // note: cols of A & rows of A† (= cols of A*) become new cols of T
//		TL(r,c) += Vout.state.A[GAUGE::L][0][s].block[0](i,j) * Vout.state.A[GAUGE::L][0][s].block[0].adjoint()(k,l);
//		TR(r,c) += Vout.state.A[GAUGE::R][0][s].block[0](i,j) * Vout.state.A[GAUGE::R][0][s].block[0].adjoint()(k,l);
//	}
	
	MatrixType Leigen10 = Vout.state.C[0].block[0].adjoint() * Vout.state.C[0].block[0];
	MatrixType Leigen01 = Vout.state.C[1].block[0].adjoint() * Vout.state.C[1].block[0];
	MatrixType Reigen10 = Vout.state.C[0].block[0] * Vout.state.C[0].block[0].adjoint();
	MatrixType Reigen01 = Vout.state.C[1].block[0] * Vout.state.C[1].block[0].adjoint();
//	MatrixType Leigen10 = eigenvectorL(TL10);
//	MatrixType Leigen01 = eigenvectorL(TL01);
//	MatrixType Reigen10 = eigenvectorR(TR10);
//	MatrixType Reigen01 = eigenvectorR(TR01);
	
//	cout << eigenvectorL(TL01) << endl;
//	cout << endl;
//	cout << eigenvectorL(TL10) << endl;
//	cout << endl;
//	cout << eigenvectorR(TR01) << endl;
//	cout << endl;
//	cout << eigenvectorR(TR10) << endl;
//	cout << endl;
//	cout << Leigen << endl << endl;
//	
//	cout << eigenvectorR(TR) << endl;
//	cout << endl;
//	cout << Reigen << endl << endl;
	
	MatrixType hL01 = make_hL(h[1], Vout.state.A[GAUGE::L][1], Vout.state.A[GAUGE::L][0], Heff[0].qloc);
	shift_L(hL01, Vout.state.A[GAUGE::L][1], Heff[0].qloc);
	hL01 += make_hL(h[0], Vout.state.A[GAUGE::L][0], Vout.state.A[GAUGE::L][1], Heff[0].qloc);
	
	MatrixType hL10 = make_hL(h[1], Vout.state.A[GAUGE::L][0], Vout.state.A[GAUGE::L][1], Heff[0].qloc);
	shift_L(hL10, Vout.state.A[GAUGE::L][0], Heff[0].qloc);
	hL10 += make_hL(h[0], Vout.state.A[GAUGE::L][1], Vout.state.A[GAUGE::L][0], Heff[0].qloc);
	
	MatrixType hR01 = make_hR(h[1], Vout.state.A[GAUGE::R][1], Vout.state.A[GAUGE::R][0], Heff[0].qloc);
	shift_R(hR01, Vout.state.A[GAUGE::R][0], Heff[0].qloc);
	hR01 += make_hR(h[0], Vout.state.A[GAUGE::R][0], Vout.state.A[GAUGE::R][1], Heff[0].qloc);
	
	MatrixType hR10 = make_hR(h[1], Vout.state.A[GAUGE::R][0], Vout.state.A[GAUGE::R][1], Heff[0].qloc);
	shift_R(hR10, Vout.state.A[GAUGE::R][1], Heff[0].qloc);
	hR10 += make_hR(h[0], Vout.state.A[GAUGE::R][1], Vout.state.A[GAUGE::R][0], Heff[0].qloc);
	
	// shifts lead to Lanczos-nonconvergence!
	
//	hL01 *= 0.5;
//	hL10 *= 0.5;
//	hR01 *= 0.5;
//	hR10 *= 0.5;
	
	double factor = 0.5;
	double eL01 = factor * (Leigen01 * hR01).trace();
	double eL10 = factor * (Leigen10 * hR10).trace();
	double eR01 = factor * (hL01 * Reigen01).trace();
	double eR10 = factor * (hL10 * Reigen10).trace();
	
	cout << "eL/R=" << eL01 << ", " << eL10 << ", " << eR01 << ", " << eR10 << endl;
	
	eL = min(eL01,eL10);
	eR = min(eR01,eR10);
	
	hR01 -= eL01 * MatrixType::Identity(M,M);
	hR10 -= eL10 * MatrixType::Identity(M,M);
	hL01 -= eR01 * MatrixType::Identity(M,M);
	hL10 -= eL10 * MatrixType::Identity(M,M);
	
	MatrixType HL01 = linearL(hL01, TL01, Reigen01);
	MatrixType HL10 = linearL(hL10, TL10, Reigen10);
	MatrixType HR01 = linearR(hR01, TR01, Leigen01);
	MatrixType HR10 = linearR(hR10, TR10, Leigen10);
	
	Heff[0].L = HL01;
	Heff[0].R = HR10; //shift_R(Heff[0].R, Vout.state.A[GAUGE::R][1], Heff[0].qloc);
	
	Heff[0].AL = Vout.state.A[GAUGE::L][1];
	Heff[0].AR = Vout.state.A[GAUGE::R][1];
	
	Heff[0].h[0] = h[1];
	Heff[0].h[1] = h[0];
	
	Heff[1].h[0] = h[0];
	Heff[1].h[1] = h[1];
	
	Heff[0].dim = D*M*M;
	Heff[1].dim = D*M*M;
	
	vector<Eigenstate<PivotVectorQ<Nq,Scalar> > > gAC(2);
	gAC[0].state.A = Vout.state.A[GAUGE::C][0];
	gAC[1].state.A = Vout.state.A[GAUGE::C][1];
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> LutzAC(LANCZOS::REORTHO::FULL);
	LutzAC.set_dimK(min(30ul, Heff[0].dim));
	LutzAC.edgeState(Heff[0],gAC[0], LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << gAC[0].energy << ", time" << LanczosTimer.info() << ", " << LutzAC.info() << endl;
	}
	
	Heff[1].L = HL10; //shift_L(Heff[1].L, Vout.state.A[GAUGE::L][0], Heff[0].qloc);
	Heff[1].R = HR01;
	
	Heff[1].AL = Vout.state.A[GAUGE::L][0];
	Heff[1].AR = Vout.state.A[GAUGE::R][0];
	
	LutzAC.set_dimK(min(30ul, Heff[1].dim));
	LutzAC.edgeState(Heff[1],gAC[1], LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << gAC[1].energy << ", time" << LanczosTimer.info() << ", " << LutzAC.info() << endl;
	}
	
	Heff[0].dim = M*M;
	Heff[1].dim = M*M;
	
	vector<Eigenstate<PivumpsVector0<Nq,Scalar> > > gC(2);
	gC[0].state.C = Vout.state.C[0];
	gC[1].state.C = Vout.state.C[1];
	
	Heff[0].L = HL10; //shift_L(Heff[0].L, Vout.state.A[GAUGE::L][0], Heff[0].qloc);
	Heff[0].R = HR10; //shift_R(Heff[0].R, Vout.state.A[GAUGE::R][1], Heff[0].qloc);
	
	Heff[0].AL = Vout.state.A[GAUGE::L][0];
	Heff[0].AR = Vout.state.A[GAUGE::R][1];
	
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivumpsVector0<Nq,Scalar>,Scalar> LutzC(LANCZOS::REORTHO::FULL);
	LutzC.set_dimK(min(30ul, Heff[0].dim));
	LutzC.edgeState(Heff[0],gC[0], LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << gC[0].energy << ", time" << LanczosTimer.info() << ", " << LutzC.info() << endl;
	}
	
	Heff[1].L = HL01;
	Heff[1].R = HR01;
	
	Heff[1].AL = Vout.state.A[GAUGE::L][1];
	Heff[1].AR = Vout.state.A[GAUGE::R][0];
	
	LutzC.set_dimK(min(30ul, Heff[1].dim));
	LutzC.edgeState(Heff[1],gC[1], LANCZOS::EDGE::GROUND, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << gC[1].energy << ", time" << LanczosTimer.info() << ", " << LutzC.info() << endl;
	}
	
	Vout.state.A[GAUGE::C][0] = gAC[0].state.A;
	Vout.state.A[GAUGE::C][1] = gAC[1].state.A;
	Vout.state.C[0]           = gC [0].state.C;
	Vout.state.C[1]           = gC [1].state.C;
	
	for (size_t l=0; l<N_sites; ++l)
	{
		(err_var>0.1)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
//		Vout.state.svdDecompose(l);
	}
	
	double epsL0, epsR0;
	Vout.state.calc_epsLR(0,epsL0,epsR0);
	double epsL1, epsR1;
	Vout.state.calc_epsLR(1,epsL1,epsR1);
	cout << "err_var=" << epsL0 << ", " << epsR0 << ", " << epsL1 << ", " << epsR1 << endl;
	err_var = max(epsL0,epsR0);
	err_var = max(err_var,epsL1);
	err_var = max(err_var,epsR1);
	
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

#endif
