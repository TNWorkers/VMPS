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
#include "GMResSolver.h"
#include "VumpsTransferMatrix.h"

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
	
	void edgeState (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, qarray<Nq> Qtot_input, 
	                double tol_eigval_input=1e-7, double tol_var_input=1e-6, 
	                size_t Dlimit=500, 
	                size_t max_iterations=50, size_t min_iterations=6);
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	void prepare (const TwoSiteHamiltonian &h2site, const vector<qarray<Nq> > &qloc_input,
	              Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t Dlimit, qarray<Nq> Qtot_input);
	void iteration1 (Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
//	void iteration2 (Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	void prepare (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t Dlimit, qarray<Nq> Qtot_input);
	void iteration1 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	void iteration2 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	void iteration4 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	void cleanup (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout);
	
	/**Returns the current error of the eigenvalue while the sweep process.*/
	inline double get_errEigval() const {return err_eigval;};
	
	/**Returns the current error of the state while the sweep process.*/
	inline double get_errState() const {return err_var;};
	
	void make_explicitT (const UmpsQ<Nq,Scalar> &Vbra, const UmpsQ<Nq,Scalar> &Vket, MatrixType &TL, MatrixType &TR);
	
private:
	
	size_t N_sites;
	double tol_eigval, tol_var;
	size_t N_iterations;
	double err_eigval, err_var, err_state=std::nan("1");
	
	vector<PivumpsMatrix<Nq,Scalar,Scalar> > Heff;
	vector<qarray<Nq> > qloc;
	std::array<boost::multi_array<Scalar,4>,2> h;
	size_t D, M, dW;
	
	double eL, eR, eoldR, eoldL;
	
	MatrixXd linearL (const MatrixType &hL, const MatrixType &TL, const MatrixType &Leigen, double e=0);
	MatrixXd linearR (const MatrixType &hR, const MatrixType &TR, const MatrixType &Reigen, double e=0);
	template<typename Atype, typename Wtype> void solve_linear (GAUGE::OPTION gauge, const Atype &A, const MatrixType &hLR, 
	                                                            const MatrixType &LReigen, const Wtype &Warray, double e, MatrixType &Hres);
	
	boost::multi_array<Scalar,4> make_Warray4 (size_t b, const MpHamiltonian &H);
	boost::multi_array<Scalar,8> make_Warray8 (size_t b, const MpHamiltonian &H);
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
//	MatrixXd eigenvectorL (const MatrixType &TL);
//	MatrixXd eigenvectorR (const MatrixType &TR);
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
	ss << "e0=" << setprecision(13) << min(eL,eR) << ", ";
	ss << "err_eigval=" << setprecision(13) << err_eigval << ", err_var=" << err_var << ", ";
	if (!isnan(err_state))
	{
		ss << "err_state=" << err_state << ", ";
	}
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
prepare (const TwoSiteHamiltonian &h2site, const vector<qarray<Nq> > &qloc_input, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t M_input, qarray<Nq> Qtot_input)
{
	N_sites = 1;
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = h2site.shape()[0];
	M = M_input;
	Heff.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Heff[l].h[0].resize(boost::extents[D][D][D][D]);
		Heff[l].h[0] = h2site;
		Heff[l].h[1].resize(boost::extents[D][D][D][D]);
//		Heff[l].h[1] = (N_sites==1)? h2site[0] : h2site[1];
		Heff[l].h[1] = h2site;
		Heff[l].qloc = qloc_input;
	}
	
	h[0].resize(boost::extents[D][D][D][D]);
	h[0] = h2site;
	h[1].resize(boost::extents[D][D][D][D]);
//	h[1] = (N_sites==1)? h2site[0] : h2site[1];
	h[1] = h2site;
	
	// resize Vout
	Vout.state = UmpsQ<Nq,Scalar>(Heff[0].qloc, N_sites, M, Qtot_input);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
	}
	
	// initial energy
//	eoldL = energy_L(Heff[0].h[0], Vout.state.A[GAUGE::L][0], Vout.state.C[0], Heff[0].qloc);
//	eoldR = energy_R(Heff[0].h[0], Vout.state.A[GAUGE::R][0], Vout.state.C[0], Heff[0].qloc);
	eoldL = std::nan("");
	eoldR = std::nan("");
	
	err_eigval = 1.;
	err_var    = 1.;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration1 (Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	MatrixType Reigen = Vout.state.C[N_sites-1].block[0] * Vout.state.C[N_sites-1].block[0].adjoint();
	MatrixType Leigen = Vout.state.C[N_sites-1].block[0].adjoint() * Vout.state.C[N_sites-1].block[0];
	
	MatrixType hR = make_hR(Heff[0].h[0], Vout.state.A[GAUGE::R][0], Heff[0].qloc);
	MatrixType hL = make_hL(Heff[0].h[0], Vout.state.A[GAUGE::L][0], Heff[0].qloc);
	
	eL = (Leigen*hR).trace();
	eR = (hL*Reigen).trace();
	
	MatrixType HR(M,M), HL(M,M);
	
	Stopwatch<> GMresTimer;
//	HL = linearL(hL, TL, Reigen, eR);
	vector<Scalar> Wdummy; // The dummy also clarifies the template parameter of solve_linear for the compiler
	solve_linear(GAUGE::L, Vout.state.A[GAUGE::L][0], hL, Reigen, Wdummy, eR, HL);
//	HR = linearR(hR, TR, Leigen, eL);
	solve_linear(GAUGE::R, Vout.state.A[GAUGE::R][0], hR, Leigen, Wdummy, eL, HR);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
//		lout << "\t•L: " << GimliL.info() << endl;
//		lout << "\t•R: " << GimliR.info() << endl;
	}
	
	Heff[0].L = HL;
	Heff[0].R = HR;
	Heff[0].AL = Vout.state.A[GAUGE::L][0];
	Heff[0].AR = Vout.state.A[GAUGE::R][0];
	Heff[0].dim = Heff[0].qloc.size() * M * M;
	
	// dynamic Lanczos error: at least 1e-7, can go down further if err_eigval is smaller, but not below 1e-14
	double tolLanczosEigval = max(min(0.01*err_eigval,1e-7),1e-14); // 1e-7
	double tolLanczosState  = max(min(0.01*err_var,1e-4),   1e-10); // 1e-4
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "current Lanczos tolerances: " << tolLanczosEigval << ", " << tolLanczosState << endl;
	}
	
	Heff[0].dim = Heff[0].qloc.size() * M * M;
	Eigenstate<PivotVectorQ<Nq,Scalar> > g1;
	g1.state.A = Vout.state.A[GAUGE::C][0];
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz1(LANCZOS::REORTHO::FULL);
	Lutz1.set_dimK(min(30ul, Heff[0].dim));
	Lutz1.edgeState(Heff[0],g1, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz1.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << g1.energy << endl;
	}
	
	Heff[0].dim = M*M;
	Eigenstate<PivumpsVector0<Nq,Scalar> > g0;
	g0.state.C = Vout.state.C[0];
	
	LanczosSolver<PivumpsMatrix<Nq,Scalar,Scalar>,PivumpsVector0<Nq,Scalar>,Scalar> Lutz0(LANCZOS::REORTHO::FULL);
	Lutz0.set_dimK(min(30ul, Heff[0].dim));
	Lutz0.edgeState(Heff[0],g0, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz0.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << g0.energy << endl;
	}
	
	Vout.state.A[GAUGE::C][0] = g1.state.A;
	Vout.state.C[0]           = g0.state.C;
	(err_var>0.1)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
	
	double epsLsq, epsRsq;
	Vout.state.calc_epsLRsq(0,epsLsq,epsRsq);
	err_var = max(sqrt(epsLsq),sqrt(epsRsq));
	
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy(0) << endl;
		lout << eigeninfo() << endl;
		lout << IterationTimer.info("full iteration") << endl;
		lout << endl;
	}
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
prepare (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout, size_t M_input, qarray<Nq> Qtot_input)
{
	assert(H.length()<=2 or H.length()==4);
	
	N_sites = H.length();
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = H.locBasis(0).size();
	M = M_input;
	dW = H.auxdim();
	
	// resize Vout
	Vout.state = UmpsQ<Nq,Scalar>(H.locBasis(0), N_sites, M, Qtot_input);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
	}
	
	// initial energy
	eoldL = std::nan("");
	eoldR = std::nan("");
	
	err_eigval = 1.;
	err_var    = 1.;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
make_explicitT (const UmpsQ<Nq,Scalar> &Vbra, const UmpsQ<Nq,Scalar> &Vket, MatrixType &TL, MatrixType &TR)
{
//	vector<vector<MatrixType> > TL(H.auxdim());
//	vector<vector<MatrixType> > TR(H.auxdim());
//	
//	for (size_t a=0; a<H.auxdim(); ++a)
//	{
//		TL[a].resize(H.auxdim());
//		TR[a].resize(H.auxdim());
//	}
//	
//	for (size_t a=0; a<H.auxdim(); ++a)
//	for (size_t b=0; b<H.auxdim(); ++b)
//	{
//		TL[a][b].resize(M*M,M*M); TL[a][b].setZero();
//		TR[a][b].resize(M*M,M*M); TR[a][b].setZero();
//	}
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<H.W[0][s1][s2].outerSize(); ++k12)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW(H.W[0][s1][s2],k12); iW; ++iW)
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	for (size_t k=0; k<M; ++k)
//	for (size_t l=0; l<M; ++l)
//	{
//		size_t a = iW.row();
//		size_t b = iW.col();
//		
//		size_t r = i + M*l; // note: rows of A & cols of A† (= rows of A*) become new rows of T
//		size_t c = j + M*k; // note: cols of A & rows of A† (= cols of A*) become new cols of T
//		
//		TL[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::L][0][s2].block[0](i,j) * Vout.state.A[GAUGE::L][0][s1].block[0].adjoint()(k,l);
//		TR[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::R][0][s2].block[0](i,j) * Vout.state.A[GAUGE::R][0][s1].block[0].adjoint()(k,l);
//	}
	
	TL.resize(M*M,M*M); TL.setZero();
	TR.resize(M*M,M*M); TR.setZero();
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
		TL(r,c) += Vket.A[GAUGE::L][0][s].block[0](i,j) * Vbra.A[GAUGE::L][0][s].block[0].adjoint()(k,l);
		TR(r,c) += Vket.A[GAUGE::R][0][s].block[0](i,j) * Vbra.A[GAUGE::R][0][s].block[0].adjoint()(k,l);
	}
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration1 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	MatrixType Reigen = Vout.state.C[0].block[0] * Vout.state.C[0].block[0].adjoint();
	MatrixType Leigen = Vout.state.C[0].block[0].adjoint() * Vout.state.C[0].block[0];
	
	vector<MatrixType> YL(dW);
	vector<MatrixType> YR(dW);
	
	for (size_t a=0; a<dW; ++a)
	{
		YL[a].resize(M,M); YL[a].setZero();
		YR[a].resize(M,M); YR[a].setZero();
	}
	
	boost::multi_array<MatrixType,LEGLIMIT> L(boost::extents[dW][1]);
	boost::multi_array<MatrixType,LEGLIMIT> R(boost::extents[dW][1]);
	
	Stopwatch<> GMresTimer;
	L[dW-1][0].resize(M,M);
	L[dW-1][0].setIdentity();
	
	for (int b=dW-2; b>=0; --b)
	{
		YL[b] = make_YL(b, H.W[0], L, Vout.state.A[GAUGE::L][0], H.locBasis(0));
		
		vector<Scalar> Wval(D);
		for (size_t s=0; s<D; ++s)
		{
			Wval[s] = H.W[0][s][s].coeff(b,b);
		}
		
//		if (TL[b][b].norm() == 0.)
		if (accumulate(Wval.begin(),Wval.end(),0) == 0.)
		{
			L[b][0] = YL[b];
		}
		else
		{
//			L[b][0] = linearL(YL[b], TL[b][b], Reigen, (YL[b] * Reigen).trace());
			solve_linear(GAUGE::L, Vout.state.A[GAUGE::L][0], YL[b], Reigen, Wval, (YL[b]*Reigen).trace(), L[b][0]);
		}
	}
	R[0][0].resize(M,M);
	R[0][0].setIdentity();
	
	for (int a=1; a<dW; ++a)
	{
		YR[a] = make_YR(a, H.W[0], R, Vout.state.A[GAUGE::R][0], H.locBasis(0));
		
		vector<Scalar> Wval(D);
		for (size_t s=0; s<D; ++s)
		{
			Wval[s] = H.W[0][s][s].coeff(a,a);
		}
		
//		if (TR[a][a].norm() == 0.)
		if (accumulate(Wval.begin(),Wval.end(),0) == 0.)
		{
			R[a][0] = YR[a];
		}
		else
		{
//			R[a][0] = linearR(YR[a], TR[a][a], Leigen, (Leigen * YR[a]).trace());
			solve_linear(GAUGE::R, Vout.state.A[GAUGE::R][0], YR[a], Leigen, Wval, (YR[a]*Leigen).trace(), R[a][0]);
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
	}
	
	// dynamic Lanczos error: at least 1e-7, can go down further if err_eigval is smaller, but not below 1e-14
	double tolLanczosEigval = max(min(0.01*err_eigval,1e-7),1e-14); // 1e-7
	double tolLanczosState  = max(min(0.01*err_var,1e-4),  1e-10); // 1e-4
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "current Lanczos tolerances: " << tolLanczosEigval << ", " << tolLanczosState << endl;
	}
	
//	HeffA.L.clear();
//	HeffA.R.clear();
//	HeffA.L.push_back(qloc3dummy,L);
//	HeffA.R.push_back(qloc3dummy,R);
//	HeffA.W = H.W[0];
	
	PivotMatrixQ<Nq,Scalar,Scalar> HeffA;
	HeffA.W = H.W[0];
	HeffA.L.push_back(qloc3dummy,L);
	HeffA.R.push_back(qloc3dummy,R);
	
//	if (HeffA.dim == 0)
	{
		precalc_blockStructure (HeffA.L, Vout.state.A[GAUGE::C][0], HeffA.W, Vout.state.A[GAUGE::C][0], HeffA.R, 
		                        H.locBasis(0), HeffA.qlhs, HeffA.qrhs);
	}
	
	// reset dim
	HeffA.dim = 0;
	for (size_t s=0; s<H.locBasis(0).size(); ++s)
	for (size_t q=0; q<Vout.state.A[GAUGE::C][0][s].dim; ++q)
	{
		HeffA.dim += Vout.state.A[GAUGE::C][0][s].block[q].rows() * Vout.state.A[GAUGE::C][0][s].block[q].cols();
	}
	
	Eigenstate<PivotVectorQ<Nq,Scalar> > gAC;
	gAC.state.A = Vout.state.A[GAUGE::C][0];
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
	Lutz.set_dimK(min(30ul, HeffA.dim));
	Lutz.edgeState(HeffA,gAC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << gAC.energy << endl;
	}
	
	Eigenstate<PivotVector0Q<Nq,Scalar> > gC;
	gC.state.A = Vout.state.C[0];
	
	HeffA.dim = 0;
	for (size_t q=0; q<Vout.state.C[0].dim; ++q)
	{
		HeffA.dim += Vout.state.C[0].block[q].rows() * Vout.state.C[0].block[q].cols();
	}
	
	LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVector0Q<Nq,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL);
	Lucy.set_dimK(min(30ul, HeffA.dim));
	Lucy.edgeState(HeffA,gC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << gC.energy << endl;
	}
	
	eL = (YL[0]*Reigen).trace();
	eR = (Leigen*YR[dW-1]).trace();
	
	Vout.state.A[GAUGE::C][0] = gAC.state.A;
	Vout.state.C[0]           = gC.state.A;
	(err_var>0.1)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
	
	double epsLsq, epsRsq;
	Vout.state.calc_epsLRsq(0,epsLsq,epsRsq);
	err_var = max(sqrt(epsLsq),sqrt(epsRsq));
	
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy(0) << endl;
		lout << eigeninfo() << endl;
		lout << IterationTimer.info("full iteration") << endl;
		lout << endl;
	}
	
//	if (N_iterations%4 == 0)
	if (err_var < 1e-4)
	{
		PivotMatrix2Q<Nq,Scalar,Scalar> Heff2;
		Heff2.W12 = H.W[0];
		Heff2.W34 = H.W[0];
		Heff2.L = HeffA.L;
		Heff2.R = HeffA.R;
		Heff2.dim = D*D*M*M;
		Heff2.qloc12 = H.locBasis(0);
		Heff2.qloc34 = H.locBasis(0);
	//	contract_R(HeffA.R, Vout.state.A[GAUGE::R][0], H.W[0], Vout.state.A[GAUGE::R][0], H.locBasis(0), Heff2.R);
		
		vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > AA;
		contract_AA(Vout.state.A[GAUGE::C][0], H.locBasis(0), Vout.state.A[GAUGE::R][0], H.locBasis(0), AA);
		
		PivotVector2Q<Nq,Scalar> Apair;
		Apair.A = AA;
		
		HxV(Heff2,Apair);
		
		MatrixType B2(Vout.state.N[GAUGE::L][0][0].block[0].cols(), Vout.state.N[GAUGE::R][0][0].block[0].rows());
		B2.setZero();
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			B2 += Vout.state.N[GAUGE::L][0][s1].block[0].adjoint() * 
			      Apair.A[s1][s2].block[0] * 
			      Vout.state.N[GAUGE::R][0][s2].block[0].adjoint();
		}
		BDCSVD<MatrixType> Jack(B2,ComputeThinU|ComputeThinV);
		
		err_state = B2.norm();
		
//		cout << "B2=" << B2.norm() << endl;
//		
//		MatrixType B2alt(Vout.state.N[GAUGE::L][0][0].block[0].cols(), Vout.state.N[GAUGE::R][0][0].block[0].rows());
//		B2alt.setZero();
//		
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		for (size_t s3=0; s3<D; ++s3)
//		for (size_t s4=0; s4<D; ++s4)
//		{
//			if (H.H2site(0,0,false)[s1][s2][s3][s4] != 0.)
//			{
//				B2alt += H.H2site(0,0,false)[s1][s2][s3][s4] * 
//				         Vout.state.N[GAUGE::L][0][s1].block[0].adjoint() * 
//				         Apair.A[s2][s4].block[0] * 
//				         Vout.state.N[GAUGE::R][0][s3].block[0].adjoint();
//			}
//		}
//		
//		cout << "B2alt=" << B2alt.norm() << endl;
		
		//		double eps_svd = 1e-4;
//		size_t Nret = (Jack.singularValues().array() > eps_svd).count();
//		Nret = min(Nret,500-D);
//		cout << "Nret=" << Nret << endl;
		
//		if (Nret > 0)
//		{
//			for (size_t s=0; s<D; ++s)
//			{
//				Vout.state.A[GAUGE::L][0][s].block[0].conservativeResize(M+Nret,M+Nret);
//				Vout.state.A[GAUGE::L][0][s].block[0].block(0,D, M,Nret) = Vout.state.N[GAUGE::L][0][s].block[0] * 
//				                                                           Jack.matrixU().leftCols(Nret);
//				Vout.state.A[GAUGE::L][0][s].block[0].bottomRows(Nret).setZero();
//				
//				Vout.state.A[GAUGE::R][0][s].block[0].conservativeResize(M+Nret,M+Nret);
//				Vout.state.A[GAUGE::R][0][s].block[0].block(D,0, Nret,M) = Jack.matrixV().adjoint().topRows(Nret) * 
//					                                                       Vout.state.N[GAUGE::R][0][s].block[0];
//				Vout.state.A[GAUGE::R][0][s].block[0].rightCols(Nret).setZero();
//				
//				Vout.state.A[GAUGE::C][0][s].block[0].conservativeResize(M+Nret,M+Nret);
//				Vout.state.A[GAUGE::C][0][s].block[0].bottomRows(Nret).setZero();
//				Vout.state.A[GAUGE::C][0][s].block[0].rightCols(Nret).setZero();
//			}
//			
//			Vout.state.C[0].block[0].conservativeResize(M+Nret,M+Nret);
//			Vout.state.C[0].block[0].bottomRows(Nret).setZero();
//			Vout.state.C[0].block[0].rightCols(Nret).setZero();
//			
//			M += Nret;
//			cout << "new M=" << M << endl;
//		}
	}
	else
	{
		err_state = std::nan("1");
	}
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
boost::multi_array<Scalar,4> VumpsSolver<Nq,MpHamiltonian,Scalar>::
make_Warray4 (size_t b, const MpHamiltonian &H)
{
	boost::multi_array<Scalar,4> Wout(boost::extents[D][D][D][D]);
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (int k12=0; k12<H.W[0][s1][s2].outerSize(); ++k12)
	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W[0][s1][s2],k12); iW12; ++iW12)
	for (int k34=0; k34<H.W[1][s3][s4].outerSize(); ++k34)
	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W[1][s3][s4],k34); iW34; ++iW34)
	{
		if (iW12.row() == b and iW34.col() == b and 
		    iW12.col() == iW34.row() and
		    H.locBasis(0)[s1]+H.locBasis(1)[s3] == H.locBasis(0)[s2]+H.locBasis(1)[s4])
		{
			Wout[s1][s2][s3][s4] = iW12.value() * iW34.value();
		}
	}
	
	return Wout;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration2 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > ApairL;
	vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > ApairR;
	contract_AA(Vout.state.A[GAUGE::L][0], H.locBasis(0), Vout.state.A[GAUGE::L][1], H.locBasis(1), ApairL);
	contract_AA(Vout.state.A[GAUGE::R][0], H.locBasis(0), Vout.state.A[GAUGE::R][1], H.locBasis(1), ApairR);
	
	typedef tuple<size_t,size_t,size_t,size_t,size_t,Scalar> Wtuple;
	std::map<int,vector<Wtuple> > WWWWbyRow;
	std::map<int,vector<Wtuple> > WWWWbyCol;
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k12=0; k12<H.W[0][s1][s2].outerSize(); ++k12)
	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W[0][s1][s2],k12); iW12; ++iW12)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (int k34=0; k34<H.W[1][s3][s4].outerSize(); ++k34)
	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W[1][s3][s4],k34); iW34; ++iW34)
	{
		if (iW12.col()==iW34.row())
		{
			if (abs(iW12.value())>1e-15 and abs(iW34.value())>1e-15 and iW12.row()>iW34.col())
			{
				auto val = iW12.value() * iW34.value();
				Wtuple r = make_tuple(iW34.col(),s1,s2,s3,s4,val);
				Wtuple c = make_tuple(iW12.row(),s1,s2,s3,s4,val);
				WWWWbyRow[iW12.row()].push_back(r);
				WWWWbyCol[iW34.col()].push_back(c);
			}
		}
	}
	
	MatrixType Reigen = Vout.state.C[N_sites-1].block[0] * Vout.state.C[N_sites-1].block[0].adjoint();
	MatrixType Leigen = Vout.state.C[N_sites-1].block[0].adjoint() * Vout.state.C[N_sites-1].block[0];
	
	vector<MatrixType> YL(dW);
	vector<MatrixType> YR(dW);
	
	for (size_t a=0; a<dW; ++a)
	{
		YL[a].resize(M,M); YL[a].setZero();
		YR[a].resize(M,M); YR[a].setZero();
	}
	
	boost::multi_array<MatrixType,LEGLIMIT> L(boost::extents[dW][1]);
	boost::multi_array<MatrixType,LEGLIMIT> R(boost::extents[dW][1]);
	
	L[dW-1][0].resize(M,M);
	L[dW-1][0].setIdentity();
	
	Stopwatch<> GMresTimer;
	
//	#pragma omp parallel sections
	{
//		#pragma omp section
		{
			for (int b=dW-2; b>=0; --b)
			{
//				YL[b] = make_YL(b, H.W[0], H.W[1], L, Vout.state.A[GAUGE::L][0], Vout.state.A[GAUGE::L][1], H.locBasis(0));
				YL[b] = make_YL(WWWWbyCol[b], L, ApairL, H.locBasis(0));
				
				boost::multi_array<Scalar,4> Warray = make_Warray4(b,H);
				
				Scalar Wsum = 0;
				for (size_t s1=0; s1<D; ++s1)
				for (size_t s2=0; s2<D; ++s2)
				for (size_t s3=0; s3<D; ++s3)
				for (size_t s4=0; s4<D; ++s4)
				{
					Wsum += Warray[s1][s2][s3][s4];
				}
				
		//		if (TL[b][b].norm() == 0.)
		//		if (accumulate(Warray.begin(),Warray.end(),0) == 0.)
				if (Wsum == 0.)
				{
					L[b][0] = YL[b];
				}
				else
				{
					double e = (YL[b]*Reigen).trace();
		//			L[b][0] = linearL(YL[b], TL[b][b], Reigen, e);
					solve_linear(GAUGE::L, ApairL, YL[b], Reigen, Warray, e, L[b][0]);
				}
			}
		}
//		#pragma omp section
		{
			R[0][0].resize(M,M);
			R[0][0].setIdentity();
			
			for (int a=1; a<dW; ++a)
			{
//				YR[a] = make_YR(a, H.W[0], H.W[1], R, Vout.state.A[GAUGE::R][0], Vout.state.A[GAUGE::R][1], H.locBasis(0));
				YR[a] = make_YR(WWWWbyRow[a], R, ApairR, H.locBasis(0));
				
				boost::multi_array<Scalar,4> Warray = make_Warray4(a,H);
				
				Scalar Wsum = 0;
				for (size_t s1=0; s1<D; ++s1)
				for (size_t s2=0; s2<D; ++s2)
				for (size_t s3=0; s3<D; ++s3)
				for (size_t s4=0; s4<D; ++s4)
				{
					Wsum += Warray[s1][s2][s3][s4];
				}
				
		//		if (TR[a][a].norm() == 0.)
		//		if (accumulate(Warray.begin(),Warray.end(),0) == 0.)
				if (Wsum == 0.)
				{
					R[a][0] = YR[a];
				}
				else
				{
					double e = (Leigen*YR[a]).trace();
		//			R[a][0] = linearR(YR[a], TR[a][a], Leigen, e);
					solve_linear(GAUGE::R, ApairR, YR[a], Leigen, Warray, e, R[a][0]);
				}
			}
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
	}
	
	vector<PivotMatrixQ<Nq,Scalar,Scalar> > HeffA(N_sites);
	vector<PivotMatrixQ<Nq,Scalar,Scalar> > HeffAC(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffA[l].W  = H.W[l];
		HeffAC[l].W = H.W[l];
	}
	
	HeffA[0].L.push_back(qloc3dummy,L);
	HeffA[N_sites-1].R.push_back(qloc3dummy,R);
	
	for (size_t l=1; l<N_sites; ++l)
	{
		contract_L(HeffA[l-1].L, Vout.state.A[GAUGE::L][l-1], H.W[l-1], Vout.state.A[GAUGE::L][l-1], H.locBasis(l-1), HeffA[l].L);
	}
	
	for (int l=N_sites-2; l>=0; --l)
	{
		contract_R(HeffA[l+1].R, Vout.state.A[GAUGE::R][l+1], H.W[l+1], Vout.state.A[GAUGE::R][l+1], H.locBasis(l+1), HeffA[l].R);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffAC[l].L = HeffA[(l+1)%N_sites].L;
		HeffAC[l].R = HeffA[l].R;
	}
	
	vector<Eigenstate<PivotVectorQ<Nq,Scalar> > > gAC(N_sites);
	vector<Eigenstate<PivotVector0Q<Nq,Scalar> > > gC(N_sites);
	
	// dynamic Lanczos error: at least 1e-7, can go down further if err_eigval is smaller, but not below 1e-14
	double tolLanczosEigval = max(min(0.01*err_eigval,1e-7),1e-14); // 1e-7
	double tolLanczosState  = max(min(0.01*err_var,1e-4),   1e-10); // 1e-4
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "current Lanczos tolerances: " << tolLanczosEigval << ", " << tolLanczosState << endl;
	}
	
	// local optimization
	for (size_t l=0; l<N_sites; ++l)
	{
//		HeffA.L.clear();
//		HeffA.R.clear();
//		HeffA.L.push_back(qloc3dummy,L);
//		HeffA.R.push_back(qloc3dummy,R);
//		HeffA.W = H.W[l];
		
//		if (l==0)
//		{
//			auto Rtmp = HeffA.R;
//			contract_R(Rtmp, Vout.state.A[GAUGE::R][1], H.W[1], Vout.state.A[GAUGE::R][1], H.locBasis(1), HeffA.R);
//			
//		}
//		else if (l==1)
//		{
//			auto Ltmp = HeffA.L;
//			contract_L(Ltmp, Vout.state.A[GAUGE::L][0], H.W[0], Vout.state.A[GAUGE::L][0], H.locBasis(0), HeffA.L);
//		}
		
		precalc_blockStructure (HeffA[l].L, Vout.state.A[GAUGE::C][l], HeffA[l].W, Vout.state.A[GAUGE::C][l], HeffA[l].R, 
		                        H.locBasis(l), HeffA[l].qlhs, HeffA[l].qrhs);
		
		// reset dim
		HeffA[l].dim = 0;
		for (size_t s=0; s<H.locBasis(l).size(); ++s)
		for (size_t q=0; q<Vout.state.A[GAUGE::C][l][s].dim; ++q)
		{
			HeffA[l].dim += Vout.state.A[GAUGE::C][l][s].block[q].rows() * Vout.state.A[GAUGE::C][l][s].block[q].cols();
		}
		
//		Eigenstate<PivotVectorQ<Nq,Scalar> > gAC;
		gAC[l].state.A = Vout.state.A[GAUGE::C][l];
		
		Stopwatch<> LanczosTimer;
		LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
		Lutz.set_dimK(min(30ul, HeffA[l].dim));
		Lutz.edgeState(HeffA[l],gAC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(AC)=" << setprecision(13) << gAC[l].energy << endl;
		}
		
//		if (l==0)
//		{
//			auto Ltmp = HeffA.L;
//			contract_L(Ltmp, Vout.state.A[GAUGE::L][0], H.W[0], Vout.state.A[GAUGE::L][0], H.locBasis(0), HeffA.L);
//		}
//		else if (l==1)
//		{
//			auto Ltmp = HeffA.L;
//			contract_L(Ltmp, Vout.state.A[GAUGE::L][1], H.W[1], Vout.state.A[GAUGE::L][1], H.locBasis(1), HeffA.L);
//		}
		
		gC[l].state.A = Vout.state.C[l];
		
		HeffAC[l].dim = 0;
		for (size_t q=0; q<Vout.state.C[l].dim; ++q)
		{
			HeffAC[l].dim += Vout.state.C[l].block[q].rows() * Vout.state.C[l].block[q].cols();
		}
		
		LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVector0Q<Nq,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL);
		Lucy.set_dimK(min(30ul, HeffAC[l].dim));
		Lucy.edgeState(HeffAC[l],gC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(C)=" << setprecision(13) << gC[l].energy << endl;
		}
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.A[GAUGE::C][l] = gAC[l].state.A;
		Vout.state.C[l]           = gC[l].state.A;
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		(err_var>0.1)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
	}
	
	eL = (YL[0]*Reigen).trace() / (N_sites * H.width());
	eR = (Leigen*YR[dW-1]).trace() / (N_sites * H.width());
	
	MatrixXd epsLRsq(N_sites,2);
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.calc_epsLRsq(l,epsLRsq(l,0),epsLRsq(l,1));
	}
	err_var = sqrt(epsLRsq.sum());
	
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy(0) << ", " << Vout.state.entropy(1) << endl;
		lout << eigeninfo() << endl;
		lout << IterationTimer.info("full iteration") << endl;
		lout << endl;
	}
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
boost::multi_array<Scalar,8> VumpsSolver<Nq,MpHamiltonian,Scalar>::
make_Warray8 (size_t b, const MpHamiltonian &H)
{
	boost::multi_array<Scalar,8> Wout(boost::extents[D][D][D][D][D][D][D][D]);
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (size_t s5=0; s5<D; ++s5)
	for (size_t s6=0; s6<D; ++s6)
	for (size_t s7=0; s7<D; ++s7)
	for (size_t s8=0; s8<D; ++s8)
	for (int k12=0; k12<H.W[0][s1][s2].outerSize(); ++k12)
	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W[0][s1][s2],k12); iW12; ++iW12)
	for (int k34=0; k34<H.W[1][s3][s4].outerSize(); ++k34)
	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W[1][s3][s4],k34); iW34; ++iW34)
	for (int k56=0; k56<H.W[2][s5][s6].outerSize(); ++k56)
	for (typename SparseMatrix<Scalar>::InnerIterator iW56(H.W[2][s5][s6],k56); iW56; ++iW56)
	for (int k78=0; k78<H.W[3][s7][s8].outerSize(); ++k78)
	for (typename SparseMatrix<Scalar>::InnerIterator iW78(H.W[3][s7][s8],k78); iW78; ++iW78)
	{
		if (iW12.row() == b and iW78.col() == b and 
		    iW12.col() == iW34.row() and
		    iW34.col() == iW56.row() and
		    iW56.col() == iW78.row() and
		    H.locBasis(0)[s1]+H.locBasis(1)[s3]+H.locBasis(2)[s5]+H.locBasis(3)[s7] 
		    == 
		    H.locBasis(0)[s2]+H.locBasis(1)[s4]+H.locBasis(2)[s6]+H.locBasis(3)[s8])
		{
			Wout[s1][s2][s3][s4][s5][s6][s7][s8] = iW12.value() * iW34.value() * iW56.value() * iW78.value();
		}
	}
	
	return Wout;
}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
iteration4 (const MpHamiltonian &H, Eigenstate<UmpsQ<Nq,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	boost::multi_array<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettL;
	AquartettL.resize(boost::extents[D][D][D][D]);
	boost::multi_array<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettR;
	AquartettR.resize(boost::extents[D][D][D][D]);
	contract_AAAA(Vout.state.A[GAUGE::L][0], H.locBasis(0), 
	              Vout.state.A[GAUGE::L][1], H.locBasis(1), 
	              Vout.state.A[GAUGE::L][2], H.locBasis(2), 
	              Vout.state.A[GAUGE::L][3], H.locBasis(3), 
	              AquartettL);
	contract_AAAA(Vout.state.A[GAUGE::R][0], H.locBasis(0), 
	              Vout.state.A[GAUGE::R][1], H.locBasis(1), 
	              Vout.state.A[GAUGE::R][2], H.locBasis(2), 
	              Vout.state.A[GAUGE::R][3], H.locBasis(3), 
	              AquartettR);
	
//	vector<tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,Scalar> > WWWWnonZeros;
	
	typedef tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,Scalar> Wtuple;
	std::map<int,vector<Wtuple> > WWWWoffDiagRow;
	std::map<int,vector<Wtuple> > WWWWoffDiagCol;
	std::map<int,vector<Wtuple> > WWWWdiag;
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k12=0; k12<H.W[0][s1][s2].outerSize(); ++k12)
	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W[0][s1][s2],k12); iW12; ++iW12)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (int k34=0; k34<H.W[1][s3][s4].outerSize(); ++k34)
	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W[1][s3][s4],k34); iW34; ++iW34)
	for (size_t s5=0; s5<D; ++s5)
	for (size_t s6=0; s6<D; ++s6)
	for (int k56=0; k56<H.W[2][s5][s6].outerSize(); ++k56)
	for (typename SparseMatrix<Scalar>::InnerIterator iW56(H.W[2][s5][s6],k56); iW56; ++iW56)
	for (size_t s7=0; s7<D; ++s7)
	for (size_t s8=0; s8<D; ++s8)
	for (int k78=0; k78<H.W[3][s7][s8].outerSize(); ++k78)
	for (typename SparseMatrix<Scalar>::InnerIterator iW78(H.W[3][s7][s8],k78); iW78; ++iW78)
	{
		if (iW12.col()==iW34.row() and 
		    iW34.col()==iW56.row() and 
		    iW56.col()==iW78.row())
		{
			if (    abs(iW12.value())>1e-15 and abs(iW34.value())>1e-15 
			    and abs(iW56.value())>1e-15 and abs(iW78.value())>1e-15
			    and iW12.row()>iW78.col())
			{
//				tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,double> val 
//				= make_tuple(iW12.row(),s1,s2,s3,s4,s5,s6,s7,s8,iW78.col(),
//				   iW12.value() * iW34.value() * iW56.value() * iW78.value());
//				WWWWnonZeros.push_back(val);
				auto val = iW12.value() * iW34.value() * iW56.value() * iW78.value();
				Wtuple r = make_tuple(iW78.col(),s1,s2,s3,s4,s5,s6,s7,s8,val);
				Wtuple c = make_tuple(iW12.row(),s1,s2,s3,s4,s5,s6,s7,s8,val);
				WWWWoffDiagRow[iW12.row()].push_back(r);
				WWWWoffDiagCol[iW78.col()].push_back(c);
			}
		}
	}
	
	MatrixType Reigen = Vout.state.C[N_sites-1].block[0] * Vout.state.C[N_sites-1].block[0].adjoint();
	MatrixType Leigen = Vout.state.C[N_sites-1].block[0].adjoint() * Vout.state.C[N_sites-1].block[0];
	
	vector<MatrixType> YL(dW);
	vector<MatrixType> YR(dW);
	
	for (size_t a=0; a<dW; ++a)
	{
		YL[a].resize(M,M); YL[a].setZero();
		YR[a].resize(M,M); YR[a].setZero();
	}
	
	boost::multi_array<MatrixType,LEGLIMIT> L(boost::extents[dW][1]);
	boost::multi_array<MatrixType,LEGLIMIT> R(boost::extents[dW][1]);
	
	L[dW-1][0].resize(M,M);
	L[dW-1][0].setIdentity();
	R[0][0].resize(M,M);
	R[0][0].setIdentity();
	
	Stopwatch<> GMresTimer;
	
//	#pragma omp parallel sections
	{
//		#pragma omp section
		{
			for (int b=dW-2; b>=0; --b)
			{
//				Stopwatch<> YTimer;
//				YL[b] = make_YL(b, H.W[0], H.W[1], H.W[2], H.W[3], L, AquartettL, H.locBasis(0));
				YL[b] = make_YL(WWWWoffDiagCol[b], L, AquartettL, H.locBasis(0));
//				cout << YTimer.info("YL") << endl;
				
				boost::multi_array<Scalar,8> Warray = make_Warray8(b,H);
				
				Scalar Wsum = 0;
				for (size_t s1=0; s1<D; ++s1)
				for (size_t s2=0; s2<D; ++s2)
				for (size_t s3=0; s3<D; ++s3)
				for (size_t s4=0; s4<D; ++s4)
				for (size_t s5=0; s5<D; ++s5)
				for (size_t s6=0; s6<D; ++s6)
				for (size_t s7=0; s7<D; ++s7)
				for (size_t s8=0; s8<D; ++s8)
				{
					Wsum += Warray[s1][s2][s3][s4][s5][s6][s7][s8];
				}
				
				if (Wsum == 0.)
				{
					L[b][0] = YL[b];
				}
				else
				{
					double e = (YL[b]*Reigen).trace();
					solve_linear(GAUGE::L, AquartettL, YL[b], Reigen, Warray, e, L[b][0]);
//					cout << YTimer.info("solve linear L") << endl;
				}
			}
		}
//		#pragma omp section
		{
			for (int a=1; a<dW; ++a)
			{
//				Stopwatch<> YTimer;
//				YR[a] = make_YR(a, H.W[0], H.W[1], H.W[2], H.W[3], R, AquartettR, H.locBasis(0));
				YR[a] = make_YR(WWWWoffDiagRow[a], R, AquartettR, H.locBasis(0));
//				cout << YTimer.info("YR") << endl;
				
				boost::multi_array<Scalar,8> Warray = make_Warray8(a,H);
				
				Scalar Wsum = 0;
				for (size_t s1=0; s1<D; ++s1)
				for (size_t s2=0; s2<D; ++s2)
				for (size_t s3=0; s3<D; ++s3)
				for (size_t s4=0; s4<D; ++s4)
				for (size_t s5=0; s5<D; ++s5)
				for (size_t s6=0; s6<D; ++s6)
				for (size_t s7=0; s7<D; ++s7)
				for (size_t s8=0; s8<D; ++s8)
				{
					Wsum += Warray[s1][s2][s3][s4][s5][s6][s7][s8];
				}
				
				if (Wsum == 0.)
				{
					R[a][0] = YR[a];
				}
				else
				{
					double e = (Leigen*YR[a]).trace();
					solve_linear(GAUGE::R, AquartettR, YR[a], Leigen, Warray, e, R[a][0]);
//					cout << YTimer.info("solve linear R") << endl;
				}
			}
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
	}
	
	vector<Eigenstate<PivotVectorQ<Nq,Scalar> > > gAC(N_sites);
	vector<Eigenstate<PivotVector0Q<Nq,Scalar> > > gC(N_sites);
	
	// dynamic Lanczos error: at least 1e-7, can go down further if err_eigval is smaller, but not below 1e-14
	double tolLanczosEigval = max(min(0.01*err_eigval,1e-7),1e-14); // 1e-7
	double tolLanczosState  = max(min(0.01*err_var,1e-4),   1e-10); // 1e-4
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "current Lanczos tolerances: " << tolLanczosEigval << ", " << tolLanczosState << endl;
	}
	
	vector<PivotMatrixQ<Nq,Scalar,Scalar> > HeffA(N_sites);
	vector<PivotMatrixQ<Nq,Scalar,Scalar> > HeffAC(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffA[l].W  = H.W[l];
		HeffAC[l].W = H.W[l];
	}
	
	HeffA[0].L.push_back(qloc3dummy,L);
	HeffA[N_sites-1].R.push_back(qloc3dummy,R);
	
	for (size_t l=1; l<N_sites; ++l)
	{
		contract_L(HeffA[l-1].L, Vout.state.A[GAUGE::L][l-1], H.W[l-1], Vout.state.A[GAUGE::L][l-1], H.locBasis(l-1), HeffA[l].L);
	}
	
	for (int l=N_sites-2; l>=0; --l)
	{
		contract_R(HeffA[l+1].R, Vout.state.A[GAUGE::R][l+1], H.W[l+1], Vout.state.A[GAUGE::R][l+1], H.locBasis(l+1), HeffA[l].R);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffAC[l].L = HeffA[(l+1)%N_sites].L;
		HeffAC[l].R = HeffA[l].R;
	}
	
	// local optimization
	for (size_t l=0; l<N_sites; ++l)
	{
//		if (HeffA.dim == 0)
		{
			precalc_blockStructure (HeffA[l].L, Vout.state.A[GAUGE::C][l], HeffA[l].W, Vout.state.A[GAUGE::C][l], HeffA[l].R, 
			                        H.locBasis(l), HeffA[l].qlhs, HeffA[l].qrhs);
		}
		
		// reset dim
		HeffA[l].dim = 0;
		for (size_t s=0; s<H.locBasis(l).size(); ++s)
		for (size_t q=0; q<Vout.state.A[GAUGE::C][l][s].dim; ++q)
		{
			HeffA[l].dim += Vout.state.A[GAUGE::C][l][s].block[q].rows() * Vout.state.A[GAUGE::C][l][s].block[q].cols();
		}
		
		gAC[l].state.A = Vout.state.A[GAUGE::C][l];
		
		Stopwatch<> LanczosTimer;
		LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
		Lutz.set_dimK(min(30ul, HeffA[l].dim));
		Lutz.edgeState(HeffA[l],gAC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(AC)=" << setprecision(13) << gAC[l].energy << endl;
		}
		
		gC[l].state.A = Vout.state.C[l];
		
		HeffAC[l].dim = 0;
		for (size_t q=0; q<Vout.state.C[l].dim; ++q)
		{
			HeffAC[l].dim += Vout.state.C[l].block[q].rows() * Vout.state.C[l].block[q].cols();
		}
		
		LanczosSolver<PivotMatrixQ<Nq,Scalar,Scalar>,PivotVector0Q<Nq,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL);
		Lucy.set_dimK(min(30ul, HeffAC[l].dim));
		Lucy.edgeState(HeffAC[l],gC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(C)=" << setprecision(13) << gC[l].energy << endl;
		}
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.A[GAUGE::C][l] = gAC[l].state.A;
		Vout.state.C[l]           = gC[l].state.A;
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		(err_var>0.1)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
	}
	
	eL = (YL[0]*Reigen).trace() / (N_sites * H.width());
	eR = (Leigen*YR[dW-1]).trace() / (N_sites * H.width());
	
	MatrixXd epsLRsq(N_sites,2);
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.calc_epsLRsq(l,epsLRsq(l,0),epsLRsq(l,1));
	}
	err_var = sqrt(epsLRsq.sum());
	
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy(0) << ", " 
		             << Vout.state.entropy(1) << ", " 
		             << Vout.state.entropy(2) << ", " 
		             << Vout.state.entropy(3) << endl;
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
		iteration1(Vout);
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
		if      (N_sites==1) {iteration1(H,Vout);}
		else if (N_sites==2) {iteration2(H,Vout);}
		else if (N_sites==4) {iteration4(H,Vout);}
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
template<typename Atype, typename Wtype>
void VumpsSolver<Nq,MpHamiltonian,Scalar>::
solve_linear (GAUGE::OPTION gauge, const Atype &A, 
               const MatrixType &hLR, const MatrixType &LReigen, 
               const Wtype &Warray, double e, MatrixType &Hres)
{
	vector<size_t> Dvec(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Dvec[l] = D;
	}
	
	TransferMatrix<Nq,Scalar> T(gauge, A, A, LReigen, Warray, Dvec);
	
	MatrixType bvec = hLR;
	bvec -= e * MatrixType::Identity(bvec.rows(),bvec.cols());
	
	GMResSolver<TransferMatrix<Nq,Scalar>,MatrixType> Gimli;
	Gimli.set_dimK(min(100ul,M*M));
	Gimli.solve_linear(T,bvec,Hres);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Gimli.info() << endl;
	}
}

//template<size_t Nq, typename MpHamiltonian, typename Scalar>
//MatrixXd VumpsSolver<Nq,MpHamiltonian,Scalar>::
//eigenvectorL (const MatrixType &TL)
//{
//	EigenSolver<MatrixType> Lutz(TL);
//	int max_index;
//	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
//	
////	cout << "max eigenvalue abs=" << Lutz.eigenvalues()(max_index) << endl;
////	cout << Lutz.eigenvalues().transpose() << endl;
//	
//	MatrixType Mout(M,M);
//	
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	{
//		size_t r = i + M*j;
//		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
//	}
//	
//	return Mout;
//}

//template<size_t Nq, typename MpHamiltonian, typename Scalar>
//MatrixXd VumpsSolver<Nq,MpHamiltonian,Scalar>::
//eigenvectorR (const MatrixType &TR)
//{
//	EigenSolver<MatrixType> Lutz(TR.adjoint());
//	int max_index;
//	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
//	
////	cout << "max eigenvalue abs=" << Lutz.eigenvalues()(max_index) << endl;
////	cout << Lutz.eigenvalues().transpose() << endl;
//	
//	MatrixType Mout(M,M);
//	
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	{
//		size_t r = i + M*j;
//		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
//	}
//	
//	return Mout;
//}

template<size_t Nq, typename MpHamiltonian, typename Scalar>
MatrixXd VumpsSolver<Nq,MpHamiltonian,Scalar>::
linearL (const MatrixType &hL, const MatrixType &TL, const MatrixType &Reigen, double e)
{
	MatrixType RxU(M*M,M*M); RxU.setZero();
	
	// projector |Reigen><1|
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	for (size_t k=0; k<M; ++k)
	{
		size_t r = i + M*j;
		size_t c = k + M*k; // delta(k,l)
		RxU(r,c) = Reigen(i,j);
	}
	
	VectorType bL(M*M);
	
	// reshape hL(i,j)-e*delta(i,j) to vector bL(r)
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		bL(r) = hL(i,j);
		
		if (i==j and e!=0)
		{
			bL(r) -= e;
		}
	}
	
	// solve left linear
//	Stopwatch<> GmresTimer;
	MatrixType LinearL = (MatrixType::Identity(M*M,M*M)-TL+RxU).adjoint();
	GMRES<MatrixType> Leonard(LinearL);
	VectorType xL = Leonard.solve(bL);
	
	MatrixType HL(M,M);
	
	// reshape xL(r) to HL(i,j)
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
linearR (const MatrixType &hR, const MatrixType &TR, const MatrixType &Leigen, double e)
{
	MatrixType UxL(M*M,M*M); UxL.setZero();
	
	// projector |1><Leigen|
	for (size_t i=0; i<M; ++i)
	for (size_t k=0; k<M; ++k)
	for (size_t l=0; l<M; ++l)
	{
		size_t r = i + M*i; // delta(i,j)
		size_t c = k + M*l;
		UxL(r,c) = Leigen(k,l);
	}
	
	VectorType bR(M*M);
	
	// reshape hR(i,j)-e*delta(i,j) to vector bR(r)
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		bR(r) = hR(i,j);
		
		if (i==j and e!=0)
		{
			bR(r) -= e;
		}
	}
	
	// solve right linear
//	Stopwatch<> GmresTimer;
	MatrixType LinearR = MatrixType::Identity(M*M,M*M)-TR+UxL;
	GMRES<MatrixType> Ronald(LinearR);
	VectorType xR = Ronald.solve(bR);
	
	MatrixType HR(M,M);
	
	// reshape xR(r) to HR(i,j)
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		HR(i,j) = xR(r);
	}
	
	return HR;
}

#endif
