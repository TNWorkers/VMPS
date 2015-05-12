#ifndef VANILLA_MPSCOMPRESSOR
#define VANILLA_MPSCOMPRESSOR

#include "LanczosSolver.h" // for isReal
#include "DmrgContractions.h"
#include "DmrgPivotStuff.h"

template<size_t D, typename Scalar>
class MpsCompressor
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	void varCompress (const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall, size_t Dcutoff_input, double tol=1e-6);
	
	template<typename MpOperator>
	void varCompress (const MpOperator &O, const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall, 
	                  size_t Dcutoff_input, double tol=1e-6);
	
	string info() const;
	
private:
	
	// for |Vsmall> ≈ |Vbig>
	vector<MatrixType> L;
	vector<MatrixType> R;
	void optimizationStep (const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall);
	void sweepStep (const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall);
	void build_L (size_t loc, const Mps<D,Scalar> &Vbra, const Mps<D,Scalar> &Vket);
	void build_R (size_t loc, const Mps<D,Scalar> &Vbra, const Mps<D,Scalar> &Vket);
	
	// for |Vsmall> ≈ O*|Vbig>
	vector<vector<MatrixType> > LW;
	vector<vector<MatrixType> > RW;
	template<typename MpOperator>
	void optimizationStep (Mps<D,Scalar> &Vbra, const MpOperator &O, const Mps<D,Scalar> &Vket);
	template<typename MpOperator>
	void sweepStep (Mps<D,Scalar> &Vsmall, const MpOperator &O, const Mps<D,Scalar> &Vbig);
	template<typename MpOperator>
	void build_LW (size_t loc, const Mps<D,Scalar> &Vbra, const MpOperator &O, const Mps<D,Scalar> &Vket);
	template<typename MpOperator>
	void build_RW (size_t loc, const Mps<D,Scalar> &Vbra, const MpOperator &O, const Mps<D,Scalar> &Vket);
	
	size_t N_sites;
	size_t N_sweepsteps, N_halfsweeps;
	double sqnormVbig, err;
	size_t Dcutoff;
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	double normdiff;
	double calc_normdiff (const Mps<D,Scalar> &Vsmall); // = |Vbig-Vsmall|
};

template<size_t D, typename Scalar>
string MpsCompressor<D,Scalar>::
info() const
{
	stringstream ss;
	ss << "MpsCompressor: Dcutoff=" << Dcutoff << ", |Vbig-Vsmall|=" << fabs(normdiff) << ", N_halfsweeps=" << N_halfsweeps;
	return ss.str();
}

template<size_t D, typename Scalar>
void MpsCompressor<D,Scalar>::
varCompress (const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall, size_t Dcutoff_input, double tol)
{
	N_sites = Vbig.length();
	sqnormVbig = dot(Vbig,Vbig);
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_input;
	
	// set L&R edges
	L.resize(N_sites);
	R.resize(N_sites);
	L[0].resize(1,1);
	L[0](0,0) = 1.;
	R[N_sites-1].resize(1,1);
	R[N_sites](0,0) = 1.;
	
	assert(Vbig.pivot == 0 or 
	       Vbig.pivot == N_sites-1 and 
	       "Please sweep to an edge first!");
	Vsmall = Vbig;
	Vsmall.N_sv = Dcutoff;
	
	if (Vbig.pivot == 0)
	{
		Vsmall.skim(DMRG::DIRECTION::RIGHT, DMRG::BROOM::BRUTAL_SVD);
		for (size_t l=1; l<N_sites; ++l)
		{
			build_L(l,Vsmall,Vbig);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	else if (Vbig.pivot == N_sites-1)
	{
		Vsmall.skim(DMRG::DIRECTION::LEFT, DMRG::BROOM::BRUTAL_SVD);
		for (int l=N_sites-2; l>=0; --l)
		{
			build_R(l,Vsmall,Vbig);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	pivot = Vsmall.pivot;
	
	normdiff = calc_normdiff(Vsmall);
//	double normdiff_change = 1.;
	size_t halfSweepRange = N_sites;
	
//	while (fabs(normdiff_change) >= tol)
	while (normdiff >= tol)
	{
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			optimizationStep(Vbig,Vsmall);
			if (j != halfSweepRange)
			{
				sweepStep(Vbig,Vsmall);
			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
//		double normdiff_new = calc_normdiff(Vsmall);
//		normdiff_change = normdiff - normdiff_new;
//		normdiff = normdiff_new;
		normdiff = calc_normdiff(Vsmall);
		
//		if (fabs(normdiff_change) >= tol)
		if (normdiff >= tol)
		{
			sweepStep(Vbig,Vsmall);
		}
	}
}

template<size_t D, typename Scalar>
void MpsCompressor<D,Scalar>::
sweepStep (const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall)
{
	Vsmall.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(++pivot,Vsmall,Vbig) : build_R(--pivot,Vsmall,Vbig);
}

template<size_t D, typename Scalar>
double MpsCompressor<D,Scalar>::
calc_normdiff (const Mps<D,Scalar> &Vsmall)
{
	double sqnormVsmall = 0.;
	for (size_t s=0; s<D; ++s)
	{
		sqnormVsmall += Vsmall.A[pivot][s].colwise().squaredNorm().sum();
	}
	return sqrt(fabs(sqnormVbig-sqnormVsmall));
	// test with:
//	Mps<D,Scalar> Vtmp = Vbig;
//	Vtmp -= Vsmall;
//	sqrt(dot(Vtmp,Vtmp))
}

template<size_t D, typename Scalar>
void MpsCompressor<D,Scalar>::
optimizationStep (const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall)
{
	for (size_t s=0; s<D; ++s)
	{
		Vsmall.A[pivot][s] = L[pivot] * Vbig.A[pivot][s] * R[pivot];
	}
}

template<size_t D, typename Scalar>
void MpsCompressor<D,Scalar>::
build_L (size_t loc, const Mps<D,Scalar> &Vbra, const Mps<D,Scalar> &Vket)
{
	L[loc] = Vbra.A[loc-1][0].adjoint() * L[loc-1] * Vket.A[loc-1][0];
	for (size_t s=1; s<D; ++s)
	{
		L[loc] += Vbra.A[loc-1][s].adjoint() * L[loc-1] * Vket.A[loc-1][s];
	}
}

template<size_t D, typename Scalar>
void MpsCompressor<D,Scalar>::
build_R (size_t loc, const Mps<D,Scalar> &Vbra, const Mps<D,Scalar> &Vket)
{
	R[loc] = Vket.A[loc+1][0] * R[loc+1] * Vbra.A[loc+1][0].adjoint();
	for (size_t s=1; s<D; ++s)
	{
		R[loc] += Vket.A[loc+1][s] * R[loc+1] * Vbra.A[loc+1][s].adjoint();
	}
}

//---------------------------compression of H*|Psi>---------------------------
// |Vsmall> ≈ O|Vbig>
// convention in program: <Vsmall|O|Vbig>

template<size_t D, typename Scalar>
template<typename MpOperator>
void MpsCompressor<D,Scalar>::
varCompress (const MpOperator &O, const Mps<D,Scalar> &Vbig, Mps<D,Scalar> &Vsmall, size_t Dcutoff_input, double tol)
{
	N_sites = Vbig.length();
	Stopwatch Chronos;
	sqnormVbig = (O.check_SQUARE()==true)? isReal(avg(Vbig,O,Vbig,true)) : isReal(avg(Vbig,O,O,Vbig));
//	sqnormVbig = isReal(avg(Vbig,O,O,Vbig));
	Chronos.check("<HH>");
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_input;
	
	Vsmall = Mps<D,Scalar>(O.length(), Dcutoff);
	Vsmall.setRandom();
	
	// prepare edges of LW & RW
	LW.resize(N_sites);
	RW.resize(N_sites);
	MatrixXd Mtmp(1,1); Mtmp << 1.;
	
	LW[0].resize(1);
	LW[0][0] = Mtmp;
	
	RW[N_sites-1].resize(1);
	RW[N_sites-1][0] = Mtmp;
	
	for (size_t l=1; l<N_sites; ++l)
	{
		LW[l].resize(O.auxdim());
		RW[l-1].resize(O.auxdim());
	}
	
	// left-to-right
//	for (size_t l=N_sites-1; l>0; --l)
//	{
//		Vsmall.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
//		build_RW(l-1,Vsmall,O,Vbig);
//	}
//	Vsmall.leftSweepStep(0, DMRG::BROOM::QR); // last sweep to get rid of large numbers
//	CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	
	// right-to-left
	for (size_t l=0; l<N_sites-1; ++l)
	{
		Vsmall.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
		build_LW(l+1,Vsmall,O,Vbig);
	}
	Vsmall.rightSweepStep(N_sites-1, DMRG::BROOM::QR); // last sweep to get rid of large numbers
	CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	
	pivot = Vsmall.pivot;
	normdiff = calc_normdiff(Vsmall);
//	double normdiff_change = 1.;
	size_t halfSweepRange = N_sites;
	
	Chronos.check("preparation");
//	while (fabs(normdiff_change) >= tol)
	while (fabs(normdiff) >= tol)
	{
		Stopwatch Chronos2;
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			optimizationStep(Vsmall,O,Vbig);
//			Chronos.check(make_string("opt.step l=",pivot));
			if (j != halfSweepRange)
			{
				sweepStep(Vsmall,O,Vbig);
			}
			++N_sweepsteps;
		}
		
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
//		double normdiff_new = calc_normdiff(Vsmall);
//		normdiff_change = normdiff-normdiff_new;
//		normdiff = normdiff_new;
		normdiff = calc_normdiff(Vsmall);
		
		bool RESIZED = false;
		if (N_halfsweeps%2 == 0 and 
		    N_halfsweeps > 0 and 
		    fabs(normdiff) >= tol)
		{
			size_t D_cutoff_new = Vsmall.calc_Dmax()+2;
			cout << "new cutoff=" << D_cutoff_new << endl;
			Vsmall.innerResize(D_cutoff_new, true);
			for (size_t l=0; l<N_sites-1; ++l)
			{
				Vsmall.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
				build_LW(l+1,Vsmall,O,Vbig);
			}
			CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
			pivot = Vsmall.pivot;
			RESIZED = true;
			size_t halfSweepRange = N_sites;
		}
		
//		if (fabs(normdiff_change) >= tol)
		if (fabs(normdiff) >= tol and
		    RESIZED == false)
		{
			sweepStep(Vsmall,O,Vbig);
		}
		Chronos2.check("half-sweep");
	}
}

template<size_t D, typename Scalar>
template<typename MpOperator>
void MpsCompressor<D,Scalar>::
sweepStep (Mps<D,Scalar> &Vsmall, const MpOperator &O, const Mps<D,Scalar> &Vbig)
{
	Vsmall.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
	(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(++pivot,Vsmall,O,Vbig) : build_RW(--pivot,Vsmall,O,Vbig);
}

template<size_t D, typename Scalar>
template<typename MpOperator>
void MpsCompressor<D,Scalar>::
optimizationStep (Mps<D,Scalar> &Vbra, const MpOperator &O, const Mps<D,Scalar> &Vket)
{
	for (size_t s=0; s<D; ++s) {Vbra.A[pivot][s].setZero();}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<O.W[pivot][s1][s2].outerSize(); ++k)
	for (SparseMatrixXd::InnerIterator iW(O.W[pivot][s1][s2],k); iW; ++iW)
	{
		if (LW[pivot][iW.row()].rows() != 0 and 
		    RW[pivot][iW.col()].rows() != 0)
		{
			Vbra.A[pivot][s1].noalias() += iW.value() * (LW[pivot][iW.row()] * Vket.A[pivot][s2] * RW[pivot][iW.col()]);
		}
	}
}

template<size_t D, typename Scalar>
template<typename MpOperator>
void MpsCompressor<D,Scalar>::
build_LW (size_t loc, const Mps<D,Scalar> &Vbra, const MpOperator &O, const Mps<D,Scalar> &Vket)
{
	contract_L(LW[loc-1], Vbra.A[loc-1], O.W[loc-1], Vket.A[loc-1], LW[loc]);
}

template<size_t D, typename Scalar>
template<typename MpOperator>
void MpsCompressor<D,Scalar>::
build_RW (size_t loc, const Mps<D,Scalar> &Vbra, const MpOperator &O, const Mps<D,Scalar> &Vket)
{
	contract_R(RW[loc+1], Vbra.A[loc+1], O.W[loc+1], Vket.A[loc+1], RW[loc]);
}

#endif
