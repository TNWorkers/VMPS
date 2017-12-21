#ifndef TDVP_PROPAGATOR
#define TDVP_PROPAGATOR

#include "tensors/DmrgContractions.h"
#include "LanczosPropagator.h" // from HELPERS
#include "pivot/DmrgPivotStuff0.h"
#include "pivot/DmrgPivotStuff2.h"
#include "Stopwatch.h" // from HELPERS

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
class TDVPPropagator
{
public:
	
	TDVPPropagator(){};
	TDVPPropagator (const Hamiltonian &H, VectorType &Vinout);
	
	string info() const;
	double memory (MEMUNIT memunit=GB) const;
	double overhead (MEMUNIT memunit=GB) const;
	
	void t_step  (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, TimeScalar dt, int N_stages=1, double tol_Lanczos=1e-8);
	
	void t_step  (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, int N_stages=1, double tol_Lanczos=1e-8);
	void t_step0 (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, int N_stages=1, double tol_Lanczos=1e-8);
	
private:
	
	vector<PivotMatrixQ<Symmetry,TimeScalar,MpoScalar> >  Heff;
	
	double x (int alg, size_t l, int N_stages);
	
	void set_blocks (const Hamiltonian &H, VectorType &Vinout);
	
	size_t N_sites;
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	void build_L (const Hamiltonian &H, const VectorType &Vinout, size_t loc);
	void build_R (const Hamiltonian &H, const VectorType &Vinout, size_t loc);
	
	double dist_max = 0.;
	double dimK_max = 0.;
	int N_stages_last = 0;
};

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
string TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
info() const
{
	stringstream ss;
	ss << "TDVPPropagator: ";
	ss << "max(dist)=" << dist_max << ", ";
	ss << "max(dimK)=" << dimK_max << ", ";
	ss << "N_stages=" << N_stages_last << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
double TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		res += Heff[l].L.memory(memunit);
//		res += Heff[l].R.memory(memunit);
//		for (size_t s1=0; s1<Heff[l].W.size(); ++s1)
//		for (size_t s2=0; s2<Heff[l].W[s1].size(); ++s2)
//		{
//			res += calc_memory(Heff[l].W[s1][s2],memunit);
//		}
//	}
	return res;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
double TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		res += Heff[l].L.overhead(memunit);
//		res += Heff[l].R.overhead(memunit);
//		res += 2. * calc_memory<size_t>(Heff[l].qlhs.size(),memunit);
//		res += 4. * calc_memory<size_t>(Heff[l].qrhs.size(),memunit);
//	}
	return res;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
TDVPPropagator (const Hamiltonian &H, VectorType &Vinout)
{
	set_blocks(H,Vinout);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
set_blocks (const Hamiltonian &H, VectorType &Vinout)
{
	N_sites = H.length();
	
	// set edges
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Symmetry::Nq>{Vinout.Qtarget(), Vinout.Qtarget(), Symmetry::qvacuum()});
	
//	Heff2.clear();
//	Heff2.resize(N_sites-1);
//	Heff2[0].L = Heff[0].L;
//	Heff2[N_sites-2].R = Heff[N_sites-1].R;
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Heff[l].W = H.W[l];
	}
	
//	for (size_t l=0; l<N_sites-1; ++l)
//	{
//		Heff2[l].W12 = H.W[l];
//		Heff2[l].W34 = H.W[l+1];
//		Heff2[l].qloc12 = H.locBasis(l);
//		Heff2[l].qloc34 = H.locBasis(l+1);
//	}
	
	// initial sweep, left-to-right:
	for (size_t l=N_sites-1; l>0; --l)
	{
		Vinout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
		build_R(H,Vinout,l-1);
	}
	CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	pivot = 0;
	
//	for (size_t l=0; l<N_sites-1; ++l)
//	{
//		Vinout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
//		build_L(H,Vinout,l+1);
//	}
//	CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
//	pivot = N_sites-1;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
double TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
x (int alg, size_t l, int N_stages)
{
	int N_updates = (alg==2)? N_sites-1 : N_sites;
	
	if (N_stages == 1)
	{
		return 0.5;
	}
	else if (N_stages == 2)
	{
		if      (l<N_updates)                      {return 0.25;}
		else if (l>=N_updates and l<2*N_updates)   {return 0.25;}
		else if (l>=2*N_updates and l<3*N_updates) {return 0.25;}
		else if (l>=3*N_updates)                   {return 0.25;}
	}
	else if (N_stages == 3)
	{
		double tripleJump1 = 1./(2.-pow(2.,1./3.));
		double tripleJump2 = 1.-2.*tripleJump1;
		
		if      (l< 2*N_updates)                   {return 0.5*tripleJump1;}
		else if (l>=2*N_updates and l<4*N_updates) {return 0.5*tripleJump2;}
		else if (l>=4*N_updates)                   {return 0.5*tripleJump1;}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t_step (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, int N_stages, double tol_Lanczos)
{
	assert(N_stages==1 or N_stages==2 or N_stages==3 and "Only N_stages=1,2,3 implemented for TDVPPropagator::t_step!");
	dist_max = 0.;
	dimK_max = 0.;
	N_stages_last = N_stages;
	
	for (size_t l=0; l<2*N_stages*(N_sites-1); ++l)
	{
		Stopwatch<> Chronos;
		bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
		size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;
		size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;
		
		// forwards
		PivotVector2Q<Symmetry,TimeScalar> Apair;
		contract_AA(Vinout.A[min(loc1,loc2)], Vinout.locBasis(min(loc1,loc2)), 
		            Vinout.A[max(loc1,loc2)], Vinout.locBasis(max(loc1,loc2)), 
		            Apair.A);
		
		PivotMatrix2Q<Symmetry,TimeScalar,MpoScalar> Heff2;
		Heff2.L = Heff[min(loc1,loc2)].L;
		Heff2.R = Heff[max(loc1,loc2)].R;
		Heff2.W12 = H.W_at(min(loc1,loc2));
		Heff2.W34 = H.W_at(max(loc1,loc2));
		Heff2.qloc12 = H.locBasis(min(loc1,loc2));
		Heff2.qloc34 = H.locBasis(max(loc1,loc2));
		Heff2.qOp12 = H.opBasis(min(loc1,loc2));
		Heff2.qOp34 = H.opBasis(max(loc1,loc2));
		
		// reset dim
		Heff2.dim = 0;
		for (size_t s1=0; s1<H.locBasis(min(loc1,loc2)).size(); ++s1)
		for (size_t s2=0; s2<H.locBasis(max(loc1,loc2)).size(); ++s2)
		for (size_t q=0; q<Apair.A[s1][s2].dim; ++q)
		{
			Heff2.dim += Apair.A[s1][s2].block[q].rows() * Apair.A[s1][s2].block[q].cols();
		}
		
		LanczosPropagator<PivotMatrix2Q<Symmetry,TimeScalar,MpoScalar>,PivotVector2Q<Symmetry,TimeScalar> > Lutz2(tol_Lanczos);
		Lutz2.t_step(Heff2, Apair, -x(2,l,N_stages)*dt.imag()); // 2-site algorithm
		if (Lutz2.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
		if (Lutz2.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
		Vinout.sweepStep2(CURRENT_DIRECTION, min(loc1,loc2), Apair.A);
		(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vinout,max(loc1,loc2)) : build_R(H,Vinout,min(loc1,loc2));
		
		pivot = Vinout.get_pivot();
		
		if ((CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT and pivot != N_sites-1) or
		    (CURRENT_DIRECTION==DMRG::DIRECTION::LEFT and pivot != 0))
		{
	//		if (Heff[pivot].dim == 0)
			{
				precalc_blockStructure (Heff[pivot].L, Vinout.A[pivot], Heff[pivot].W, Vinout.A[pivot], Heff[pivot].R, 
			                            H.locBasis(pivot), H.opBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
			}
			
			// reset dim
			Heff[pivot].dim = 0;
			for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
			for (size_t q=0; q<Vinout.A[pivot][s].dim; ++q)
			{
				Heff[pivot].dim += Vinout.A[pivot][s].block[q].rows() * Vinout.A[pivot][s].block[q].cols();
			}
			
			PivotVectorQ<Symmetry,TimeScalar> Asingle;
			Asingle.A = Vinout.A[pivot];
			
			LanczosPropagator<PivotMatrixQ<Symmetry,TimeScalar,MpoScalar>, PivotVectorQ<Symmetry,TimeScalar> > Lutz(tol_Lanczos);
			Lutz.t_step(Heff[pivot], Asingle, +x(2,l,N_stages)*dt.imag()); // 2-site algorithm
			if (Lutz.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
			if (Lutz.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
			Vinout.A[pivot] = Asingle.A;
		}
	}
	
//	double norm_Psi_t = Vref.squaredNorm();
//	double norm_Psi_dt = Vinout.squaredNorm();
//	double overlap = dot(Vref,Vinout).real();
//	double eta = (norm_Psi_t+norm_Psi_dt-2.*overlap)/pow(dt.imag(),2);
//	
//	double avgH = avg(Vinout,H,Vinout).real();
//	double avgHsq = avg(Vinout,H,Vinout,true).real();
//	
//	cout << "error: " << pow(avgHsq-avgH,2)-pow(eta,2) << "\t" << avgHsq-avgH << "\t" << eta << endl;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t_step0 (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, int N_stages, double tol_Lanczos)
{
	dist_max = 0.;
	dimK_max = 0.;
	N_stages_last = N_stages;
	
//	VectorType Vref = Vinout;
	
	for (size_t l=0; l<2*N_stages*N_sites; ++l)
	{
		bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
		
		// forwards
		PivotVectorQ<Symmetry,TimeScalar> Asingle;
		Asingle.A = Vinout.A[pivot];
		
//		if (Heff[pivot].dim == 0)
		{
			precalc_blockStructure (Heff[pivot].L, Vinout.A[pivot], Heff[pivot].W, Vinout.A[pivot], Heff[pivot].R, 
			                        H.locBasis(pivot), H.opBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
		}
		
		// reset dim
		Heff[pivot].dim = 0;
		for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
		for (size_t q=0; q<Vinout.A[pivot][s].dim; ++q)
		{
			Heff[pivot].dim += Vinout.A[pivot][s].block[q].rows() * Vinout.A[pivot][s].block[q].cols();
		}
		
		LanczosPropagator<PivotMatrixQ<Symmetry,TimeScalar,MpoScalar>, PivotVectorQ<Symmetry,TimeScalar> > Lutz(tol_Lanczos);
		Lutz.t_step(Heff[pivot], Asingle, -x(1,l,N_stages)*dt.imag()); // 1-site algorithm
		if (Lutz.get_dist() > dist_max) {dist_max = Lutz.get_dist();}
		if (Lutz.get_dimK() > dimK_max) {dimK_max = Lutz.get_dimK();}
		Vinout.A[pivot] = Asingle.A;
		
		if ((l+1)%N_sites != 0)
		{
			PivotVector0Q<Symmetry,TimeScalar> Azero;
			int old_pivot = pivot;
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? Vinout.rightSplitStep(pivot,Azero.A) : Vinout.leftSplitStep(pivot,Azero.A);
			pivot = Vinout.get_pivot();
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vinout,pivot) : build_R(H,Vinout,pivot);
			
			LanczosPropagator<PivotMatrixQ<Symmetry,TimeScalar,MpoScalar>, PivotVector0Q<Symmetry,TimeScalar> > Lutz0(tol_Lanczos);
			
			// set Heff0
			PivotMatrixQ<Symmetry,TimeScalar,MpoScalar> Heff0;
			Heff0.L = (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? Heff[old_pivot+1].L : Heff[old_pivot].L;
			Heff0.R = (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? Heff[old_pivot].R   : Heff[old_pivot-1].R;
			Heff0.W = (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? Heff[old_pivot+1].W : Heff[old_pivot-1].W;
			
			// reset dim
			Heff0.dim = 0;
			for (size_t q=0; q<Azero.A.dim; ++q)
			{
				Heff0.dim += Azero.A.block[q].rows() * Azero.A.block[q].cols();
			}
			
			Lutz0.t_step(Heff0, Azero, +x(1,l,N_stages)*dt.imag()); // 1-site algorithm
			if (Lutz0.get_dist() > dist_max) {dist_max = Lutz0.get_dist();}
			if (Lutz0.get_dimK() > dimK_max) {dimK_max = Lutz0.get_dimK();}
			
			Vinout.absorb(pivot,CURRENT_DIRECTION,Azero.A);
		}
	}
	
//	double norm_Psi_t = Vref.squaredNorm();
//	double norm_Psi_dt = Vinout.squaredNorm();
//	double overlap = dot(Vref,Vinout).real();
//	double eta = (norm_Psi_t+norm_Psi_dt-2.*overlap)/dt.imag();
//	
//	double avgH = avg(Vinout,H,Vinout).real();
//	double avgHsq = avg(Vinout,H,Vinout,true).real();
//	
//	cout << "error: " << pow(avgH-avgHsq,2)-pow(eta,2) << endl;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t_step (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, TimeScalar dt, int N_stages, double tol_Lanczos)
{
	Vout = Vin;
	set_blocks(H,Vout);
	t_step(H,Vout,dt,N_stages,tol_Lanczos);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
inline void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
build_L (const Hamiltonian &H, const VectorType &Vinout, size_t loc)
{
	if (loc != 0)
	{
		contract_L(Heff[loc-1].L, Vinout.A[loc-1], H.W[loc-1], Vinout.A[loc-1], H.locBasis(loc-1), H.opBasis(loc-1), Heff[loc].L);
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
inline void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
build_R (const Hamiltonian &H, const VectorType &Vinout, size_t loc)
{
	if (loc != N_sites-1)
	{
		contract_R(Heff[loc+1].R, Vinout.A[loc+1], H.W[loc+1], Vinout.A[loc+1], H.locBasis(loc+1), H.opBasis(loc+1), Heff[loc].R);
	}
}

#endif
