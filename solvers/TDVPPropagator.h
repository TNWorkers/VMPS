#ifndef TDVP_PROPAGATOR
#define TDVP_PROPAGATOR

#include "Stopwatch.h" // from TOOLS
#include "LanczosPropagator.h" // from ALGS

#include "pivot/DmrgPivotMatrix0.h"
#include "pivot/DmrgPivotMatrix2.h"
//include "tensors/DmrgContractions.h"

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
	
	void t_step_adaptive (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, const vector<bool> &TWO_STEP_AT, int N_stages=1, double tol_Lanczos=1e-8);
	
	inline VectorXd get_deltaE() const {return deltaE;};
	inline double get_t_tot() const {return t_tot;};
	inline vector<size_t> get_dimK2_log() const {return dimK2_log;};
	inline vector<size_t> get_dimK1_log() const {return dimK1_log;};
	inline vector<size_t> get_dimK0_log() const {return dimK0_log;};
	
private:
	
	void t_step_pivot  (double x, const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, double tol_Lanczos=1e-8);
	void t0_step_pivot (bool BACK, double x, const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, double tol_Lanczos=1e-8, bool TURN_FIRST=true);
	
	void test_edge_eigenvector (const PivotVector<Symmetry,TimeScalar> &Asingle);
	VectorXd deltaE;
	VectorXd dimKlog;
	
	vector<PivotMatrix1<Symmetry,TimeScalar,MpoScalar> >  Heff;
	PivotMatrix1<Symmetry,TimeScalar,MpoScalar> HeffLast;
	PivotMatrix1<Symmetry,TimeScalar,MpoScalar> HeffFrst;
	
	double x (int alg, size_t l, int N_stages);
	
	void set_blocks (const Hamiltonian &H, VectorType &Vinout);
	
	size_t N_sites;
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	void build_L (const Hamiltonian &H, const VectorType &Vinout, int loc);
	void build_R (const Hamiltonian &H, const VectorType &Vinout, int loc);
	
	double dist_max = 0.;
	double dimK_max = 0.;
	vector<size_t> dimK0_log;
	vector<size_t> dimK1_log;
	vector<size_t> dimK2_log;
	int N_stages_last = 0;
	
	double t_0site = 0;
	double t_1site = 0;
	double t_2site = 0;
	double t_ohead = 0; // precalc
	double t_contr = 0; // contract & sweep
	double t_tot   = 0; // full time step
};

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
string TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
info() const
{
	stringstream ss;
	ss << "TDVPPropagator: ";
	ss << "max(dist)=" << dist_max << ", ";
//	ss << "max(dimK)=" << dimK_max << ", ";
	if (dimK2_log.size() > 0)
	{
		ss << "dimK2=" << *min_element(dimK2_log.begin(), dimK2_log.end()) << "~" << *max_element(dimK2_log.begin(), dimK2_log.end()) << ", ";
	}
	if (dimK1_log.size() > 0)
	{
		ss << "dimK1=" << *min_element(dimK1_log.begin(), dimK1_log.end()) << "~" << *max_element(dimK1_log.begin(), dimK1_log.end()) << ", ";
	}
	if (dimK0_log.size() > 0)
	{
		ss << "dimK0=" << *min_element(dimK0_log.begin(), dimK0_log.end()) << "~" << *max_element(dimK0_log.begin(), dimK0_log.end()) << ", ";
	}
//	ss << "N_stages=" << N_stages_last << ", ";
//	ss << "mem=" << round(memory(GB),3) << "GB, ";
	ss << "δE@edge: L=" << deltaE(0) << ", R=" << deltaE(deltaE.rows()-1) << ", ";
	ss << "overhead=" << round(overhead(MB),3) << "MB, ";
	ss << "t[s]=" << t_tot << ", "
	   << "t0=" << round(t_0site/t_tot*100.,0) << "%, "
	   << "t1=" << round(t_1site/t_tot*100.) << "%, "
	   << "t2=" << round(t_2site/t_tot*100.) << "%, "
	   << "t_ohead=" << round(t_ohead/t_tot*100.) << "%, "
	   << "t_contr=" << round(t_contr/t_tot*100.) << "%";
	
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
	assert(H.HAS_TWO_SITE_DATA() and "You need to call H.precalc_TwoSiteData() before dynamics!");
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
set_blocks (const Hamiltonian &H, VectorType &Vinout)
{
	N_sites = H.length();
	
	// set edges
	Heff.clear();
	Heff.resize(N_sites);
//	Heff[0].L.setVacuum();
//	Heff[N_sites-1].R.setTarget(qarray3<Symmetry::Nq>{Vinout.Qtarget(), Vinout.Qtarget(), Symmetry::qvacuum()});
	Heff[0].L         = Vinout.get_boundaryTensor(DMRG::DIRECTION::LEFT);
	Heff[N_sites-1].R = Vinout.get_boundaryTensor(DMRG::DIRECTION::RIGHT);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Heff[l].W = H.W[l];
	}
	
	// initial sweep, right-to-left:
	for (size_t l=N_sites-1; l>0; --l)
	{
		Vinout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
		build_R(H,Vinout,l-1);
	}
	CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	pivot = 0;
	
	deltaE.resize(N_sites);
	
	// initial sweep, left-to-right:
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
	double out;
	int N_updates = (alg==2)? N_sites-1 : N_sites;
	
	if (N_stages == 1)
	{
		out = 0.5;
	}
	else if (N_stages == 2)
	{
		if      (l<N_updates)                      {out = 0.25;}
		else if (l>=N_updates and l<2*N_updates)   {out = 0.25;}
		else if (l>=2*N_updates and l<3*N_updates) {out = 0.25;}
		else if (l>=3*N_updates)                   {out = 0.25;}
	}
	else if (N_stages == 3)
	{
		double tripleJump1 = 1./(2.-pow(2.,1./3.));
		double tripleJump2 = 1.-2.*tripleJump1;
		
		if      (l< 2*N_updates)                   {out = 0.5*tripleJump1;}
		else if (l>=2*N_updates and l<4*N_updates) {out = 0.5*tripleJump2;}
		else if (l>=4*N_updates)                   {out = 0.5*tripleJump1;}
	}
	return out;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t_step (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, int N_stages, double tol_Lanczos)
{
	assert(N_stages==1 or N_stages==2 or N_stages==3 and "Only N_stages=1,2,3 implemented for TDVPPropagator::t_step!");
	dist_max = 0.;
	dimK_max = 0;
	N_stages_last = N_stages;
	
	t_0site = 0;
	t_1site = 0;
	t_2site = 0;
	t_ohead = 0;
	t_contr = 0;
	t_tot   = 0;
	
	Stopwatch<> Wtot;
	
	dimK2_log.clear();
	dimK1_log.clear();
	dimK0_log.clear();
	
	for (size_t l=0; l<2*N_stages*(N_sites-1); ++l)
	{
		Stopwatch<> Chronos;
		turnaround(pivot, N_sites, CURRENT_DIRECTION);
		
		// 2-site propagation
		size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;
		size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;
//		cout << "2site between: " << loc1 << "," << loc2 << ", pivot=" << pivot << endl;
		
		Stopwatch<> Wc;
		PivotVector<Symmetry,TimeScalar> Apair(Vinout.A[loc1], Vinout.locBasis(loc1),
		                                       Vinout.A[loc2], Vinout.locBasis(loc2),
		                                       Vinout.QoutTop[loc1], Vinout.QoutBot[loc1]);
		t_contr += Wc.time(SECONDS);
		PivotMatrix2<Symmetry,TimeScalar,MpoScalar> Heff2(Heff[loc1].L, Heff[loc2].R, 
		                                                  H.W_at(loc1), H.W_at(loc2), 
		                                                  H.locBasis(loc1), H.locBasis(loc2), 
		                                                  H.opBasis(loc1), H.opBasis(loc2));
		
		Stopwatch<> Woh2;
		precalc_blockStructure (Heff[loc1].L, Apair.data, Heff2.W12, Heff2.W34, Apair.data, Heff[loc2].R, 
		                        H.locBasis(loc1), H.locBasis(loc2), H.opBasis(loc1), H.opBasis(loc2), 
		                        H.TSD[loc1], 
		                        Heff2.qlhs, Heff2.qrhs, Heff2.factor_cgcs);
		t_ohead += Woh2.time(SECONDS);
		
//		LanczosPropagator<PivotMatrix2<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar> > Lutz2(tol_Lanczos);
		LanczosPropagator<PivotMatrix2<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz2(tol_Lanczos);
		Stopwatch<> W2;
//		Lutz2.t_step(Heff2, Apair, -x(2,l,N_stages)*dt.imag()); // 2-site algorithm
		Lutz2.t_step(Heff2, Apair, x(2,l,N_stages)*dt); // 2-site algorithm
//		cout << Lutz2.info() << endl;
		t_2site += W2.time(SECONDS);
		
		dimK2_log.push_back(Lutz2.get_dimK());
		if (Lutz2.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
		if (Lutz2.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
		
		Stopwatch<> Ws;
		Vinout.sweepStep2(CURRENT_DIRECTION, loc1, Apair.data);
		t_contr += Ws.time(SECONDS);
		(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vinout,loc2) : build_R(H,Vinout,loc1);
		pivot = Vinout.get_pivot();
		
		// 1-site propagation
		if ((CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT and pivot != N_sites-1) or
		    (CURRENT_DIRECTION==DMRG::DIRECTION::LEFT and pivot != 0))
		{
			Stopwatch<> Woh1;
			precalc_blockStructure (Heff[pivot].L, Vinout.A[pivot], Heff[pivot].W, Vinout.A[pivot], Heff[pivot].R, 
			                        H.locBasis(pivot), H.opBasis(pivot), 
			                        Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
			t_ohead += Woh1.time(SECONDS);
			
			PivotVector<Symmetry,TimeScalar> Asingle(Vinout.A[pivot]);
			
//			lout << "1site at: " << pivot << endl;
//			LanczosPropagator<PivotMatrix1<Symmetry,TimeScalar,MpoScalar>, PivotVector<Symmetry,TimeScalar> > Lutz(tol_Lanczos);
			LanczosPropagator<PivotMatrix1<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz(tol_Lanczos);
			Stopwatch<> W1;
//			Lutz.t_step(Heff[pivot], Asingle, +x(2,l,N_stages)*dt.imag()); // 1-site algorithm
			Lutz.t_step(Heff[pivot], Asingle, -x(2,l,N_stages)*dt); // 1-site algorithm
//			cout << Lutz.info() << endl;
			t_1site += W1.time(SECONDS);
			
			dimK1_log.push_back(Lutz2.get_dimK());
			if (Lutz.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
			if (Lutz.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
			
			Vinout.A[pivot] = Asingle.data;
		}
	}
	
//	lout << "final pivot=" << pivot << endl;
	
	t_tot = Wtot.time(SECONDS);
	
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

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
//void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
//t_step3 (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, int N_stages, double tol_Lanczos)
//{
//	assert(N_stages==1 or N_stages==2 or N_stages==3 and "Only N_stages=1,2,3 implemented for TDVPPropagator::t_step!");
//	dist_max = 0.;
//	dimK_max = 0;
//	N_stages_last = N_stages;
//	
//	t_0site = 0;
//	t_1site = 0;
//	t_2site = 0;
//	t_ohead = 0;
//	t_contr = 0;
//	t_tot   = 0;
//	
//	Stopwatch<> Wtot;
//	
//	dimK2_log.clear();
//	dimK1_log.clear();
//	dimK0_log.clear();
//	
//	for (size_t l=0; l<2*N_stages*(N_sites-2); ++l)
//	{
//		Stopwatch<> Chronos;
//		turnaround(pivot, N_sites, CURRENT_DIRECTION);
//		
//		// 3-site propagation
//		size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-2;
//		size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot-1;
//		size_t loc3 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+2 : pivot;
//		
//		Stopwatch<> Wc;
//		PivotVector<Symmetry,TimeScalar> Apair(Vinout.A[loc1], Vinout.locBasis(loc1),
//		                                       Vinout.A[loc2], Vinout.locBasis(loc2),
//		                                       Vinout.QoutTop[loc1], Vinout.QoutBot[loc1]);
//		t_contr += Wc.time(SECONDS);
//		PivotMatrix2<Symmetry,TimeScalar,MpoScalar> Heff2(Heff[loc1].L, Heff[loc2].R, 
//		                                                  H.W_at(loc1), H.W_at(loc2), 
//		                                                  H.locBasis(loc1), H.locBasis(loc2), 
//		                                                  H.opBasis(loc1), H.opBasis(loc2));
//		
//		Stopwatch<> Woh2;
//		precalc_blockStructure (Heff[loc1].L, Apair.data, Heff2.W12, Heff2.W34, Apair.data, Heff[loc2].R, 
//		                        H.locBasis(loc1), H.locBasis(loc2), H.opBasis(loc1), H.opBasis(loc2), 
//		                        H.TSD[loc1], 
//		                        Heff2.qlhs, Heff2.qrhs, Heff2.factor_cgcs);
//		t_ohead += Woh2.time(SECONDS);
//		
//		LanczosPropagator<PivotMatrix2<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz2(tol_Lanczos);
//		Stopwatch<> W2;
//		Lutz2.t_step(Heff2, Apair, x(2,l,N_stages)*dt); // 2-site algorithm
//		t_2site += W2.time(SECONDS);
//		
//		dimK2_log.push_back(Lutz2.get_dimK());
//		if (Lutz2.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
//		if (Lutz2.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
//		
//		Stopwatch<> Ws;
//		Vinout.sweepStep2(CURRENT_DIRECTION, loc1, Apair.data);
//		t_contr += Ws.time(SECONDS);
//		(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vinout,loc2) : build_R(H,Vinout,loc1);
//		pivot = Vinout.get_pivot();
//		
//		// 1-site propagation
//		if ((CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT and pivot != N_sites-1) or
//		    (CURRENT_DIRECTION==DMRG::DIRECTION::LEFT and pivot != 0))
//		{
//			Stopwatch<> Woh1;
//			precalc_blockStructure (Heff[pivot].L, Vinout.A[pivot], Heff[pivot].W, Vinout.A[pivot], Heff[pivot].R, 
//			                        H.locBasis(pivot), H.opBasis(pivot), 
//			                        Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
//			t_ohead += Woh1.time(SECONDS);
//			
//			PivotVector<Symmetry,TimeScalar> Asingle(Vinout.A[pivot]);
//			
//			LanczosPropagator<PivotMatrix1<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz(tol_Lanczos);
//			Stopwatch<> W1;
//			Lutz.t_step(Heff[pivot], Asingle, -x(2,l,N_stages)*dt); // 1-site algorithm
//			t_1site += W1.time(SECONDS);
//			
//			dimK1_log.push_back(Lutz2.get_dimK());
//			if (Lutz.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
//			if (Lutz.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
//			
//			Vinout.A[pivot] = Asingle.data;
//		}
//	}
//	
//	t_tot = Wtot.time(SECONDS);
//}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t_step0 (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, int N_stages, double tol_Lanczos)
{
	dist_max = 0.;
	dimK_max = 0.;
	N_stages_last = N_stages;
	
	t_0site = 0;
	t_1site = 0;
	t_2site = 0;
	t_ohead = 0;
	t_contr = 0;
	t_tot = 0;
	
	Stopwatch<> Wtot;
	
	dimK2_log.clear();
	dimK1_log.clear();
	dimK0_log.clear();
	
//	VectorType Vref = Vinout;
	
	for (size_t l=0; l<2*N_stages*N_sites; ++l)
	{
		turnaround(pivot, N_sites, CURRENT_DIRECTION);
		
		// 1-site propagation
		PivotVector<Symmetry,TimeScalar> Asingle(Vinout.A[pivot]);
		Stopwatch<> Woh1;
		precalc_blockStructure (Heff[pivot].L, Vinout.A[pivot], Heff[pivot].W, Vinout.A[pivot], Heff[pivot].R, 
		                        H.locBasis(pivot), H.opBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
		t_ohead += Woh1.time(SECONDS);
		
//		lout << "1site at: " << pivot << endl;
		LanczosPropagator<PivotMatrix1<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz(tol_Lanczos);
		
		Stopwatch<> W1;
//		Lutz.t_step(Heff[pivot], Asingle, -x(1,l,N_stages)*dt.imag()); // 1-site algorithm
		Lutz.t_step(Heff[pivot], Asingle, x(1,l,N_stages)*dt); // 1-site algorithm
		t_1site += W1.time(SECONDS);
		
		dimK1_log.push_back(Lutz.get_dimK());
		if (Lutz.get_dist() > dist_max) {dist_max = Lutz.get_dist();}
		if (Lutz.get_dimK() > dimK_max) {dimK_max = Lutz.get_dimK();}
		Vinout.A[pivot] = Asingle.data;
		
		// 0-site propagation
		if ((l+1)%N_sites != 0)
		{
			PivotVector<Symmetry,TimeScalar> Azero;
			int old_pivot = pivot;
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? Vinout.rightSplitStep(pivot,Azero.data[0]) : Vinout.leftSplitStep(pivot,Azero.data[0]);
			pivot = Vinout.get_pivot();
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vinout,pivot) : build_R(H,Vinout,pivot);
			
//			if (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)
//			{
//				lout << "0site between " << old_pivot << "," << old_pivot+1 << endl;
//			}
//			else
//			{
//				lout << "0site between " << old_pivot-1 << "," << old_pivot << endl;
//			}
			LanczosPropagator<PivotMatrix0<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz0(tol_Lanczos);
			
			PivotMatrix0<Symmetry,TimeScalar,MpoScalar> Heff0;
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)?
			Heff0 = PivotMatrix0<Symmetry,TimeScalar,MpoScalar>(Heff[old_pivot+1].L, Heff[old_pivot].R):
			Heff0 = PivotMatrix0<Symmetry,TimeScalar,MpoScalar>(Heff[old_pivot].L, Heff[old_pivot-1].R);
			
			Stopwatch<> W0;
//			Lutz0.t_step(Heff0, Azero, +x(1,l,N_stages)*dt.imag()); // 0-site algorithm
			Lutz0.t_step(Heff0, Azero, -x(1,l,N_stages)*dt); // 0-site algorithm
			t_0site += W0.time(SECONDS);
			
			dimK0_log.push_back(Lutz0.get_dimK());
			if (Lutz0.get_dist() > dist_max) {dist_max = Lutz0.get_dist();}
			if (Lutz0.get_dimK() > dimK_max) {dimK_max = Lutz0.get_dimK();}
			
			Vinout.absorb(pivot, CURRENT_DIRECTION, Azero.data[0]);
		}
	}
	
//	lout << "final pivot=" << pivot << endl;
	
	t_tot = Wtot.time(SECONDS);
	
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
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t_step_pivot (double x, const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, double tol_Lanczos)
{
	turnaround(pivot, N_sites, CURRENT_DIRECTION);
	
	// 2-site propagation
	size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;
	size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;
//	cout << "2site between: " << loc1 << "," << loc2 << ", pivot=" << pivot << endl;
	
	Stopwatch<> Wc;
	PivotVector<Symmetry,TimeScalar> Apair(Vinout.A[loc1], Vinout.locBasis(loc1),
	                                       Vinout.A[loc2], Vinout.locBasis(loc2),
	                                       Vinout.QoutTop[loc1], Vinout.QoutBot[loc1]);
	t_contr += Wc.time(SECONDS);
	PivotMatrix2<Symmetry,TimeScalar,MpoScalar> Heff2(Heff[loc1].L, Heff[loc2].R, 
	                                                  H.W_at(loc1), H.W_at(loc2), 
	                                                  H.locBasis(loc1), H.locBasis(loc2), 
	                                                  H.opBasis(loc1), H.opBasis(loc2));
	
	Stopwatch<> Woh2;
	precalc_blockStructure (Heff[loc1].L, Apair.data, Heff2.W12, Heff2.W34, Apair.data, Heff[loc2].R, 
	                        H.locBasis(loc1), H.locBasis(loc2), H.opBasis(loc1), H.opBasis(loc2), 
	                        H.TSD[loc1], 
	                        Heff2.qlhs, Heff2.qrhs, Heff2.factor_cgcs);
	t_ohead += Woh2.time(SECONDS);
	
	LanczosPropagator<PivotMatrix2<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz2(tol_Lanczos);
	Stopwatch<> W2;
//	Lutz2.t_step(Heff2, Apair, -x*dt.imag()); // 2-site algorithm
	Lutz2.t_step(Heff2, Apair, x*dt); // 2-site algorithm
//	cout << Lutz2.info() << endl;
	t_2site += W2.time(SECONDS);
	
	dimK2_log.push_back(Lutz2.get_dimK());
	if (Lutz2.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
	if (Lutz2.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
	
	Stopwatch<> Ws;
	Vinout.sweepStep2(CURRENT_DIRECTION, loc1, Apair.data);
	t_contr += Ws.time(SECONDS);
	
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vinout,loc2) : build_R(H,Vinout,loc1);
	pivot = Vinout.get_pivot();
	
	// 1-site propagation
	if ((CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT and pivot != N_sites-1) or
	    (CURRENT_DIRECTION==DMRG::DIRECTION::LEFT and pivot != 0))
	{
		Stopwatch<> Woh1;
		precalc_blockStructure (Heff[pivot].L, Vinout.A[pivot], Heff[pivot].W, Vinout.A[pivot], Heff[pivot].R, 
		                        H.locBasis(pivot), H.opBasis(pivot), 
		                        Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
		t_ohead += Woh1.time(SECONDS);
		
		PivotVector<Symmetry,TimeScalar> Asingle(Vinout.A[pivot]);
//		test_edge_eigenvector(Asingle);
		
//		cout << "1site at: " << pivot << endl;
		LanczosPropagator<PivotMatrix1<Symmetry,TimeScalar,MpoScalar>, PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz(tol_Lanczos);
		Stopwatch<> W1;
//		Lutz.t_step(Heff[pivot], Asingle, +x*dt.imag()); // 1-site algorithm
		Lutz.t_step(Heff[pivot], Asingle, -x*dt); // 1-site algorithm
		deltaE(pivot) = Lutz.get_deltaE();
//		cout << Lutz.info() << endl;
		t_1site += W1.time(SECONDS);
		
		dimK1_log.push_back(Lutz.get_dimK());
		if (Lutz.get_dist() > dist_max) {dist_max = Lutz2.get_dist();}
		if (Lutz.get_dimK() > dimK_max) {dimK_max = Lutz2.get_dimK();}
		
		Vinout.A[pivot] = Asingle.data;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t0_step_pivot (bool BACK, double x, const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, double tol_Lanczos, bool TURN_FIRST)
{
	if (TURN_FIRST) turnaround(pivot, N_sites, CURRENT_DIRECTION);
	
	// 1-site propagation
	PivotVector<Symmetry,TimeScalar> Asingle(Vinout.A[pivot]);
	Stopwatch<> Woh1;
	precalc_blockStructure (Heff[pivot].L, Vinout.A[pivot], Heff[pivot].W, Vinout.A[pivot], Heff[pivot].R, 
	                        H.locBasis(pivot), H.opBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
	t_ohead += Woh1.time(SECONDS);
	
//	test_edge_eigenvector(Asingle);
	
//	cout << "1site at: " << pivot << endl;
	LanczosPropagator<PivotMatrix1<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz(tol_Lanczos);
	
	Stopwatch<> W1;
//	Lutz.t_step(Heff[pivot], Asingle, -x*dt.imag()); // 1-site algorithm
	Lutz.t_step(Heff[pivot], Asingle, x*dt); // 1-site algorithm
	deltaE(pivot) = Lutz.get_deltaE();
	t_1site += W1.time(SECONDS);
	
	dimK1_log.push_back(Lutz.get_dimK());
	if (Lutz.get_dist() > dist_max) {dist_max = Lutz.get_dist();}
	if (Lutz.get_dimK() > dimK_max) {dimK_max = Lutz.get_dimK();}
	Vinout.A[pivot] = Asingle.data;
	
	// 0-site propagation
	if (BACK)
	{
		PivotVector<Symmetry,TimeScalar> Azero;
		int old_pivot = pivot;
		if (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)
		{
//			cout << "split to RIGHT at pivot=" << pivot << endl;
			Vinout.rightSplitStep(pivot,Azero.data[0]);
		}
		else
		{
//			cout << "split to LEFT at pivot=" << pivot << endl;
			Vinout.leftSplitStep(pivot,Azero.data[0]);
		}
		pivot = Vinout.get_pivot();
		if (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)
		{
			if (old_pivot+1 == N_sites)
			{
				build_L(H,Vinout,N_sites);
			}
			else
			{
				build_L(H,Vinout,pivot);
			}
		}
		else
		{
			if (old_pivot-1 == -1)
			{
				build_R(H,Vinout,-1);
			}
			else
			{
				build_R(H,Vinout,pivot);
			}
		}
		
//		if (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)
//		{
//			cout << "0site between " << old_pivot << "," << old_pivot+1 << endl;
//		}
//		else
//		{
//			cout << "0site between " << old_pivot-1 << "," << old_pivot << endl;
//		}
		LanczosPropagator<PivotMatrix0<Symmetry,TimeScalar,MpoScalar>,PivotVector<Symmetry,TimeScalar>,TimeScalar> Lutz0(tol_Lanczos);
		
		PivotMatrix0<Symmetry,TimeScalar,MpoScalar> Heff0;
		if (CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)
		{
			if (old_pivot+1 == N_sites)
			{
				Heff0 = PivotMatrix0<Symmetry,TimeScalar,MpoScalar>(HeffLast.L, Heff[old_pivot].R);
			}
			else
			{
				Heff0 = PivotMatrix0<Symmetry,TimeScalar,MpoScalar>(Heff[old_pivot+1].L, Heff[old_pivot].R);
			}
		}
		else
		{
			if (old_pivot-1 == -1)
			{
				Heff0 = PivotMatrix0<Symmetry,TimeScalar,MpoScalar>(Heff[old_pivot].L, HeffFrst.R);
			}
			else
			{
				Heff0 = PivotMatrix0<Symmetry,TimeScalar,MpoScalar>(Heff[old_pivot].L, Heff[old_pivot-1].R);
			}
		}
		
		Stopwatch<> W0;
//		Lutz0.t_step(Heff0, Azero, +x*dt.imag()); // 0-site algorithm
		Lutz0.t_step(Heff0, Azero, -x*dt); // 0-site algorithm
//		cout << Lutz0.info() << endl;
		t_0site += W0.time(SECONDS);
		
		dimK0_log.push_back(Lutz0.get_dimK());
		if (Lutz0.get_dist() > dist_max) {dist_max = Lutz0.get_dist();}
		if (Lutz0.get_dimK() > dimK_max) {dimK_max = Lutz0.get_dimK();}
		
		if (!TURN_FIRST) turnaround(pivot, N_sites, CURRENT_DIRECTION);
		
//		cout << "absorb at pivot=" << pivot << " going " << CURRENT_DIRECTION << endl;
		Vinout.absorb(pivot, CURRENT_DIRECTION, Azero.data[0]);
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
t_step_adaptive (const Hamiltonian &H, VectorType &Vinout, TimeScalar dt, const vector<bool> &TWO_STEP_AT, int N_stages, double tol_Lanczos)
{
	assert(N_stages==1 and "Only N_stages=1 implemented for TDVPPropagator::t_step_adaptive!");
	dist_max = 0.;
	dimK_max = 0;
	N_stages_last = N_stages;
	
	t_0site = 0;
	t_1site = 0;
	t_2site = 0;
	t_ohead = 0;
	t_contr = 0;
	t_tot   = 0;
	
	Stopwatch<> Wtot;
	
	dimK2_log.clear();
	dimK1_log.clear();
	dimK0_log.clear();
	
	for (size_t l=0; l<N_sites-1; ++l)
	{
		if (TWO_STEP_AT[l] == true)
		{
			t_step_pivot(x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
		}
		else
		{
			t0_step_pivot(true,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
		}
	}
	
	if (TWO_STEP_AT[N_sites-2])
	{
		t_step_pivot(x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
	}
	else
	{
		// at N_sites-1 RIGHT
		// at N_sites-1 LEFT
//		if (Vinout.Boundaries.IS_TRIVIAL())
//		{
			t0_step_pivot(false,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
			t0_step_pivot(true,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
//		}
//		else
//		{
//			t0_step_pivot(true,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos,false);
////			cout << endl;
//			t0_step_pivot(true,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
//		}
	}
	
	for (int l=N_sites-3; l>=0; --l)
	{
		if (TWO_STEP_AT[l] == true)
		{
			t_step_pivot(x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
		}
		else
		{
			t0_step_pivot(true,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
		}
	}
	
	if (!TWO_STEP_AT[0])
	{
//		if (Vinout.Boundaries.IS_TRIVIAL())
//		{
			t0_step_pivot(false,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos);
//		}
//		else
//		{
//			t0_step_pivot(true,x(1,0,N_stages),H,Vinout,dt,tol_Lanczos,false);
//		}
	}
	
//	cout << "final pivot=" << pivot << endl << endl;
	
	t_tot = Wtot.time(SECONDS);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
test_edge_eigenvector (const PivotVector<Symmetry,TimeScalar> &Asingle)
{
	if (pivot == 0 or pivot == N_sites-1)
	{
		auto V = Asingle;
		normalize(V);
		PivotVector<Symmetry,TimeScalar> HV;
		HxV(Heff[pivot],V,HV);
		double res = abs(dot(HV,HV).real()-pow(dot(V,HV).real(),2));
//		if (pivot==0)
//		{
//			EvarL = res;
//		}
//		else
//		{
//			EvarR = res;
//		}
//		lout << "pivot=" << pivot << ", eigenstate test=" << res << endl;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
inline void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
build_L (const Hamiltonian &H, const VectorType &Vinout, int loc)
{
	if (loc != 0)
	{
		if (loc == N_sites)
		{
			contract_L(Heff[loc-1].L, Vinout.A[loc-1], H.W[loc-1], Vinout.A[loc-1], H.locBasis(loc-1), H.opBasis(loc-1), HeffLast.L);
		}
		else
		{
			contract_L(Heff[loc-1].L, Vinout.A[loc-1], H.W[loc-1], Vinout.A[loc-1], H.locBasis(loc-1), H.opBasis(loc-1), Heff[loc].L);
		}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar, typename VectorType>
inline void TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,VectorType>::
build_R (const Hamiltonian &H, const VectorType &Vinout, int loc)
{
	if (loc != N_sites-1)
	{
		if (loc == -1)
		{
			contract_R(Heff[loc+1].R, Vinout.A[loc+1], H.W[loc+1], Vinout.A[loc+1], H.locBasis(loc+1), H.opBasis(loc+1), HeffFrst.R);
		}
		else
		{
			contract_R(Heff[loc+1].R, Vinout.A[loc+1], H.W[loc+1], Vinout.A[loc+1], H.locBasis(loc+1), H.opBasis(loc+1), Heff[loc].R);
		}
	}
}

#endif
