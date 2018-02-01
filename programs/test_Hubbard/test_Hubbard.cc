#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
//#define USE_HDF5_STORAGE
//#define EIGEN_USE_THREADS

// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
//#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

//Also calculate SU2xSU2, implies no tPrime
#define SU2XSU2

#include <iostream>
#include <fstream>
#include <complex>
#include <variant>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

// ED stuff
#include "HubbardModel.h"
#include "LanczosWrappers.h"
#include "LanczosSolver.h"
#include "Photo.h"
#include "Auger.h"

#include "solvers/DmrgSolver.h"

#include "models/Hubbard.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardSU2xU1.h"
#ifdef SU2XSU2
#include "models/HubbardSU2xSU2.h"
#endif

template<typename Scalar>
string to_string_prec (Scalar x, int n=14)
{
	ostringstream ss;
	ss << setprecision(n) << x;
	return ss.str();
}

complex<double> Ptot (const MatrixXd &densityMatrix, int Lx)
{
	complex<double> P=0.;
	int L_2 = static_cast<int>(Lx)/2;
	for (int i=0; i<Lx; ++i)
	for (int j=0; j<Lx; ++j)
	for (int n=-L_2; n<L_2; ++n)
	{
		double k = 2.*M_PI*n/Lx;
		P += k * exp(-1.i*k*static_cast<double>(i-j)) * densityMatrix(i,j);
	}
	P /= (Lx*Lx);
	return P;
}

bool CALC_DYNAMICS;
size_t L, Lx, Ly;
double t, tPrime, U, mu, Bz;
int Nup, Ndn, N;
double alpha;
double t_U0, t_U1, t_SU2, t_SU2xSU2;
int Dinit, Dlimit, Imin, Imax;
double tol_eigval, tol_state;
double dt;
int i0;
DMRG::VERBOSITY::OPTION VERB;
double overlap_ED = 0.;
double overlap_U1_zipper = 0.;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	Lx = args.get<size_t>("Lx",4); L=Lx;
	Ly = args.get<size_t>("Ly",1);
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",8.);
	mu = args.get<double>("mu",0.5*U);
	Nup = args.get<int>("Nup",L/2);
	Ndn = args.get<int>("Ndn",L/2);
	N = Nup+Ndn;
	cout << "Nup=" << Nup << ", Ndn=" << Ndn << ", N=" << N << endl;
	alpha = args.get<double>("alpha",1.);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	i0 = args.get<int>("i0",L/2);
	dt = 0.2;
	int V = L*Ly; int Vsq = V*V;
	
	Dinit  = args.get<int>("Dmin",2);
	Dlimit = args.get<int>("Dmax",100);
	Imin   = args.get<int>("Imin",2);
	Imax   = args.get<int>("Imax",50);
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_state  = args.get<double>("tol_state",1e-5);
	
	CALC_DYNAMICS = args.get<bool>("CALC_DYN",0);
	
	lout << args.info() << endl;
	lout.set(make_string("Lx=",Lx,"_Ly=",Ly,"_t=",t,"_t'=",tPrime,"_U=",U,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------ED-----------
	lout << endl << "--------ED---------" << endl << endl;
	
	InteractionParams params;
	params.set_U(U);
	(tPrime!=0) ? params.set_hoppings({-t,-tPrime}):params.set_hoppings({-t});
//	MatrixXd BondMatrix(Lx*Ly,Lx*Ly); BondMatrix.setZero();
//	BondMatrix(0,1) = -t;
//	BondMatrix(1,0) = -t;
//	
//	BondMatrix(0,2) = -t;
//	BondMatrix(2,0) = -t;
//	
//	BondMatrix(2,3) = -t;
//	BondMatrix(3,2) = -t;
//	
//	BondMatrix(1,3) = -t;
//	BondMatrix(3,1) = -t;
	
//	HubbardModel H_ED(Lx*Ly,Nup,Ndn,U,BondMatrix.sparseView(), BC_DANGLING);
	HubbardModel H_ED(Lx*Ly,Nup,Ndn,params, BC_DANGLING);
	lout << H_ED.info() << endl;
	Eigenstate<VectorXd> g_ED;
	LanczosSolver<HubbardModel,VectorXd,double> Lutz;
	Lutz.edgeState(H_ED,g_ED,LANCZOS::EDGE::GROUND);
	
//	HubbardModel H_EDm(Lx*Ly,Nup-1,Ndn,U,BondMatrix.sparseView(), BC_DANGLING);
	HubbardModel H_EDm(Lx*Ly,Nup-1,Ndn,params, BC_DANGLING);
	Eigenstate<VectorXd> g_EDm;
	Lutz.edgeState(H_EDm,g_EDm,LANCZOS::EDGE::GROUND);
	
//	HubbardModel H_EDmm(Lx*Ly,Nup-1,Ndn-1,U,BondMatrix.sparseView(), BC_DANGLING);
	HubbardModel H_EDmm(Lx*Ly,Nup-1,Ndn-1,params, BC_DANGLING);
	Eigenstate<VectorXd> g_EDmm;
	Lutz.edgeState(H_EDmm,g_EDmm,LANCZOS::EDGE::GROUND);
	
	for (int l=0; l<L; ++l)
	{
		Photo Ph(H_EDm,H_ED,UP,l);
		cout << "l=" << l << ", <c>=" << avg(g_EDm.state, (Ph.Operator()).eval(), g_ED.state) << endl;
	}
	
	Auger A(H_EDmm, H_ED, i0);
	VectorXd OxV_ED = A.Operator() * g_ED.state;
	double overlap_ED = g_EDmm.state.dot(OxV_ED);
	
	lout << "Emin=" << g_ED.energy << ", Emin/V=" << to_string_prec(g_ED.energy/V) << endl;
	
	MatrixXd densityMatrix_ED(L,L); densityMatrix_ED.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_ED(i,j) = avg(g_ED.state, H_ED.hopping_element(j,i,UP), g_ED.state)+
		                        avg(g_ED.state, H_ED.hopping_element(j,i,DN), g_ED.state);
	}
	lout << "<cdagc>=" << endl << densityMatrix_ED << endl;
	
	ArrayXd d_ED(L); d_ED=0.;
	ArrayXd h_ED(L); h_ED=0.;
	for (size_t i=0; i<L; ++i) 
	{
		d_ED(i) = avg(g_ED.state, H_ED.d(i), g_ED.state);
		h_ED(i) = 1.-avg(g_ED.state, H_ED.n(i), g_ED.state)+d_ED(i);
	}
	lout << "<d>=" << endl << d_ED << endl;
	lout << "<h>=" << endl << h_ED << endl;
	
	//--------U(0)---------
	lout << endl << "--------U(0)---------" << endl << endl;
	
	Stopwatch<> Watch_U0;
	VMPS::Hubbard H_U0(Lx,{{"t",t},{"tPrime",tPrime},{"U",U},{"mu",mu},{"Ly",Ly}});
	lout << H_U0.info() << endl;
	Eigenstate<VMPS::Hubbard::StateXd> g_U0;
	
	VMPS::Hubbard::Solver DMRG_U0(VERB);
	DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, 10.*tol_eigval,10.*tol_state, Dinit,3*Dlimit, Imax,Imin, 0.1);
	
	lout << endl;
	double Ntot = 0.;
	for (size_t lx=0; lx<Lx; ++lx)
	for (size_t ly=0; ly<Ly; ++ly)
	{
		double n_l = avg(g_U0.state, H_U0.n(UPDN,lx,ly), g_U0.state);
		cout << "lx=" << lx << ", ly=" << ly << "\tn=" << n_l << endl;
		Ntot += n_l;
	}
	
	double Emin_U0 = g_U0.energy+mu*Ntot;
	double emin_U0 = Emin_U0/V;
	lout << "correction for mu: E=" << to_string_prec(Emin_U0) << ", E/V=" << to_string_prec(emin_U0) << endl;
	
	t_U0 = Watch_U0.time();
	
//	// compressor
//	
//	VMPS::Hubbard::StateXd Hxg_U0;
//	HxV(H_U0,g_U0.state,Hxg_U0,VERB);
//	double E_U0_compressor = g_U0.state.dot(Hxg_U0);
	
	// zipper
	
	VMPS::Hubbard::StateXd Oxg_U0;
	Oxg_U0.eps_svd = 1e-15;
	OxV(H_U0,g_U0.state,Oxg_U0,DMRG::BROOM::SVD);
	double E_U0_zipper = g_U0.state.dot(Oxg_U0);
	
	//--------U(1)---------
	lout << endl << "--------U(1)---------" << endl << endl;
	
	Stopwatch<> Watch_U1;
	
	VMPS::HubbardU1xU1 H_U1(Lx,{{"t",t},{"tPrime",tPrime},{"U",U},{"Ly",Ly}});
	lout << H_U1.info() << endl;
	Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1;
	
	VMPS::HubbardU1xU1::Solver DMRG_U1(VERB);
	DMRG_U1.edgeState(H_U1, g_U1, {Nup,Ndn}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	t_U1 = Watch_U1.time();
	
	Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1m;
	DMRG_U1.set_verbosity(DMRG::VERBOSITY::SILENT);
	DMRG_U1.edgeState(H_U1, g_U1m, {Nup-1,Ndn}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	lout << "g_U1m.energy=" << g_U1m.energy << endl;
	
	ArrayXd c_U1(L);
	for (int l=0; l<L; ++l)
	{
		c_U1(l) = avg(g_U1m.state, H_U1.c(UP,l), g_U1.state);
		cout << "l=" << l << ", <c>=" << c_U1(l) << endl;
	}
	
	// observables
	
	MatrixXd densityMatrix_U1(L,L); densityMatrix_U1.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_U1(i,j) = avg(g_U1.state, H_U1.cdagc(UP,i,j), g_U1.state)+
		                        avg(g_U1.state, H_U1.cdagc(DN,i,j), g_U1.state);
	}
	lout << "<cdagc>=" << endl << densityMatrix_U1 << endl;
	
	MatrixXd densityMatrix_U1B(L,L); densityMatrix_U1B.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_U1B(i,j) = avg(g_U1.state, H_U1.cdag(UP,i), H_U1.c(UP,j), g_U1.state)+
		                         avg(g_U1.state, H_U1.cdag(DN,i), H_U1.c(DN,j), g_U1.state);
	}
	lout << endl << densityMatrix_U1B << endl;
	lout << "diff=" << (densityMatrix_U1-densityMatrix_U1B).norm() << endl;
	
	lout << "P U(1): " << Ptot(densityMatrix_U1,Lx) << "\t" << Ptot(densityMatrix_U1B,Lx) << endl;
	
	ArrayXd d_U1(L); d_U1=0.;
	for (size_t i=0; i<L; ++i) 
	{
		d_U1(i) = avg(g_U1.state, H_U1.d(i), g_U1.state);
	}
	lout << "<d>=" << endl << d_U1 << endl;
	
//	// compressor
//	
//	VMPS::HubbardU1xU1::StateXd Hxg_U1;
//	HxV(H_U1,g_U1.state,Hxg_U1,VERB);
//	double E_U1_compressor = g_U1.state.dot(Hxg_U1);
//	
	// zipper
	
	VMPS::HubbardU1xU1::StateXd Oxg_U1;
	Oxg_U1.eps_svd = 1e-15;
	OxV(H_U1.cc(i0), g_U1.state, Oxg_U1, DMRG::BROOM::SVD);
	Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1mm;
	DMRG_U1.edgeState(H_U1, g_U1mm, {Nup-1,Ndn-1}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, 
	                  tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	overlap_U1_zipper = g_U1mm.state.dot(Oxg_U1);
	
	// --------SU(2)---------
	lout << endl << "--------SU(2)---------" << endl << endl;
	
	Stopwatch<> Watch_SU2;
	
	VMPS::HubbardSU2xU1 H_SU2(Lx,{{"t",t},{"tPrime",tPrime},{"U",U},{"Ly",Ly}});
	lout << H_SU2.info() << endl;
	Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2;
	
	VMPS::HubbardSU2xU1::Solver DMRG_SU2(VERB);
	DMRG_SU2.edgeState(H_SU2, g_SU2, {Nup-Ndn+1,N}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, 
	                   tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	t_SU2 = Watch_SU2.time();
	
	// observables
	
	Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2m;
	DMRG_SU2.set_verbosity(DMRG::VERBOSITY::SILENT);
	DMRG_SU2.edgeState(H_SU2, g_SU2m, {abs(Nup-1-Ndn)+1,N-1}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, 
	                   tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	lout << "g_SU2m.energy=" << g_SU2m.energy << endl;
	
	ArrayXd c_SU2(L);
	for (int l=0; l<L; ++l)
	{
		c_SU2(l) = avg(g_SU2m.state, H_SU2.c(l), g_SU2.state);
		cout << "l=" << l << ", <c>=" << c_SU2(l) << "\t" << c_SU2(l)/c_U1(l) << endl;
	}
	
	MatrixXd densityMatrix_SU2(L,L); densityMatrix_SU2.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_SU2(i,j) = avg(g_SU2.state, H_SU2.cdagc(i,j), g_SU2.state);
	}
	lout << densityMatrix_SU2 << endl;
	
	MatrixXd densityMatrix_SU2B(L,L); densityMatrix_SU2B.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_SU2B(i,j) = sqrt(2.)*avg(g_SU2.state, H_SU2.cdag(i), H_SU2.c(j), g_SU2.state);
	}
	lout << endl << densityMatrix_SU2B << endl;
	lout << "diff=" << (densityMatrix_SU2-densityMatrix_SU2B).norm() << endl;
	
	lout << "P SU(2): " << Ptot(densityMatrix_SU2,Lx) << "\t" << Ptot(densityMatrix_SU2B,Lx) << endl;
	
	ArrayXd d_SU2(L); d_SU2=0.;
	for (size_t i=0; i<L; ++i) 
	{
		d_SU2(i) = avg(g_SU2.state, H_SU2.d(i), g_SU2.state);
	}
	lout << "<d>=" << endl << d_SU2 << endl;
	
#ifdef SU2XSU2
	// --------SU(2)xSU(2)---------
	lout << endl << "--------SU(2)xSU(2)---------" << endl << endl;
	
	Stopwatch<> Watch_SU2xSU2;
	
	vector<Param> paramsSU2xSU2;
	paramsSU2xSU2.push_back({"U",U,0});
	paramsSU2xSU2.push_back({"U",U,1});
	paramsSU2xSU2.push_back({"subL",SUB_LATTICE::A,0});
	paramsSU2xSU2.push_back({"subL",SUB_LATTICE::B,1});
	paramsSU2xSU2.push_back({"Ly",Ly,0});
	paramsSU2xSU2.push_back({"Ly",Ly,1});
	VMPS::HubbardSU2xSU2 H_SU2xSU2(Lx,paramsSU2xSU2);
	lout << H_SU2xSU2.info() << endl;
	Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2;
	
	VMPS::HubbardSU2xSU2::Solver DMRG_SU2xSU2(VERB);
	DMRG_SU2xSU2.edgeState(H_SU2xSU2, g_SU2xSU2, {abs(Nup-Ndn)+1,V-(Nup+Ndn)+1}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, 
	                       tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha); //Todo: check Pseudospin quantum number... (1 <==> half filling)
	
	double Emin_SU2xSU2 = g_SU2xSU2.energy-0.5*U*(V-Nup-Ndn);
	double emin_SU2xSU2 = Emin_SU2xSU2/V;
	t_SU2xSU2 = Watch_SU2xSU2.time();
	
	// observables
	
	 Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2m;
	 DMRG_SU2xSU2.set_verbosity(DMRG::VERBOSITY::SILENT);
	 DMRG_SU2xSU2.edgeState(H_SU2xSU2, g_SU2xSU2m, {abs(Nup-1-Ndn)+1,V-(Nup+Ndn)+2}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, 
	                    tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	lout << "g_SU2xSU2m.energy=" << g_SU2xSU2m.energy-0.5*U*(V-Nup+1-Ndn) << endl;
	
	ArrayXd c_SU2xSU2(L);
	for (int l=0; l<L; ++l)
	{
		c_SU2xSU2(l) = avg(g_SU2xSU2m.state, H_SU2xSU2.c(l), g_SU2xSU2.state);
		cout << "l=" << l << ", <c>=" << c_SU2xSU2(l) << "\t" << c_SU2xSU2(l)/c_U1(l) << endl;
	}
	
	MatrixXd densityMatrix_SU2xSU2(L,L); densityMatrix_SU2xSU2.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_SU2xSU2(i,j) = avg(g_SU2xSU2.state, H_SU2xSU2.cdagc(i,j), g_SU2xSU2.state);
	}
	lout << 0.5*densityMatrix_SU2xSU2 << endl;
	
	MatrixXd densityMatrix_SU2xSU2B(L,L); densityMatrix_SU2xSU2B.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_SU2xSU2B(i,j) = sqrt(2.)*sqrt(2.)*avg(g_SU2xSU2.state, H_SU2xSU2.cdag(i), H_SU2xSU2.c(j), g_SU2xSU2.state);
	}
	lout << endl << 0.5*densityMatrix_SU2xSU2B << endl; //factor 1/2 because we have computed cdagc+cdagc
	lout << "diff=" << (densityMatrix_SU2xSU2-densityMatrix_SU2xSU2B).norm() << endl;
	
	lout << "P SU(2): " << Ptot(0.5*densityMatrix_SU2xSU2,Lx) << "\t" << Ptot(0.5*densityMatrix_SU2xSU2B,Lx) << endl;
	
	ArrayXd nh_SU2xSU2(L); nh_SU2xSU2=0.;
	ArrayXd ns_SU2xSU2(L); ns_SU2xSU2=0.;
	for (size_t i=0; i<L; ++i) 
	{
		nh_SU2xSU2(i) = avg(g_SU2xSU2.state, H_SU2xSU2.nh(i), g_SU2xSU2.state);
		ns_SU2xSU2(i) = avg(g_SU2xSU2.state, H_SU2xSU2.ns(i), g_SU2xSU2.state);
	}
	lout << "<nh>=" << endl << nh_SU2xSU2 << endl;
	lout << "error(<nh>=<h>+<d>)=" << (nh_SU2xSU2-d_ED-h_ED).matrix().norm() << endl;
#endif
	
	
	//--------output---------
	TextTable T( '-', '|', '+' );
	
	T.add("");
	T.add("ED");
	T.add("U(0)");
	T.add("U(1)⊗U(1)");
	T.add("SU(2)⊗U(1)");
#ifdef SU2XSU2
	T.add("SU(2)⊗SU(2)");
#endif
	T.endOfRow();
	
	T.add("E/V");
	T.add(to_string_prec(g_ED.energy/V));
	T.add(to_string_prec(emin_U0));
	T.add(to_string_prec(g_U1.energy/V));
	T.add(to_string_prec(g_SU2.energy/V));
#ifdef SU2XSU2
	T.add(to_string_prec(emin_SU2xSU2));
#endif
	T.endOfRow();
	
	T.add("E/V diff");
	T.add("-");
	T.add(to_string_prec(abs(Emin_U0-g_ED.energy)/V));
	T.add(to_string_prec(abs(g_U1.energy-g_ED.energy)/V));
	T.add(to_string_prec(abs(g_SU2.energy-g_ED.energy)/V));
#ifdef SU2XSU2
	T.add(to_string_prec(abs(Emin_SU2xSU2-g_ED.energy)/V));
#endif
	T.endOfRow();
	
//	T.add("E/L Compressor"); T.add(to_string_prec(E_U0_compressor/V)); T.add(to_string_prec(E_U1_compressor/V)); T.add("-"); T.endOfRow();
	
	T.add("OxV");
	T.add(to_string_prec(overlap_ED));
	T.add("-");
	T.add(to_string_prec(overlap_U1_zipper));
	T.add("-");
#ifdef SU2XSU2
	T.add("-");
#endif
	T.endOfRow();
	
	T.add("OxV zipper rel. err.");
	T.add("0");
	T.add("-");
	T.add(to_string_prec(abs(abs(overlap_U1_zipper) -abs(overlap_ED))/abs(overlap_ED)));
	T.add("-");
#ifdef SU2XSU2
	T.add("-");
#endif
	T.endOfRow();
	
	T.add("t/s");
	T.add("-");
	T.add(to_string_prec(t_U0,2));
	T.add(to_string_prec(t_U1,2));
	T.add(to_string_prec(t_SU2,2));
#ifdef SU2XSU2
	T.add(to_string_prec(t_SU2xSU2,2));
#endif
	T.endOfRow();
	
	T.add("t gain");
	T.add("-");
	T.add(to_string_prec(t_U0/t_SU2,2));
	T.add(to_string_prec(t_U1/t_SU2,2));
	T.add("1");
#ifdef SU2XSU2
	T.add(to_string_prec(t_SU2xSU2/t_SU2,2));
#endif
	T.endOfRow();
	
	T.add("observables diff");
	T.add("0");
	T.add("-");
	T.add(to_string_prec((densityMatrix_U1-densityMatrix_ED).norm()));
	T.add(to_string_prec((densityMatrix_SU2-densityMatrix_ED).norm()));
#ifdef SU2XSU2
	T.add(to_string_prec((0.5*densityMatrix_SU2xSU2-densityMatrix_ED).norm()));
#endif
	T.endOfRow();
	
	T.add("Dmax");
	T.add("-");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U1.state.calc_Dmax()));
	T.add(to_string(g_SU2.state.calc_Dmax()));
#ifdef SU2XSU2
	T.add(to_string(g_SU2xSU2.state.calc_Dmax()));
#endif
	T.endOfRow();
	
	T.add("Mmax");
	T.add("-");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U1.state.calc_Mmax()));
	T.add(to_string(g_SU2.state.calc_Mmax()));
#ifdef SU2XSU2
	T.add(to_string(g_SU2xSU2.state.calc_Mmax()));
#endif
	T.endOfRow();
	
	lout << endl << T;
}
