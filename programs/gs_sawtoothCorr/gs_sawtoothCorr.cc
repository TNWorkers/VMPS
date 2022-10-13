#if defined(BLAS) or defined(BLIS) or defined(MKL)
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define USE_WIG_SU2_COEFFS
#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define LINEARSOLVER_DIMK 100
#define HELPERS_IO_TABLE

#include <iostream>
#include <fstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
#define USING_SU2

#include "models/ParamCollection.h"
#include "DmrgLinearAlgebra.h"
#include "EigenFiles.h"

double calc_Stot (const MODEL &H, const MODEL::StateXd &Psi)
{
	double res = 0.;
	#pragma omp parallel for collapse(2) reduction(+:res)
	for (int i=0; i<Psi.length(); ++i)
	for (int j=0; j<Psi.length(); ++j)
	{
		res += avg(Psi, H.SdagS(i,j), Psi);
	}
	return res;
}

struct SaveData
{
	VectorXd Savg;
	MatrixXd SdagS;
	double SdagStot;
	double E;
	double e;
	double var = -1.;
	
	void save (string label)
	{
		HDF5Interface target(label+".h5",WRITE);
		target.save_vector(Savg,"Savg","");
		target.save_matrix(SdagS,"SdagS","");
		target.save_scalar(E,"E","");
		target.save_scalar(e,"e","");
		target.save_scalar(var,"var","");
		target.save_scalar(SdagStot,"SdagStot","");
		target.close();
	}
};

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	int L = args.get<int>("L",100);
	
	int Slimit = args.get<int>("Slimit",L/2);
	#if defined(USE_WIG_SU2_COEFFS)
	lout << "initializing CGC tables for Slimit=" << Slimit << "..." << endl;
//	Sym::initialize(120, "./cgc_hash/table_120.3j", "./cgc_hash/table_120.6j");
	Sym::initialize(Slimit);
	#endif
	
	int S = args.get<int>("S",27);
	int M = args.get<int>("M",0);
	#ifdef USING_U1
	qarray<MODEL::Symmetry::Nq> Q = {M};
	#elif defined(USING_U0)
	qarray<MODEL::Symmetry::Nq> Q = {};
	#elif defined(USING_SU2)
	qarray<MODEL::Symmetry::Nq> Q = {2*S+1};
	#endif
	lout << "Q=" << Q << endl;
	
	double J = args.get<double>("J",-1.);
	double JpA = args.get<double>("JpA",1.);
	double JpB = args.get<double>("JpB",0.);
	
	size_t D = args.get<size_t>("D",2ul);
	size_t maxPower = args.get<size_t>("maxPower",2ul);
	bool COMPRESS = args.get<bool>("COMPRESS",true);
	bool CALC_VAR = args.get<bool>("CALC_VAR",false);
	size_t Mlimit = args.get<size_t>("Mlimit",2000ul);
	bool LOAD_AVG = args.get<bool>("LOAD_AVG",false);
	bool LOAD_SDAGS = args.get<bool>("LOAD_SDAGS",false);
	int Nl0 = args.get<int>("Nl0",L);
	int dmax = args.get<int>("dmax",L/2);
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	string base = make_string("L=",L,"_J=",J,",",J,"_Jprime=",JpA,",",JpB,"_D=",D);
	base += make_string("_S=",S);
	base += "_PBC=1";
	lout << base << endl;
	lout.set(base+".log",wd+"log");
	
	lout << args.info() << endl;
	#ifdef _OPENMP
	omp_set_nested(1);
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	vector<Param> params;
	
	ArrayXXd Jfull_uncompressed = create_1D_PBC_AB(L,J,J,JpA,JpB,false);
	CuthillMcKeeCompressor CMK(Jfull_uncompressed,false); // PRINT=false
	
	ArrayXXd Jfull = create_1D_PBC_AB(L,J,J,JpA,JpB,COMPRESS);
	params.push_back({"Jfull",Jfull});
	
	MODEL H(L,params);
	lout << H.info() << endl;
	
	Eigenstate<MODEL::StateXd> g;
	string filename = make_string("gs_",base,"_Mmax=",Mlimit);
	string folder = make_string("../gs_sawtooth/SU2/Mlimit=",Mlimit,"/state/");
	lout << folder+filename << endl;
	g.state.load(folder+filename);
	
	lout << g.state.info() << endl;
	
	MatrixXd Savgl(L,2); Savgl.setZero();
	MatrixXd Savgt(L,2); Savgt.setZero();
	
	if (LOAD_AVG)
	{
		Savgl = loadMatrix(make_string("obs/Savgl_",base,"_Mmax=",Mlimit,".dat"));
		Savgt = loadMatrix(make_string("obs/Savgt_",base,"_Mmax=",Mlimit,".dat"));
		lout << Savgl << endl;
	}
	else
	{
		if (S!=0)
		{
			Stopwatch<> Timer;
			#pragma omp parallel for schedule(dynamic)
			for (int l=0; l<L; ++l)
			{
				#if defined(USE_WIG_SU2_COEFFS)
				wig_thread_temp_init(2*Slimit);
				#endif
			
				int t = CMK.get_transform()[l];
				double val = avg(g.state, H.S(t), g.state) * S/sqrt(S*(S+1.));
				Savgl(l,0) = l;
				Savgl(l,1) = val;
				Savgt(t,0) = t;
				Savgt(t,1) = val;
				#pragma omp critical
				{
					lout << "l=" << l << ", t=" << t << ", Savg=" << val << endl;
				}
			}
			lout << Timer.info("Savg") << endl;
			saveMatrix(Savgl, make_string("obs/Savgl_",base,"_Mmax=",Mlimit,".dat"));
			saveMatrix(Savgt, make_string("obs/Savgt_",base,"_Mmax=",Mlimit,".dat"));
		}
	}
	
	lout << "sum=" << Savgl.col(1).sum() << " must be equal to S=" << S << endl;
	lout << endl;
	
	MatrixXd Savg_nrm, Savg_red;
	if (LOAD_SDAGS)
	{
		Savg_nrm = loadMatrix(make_string("obs/SdagSnrm_","d_",base,"_Mmax=",Mlimit,".dat"));
		Savg_red = loadMatrix(make_string("obs/SdagSred_","d_",base,"_Mmax=",Mlimit,".dat"));
	}
	else
	{
		Savg_nrm.resize(L/2+1,1); Savg_nrm.setZero(); // x=distance, y=i0
		Savg_red.resize(L/2+1,1); Savg_red.setZero();
	}
	
	Stopwatch<> SweepTimer;
	if (S == 0) g.state.sweep(0,DMRG::BROOM::QR);
	//lout << g.state.get_pivot() << endl;
	lout << SweepTimer.info("QR sweep") << endl;
	
	for (int l0=Savg_nrm.cols()-1; l0<Nl0; ++l0)
	{
		lout << "l0=" << l0 << endl;
		Savg_nrm.conservativeResize(Savg_nrm.rows(),Savg_nrm.cols()+1);
		Savg_red.conservativeResize(Savg_red.rows(),Savg_red.cols()+1);
		
		Stopwatch<> Timer;
		
		#pragma omp parallel for
		for (int d=0; d<=dmax; ++d)
		{
			#if defined(USE_WIG_SU2_COEFFS)
			wig_thread_temp_init(2*Slimit);
			#endif
			
			int inew = CMK.get_transform()[l0];
			int jnew = CMK.get_transform()[(l0+d)%L];
			
			double val;
			if (S != 0)
			{
				val = avg(g.state, H.SdagS(inew,jnew), g.state);
			}
			else
			{
				val = g.state.locAvg(H.SdagS(inew,jnew), max(inew,jnew));
			}
			double val_red = val - Savgt(inew,1)*Savgt(jnew,1);
			//double val_red = val - avg(g.state, H.S(inew), g.state) * avg(g.state, H.S(jnew), g.state) * pow(S/sqrt(S*(S+1.)),2);
			
			#pragma omp critical
			{
				lout << "l0=" << l0 << ", ld=" << l0+d << ", inew=" << inew << ", jnew=" << jnew << ", d=" << d << "\t" << val;
				if (S!=0)
				{
					lout << "\t" << val_red;
				}
				lout << endl;
				
				Savg_nrm(d,0) = d;
				Savg_nrm(d,l0+1) = val;
				
				Savg_red(d,0) = d;
				Savg_red(d,l0+1) = val_red;
			}
		}
		lout << Timer.info(make_string("l0=",l0)) << endl;
		if (S!=0) saveMatrix(Savg_nrm, make_string("obs/SdagSnrm_","d_",base,"_Mmax=",Mlimit,".dat"));
		saveMatrix(Savg_red, make_string("obs/SdagSred_","d_",base,"_Mmax=",Mlimit,".dat"));
	}
	
	if (CALC_VAR)
	{
		g.energy = avg(g.state,H,g.state);
		double var = abs(avg(g.state,H,g.state,2)-pow(g.energy,2))/L;
		lout << "varE=" << var << endl;
	}
}
