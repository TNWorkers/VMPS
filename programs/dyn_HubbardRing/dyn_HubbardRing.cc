#include "EigenFiles.h"

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t L = args.get<size_t>("L",6);
	size_t N = args.get<size_t>("N",L);
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",1.);
	
	string phigFile = args.get<string>("phigFile","");
	string phipFile = args.get<string>("phipFile","");
	VectorXd phig(L), phip(L);
	if (phigFile == "")
	{
		for (int i=0; i<L; ++i) phig(i) = args.get<double>("phig",0.);
	}
	else
	{
		phig = loadMatrix(phigFile);
	}
	if (phipFile == "")
	{
		for (int i=0; i<L; ++i) phip(i) = args.get<double>("phip",0.5);
	}
	else
	{
		phip = loadMatrix(phipFile);
	}
	
	double tol_compr = args.get<double>("tol_compr",1e-7);
	bool CALC_QUENCH = args.get<bool>("CALC_QUENCH",false);
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	string base = make_string("L=",L,"_U=",U);
	lout << base << endl;
	lout.set(base+".log",wd+"log");
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.Minit = args.get<size_t>("Minit",20ul);
	GlobParam.Mlimit = args.get<size_t>("Mlimit",400ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",20ul);
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",1ul);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",20ul);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam.tol_state = args.get<double>("tol_state",1e-4);
	
	DMRG::CONTROL::DYN  DynParam;
	size_t Mincr_per = args.get<size_t>("Mincr_per",2ul);
	DynParam.Mincr_per = [Mincr_per] (size_t i) {return Mincr_per;};
	size_t Mincr_abs = args.get<size_t>("Mincr_abs",60ul);
	DynParam.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
	size_t start_alpha = args.get<size_t>("start_alpha",0);
	size_t end_alpha = args.get<size_t>("end_alpha",GlobParam.max_halfsweeps-3);
	double alpha = args.get<double>("alpha",100.);
	DynParam.max_alpha_rsvd = [start_alpha, end_alpha, alpha] (size_t i) {return (i>=start_alpha and i<end_alpha)? alpha:0.;};
	
	vector<Param> params;
	params.push_back({"Uph",U});
	params.push_back({"maxPower",2ul});
	
	ArrayXXd tFull = create_1D_PBC(L,1.,0.,true);
	ArrayXXcd tFullc = tFull.cast<complex<double> >();
	params.push_back({"tFull",tFullc});
	
	#if defined(USING_SU2_COMPLEX)
	vector<SUB_LATTICE> G(L);
	G[0] = static_cast<SUB_LATTICE>(1);
	for (int l=1; l<=L-3; l+=2)
	{
		int fac = -1*G[l-1];
		G[l  ] = static_cast<SUB_LATTICE>(fac);
		G[l+1] = static_cast<SUB_LATTICE>(fac);
	}
	G[L-1] = static_cast<SUB_LATTICE>(-1*G[L-2]);
	for (int l=0; l<L; ++l) lout << G[l];
	lout << endl;
	params.push_back({"G",G});
	#endif
	
	if (L<=20)
	{
		lout << tFullc << endl;
	}
	
	MODEL Htmp(L,params);
	lout << "Htmp:" << Htmp.info() << endl;
	
	Mpo<MODEL::Symmetry,MODEL::Scalar_> Hmpo0 = Htmp;
	Mpo<MODEL::Symmetry,MODEL::Scalar_> Hmpop = Htmp;
	
	for (int i=0; i<L; ++i)
	{
		Mpo<MODEL::Symmetry,MODEL::Scalar_> SFp = Htmp.Sp(i); SFp.scale(exp(+1.i*M_PI*phig[i]));
		Mpo<MODEL::Symmetry,MODEL::Scalar_> SFm = Htmp.Sm(i); SFm.scale(exp(-1.i*M_PI*phig[i]));
		Mpo<MODEL::Symmetry,MODEL::Scalar_> Sftot = sum(SFp,SFm);
		
		Hmpo0 = sum(Hmpo0,Sftot);
		
		SFp = Htmp.Sp(i); SFp.scale(exp(+1.i*M_PI*phip[i]));
		SFm = Htmp.Sm(i); SFm.scale(exp(-1.i*M_PI*phip[i]));
		Sftot = sum(SFp,SFm);
		
		Hmpop = sum(Hmpop,Sftot);
	}
	
	MODEL H0(Hmpo0,params);
	lout << "H0:" << H0.info() << endl;
	
	MODEL Hp(Hmpop,params);
	lout << "Hp:" << Hp.info() << endl;
	
	Eigenstate<MODEL::StateXcd> g;
	MODEL::Solver DMRG(VERB);
	DMRG.userSetGlobParam();
	DMRG.userSetDynParam();
	DMRG.DynParam = DynParam;
	DMRG.GlobParam = GlobParam;
	
	DMRG.edgeState(H0, g, Q);
	lout << endl << "var/L=" << abs(avg(g.state, H0, H0, g.state)-pow(g.energy,2))/L << endl << endl;
	
	if (CALC_QUENCH)
	{
		auto Psi = g.state.cast<complex<double> >();
		Psi.eps_truncWeight = tol_compr;
		TDVPPropagator<MODEL,MODEL::Symmetry,MODEL::Scalar_,complex<double>,Mps<MODEL::Symmetry,complex<double>>> TDVP(Hp,Psi);
		
		double dt = 0.1;
		double tmax = 20.;
		int tpoints = int(tmax/dt+1);
		
		IntervalIterator it(0.,tmax,tpoints);
		
		for (it=it.begin(3); it!=it.end(); ++it)
		{
			double E = isReal(avg(Psi, Hp, Psi));
			
			int l = L/2;
			//for (int l=0; l<L; ++l)
			{
				double nUP = isReal(avg(Psi, Hp.n<UP>(l), Psi));
				double nDN = isReal(avg(Psi, Hp.n<DN>(l), Psi));
				lout << "l=" << l << ", nUP=" << nUP << ", nDN=" << nDN << endl;
				if (l==L/2)
				{
					it << nUP, nDN, E;
				}
			}
			it.save("n(t).dat");
			
			Stopwatch<> Timer;
			TDVP.t_step(Hp, Psi, 1.i*dt);
			lout << Timer.info(make_string("propagated to t=",*it)) << endl;
			lout << TDVP.info() << endl;
			lout << Psi.info() << endl;
			lout << endl;
		}
	}
}
