#include "EigenFiles.h"

int get_next (int i, int L)
{
	int res;
	if      (i%2==0 and i<L-2)  {res = i+2;}
	else if (i%2==0 and i==L-2) {res = L-1;}
	else if (i%2==1 and i>1)    {res = i-2;}
	else if (i%2==1 and i==1)   {res = 0;}
	return res;
}

template<SPIN_INDEX sigma>
Mpo<MODEL::Symmetry,MODEL::Scalar_> P (const MODEL &H, int i, int j)
{
	Mpo<MODEL::Symmetry,MODEL::Scalar_> res = H.Identity();
	
	res = diff(res,H.n<sigma>(i));
	res = diff(res,H.n<sigma>(j));
	
	res = sum(res,H.cdagc<sigma,sigma>(i,j));
	
	#if not defined(USING_SU2_COMPLEX)
	res = sum(res,H.cdagc<sigma,sigma>(j,i));
	#endif
	
	return res;
}

// for testing purposes:
template<SPIN_INDEX sigma>
Mpo<MODEL::Symmetry,MODEL::Scalar_> cdagc_ij_ji (const MODEL &H, int i, int j)
{
	Mpo<MODEL::Symmetry,MODEL::Scalar_> res = H.cdagc<sigma,sigma>(i,j);
	#if not defined(USING_SU2_COMPLEX)
	res = sum(res,H.cdagc<sigma,sigma>(j,i));
	#endif
	
	return res;
}

struct SaveData
{
	SaveData (int L)
	{
		nUP.resize(L); nUP.setZero();
		nDN.resize(L); nDN.setZero();
		nh.resize(L); nh.setZero();
		Sp.resize(L); Sp.setZero();
		Sm.resize(L); Sm.setZero();
		Sx.resize(L); Sx.setZero();
		Sy.resize(L); Sy.setZero();
	};
	
	double var;
	double E, e;
	double Ekin, ekin;
	
	VectorXd nUP, nDN, nh;
	VectorXcd Sp, Sm;
	VectorXd Sx, Sy;
	
	double absAvgU, argAvgU, j;
	
	void save (string label)
	{
		HDF5Interface target(label+".h5",WRITE);
		
		target.save_scalar(var,"var","");
		target.save_scalar(E,"E","");
		target.save_scalar(e,"e","");
		target.save_scalar(Ekin,"Ekin","");
		target.save_scalar(ekin,"ekin","");
		
		target.save_vector(nUP,"nUP","");
		target.save_vector(nDN,"nDN","");
		target.save_vector(nh,"nh","");
		target.save_vector(Sp,"Sp","");
		target.save_vector(Sm,"Sm","");
		target.save_vector(Sx,"Sx","");
		target.save_vector(Sy,"Sy","");
		
		target.save_scalar(absAvgU,"absAvgU","");
		target.save_scalar(argAvgU,"argAvgU","");
		target.save_scalar(j,"j","");
		
		target.close();
	}
};

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t L = args.get<size_t>("L",6);
	size_t N = args.get<size_t>("N",L);
	double T = args.get<double>("T",0.);
	//qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
	#if defined(USING_SU2_COMPLEX)
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(static_cast<int>(2*T+1));
	#else
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
	#endif
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",1.);
	double t = args.get<double>("t",1.);
	double fp = args.get<double>("fp",0.);
	bool TEST = args.get<bool>("TEST",false);
	
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
	string base = make_string("L=",L,"_t=",t,"_U=",U,"_fp=",fp);
	lout << base << endl;
	lout.set(base+".log",wd+"log");
	
	SaveData data(L);
	
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
	size_t start_2site = args.get<size_t>("start_2site",0ul);
	size_t end_2site = args.get<size_t>("end_2site",6); //GlobParam.max_halfsweeps-3
	size_t period_2site = args.get<size_t>("period_2site",1ul);
	DynParam.iteration = [start_2site,end_2site,period_2site] (size_t i) {return (i>=start_2site and i<=end_2site and i%period_2site==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	
	vector<Param> params;
	params.push_back({"Uph",U});
	params.push_back({"maxPower",2ul});
	
	ArrayXXd tFull = create_1D_PBC(L,t,0.,true);
	ArrayXXcd tFullc = tFull.cast<complex<double> >();
	params.push_back({"tFull",tFullc});
	
	if (L<=20) lout << tFull << endl;
	
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
	
	auto params_kin = params;
	MODEL Hkin(L,params_kin);
	lout << "Hkin=" << Hkin.info() << endl; 
	
	MODEL Htmp(L,params);
	lout << "Htmp:" << Htmp.info() << endl;
	
	Mpo<MODEL::Symmetry,MODEL::Scalar_> Hmpo0 = Htmp;
	Mpo<MODEL::Symmetry,MODEL::Scalar_> Hmpop = Htmp;
	
	if (fp != 0.)
	{
		for (int i=0; i<L; ++i)
		{
			Mpo<MODEL::Symmetry,MODEL::Scalar_> SFp = Htmp.Sp(i); SFp.scale(fp*exp(+1.i*M_PI*phig[i]));
			Mpo<MODEL::Symmetry,MODEL::Scalar_> SFm = Htmp.Sm(i); SFm.scale(fp*exp(-1.i*M_PI*phig[i]));
			Mpo<MODEL::Symmetry,MODEL::Scalar_> Sftot = sum(SFp,SFm);
			
			Hmpo0 = sum(Hmpo0,Sftot);
			
			SFp = Htmp.Sp(i); SFp.scale(fp*exp(+1.i*M_PI*phip[i]));
			SFm = Htmp.Sm(i); SFm.scale(fp*exp(-1.i*M_PI*phip[i]));
			Sftot = sum(SFp,SFm);
			
			Hmpop = sum(Hmpop,Sftot);
		}
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
	
	/////// Ground state calculation ///////
	DMRG.edgeState(H0, g, Q);
	
	data.E = g.energy;
	data.e = data.E/L;
	
	/////// variance ///////
	data.var = abs(avg(g.state, H0, g.state, 2)-pow(g.energy,2))/L;
	lout << endl << "var/L=" << data.var << endl << endl;
	
	/////// Observables ///////
	for (int i=0; i<L; ++i)
	{
		data.nUP(i) = real(avg(g.state, H0.n<UP>(i), g.state));
		data.nDN(i) = real(avg(g.state, H0.n<DN>(i), g.state));
		data.nh(i)  = real(avg(g.state, H0.nh(i), g.state));
		data.Sp(i) = avg(g.state, H0.Sp(i), g.state);
		data.Sm(i) = avg(g.state, H0.Sm(i), g.state);
		data.Sx(i) = real(avg(g.state, H0.Scomp(SX,i), g.state));
		data.Sy(i) = real(-1.i*avg(g.state, H0.Scomp(iSY,i), g.state));
	}
	lout << "nUP=" << data.nUP.transpose() << endl;
	lout << "nDN=" << data.nDN.transpose() << endl;
	lout << "nh=" << data.nh.transpose() << endl;
	lout << "Sp=" << data.Sp.transpose() << endl;
	lout << "Sm=" << data.Sm.transpose() << endl;
	lout << "Sx=" << data.Sx.transpose() << endl;
	lout << "Sy=" << data.Sy.transpose() << endl;
	
	for (int i=0; i!=1; i=get_next(i,L))
	{
		lout << "i=" << i << ", nn<UP>=" << real(avg(g.state, H0.n<UP>(i), H0.n<UP>(get_next(i,L)), g.state)) << ", nn<DN>=" << real(avg(g.state, H0.n<DN>(i), H0.n<DN>(get_next(i,L)), g.state)) << endl;
	}
	
	data.Ekin = real(avg(g.state, Hkin, g.state));
	data.ekin = data.Ekin/L;
	lout << "Ekin=" << data.Ekin << endl;
	lout << endl;
	
	/*for (int l=0; l<L-1; ++l)
	{
		j = next(i,L);\
		k = next(next(i,L),L);
		lout << avg(g.state, H0.n<UP>(l), g.state) << endl;
		lout << avg(g.state, H0.nh(l), g.state) << endl;
		lout << avg(g.state, H0.SpSm(l,j), g.state) << endl;
		lout << avg(g.state, H0.SmSp(l,l+1), g.state) << endl;
		lout << avg(g.state, H0.SzSz(l,l+1), g.state) << endl;
		lout << avg(g.state, H0.nn<UP,DN>(l,l+1), g.state) << endl;
	}*/
	
	/////// Angular momentum ///////
	Mpo<MODEL::Symmetry,MODEL::Scalar_> Ushift_up = Htmp.Identity();
	Mpo<MODEL::Symmetry,MODEL::Scalar_> Ushift_dn = Htmp.Identity();
	//Mpo<MODEL::Symmetry,MODEL::Scalar_> Ushift = Htmp.Identity();
	for (int i=0; i!=1; i=get_next(i,L))
	{
		int j = get_next(i,L);
		Ushift_up = prod(P<UP>(Htmp,i,j),Ushift_up);
		Ushift_dn = prod(P<DN>(Htmp,i,j),Ushift_dn);
		//Ushift = prod(Ushift_up,Ushift_dn);
	}
	lout << Ushift_up.info() << endl;
	lout << Ushift_dn.info() << endl;
	//lout << Ushift.info() << endl;
	
	complex<double> avgUshift = avg(g.state, Ushift_up, Ushift_dn, g.state);
	data.absAvgU = abs(avgUshift);
	data.argAvgU = arg(avgUshift);
	data.j = L/(2*M_PI)*arg(avgUshift);
	lout << termcolor::blue << "<Ushift>: " << "abs=" << data.absAvgU << ", arg=" << data.argAvgU << ", j=" << data.j << termcolor::reset << endl;
	
	/////// Save data ///////
	data.save(make_string("HubbardRing_",base,".h5"));
	
	/////// Test stuff ///////
	if (TEST)
	{
		MatrixXcd Lambda(L,L);
		for (int i=0; i<L; ++i)
		for (int j=0; j<L; ++j)
		{
			Lambda(i,j) = avg(g.state, Htmp.cdagc<UP,UP>(i,j), g.state);
			#if not defined(USING_SU2_COMPLEX)
			Lambda(i,j) += avg(g.state, Htmp.cdagc<UP,UP>(j,i), g.state);
			#endif
		}
		lout << Lambda.real() << endl;
		
		lout << endl;
		
		lout << "pow=1" << endl;
		lout << avg(g.state, cdagc_ij_ji<UP>(Htmp,0,1), g.state) << endl;
		
		lout << "pow=2" << endl;
		lout << avg(g.state, cdagc_ij_ji<UP>(Htmp,0,1), cdagc_ij_ji<UP>(Htmp,0,1), g.state) << endl;
		
		lout << "MPO pow=2" << endl;
		auto Ctmp = cdagc_ij_ji<UP>(Htmp,0,1);
		Ctmp.calc(2);
		lout << avg(g.state, Ctmp, g.state, 2) << endl;
		
		lout << "n-formula:" << endl;
		lout << avg(g.state, Htmp.n<UP>(0), g.state) + avg(g.state, Htmp.n<UP>(1), g.state) -2.*avg(g.state, Htmp.n<UP>(0), Htmp.n<UP>(1), g.state) << endl;
		lout << "ncorr=" << avg(g.state, Htmp.n<UP>(0), Htmp.n<UP>(1), g.state) << endl;
		
		lout << "OxV_exact" << endl;
		MODEL::StateXcd Psitmp;
		OxV_exact(cdagc_ij_ji<UP>(Htmp,0,1), g.state, Psitmp, 2., DMRG::VERBOSITY::SILENT);
		lout << "pow=1: " << dot(g.state,Psitmp) << endl;
		lout << "pow=2: " << dot(Psitmp,Psitmp) << endl;
		
		lout << endl;
		
//		Mpo<MODEL::Symmetry,MODEL::Scalar_> Tforw = Htmp.cdagc<UP,UP>(0,get_next(0,L));
//		for (int i=get_next(0,L); i!=0; i=get_next(i,L))
//		{
//			Tforw = sum(Tforw,Htmp.cdagc<UP,UP>(i,get_next(i,L)));
//		}
//		lout << Tforw.info() << endl;
//		
//		lout << avg(g.state, Ushift_up, g.state) << endl;
//		lout << avg(g.state, Tforw, g.state) << endl;
	}
	/////// Test stuff ///////
	
	if (CALC_QUENCH)
	{
		auto Psi = g.state.cast<complex<double> >();
		Psi.eps_truncWeight = tol_compr;
		TDVPPropagator<MODEL,MODEL::Symmetry,MODEL::Scalar_,complex<double>,Mps<MODEL::Symmetry,complex<double>>> TDVP(Hp,Psi);
		
		double dt = 0.1;
		double tmax = 20.;
		int tpoints = int(tmax/dt+1);
		
		IntervalIterator it(0.,tmax,tpoints);
		MatrixXd nUPdata(tpoints,L+1); nUPdata.setZero();
		MatrixXd nDNdata(tpoints,L+1); nDNdata.setZero();
		
		for (it=it.begin(3); it!=it.end(); ++it)
		{
			double time = *it;
			nUPdata(it.index(),0) = time;
			nDNdata(it.index(),0) = time;
			double E = isReal(avg(Psi, Hp, Psi));
			
			//int l = L/2;
			//int l = 1  ; 
			for (int l=0; l<L; ++l)
			{
				double nUP = isReal(avg(Psi, Hp.n<UP>(l), Psi));
				double nDN = isReal(avg(Psi, Hp.n<DN>(l), Psi));
				lout << "l=" << l << ", nUP=" << nUP << ", nDN=" << nDN << endl;
				//it << nUP, nDN, E;
				nUPdata(it.index(),1+l) = nUP;
				nDNdata(it.index(),1+l) = nDN;
			}
			//it.save("n(t)_l1.dat");
			
			Stopwatch<> Timer;
			TDVP.t_step(Hp, Psi, 1.i*dt);
			lout << Timer.info(make_string("propagated to t=",*it)) << endl;
			lout << TDVP.info() << endl;
			lout << Psi.info() << endl;
			lout << endl;
			
			saveMatrix(nUPdata,make_string("nUP_",base,".dat"));
			saveMatrix(nDNdata,make_string("nDN_",base,".dat"));
		}
	}
}
