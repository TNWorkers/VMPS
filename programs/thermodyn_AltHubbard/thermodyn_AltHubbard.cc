int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t Ly = args.get<size_t>("Ly",1ul);
	assert(Ly==1 and "Please choose Ly=1 for now!");
	int dLphys = (Ly==2)? 1:2;
	size_t L = args.get<size_t>("L",40ul);
	
	//double S = args.get<int>("S",0);
	int N = args.get<int>("N",2*L);
	//double T = (N-2*L)/2;
	double UA = args.get<double>("UA",0.);
	double UB = args.get<double>("UB",8.);
	
	int i0 = args.get<int>("i",L/2);
	int j0 = args.get<int>("j",L/2);
	assert(i0>=0 and i0<L);
	assert(j0>=0 and j0<L);
	bool PBC = args.get<bool>("PBC",false);
	
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N,2*L);
	
	bool SAVE_BETA = args.get<bool>("SAVE_BETA",true);
	bool LOAD_BETA = args.get<bool>("LOAD_BETA",false);
	bool RELOAD = args.get<bool>("RELOAD",false);
	bool CALC_SPEC = args.get<bool>("CALC_SPEC",true);
	size_t maxPower = args.get<size_t>("maxPower",2ul);
	
	int Ntaylor= args.get<double>("Ntaylor",0);
	double dbeta = args.get<double>("dbeta",0.1);
	double beta = args.get<double>("beta",1.);
	double tol_compr_beta = args.get<double>("tol_compr_beta",1e-10);
	double betaswitch = args.get<double>("betaswitch",1e4);
	
	double tol_DeltaS = args.get<double>("tol_DeltaS",1e-3);
	size_t MlimitKet = args.get<size_t>("MlimitKet",1000ul);
	size_t MlimitBra = args.get<size_t>("MlimitBra",1000ul);
	double dt_back = args.get<double>("dt_back",0.1);
	double dt_forw = args.get<double>("dt_forw",0.1);
	double tmax = args.get<double>("tmax",12.);
	lout << "tmax=" << tmax << endl;
	double tol_compr_forw = args.get<double>("tol_compr_forw",1e-5);
	double tol_compr_back = args.get<double>("tol_compr_back",1e-5);
	int Nt;
	if (dt_forw==0)
	{
		Nt = tmax/dt_back+1;
	}
	else if (dt_back==0)
	{
		Nt = tmax/dt_forw+1;
	}
	else
	{
		Nt = tmax/min(dt_forw,dt_back)+1;
	}
	double factor = args.get<double>("factor",+1.);
	
	size_t Mbeta = args.get<size_t>("Mbeta",1000ul);
	size_t Mlimit = args.get<size_t>("Mlimit",1000ul);
	double tol_OxV = 1e-7; // val>1 = do not compress
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	string param_base = make_string("UA=",UA,"_UB=",UB);
	param_base += make_string("_beta=",beta);
	string base = make_string("L=",L,"_Ly=",Ly,"_i=",i0,"_j=",j0,"_PBC=",PBC,"_",param_base);
	string tbase = make_string("tmax=",tmax,"_dt=",dt_forw,",",dt_back,"_tol_DeltaS=",tol_DeltaS,"_tol_compr=",tol_compr_forw,",",tol_compr_back);
	lout.set(base+".log",wd+"log");
	
	lout << base << endl;
	lout << boost::asio::ip::host_name() << endl;
	lout << args.info() << endl;
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	// basic params for Ly=2 (anicllas on same site)
	ArrayXXd tFull = (PBC)? create_1D_PBC(L,1.,0.,false) : create_1D_OBC(L,1.,0.); // COMPRESSED=false
	vector<SUB_LATTICE> G;
	for (int l=0; l<L; l+=2)
	{
		G.push_back(static_cast<SUB_LATTICE>(1));
		G.push_back(static_cast<SUB_LATTICE>(-1));
	}
	lout << "Original sublattice structure:" << endl;
	for (size_t l=0; l<L; ++l) {lout << G[l];}
	lout << endl;
	sublattice_check(tFull,G);
	
	vector<size_t> transform(L);
	for (int i=0; i<transform.size(); ++i) transform[i] = i;
	CuthillMcKeeCompressor<double> CMK;
	if (PBC)
	{
		CMK = CuthillMcKeeCompressor(tFull,true);
		CMK.apply_compression(tFull);
		CMK.apply_compression(G);
		transform = CMK.get_transform();
	}
	sublattice_check(tFull,G);
	
	bool PRINT_FREE = args.get<bool>("PRINT_FREE",false);
	if (PRINT_FREE)
	{
		SelfAdjointEigenSolver<MatrixXd> Eugen(-1.*tFull.matrix());
		lout << Eugen.eigenvalues().transpose() << endl;
		VectorXd occ = Eugen.eigenvalues().head(L/2);
		VectorXd unocc = Eugen.eigenvalues().tail(L-L/2);
		lout << "orbital energies occupied:" << endl << occ.transpose()  << endl;
		lout << "orbital energies unoccupied:" << endl << unocc.transpose()  << endl << endl;
		double E0 = 2.*occ.sum();
		lout << setprecision(16) << "non-interacting fermions: E0=" << E0 << ", E0/L=" << E0/(L) << setprecision(6) << endl << endl;
	}
	
	lout << "old indices: i=" << i0 << ", j=" << j0 << endl;
	lout << "new indices: i=" << transform[i0]*dLphys << ", j=" << transform[j0]*dLphys << endl;
	
	// basic params extended to thermal for Ly=1 (physical sites followed by ancillas)
	ArrayXXd tFullExt_beta = extend_to_thermal(tFull,0.);
	ArrayXXd tFullExt_beta_t = extend_to_thermal(tFull,factor);
	vector<SUB_LATTICE> Gext;
	for (int l=0; l<L; ++l)
	{
		Gext.push_back(G[l]);
		Gext.push_back(flip_sublattice(G[l]));
		//lout << "l=" << l << ", " << G[l] << ", " << flip_sublattice(G[l]) << endl;
	}
	lout << "Extended sublattice structure:" << endl;
	for (size_t l=0; l<2*L; ++l) {lout << Gext[l];}
	lout << endl;
	sublattice_check(tFullExt_beta,Gext);
	sublattice_check(tFullExt_beta_t,Gext);
	
	//lout << tFullExt_beta_t << endl;
	
	// params for infinite temperature
	vector<Param> params_Tinf;
	params_Tinf.push_back({"Ly",Ly});
	if (Ly==1)
	{
		ArrayXXd tFullInf(2*L,2*L); tFullInf = 0.;
		for (int l=0; l<2*L; l+=2)
		{
			tFullInf(l,l+1) = 1.;
			tFullInf(l+1,l) = 1.;
		}
		params_Tinf.push_back({"tFull",tFullInf});
		params_Tinf.push_back({"G",Gext});
		sublattice_check(tFullInf,Gext);
	}
	else
	{
		params_Tinf.push_back({"tRung",1.});
		params_Tinf.push_back({"G",G});
	}
	params_Tinf.push_back({"t",0.});
	params_Tinf.push_back({"maxPower",1ul});
	
	// params for finite temperature
	vector<Param> params_Tfin;
	params_Tfin.push_back({"Ly",Ly});
	if (Ly==1)
	{
		params_Tfin.push_back({"tFull",tFullExt_beta});
		
		for (int l=0; l<L; ++l)
		{
			double Uval = (l%2==0)? UA:UB;
			params_Tfin.push_back({"Uph",Uval,transform[l]*dLphys});
			params_Tfin.push_back({"Uph",0.,transform[l]*dLphys+1});
			lout << "l=" << l << ", Uval=" << Uval << ", l+1=" << l+1 << ", Uval=0" << endl;
		}
		params_Tfin.push_back({"G",Gext});
	}
	else
	{
		params_Tfin.push_back({"tFull",tFull});
		
		ArrayXd UAorb(2); UAorb << UA, 0.;
		ArrayXd UBorb(2); UBorb << UB, 0.;
		for (int l=0; l<L; ++l)
		{
			ArrayXd Uval = (l%2==0)? UAorb:UBorb;
			params_Tfin.push_back({"Uphorb",Uval,transform[l]*dLphys});
		}
		
		params_Tfin.push_back({"G",G});
	}
	params_Tfin.push_back({"t",0.});
	params_Tfin.push_back({"tRung",0.});
	params_Tfin.push_back({"maxPower",maxPower});
	
	// params for time propagation
	vector<Param> pparams;
	pparams.push_back({"Ly",Ly});
	if (Ly==1)
	{
		pparams.push_back({"tFull",tFullExt_beta_t});
		
		for (int l=0; l<L; ++l)
		{
			double Uval = (l%2==0)? UA:UB;
			pparams.push_back({"Uph",+Uval,transform[l]*dLphys});
			pparams.push_back({"Uph",-Uval,transform[l]*dLphys+1});
			lout << "l=" << l << ", Uval=" << Uval << ", l+1=" << l+1 << ", Uval=" << -Uval << endl;
		}
		
		pparams.push_back({"G",Gext});
	}
	else
	{
		pparams.push_back({"tFull",tFull});
		ArrayXXd tFullA = factor*tFull;
		pparams.push_back({"tFullA",tFullA});
		pparams.push_back({"G",G});
		
		ArrayXd UAorb(2); UAorb << UA, -UA;
		ArrayXd UBorb(2); UBorb << UB, -UB;
		for (int l=0; l<L; ++l)
		{
			ArrayXd Uval = (l%2==0)? UAorb:UBorb;
			pparams.push_back({"Uphorb",Uval,transform[l]*dLphys});
		}
		
		pparams.push_back({"G",G});
	}
	pparams.push_back({"t",0.});
	pparams.push_back({"tRung",0.});
	pparams.push_back({"maxPower",maxPower});
	
	// Aufbau des Modells bei β=0
	//MODEL H_Tinf(dLphys*L, Tinf_params_fermions(L,Ly,1ul)); //maxPower=1
	MODEL H_Tinf(dLphys*L, params_Tinf);
	lout << endl << "β=0 Entangler " << H_Tinf.info() << endl;
	
	// Modell fuer die β-Propagation
	MODEL H_Tfin(dLphys*L, params_Tfin); H_Tfin.precalc_TwoSiteData();
	lout << endl << "physical Hamiltonian " << H_Tfin.info() << endl << endl;
	
	// Modell fuer die t-propagation
	MODEL Hp(dLphys*L, pparams); Hp.precalc_TwoSiteData();
	lout << endl << "propagation Hamiltonian " << Hp.info() << endl << endl;
	
	SpectralManager<MODEL> SpecMan({"SSF"},Hp,VERB);
	SpecMan.beta_propagation<MODEL>(H_Tfin, H_Tinf, 1, dLphys, beta, dbeta, tol_compr_beta, Mbeta, Q, log(4), betaswitch, "thermodyn", base+make_string("_Mlimit=",Mbeta), LOAD_BETA, SAVE_BETA, VERB, {}, {}, Ntaylor);
	
	MODEL::StateXcd PhiT = SpecMan.get_PhiT().cast<complex<double> >();
	
	bool TEST_EIGEN = args.get<bool>("TEST_EIGEN",false);
	if (TEST_EIGEN)
	{
		// test that PhiT is an eigenstate of Hp:
		auto avgHp = avg(PhiT, Hp, PhiT);
		auto avgHpHp = (maxPower>=2)? avg(PhiT,Hp,PhiT,2) : avg(PhiT,Hp,Hp,PhiT);
		double test = abs(avgHpHp-avgHp);
		if (test < 1e-4)
		{
			lout << termcolor::green;
		}
		else
		{
			lout << termcolor::red;
		}
		lout << "<Hp*Hp>=" << avgHpHp << endl;
		lout << "<Hp>=" << avgHp << ", <Hp>^2=" << avgHp*avgHp << endl;
		lout << "eigenstate test for PhiT: var=<Hp*Hp>-<Hp>^2=" << abs(avgHpHp-avgHp*avgHp) << termcolor::reset << endl;
		lout << endl;
	}
	
	MODEL::StateXcd PsiKet, PsiBra;
	OxV_exact(Hp.S(transform[i0]*dLphys,0), PhiT, PsiBra, tol_OxV);
	OxV_exact(Hp.S(transform[j0]*dLphys,0), PhiT, PsiKet, tol_OxV);
	double dagfac = sqrt(3.);
	
	PsiKet.eps_truncWeight = tol_compr_forw;
	PsiKet.min_Nsv = PsiKet.calc_Mmax();
	int MstartKet = PsiKet.calc_Mmax();
	PsiKet.max_Nsv = MlimitKet;
	
	PsiBra.eps_truncWeight = tol_compr_forw;
	PsiBra.min_Nsv = PsiBra.calc_Mmax();
	int MstartBra = PsiBra.calc_Mmax();
	PsiBra.max_Nsv = MlimitBra;
	
	MODEL::StateXcd PsiBra_prev = PsiBra;
	
	lout << PsiKet.info() << endl;
	lout << PsiBra.info() << endl;
	lout << endl;
	
	MatrixXd Moverlap(0,3);
	double time = 0.;
	double time_prev = 0.;
	
	TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,MODEL::StateXcd> TDVPket(Hp,PsiKet);
	TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,MODEL::StateXcd> TDVPbra(Hp,PsiBra);
	
	EntropyObserver<MODEL::StateXcd> SobsKet(Hp.length(), Nt, VERB, tol_DeltaS);
	EntropyObserver<MODEL::StateXcd> SobsBra(Hp.length(), Nt, VERB, tol_DeltaS);
	vector<bool> TWO_SITE_KET = SobsKet.TWO_SITE(VERB, PsiKet, 1.);
	vector<bool> TWO_SITE_BRA = SobsBra.TWO_SITE(VERB, PsiBra, 1.);
	
	int itKet = 0;
	int itBra = 0;
	Stopwatch<> TpropTimer;
	string filename = make_string("SSF_",base,"_",tbase,".dat");
	
	while (time < tmax)
	{
		// propagate forwards
		lout << "t=" << time << ", it=" << itKet << ", " << itBra << endl;
		complex<double> res, res_prev;
		
		if (itKet==0)
		{
			Moverlap.conservativeResize(Moverlap.rows()+1,Moverlap.cols());
			
			Moverlap(Moverlap.rows()-1,0) = time;
			res = dagfac*dot(PsiBra,PsiKet);
			
			Moverlap(Moverlap.rows()-1,1) = res.real();
			Moverlap(Moverlap.rows()-1,2) = res.imag();
		}
		else
		{
			Moverlap.conservativeResize(Moverlap.rows()+2,Moverlap.cols());
			
			Moverlap(Moverlap.rows()-2,0) = time_prev;
			Moverlap(Moverlap.rows()-1,0) = time;
			
			res_prev = dagfac*dot(PsiBra_prev,PsiKet);
			res = dagfac*dot(PsiBra,PsiKet);
			
			Moverlap(Moverlap.rows()-2,1) = res_prev.real();
			Moverlap(Moverlap.rows()-2,2) = res_prev.imag();
			
			Moverlap(Moverlap.rows()-1,1) = res.real();
			Moverlap(Moverlap.rows()-1,2) = res.imag();
			
			lout << "save results at t=" << time_prev << ", res=" << res_prev << endl;
		}
		lout << "save results at t=" << time << ", res=" << res << endl;
		lout << "file=" << wd+filename << endl;
		saveMatrix(Moverlap,make_string(wd,filename));
		
		Stopwatch<> StepTimer;
		
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				// propagate forward
				//-----------------------------------------------------------
				if (tol_DeltaS == 0.)
				{
					TDVPket.t_step(Hp, PsiKet, -1.i*dt_forw, 1);
				}
				else
				{
					TDVPket.t_step_adaptive(Hp, PsiKet, -1.i*dt_forw, TWO_SITE_KET, 1);
				}
				//-----------------------------------------------------------
				
				if (tol_DeltaS > 0.)
				{
					auto PsiTmp = PsiKet; PsiTmp.entropy_skim();
					lout << "ket: ";
					TWO_SITE_KET = SobsKet.TWO_SITE(itKet, PsiTmp);
				}
			}
			#pragma omp section
			{
				// propagate back
				PsiBra_prev = PsiBra;
				//-----------------------------------------------------------
				if (tol_DeltaS == 0.)
				{
					TDVPbra.t_step(Hp, PsiBra, +1.i*dt_back, 1);
				}
				else
				{
					TDVPbra.t_step_adaptive(Hp, PsiBra, +1.i*dt_back, TWO_SITE_BRA, 1);
				}
				//-----------------------------------------------------------
				
				if (tol_DeltaS > 0.)
				{
					auto PsiTmp = PsiBra; PsiTmp.entropy_skim();
					lout << "bra: ";
					TWO_SITE_BRA = SobsBra.TWO_SITE(itBra, PsiTmp);
				}
			}
		}
		
		itKet += 1;
		itBra += 1;
		time += dt_forw+dt_back;
		time_prev = time-dt_back;
		
		lout << StepTimer.info("time step") << endl;
		lout << "ket: " << TDVPket.info() << endl;
		lout << "ket: " << PsiKet.info() << endl;
		lout << "bra: " << TDVPbra.info() << endl;
		lout << "bra: " << PsiBra.info() << endl;
		
		lout << TpropTimer.info("total time",false) << endl;
		lout << endl;
	}
}
