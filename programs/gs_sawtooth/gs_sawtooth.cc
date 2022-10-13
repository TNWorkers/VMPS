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
	double overlap;
	double fidelity;
	double dJ;
	VectorXd E_excited, var_excited;
	
	void save (string label)
	{
		HDF5Interface target(label+".h5",WRITE);
		target.save_vector(Savg,"Savg","");
		target.save_matrix(SdagS,"SdagS","");
		target.save_scalar(E,"E","");
		target.save_scalar(e,"e","");
		target.save_scalar(var,"var","");
		target.save_scalar(SdagStot,"SdagStot","");
		target.save_scalar(overlap,"overlap","");
		target.save_scalar(fidelity,"fidelity","");
		target.save_scalar(dJ,"dJ","");
		if (E_excited.rows() > 0)
		{
			target.save_vector(E_excited,"E_excited","");
			target.save_vector(var_excited,"var_excited","");
		}
		target.close();
	}
};

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	int L = args.get<int>("L",100);
	
	int Slimit = args.get<int>("Slimit",L/2);
	#if defined(USING_SU2)
	lout << "initializing CGC tables for Slimit=" << Slimit << "..." << endl;
	Sym::initialize(Slimit);
	#endif
	
	int S = args.get<int>("S",0);
	int M = args.get<int>("M",0);
	#ifdef USING_U1
	qarray<MODEL::Symmetry::Nq> Q = {M};
	#elif defined(USING_U0)
	qarray<MODEL::Symmetry::Nq> Q = {};
	#elif defined(USING_SU2)
	qarray<MODEL::Symmetry::Nq> Q = {2*S+1};
	#endif
	lout << "Q=" << Q << endl;
	
	double JA = args.get<double>("JA",-1.);
	double JB = args.get<double>("JB",-1.);
	double JpA = args.get<double>("JpA",0.);
	double JpB = args.get<double>("JpB",0.);
	
	double J = args.get<double>("J",JA);
	double Jprime = args.get<double>("Jprime",JpA);
	
	double R = args.get<double>("R",0.);
	double Bz = args.get<double>("Bz",0.);
	double Bzstag = args.get<double>("Bzstag",0.);
	double Kz = args.get<double>("Kz",0.);
	double Bx = args.get<double>("Bx",0.);
	size_t D = args.get<size_t>("D",2ul); // 3ul
	size_t maxPower = args.get<size_t>("maxPower",2ul);
	bool PBC = args.get<bool>("PBC",true);
	bool SOFT_EDGE = args.get<bool>("SOFT_EDGE",false);
	bool COMPRESS = args.get<bool>("COMPRESS",true);
	int betaNum = args.get<int>("betaNum",0); //-1
	int betaDen = args.get<int>("betaDen",3);
	int Neigen = args.get<int>("Neigen",1);
	bool CALC_VAR = args.get<bool>("CALC_VAR",true);
	bool CALC_CORR = args.get<bool>("CALC_CORR",false);
	bool CALC_STOT = args.get<bool>("CALC_STOT",true);
	bool CALC_AVG = args.get<bool>("CALC_AVG",false);
	bool CALC_GS = args.get<bool>("CALC_GS",true);
	
	bool CALC_FIDELITY = args.get<bool>("CALC_FIDELITY",false);
	double dJ = args.get<double>("dJ",0.01);
	
	string LOAD = args.get<string>("LOAD","");
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	//string base = make_string("J=",J,"_beta=",betaNum,"|",betaDen,"_D=",D,"_S=",S,"_SOFT=",SOFT_EDGE);
	string base = make_string("L=",L,"_J=",JA,",",JB,"_Jprime=",JpA,",",JpB,"_D=",D);
	#if defined(USING_SU2)
	base += make_string("_S=",S);
	#endif
	#if defined(USING_U1)
	base += make_string("_M=",M);
	#endif
	#if defined(USING_U1) or defined(USING_U0)
	base += make_string("_Bz=",Bz);
	#endif
	#if defined(USING_U0)
	base += make_string("_Bx=",Bx);
	#endif
	if (PBC)
	{
		base += "_PBC=1";
	}
	lout << base << endl;
	lout.set(base+".log",wd+"log");
	
	lout << args.info() << endl;
	#ifdef _OPENMP
	omp_set_nested(1);
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.Minit = args.get<size_t>("Minit",20ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",20ul);
	GlobParam.Mlimit = args.get<size_t>("Mlimit",1000ul);
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",36ul);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",40ul);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-12);
	GlobParam.tol_state = args.get<double>("tol_state",1e-10);
	GlobParam.savePeriod = args.get<size_t>("savePeriod",0);
	GlobParam.CALC_S_ON_EXIT = false;
	
	DMRG::CONTROL::DYN  DynParam;
	double eps_truncWeight = args.get<double>("eps_truncWeight",0);
	DynParam.eps_truncWeight = [eps_truncWeight] (size_t i) {return eps_truncWeight;};
	
	size_t Mincr_per = args.get<size_t>("Mincr_per",2ul);
	DynParam.Mincr_per = [Mincr_per,LOAD] (size_t i) {return (i==0 and LOAD!="")? 0:Mincr_per;}; // if LOAD, resize before first step
	
	size_t Mincr_abs = args.get<size_t>("Mincr_abs",60ul);
	DynParam.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
	
	int max_Nrich = args.get<int>("max_Nrich",-1);
	DynParam.max_Nrich = [max_Nrich] (size_t i) {return max_Nrich;};
	
	size_t start_2site = args.get<size_t>("start_2site",0ul);
	size_t end_2site = args.get<size_t>("end_2site",0); //GlobParam.max_halfsweeps-3
	size_t period_2site = args.get<size_t>("period_2site",2ul);
	DynParam.iteration = [start_2site,end_2site,period_2site] (size_t i) {return (i>=start_2site and i<=end_2site and i%period_2site==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	
	// alpha
	size_t start_alpha = args.get<size_t>("start_alpha",0);
	size_t end_alpha = args.get<size_t>("end_alpha",GlobParam.max_halfsweeps-3);
	double alpha = args.get<double>("alpha",100.);
	DynParam.max_alpha_rsvd = [start_alpha, end_alpha, alpha] (size_t i) {return (i>=start_alpha and i<end_alpha)? alpha:0.;};
	
	vector<Param> params;
	params.push_back({"D",D});
	params.push_back({"maxPower",maxPower});
	
	vector<int> Llist;
	string Linfo;
	if (L==-1)
	{
		Llist = args.get_list<int>("Llist",{20, 40, 60, 80, 100, 200});
		Linfo = str(Llist);
	}
	else
	{
		Llist = {L};
		Linfo = make_string("L=",L);
	}
	
	SaveData data;
	
	// OBC
	if (!PBC)
	{
		//vector<tuple<int,double,double>> res;
		
		for (const auto &L:Llist)
		{
			vector<Param> paramsOBC;
			if (SOFT_EDGE)
			{
				for (size_t i=0; i<L; ++i)
				{
					if (i==0 or i==L-1)
					{
						paramsOBC.push_back({"D",2ul,i});
					}
					else
					{
						paramsOBC.push_back({"D",D,i});
					}
				}
			}
			else
			{
				paramsOBC.push_back({"D",D});
			}
			paramsOBC.push_back({"maxPower",maxPower});
			cout << "maxPower=" << maxPower << endl;
			auto [J, R, offset] = params_bilineraBiquadratic_beta(boost::rational<int>(betaNum,betaDen));
			lout << "J=" << J << ", R=" << R << ", loc.offset=" << offset << endl;
			for (size_t l=0; l<L; l+=2)
			{
				paramsOBC.push_back({"J",JA,l});
				paramsOBC.push_back({"J",JB,l+1});
				paramsOBC.push_back({"Jprime",JpA,l});
				paramsOBC.push_back({"Jprime",JpB,l+1});
			}
			paramsOBC.push_back({"R",R});
			
			for (size_t l=0; l<L; l+=2)
			{
				paramsOBC.push_back({"Bz",Bz+Bzstag,l});
				paramsOBC.push_back({"Bz",Bz-Bzstag,l+1});
			}
			paramsOBC.push_back({"Kz",Kz});
			
			MODEL H(L,paramsOBC);
			if (offset != 0.)
			{
				H.scale(1.,offset*(L-1));
			}
			lout << H.info() << endl;
			
			Eigenstate<MODEL::StateXd> g;
			if (LOAD!="")
			{
				g.state.load(LOAD,g.energy);
			}
			if (CALC_GS)
			{
				MODEL::Solver DMRG(VERB);
				DMRG.GlobParam = GlobParam;
				DMRG.DynParam = DynParam;
				DMRG.userSetGlobParam();
				DMRG.userSetDynParam();
				DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
				g.state.save(make_string(wd,"state/gs_",base,"_Mmax=",GlobParam.Mlimit),"",g.energy);
				data.E = g.energy;
				data.e = g.energy/L;
			}
			if (CALC_VAR and offset==0.)
			{
				double var = abs(avg(g.state,H,g.state,2)-pow(g.energy,2))/L;
				data.var = var;
				lout << "varE=" << var << endl;
			}
			if (CALC_AVG)
			{
				data.Savg.resize(L);
				#if defined(USING_SU2)
				for (int l=0; l<L; ++l)
				{
					double res = (S==0)? 0. : avg(g.state, H.S(l), g.state) * S/sqrt(S*(S+1.));
					data.Savg(l) = res;
					lout << l << "\t" << res << endl;
				}
//				for (int l=0; l<2; ++l)
//				for (int m=S; m>=0; --m)
//				{
//					double 3j = coupl_3j_base(2*S+1, 3, 2*S+1, 2*m, 0, -2*m);
//					double cgc = 3j * sqrt(2*S+1) * pow(-1,S-1+m);
//					double res = (S==0)? 0. : avg(g.state, H.S(l), g.state) * cgc; //* S/sqrt(S*(S+1.));
//					lout << "l=" << l << ", M=" << m << ", res=" << res << ", cgc=" << cgc << ", 3j=" << 3j << endl;
//				}
				#else
				for (int l=0; l<L; ++l)
				{
					double resx = 0.;
					#if defined(USING_U0)
					resx = avg(g.state, H.Scomp(SX,l), g.state);
					#endif
					double resz = avg(g.state, H.Sz(l), g.state);
					double res = sqrt(resx*resx+resz*resz);
					data.Savg(l) = res;
					lout << l << "\t" << res << endl;
				}
				#endif
			}
			if (CALC_CORR)
			{
				data.SdagS.resize(L,L);
				data.SdagS.setZero();
				#pragma omp parallel for schedule(dynamic)
				for (int i=0; i<L; ++i)
				for (int j=0; j<i; ++j)
				{
					#if defined(USE_WIG_SU2_COEFFS)
					wig_thread_temp_init(2*Slimit);
					#endif
					
					double res = avg(g.state, H.SdagS(i,j), g.state);
					data.SdagS(i,j) = res;
					data.SdagS(j,i) = res;
				}
				for (int i=0; i<L; ++i)
				{
					data.SdagS(i,i) = 0.5*(D-1) * 0.5*(D+1);
				}
				
				double SdagStot = data.SdagS.sum();
				lout << "SdagS=" << SdagStot << endl;
				data.SdagStot = SdagStot;
			}
			
			g.state.entropy_skim();
			lout << g.state.entropy().transpose() << endl;
			lout << "entanglementSpectrum:" << endl;
			ArrayXd Sspec = g.state.entanglementSpectrumLoc(L/2).second;
			std::sort(Sspec.data(), Sspec.data()+Sspec.size(), std::greater<double>());
			lout << Sspec.head(min(int(Sspec.rows()),30)) << endl;
			// excited states
			if (Neigen>1)
			{
				vector<Eigenstate<MODEL::StateXd>> spectrum(Neigen);
				spectrum[0] = g;
				vector<MODEL::Solver> DMRGex(Neigen);
				
				for (int i=1; i<Neigen; ++i)
				{
					cout << "i=" << i << endl;
					DMRGex[i] = MODEL::Solver(VERB);
					DMRGex[i].GlobParam = GlobParam;
					DMRGex[i].DynParam = DynParam;
					DMRGex[i].userSetGlobParam();
					DMRGex[i].userSetDynParam();
					for (int j=0; j<i; ++j)
					{
						DMRGex[i].push_back(spectrum[j].state);
					}
					DMRGex[i].edgeState(H, spectrum[i], Q, LANCZOS::EDGE::GROUND);
					lout << "SdagS=" << calc_Stot(H,spectrum[i].state) << endl;
				}
				lout << endl;
			}
		}
	}
	else // PBC
	{
		for (const auto &L:Llist)
		{
			vector<Param> paramsPBC = params;
			
			ArrayXXd Jfull_uncompressed = create_1D_PBC_AB(L,JA,JB,JpA,JpB,false);
			CuthillMcKeeCompressor CMK(Jfull_uncompressed,false); // PRINT=false
			
			ArrayXXd Jfull = create_1D_PBC_AB(L,JA,JB,JpA,JpB,COMPRESS);
			paramsPBC.push_back({"Jfull",Jfull});
			#if defined(USING_U1) or defined(USING_U0)
			paramsPBC.push_back({"Bz",Bz});
			#endif
			
			MODEL H(L,paramsPBC);
			lout << H.info() << endl;
			
			Eigenstate<MODEL::StateXd> g;
			if (LOAD!="")
			{
				g.state.load(LOAD,g.energy);
				lout << g.state.info() << endl;
				double avgH = avg(g.state,H,g.state);
				double avgHsq = avg(g.state,H,g.state);
				data.E = avgH;
				data.e = avgH/L;
				data.var = abs(avgHsq-pow(avgH,2))/L;
			}
			if (CALC_GS)
			{
				MODEL::Solver DMRG(VERB);
				DMRG.GlobParam = GlobParam;
				DMRG.userSetGlobParam();
				DMRG.DynParam = DynParam;
				DMRG.userSetGlobParam();
				DMRG.userSetDynParam();
				DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
				g.state.save(make_string(wd,"state/gs_",base,"_Mmax=",GlobParam.Mlimit),"",g.energy);
				data.E = g.energy;
				data.e = g.energy/L;
				
				double SdagS1 = avg(g.state, H.SdagS(0,1), g.state);
				double SdagS2 = avg(g.state, H.SdagS(1,2), g.state);
				lout << termcolor::blue << "dimer=" << abs(SdagS1-SdagS2) << termcolor::reset << endl;
			}
			if (CALC_FIDELITY)
			{
				vector<Param> paramsPBCf = params;
				ArrayXXd Jfullf = create_1D_PBC_AB(L,J+dJ,J+dJ,JpA,0.,COMPRESS);
				paramsPBCf.push_back({"Jfull",Jfullf});
				
				MODEL Hf(L,paramsPBCf);
				lout << Hf.info() << endl;
				Eigenstate<MODEL::StateXd> gf;
				
				MODEL::Solver DMRGf(VERB);
				DMRGf.GlobParam = GlobParam;
				DMRGf.userSetGlobParam();
				DMRGf.DynParam = DynParam;
				DMRGf.userSetGlobParam();
				DMRGf.userSetDynParam();
				DMRGf.edgeState(Hf, gf, Q, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
				
				data.overlap = dot(g.state, gf.state);
				data.dJ = dJ;
				data.fidelity = abs(1.-abs(gf.state.dot(g.state)))/pow(dJ,2.);
				lout << "overlap=" << data.overlap << ", fidelity=" << data.fidelity << endl;
			}
			if (CALC_AVG)
			{
				vector<tuple<int,double>> Savg;
				for (int l=0; l<L; ++l)
				{
					int lorig = CMK.get_inverse()[l];
					double res = 0.;
					double resx = 0.;
					double resz = 0.;
					#ifdef USING_SU2
					res = (S==0)? 0. : avg(g.state, H.S(l), g.state) * S/sqrt(S*(S+1.));
					#elif defined(USING_U0)
					resx = avg(g.state, H.Scomp(SX,l), g.state);
					resz = avg(g.state, H.Scomp(SZ,l), g.state);
					res = sqrt(resx*resx+resz*resz);
					#elif defined(USING_U1)
					resz = avg(g.state, H.Scomp(SZ,l), g.state);
					res = resz;
					#endif
					if (COMPRESS)
					{
						Savg.push_back(make_tuple(lorig,res));
					}
					else
					{
						Savg.push_back(make_tuple(l,res));
					}
				}
				std::sort(Savg.begin(), Savg.end());
				data.Savg.resize(L);
				
				lout << "sorted:" << endl;
				for (int l=0; l<Savg.size(); ++l)
				{
					data.Savg(l) = get<1>(Savg[l]);
					lout << get<0>(Savg[l]) << "\t" << get<1>(Savg[l]) << endl;
				}
			}
			if (CALC_VAR)
			{
				double var = abs(avg(g.state,H,g.state,2)-pow(g.energy,2))/L;
				lout << "varE=" << var << endl;
				data.var = var;
			}
			
			if (CALC_CORR)
			{
				data.SdagS.resize(L,L);
				data.SdagS.setZero();
				#pragma omp parallel for schedule(dynamic)
				for (int i=0; i<L; ++i)
				for (int j=0; j<i; ++j)
				{
					#if defined(USE_WIG_SU2_COEFFS)
					wig_thread_temp_init(2*Slimit);
					#endif
					
					int inew = CMK.get_transform()[i];
					int jnew = CMK.get_transform()[j];
					double res = avg(g.state, H.SdagS(inew,jnew), g.state);
//					#ifdef USING_SU2
//					double avgi = avg(g.state, H.S(inew), g.state) * S/sqrt(S*(S+1.));
//					double avgj = avg(g.state, H.S(jnew), g.state) * S/sqrt(S*(S+1.));
//					#else
//					double avgi = avg(g.state, H.Scomp(SZ,inew), g.state);
//					double avgj = avg(g.state, H.Scomp(SZ,jnew), g.state);
//					#endif
					data.SdagS(i,j) = res;// - avgi*avgj;
					data.SdagS(j,i) = res;// - avgi*avgj;
//					#pragma omp critical
//					{
//						lout << "i=" << i << ", j=" << j << ", SdagS=" << res << endl;
//					}
				}
				for (int i=0; i<L; ++i)
				{
					data.SdagS(i,i) = 0.5*(D-1) * 0.5*(D+1);
				}
				
				//lout << data.SdagS.col(0) << endl;
				double SdagStot = data.SdagS.sum();
				lout << "SdagStot=" << SdagStot << " => Stot=" << -0.5+sqrt(0.25+SdagStot) << endl;
				data.SdagStot = SdagStot;
			}
			#if defined(USING_U1) or defined(USING_U0)
			if (CALC_STOT)
			{
				double Stot_pm = avg(g.state, H.Scomptot(SP), H.Scomptot(SM), g.state);
				double Stot_mp = avg(g.state, H.Scomptot(SM), H.Scomptot(SP), g.state);
				double Stot_zz = avg(g.state, H.Scomptot(SZ), H.Scomptot(SZ), g.state);
				
				double SdagStot = 0.5*(Stot_pm+Stot_mp)+Stot_zz;
				lout << "SdagStot=" << SdagStot << " => Stot=" << -0.5+sqrt(0.25+SdagStot) << endl;
			}
			#endif
			
			// excited states
			if (Neigen>1)
			{
				vector<Eigenstate<MODEL::StateXd>> spectrum(Neigen);
				spectrum[0] = g;
				vector<MODEL::Solver> DMRGex(Neigen);
				
				data.E_excited.resize(Neigen-1);
				data.var_excited.resize(Neigen-1);
				
				for (int i=1; i<Neigen; ++i)
				{
					cout << "i=" << i << endl;
					DMRGex[i] = MODEL::Solver(VERB);
					DMRGex[i].GlobParam = GlobParam;
					DMRGex[i].DynParam = DynParam;
					DMRGex[i].userSetGlobParam();
					DMRGex[i].userSetDynParam();
					for (int j=0; j<i; ++j)
					{
						DMRGex[i].push_back(spectrum[j].state);
					}
					DMRGex[i].edgeState(H, spectrum[i], Q, LANCZOS::EDGE::GROUND);
					
					double var = abs(avg(spectrum[i].state,H,spectrum[i].state,2)-pow(spectrum[i].energy,2))/L;
					
					data.E_excited(i-1) = spectrum[i].energy;
					data.var_excited(i-1) = var;
					
					lout << "i=" << i << ", E=" << data.E_excited(i-1) << ", E/L=" << data.E_excited(i-1)/L << ", var=" << var << endl;
				}
				lout << endl;
			}
		}
	}
	
	data.save(make_string(wd,"obs/",base,"_Mlimit=",GlobParam.Mlimit));
	lout << "saved to: " << make_string(wd,base,"_Mlimit=",GlobParam.Mlimit) << endl;
	
}
