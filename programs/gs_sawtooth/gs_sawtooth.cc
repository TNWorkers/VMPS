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
	VectorXd E_excited;
	
	void save (string label)
	{
		HDF5Interface target(label+".h5",WRITE);
		target.save_vector(Savg,"Savg","");
		target.save_matrix(SdagS,"SdagS","");
		target.save_scalar(E,"E","");
		target.save_scalar(e,"e","");
		target.save_scalar(var,"var","");
		target.save_scalar(SdagStot,"SdagStot","");
		if (E_excited.rows() > 0)
		{
			target.save_vector(E_excited,"E_excited","");
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
	bool CALC_AVG = args.get<bool>("CALC_AVG",false);
	
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
	GlobParam.Minit = args.get<size_t>("Minit",10ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam.Mlimit = args.get<size_t>("Mlimit",1000ul);
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",36ul);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",40ul);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-12);
	GlobParam.tol_state = args.get<double>("tol_state",1e-10);
	GlobParam.CALC_S_ON_EXIT = false;
	
	DMRG::CONTROL::DYN  DynParam;
	double eps_svd = args.get<double>("eps_svd",1e-10);
	DynParam.eps_svd = [eps_svd] (size_t i) {return eps_svd;};
	
	size_t Mincr_per = args.get<size_t>("Mincr_per",2ul);
	DynParam.Mincr_per = [Mincr_per,LOAD] (size_t i) {return (i==0 and LOAD!="")? 0:Mincr_per;}; // if LOAD, resize before first step
	
	size_t Mincr_abs = args.get<size_t>("Mincr_abs",60ul);
	DynParam.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
	
	//size_t start_2site = args.get<size_t>("start_2site",0ul);
	//size_t end_2site = args.get<size_t>("end_2site",20ul);
	//DynParam.iteration = [start_2site,end_2site] (size_t i) {return (i>=start_2site and i<=end_2site and i%2==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	
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
			MODEL::Solver DMRG(VERB);
			DMRG.GlobParam = GlobParam;
			DMRG.DynParam = DynParam;
			DMRG.userSetGlobParam();
			DMRG.userSetDynParam();
			DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
			g.state.save(make_string(wd,"state/gs_",base,"_Mmax=",g.state.calc_Mmax()),"",g.energy);
			data.E = g.energy;
			data.e = g.energy/L;
			
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
			
//			ofstream MagFiler(make_string("avgS","_L=",L,"_",base,".dat"));
//			double SzSum = 0.;
//			
//			#ifdef USING_U1
//			for (int l=0; l<L; ++l)
//			{
//				double avgSz = avg(g.state, H.Sz(l), g.state);
//				SzSum += avgSz;
////				double avgSx = avg(g.state, H.Scomp(SX,l), g.state);
//				lout << "l=" << l << ", <Sz>=" << avgSz << ", sum=" << SzSum << endl; // << ", <Sx>=" << avgSx
//				MagFiler << l << "\t" << avgSz << "\t" << SzSum << endl; //  << "\t" << avgSx
//			}
//			lout << "<Sz(0)*Sz(L-1)>=" << avg(g.state, H.SzSz(0,L-1), g.state)  << endl;
//			lout << "<Sz(0)*Sz(L-1)>-<Sz(0)><Sz(L-1)>=" << avg(g.state, H.SzSz(0,L-1), g.state) - avg(g.state, H.Sz(0), g.state)*avg(g.state, H.Sz(L-1), g.state)  << endl;
//			lout << "SdagS=" << calc_Stot(H,g.state) << endl;
//			lout << "SzSz=" << avg(g.state, H.SzSz(L/2,L/2+1), g.state) << endl;
//			if (D == 3)
//			{
//				for (int d=0; d<L-1; d+=1)
//				{
//					lout << "d=" << d << ", string=" << avg(g.state, H.String(STRINGZ,d), g.state) << endl;
//				}
//				for (int d=1; d<L/2; d+=1)
//				{
//					lout << "d=" << d << ", string*string=" 
//						 << avg(g.state, H.String(STRINGZ,L/2), H.String(STRINGZ,L/2+d), g.state) << "\t" 
//						 << avg(g.state, H.StringCorr(STRINGZ,L/2,L/2+d), g.state) << endl;
//				}
//			}
//			
//			#elif defined(USING_U0)
//			for (int l=0; l<L; ++l)
//			{
//				double avgSz = avg(g.state, H.Scomp(SZ,l), g.state);
//				SzSum += avgSz;
//				double avgSx = avg(g.state, H.Scomp(SX,l), g.state);
//				double avgiSy = avg(g.state, H.Scomp(iSY,l), g.state);
//				lout << "l=" << l << ", Sz=" << avgSz << ", Sx=" << avgSx  << ", iSy=" << avgiSy << ", length=" << sqrt(avgSz*avgSz+avgSx*avgSx-avgiSy*avgiSy) << endl;
//				MagFiler << l << "\t" << avgSz << "\t" << SzSum << endl;
//			}
//			lout << "<Sz(0)*Sz(L-1)>=" << avg(g.state, H.ScompScomp(SZ,SZ,0,L-1), g.state) << endl;
//			lout << "<Sx(0)*Sx(L-1)>=" << avg(g.state, H.ScompScomp(SX,SX,0,L-1), g.state) << endl;
//			lout << "<Sz(0)*Sz(L-1)>-<Sz(0)><Sz(L-1)>=" << avg(g.state, H.ScompScomp(SZ,SZ,0,L-1), g.state) - avg(g.state, H.Scomp(SZ,0), g.state)*avg(g.state, H.Scomp(SZ,L-1), g.state) << endl;
//			lout << "<Sx(0)*Sx(L-1)>-<Sx(0)><Sx(L-1)>=" << avg(g.state, H.ScompScomp(SX,SX,0,L-1), g.state) - avg(g.state, H.Scomp(SX,0), g.state)*avg(g.state, H.Scomp(SX,L-1), g.state) << endl;
//			lout << "SdagS=" << calc_Stot(H,g.state) << endl;
//			lout << "SzSz=" << avg(g.state, H.SzSz(L/2,L/2+1), g.state) << endl;
//			if (D == 3)
//			{
//				for (int d=0; d<L-1; d+=1)
//				{
//					lout << "d=" << d 
//						 << ", stringz=" << avg(g.state, H.String(STRINGZ,d), g.state)
//						 << ", stringx=" << avg(g.state, H.String(STRINGX,d), g.state)
//						 << ", stringy=" << avg(g.state, H.String(STRINGY,d), g.state)
//						 << endl;
//				}
//				for (int d=1; d<L/2; d+=1)
//				{
//					lout << "d=" << d 
//						 << ", stringz*stringz=" << avg(g.state, H.String(STRINGZ,L/2), H.String(STRINGZ,L/2+d), g.state) << "\t" << avg(g.state, H.StringCorr(STRINGZ,L/2,L/2+d), g.state)
//						 << ", stringx*stringx=" << avg(g.state, H.String(STRINGX,L/2), H.String(STRINGX,L/2+d), g.state) << "\t" << avg(g.state, H.StringCorr(STRINGX,L/2,L/2+d), g.state)
//						 << ", stringy*stringy=" << -avg(g.state, H.String(STRINGY,L/2), H.String(STRINGY,L/2+d), g.state) << "\t" << -avg(g.state, H.StringCorr(STRINGY,L/2,L/2+d), g.state)
//						 << endl;
//				}
//			}
//			#else
//			for (int l=0; l<L; ++l)
//			{
//				double avgS = avg(g.state, H.S(l), g.state) * S/sqrt(S*(S+1.));;
////				lout << "S=" << avgS << endl;
//				MagFiler << l << "\t" << avgS << endl;
//			}
//			lout << "<S(0)*S(L-1)>=" << avg(g.state, H.SdagS(0,L-1), g.state) << endl;
//			#endif
//			MagFiler.close();
			
			// excited states
			if (Neigen>1)
			{
				vector<Eigenstate<MODEL::StateXd>> spectrum(Neigen);
				spectrum[0] = g;
				vector<MODEL::Solver> DMRGex(Neigen);
				
				for (int i=1; i<Neigen; ++i)
				{
					cout << "i=" << i << endl;
					DMRGex[i] = MODEL::Solver(DMRG::VERBOSITY::ON_EXIT);
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
					
					#ifndef USING_SU2
					if (i==1)
					{
						ofstream MagFilerExc(make_string("avgSexc","_L=",L,"_",base,".dat"));
						double SzSum = 0.;
						for (int l=0; l<L; ++l)
						{
							double avgSz = avg(spectrum[i].state, H.Scomp(SZ,l), spectrum[i].state);
							SzSum += avgSz;
							lout << "l=" << l << ", Sz=" << avgSz << ", Szsum=" << SzSum << endl;
							MagFilerExc << l << "\t" << avgSz << "\t" << SzSum << endl;
						}
						MagFilerExc.close();
//						if (D==2) save_state("./Heisenberg/Heisenberg", D, spectrum[i].state);
					}
					#endif
				}
				lout << endl;
			}
			
			//res.push_back(make_tuple(L,g.energy,g.energy/L));
		}
//		ofstream FilerOBC(make_string("E_",base,"_",Linfo,"_BC=OBC",".dat"));
//		for (const tuple<int,double,double> &x:res)
//		{
//			FilerOBC << setprecision(16) << get<0>(x) << "\t" << get<1>(x) << "\t" << get<2>(x) << setprecision(6) << endl;
//		}
//		FilerOBC.close();
	}
	else // PBC
	{
		//vector<tuple<int,double,double>> res;
		
		for (const auto &L:Llist)
		{
			vector<Param> paramsPBC = params;
//			paramsPBC.push_back({"J",0.});
//			auto [J, R, offset] = params_bilineraBiquadratic(boost::rational<int>(betaNum,betaDen));
//			lout << "J=" << J << ", R=" << R << ", loc.offset=" << offset << endl;
//			ArrayXXd Jfull = create_1D_PBC(L,J,Jprime,COMPRESS);
//			ArrayXXd Rfull = create_1D_PBC(L,R,0.,COMPRESS);
//			lout << Jfull << endl;
//			paramsPBC.push_back({"Jfull",Jfull});
//			paramsPBC.push_back({"Rfull",Rfull});
			
			ArrayXXd Jfull_uncompressed = create_1D_PBC_AB(L,JA,JB,JpA,JpB,false);
			//vector<size_t> transform = transform_CuthillMcKee(Jfull_uncompressed,false);
			CuthillMcKeeCompressor CMK(Jfull_uncompressed,false); // PRINT=false
			
			ArrayXXd Jfull = create_1D_PBC_AB(L,JA,JB,JpA,JpB,COMPRESS);
//			lout << "Jfull=" << Jfull << endl;
			paramsPBC.push_back({"Jfull",Jfull});
			
			MODEL H(L,paramsPBC);
//			H.scale(1.,offset*L);
			lout << H.info() << endl;
			
			Eigenstate<MODEL::StateXd> g;
			if (LOAD!="")
			{
				g.state.load(LOAD,g.energy);
			}
			MODEL::Solver DMRG(VERB);
			DMRG.GlobParam = GlobParam;
			DMRG.userSetGlobParam();
			DMRG.DynParam = DynParam;
			DMRG.userSetGlobParam();
			DMRG.userSetDynParam();
			DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
			g.state.save(make_string(wd,"state/gs_",base,"_Mmax=",g.state.calc_Mmax()),"",g.energy);
			data.E = g.energy;
			data.e = g.energy/L;
			
			if (CALC_AVG)
			{
				vector<tuple<int,double>> Savg;
				for (int l=0; l<L; ++l)
				{
					int lorig = CMK.get_inverse()[l];
					double res;
					#ifdef USING_SU2
					res = (S==0)? 0. : avg(g.state, H.S(l), g.state) * S/sqrt(S*(S+1.));
					#else
					double resx = avg(g.state, H.Scomp(SX,l), g.state);
					double resz = avg(g.state, H.Scomp(SZ,l), g.state);
					res = sqrt(resx*resx+resz*resz);
					#endif
					if (COMPRESS)
					{
						Savg.push_back(make_tuple(lorig,res));
					}
					else
					{
						Savg.push_back(make_tuple(l,res));
					}
					//lout << "l=" << l << ", lorig=" << lorig << ", avgS=" << res << endl;
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
					data.SdagS(i,j) = res;
					data.SdagS(j,i) = res;
					#pragma omp critical
					{
						lout << "i=" << i << ", j=" << j << ", SdagS=" << res << endl;
					}
				}
				for (int i=0; i<L; ++i)
				{
					data.SdagS(i,i) = 0.5*(D-1) * 0.5*(D+1);
				}
				
				lout << "SdagS=" << endl << data.SdagS << endl;
				double SdagStot = data.SdagS.sum();
				lout << "SdagStot=" << SdagStot << endl;
				data.SdagStot = SdagStot;
			}
			
//			Eigenstate<MODEL::StateXd> g_;
//			MODEL::Solver DMRG_(VERB);
//			DMRG_.GlobParam = GlobParam;
//			DMRG_.userSetGlobParam();
//			DMRG_.DynParam = DynParam;
//			DMRG_.userSetGlobParam();
//			DMRG_.userSetDynParam();
//			DMRG_.edgeState(H_, g_, Q, LANCZOS::EDGE::GROUND);
//			
//			lout << "DOT=" << dot(g.state,g_.state) << endl;
//			lout << avg(g_.state,H,g_.state) << "\t" << avg(g.state,H_,g.state) << endl;
//			lout << avg(g_.state,H_,g_.state) << "\t" << avg(g.state,H,g.state) << endl;
//			assert(1!=1);
			
//			#if defined(USING_U1) or defined(USING_U0)
//			lout << "<SzSz>=" << avg(g.state, H.ScompScomp(SZ,SZ,L/2,L/2+2), g.state) << endl;
//			lout << "<QzQz>=" << avg(g.state, H.QcompQcomp(QZ,QZ,L/2,L/2+2), g.state) << endl;
//			lout << "0.5*<QpQm>=" << 0.5*avg(g.state, H.QcompQcomp(QP,QM,L/2,L/2+2), g.state) << endl;
//			lout << "0.5*<QpzQmz>=" << 0.5*avg(g.state, H.QcompQcomp(QPZ,QMZ,L/2,L/2+2), g.state) << endl;
//			lout << "0.5*<QmzQpz>=" << 0.5*avg(g.state, H.QcompQcomp(QMZ,QPZ,L/2,L/2+2), g.state) << endl;
//			lout << "0.5*<QpQm>=" << 0.5*avg(g.state, H.QcompQcomp(QP,QM,L/2,L/2+2), g.state) << endl;
//			lout << "0.5*<QmQp>=" << 0.5*avg(g.state, H.QcompQcomp(QM,QP,L/2,L/2+2), g.state) << endl;
//			lout << "<SzSz(loc)>=" << avg(g.state, H.Sz(L/2), H.Sz(L/2), g.state) << endl;
//			auto SzOp1 = H.Sz(L/2);
//			SzOp1 = prod(SzOp1,SzOp1);
//			auto SzOp2 = H.Sz(L/2+2);
//			SzOp2 = prod(SzOp2,SzOp2);
//			lout << "<SzSz(nloc)>=" << avg(g.state, SzOp1, SzOp2, g.state) << endl;
			
//			for (int d=0; d<L-1; d+=1)
//			{
//				lout << "d=" << d 
//				     << ", stringz=" << avg(g.state, H.String(STRINGZ,d), g.state) 
//				     << ", stringx=" << avg(g.state, H.String(STRINGX,d), g.state)
//				     << ", stringy=" << avg(g.state, H.String(STRINGY,d), g.state) 
//				     << endl;
//			}
//			for (int d=1; d<L/2; d+=1)
//			{
//				lout << "d=" << d 
//				     << ", stringz*stringz=" << avg(g.state, H.String(STRINGZ,L/2), H.String(STRINGZ,L/2+d), g.state) << "\t" << avg(g.state, H.StringCorr(STRINGZ,L/2,L/2+d), g.state)
//				     << ", stringx*stringx=" << avg(g.state, H.String(STRINGX,L/2), H.String(STRINGX,L/2+d), g.state) << "\t" << avg(g.state, H.StringCorr(STRINGX,L/2,L/2+d), g.state)
//				     << ", stringy*stringy=" << -avg(g.state, H.String(STRINGY,L/2), H.String(STRINGY,L/2+d), g.state) << "\t" << -avg(g.state, H.StringCorr(STRINGY,L/2,L/2+d), g.state)
//				     << endl;
//			}
//			#endif
//			lout << "<SS>=" << avg(g.state, H.SdagS(L/2,L/2+2), g.state) << endl;
//			lout << "<QQ>=" << avg(g.state, H.QdagQ(L/2,L/2+2), g.state) << endl;
////			
//			MODEL::Operator P1 = H.SdagS(0,3); P1.scale(2.,0.5);
//			MODEL::Operator P2 = H.SdagS(2,3); P2.scale(2.,0.5);
//			MODEL::Operator P3 = H.SdagS(1,2); P3.scale(2.,0.5);
//			MODEL::Operator Pprod = prod(P2,P1);
//			Pprod = prod(P3,Pprod);
//			cout << Pprod.info() << endl;
//			
//			cout << "H.SdagS(0,2)=" << avg(g.state, H.SdagS(0,2), g.state) << endl;
//			cout << "H.SdagS(1,3)=" << avg(g.state, H.SdagS(1,3), g.state) << endl;
//			
////			MODEL::StateXd tmp1, tmp2, tmp3;
////			OxV_exact(P1,g.state,tmp1, 2.);
////			cout << "dot(tmp1,tmp1)=" << dot(tmp1,tmp1) << endl;
////			OxV_exact(P2,tmp1,tmp2, 2.);
////			cout << "dot(tmp2,tmp2)=" << dot(tmp2,tmp2) << endl;
////			OxV_exact(P3,tmp2,tmp3, 2.);
////			cout << "dot(tmp3,tmp3)=" << dot(tmp3,tmp3) << endl;
////			cout << tmp3.info() << endl;
////			cout << "dot(g,tmp3)=" << dot(g.state,tmp3) << endl;
//			cout << "avg(g,Pprod,g)=" << avg(g.state, Pprod, g.state) << endl;
			
			// excited states
			if (Neigen>1)
			{
				vector<Eigenstate<MODEL::StateXd>> spectrum(Neigen);
				spectrum[0] = g;
				vector<MODEL::Solver> DMRGex(Neigen);
				
				data.E_excited.resize(Neigen-1);
				
				for (int i=1; i<Neigen; ++i)
				{
					cout << "i=" << i << endl;
					DMRGex[i] = MODEL::Solver(DMRG::VERBOSITY::ON_EXIT);
					DMRGex[i].GlobParam = GlobParam;
					DMRGex[i].DynParam = DynParam;
					DMRGex[i].userSetGlobParam();
					DMRGex[i].userSetDynParam();
					for (int j=0; j<i; ++j)
					{
						DMRGex[i].push_back(spectrum[j].state);
					}
					DMRGex[i].edgeState(H, spectrum[i], Q, LANCZOS::EDGE::GROUND);
					//lout << "SdagS=" << calc_Stot(H, spectrum[i].state) << endl;
					
					data.E_excited(i-1) = spectrum[i].energy;
//					cout << "avg(spectrum[i].state,Pprod,spectrum[i].state)=" << avg(spectrum[i].state, Pprod, spectrum[i].state) << endl;
				}
				lout << endl;
			}
			
			//res.push_back(make_tuple(L,g.energy,g.energy/L));
		}
//		ofstream FilerPBC(make_string("E_",base,"_",Linfo,"_BC=PBC",".dat"));
//		for (const tuple<int,double,double> &x:res)
//		{
//			FilerPBC << setprecision(16) << get<0>(x) << "\t" << get<1>(x) << "\t" << get<2>(x) << setprecision(6) << endl;
//		}
//		FilerPBC.close();
	}
	
	data.save(make_string(wd,"obs/",base,"_Mlimit=",GlobParam.Mlimit));
	lout << "saved to: " << make_string(wd,base,"_Mlimit=",GlobParam.Mlimit) << endl;
	
}
