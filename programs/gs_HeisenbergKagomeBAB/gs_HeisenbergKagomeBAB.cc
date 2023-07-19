SaveData obs;

int L;
size_t D;
MODEL H;

//MODEL Projector (int i, int j, double factor)
//{
//	ArrayXXd Jfull(L,L); Jfull.setZero();
//	Jfull(i,j) = -1.;
//	Jfull(j,i) = -1.;
//	vector<Param> params;
//	params.push_back({"D",D});
//	params.push_back({"maxPower",1ul});
//	params.push_back({"J",0.});
//	params.push_back({"Offset",factor/L});
//	params.push_back({"Jfull",Jfull});
//	//lout << "factor/L=" << factor/L << endl;
//	MODEL Hout(L,params);
//	return Hout;
//}

OPERATOR Projector (int i, int j)
{
	OPERATOR Id = H.Identity();
	Id.scale(0.25);
	OPERATOR Mout = H.SdagS(i,j,0,0,-1.);
	Mout = sum(Id,Mout);
	return Mout;
}

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	int Lx = args.get<int>("Lx",11);
	assert(Lx%2==1);
	int Ly = args.get<int>("Ly",4);
	int S = args.get<int>("S",0);
	int M = args.get<int>("M",0);
	double F = args.get<double>("F",0.);
	#ifdef USING_U1
	qarray<MODEL::Symmetry::Nq> Q = {M};
	#elif defined(USING_U0)
	qarray<MODEL::Symmetry::Nq> Q = {};
	#elif defined(USING_SU2)
	qarray<MODEL::Symmetry::Nq> Q = {2*S+1};
	#endif
	lout << "Q=" << Q << endl;
	
	double J = args.get<double>("J",1.);
	double Jprime = args.get<double>("Jprime",Jprime);
	D = args.get<size_t>("D",2ul);
	
	size_t maxPower = args.get<size_t>("maxPower",2ul);
	bool CALC_VAR = args.get<bool>("CALC_VAR",true);
	bool CALC_CORR = args.get<bool>("CALC_CORR",true);
	bool CALC_GS = args.get<bool>("CALC_GS",true);
	
	size_t Mlimit = args.get<size_t>("Mlimit",1000ul);
	string LOAD = args.get<string>("LOAD","");
	bool VERBOSE = args.get<bool>("VERBOSE",false);
	
	// overwrite beta params in case of LOAD
	if (LOAD!="")
	{
		string LOAD_ = LOAD;
		std::string append_str = ".h5";
		size_t pos = LOAD_.rfind(append_str);
		if (pos != std::string::npos && pos == LOAD_.size() - append_str.size())
		{
			LOAD_ = LOAD_.substr(0,pos);
		}
		
		vector<string> parsed_params;
		boost::split(parsed_params, LOAD_, [](char c){return c == '_';});
		for (int i=0; i<parsed_params.size(); ++i)
		{
			vector<string> parsed_vals;
			boost::split(parsed_vals, parsed_params[i], [](char c){return c == '=';});
			for (int j=0; j<parsed_vals.size(); ++j)
			{
				if (parsed_vals[j] == "Mlimit")
				{
					Mlimit = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: Mlimit=" << Mlimit << endl;
				}
				else if (parsed_vals[j] == "Lx")
				{
					Lx = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: Lx=" << Lx << endl;
				}
				if (parsed_vals[j] == "Ly")
				{
					Ly = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: Ly=" << Ly << endl;
				}
			}
		}
	}
	
	int Nevn = Lx/2+1;
	int Nodd = Lx/2;
	L = Ly/2*Nevn+Ly*Nodd;
	lout << "Nevn=" << Nevn << ", Nodd=" << Nodd << ", Lx=" << Lx << ", Ly=" << Ly << ", L=" << L << endl; 
	
	bool VERBOSE_LATTICE = args.get<bool>("VERBOSE_LATTICE",false);
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	string base = make_string("Lx=",Lx,"_Ly=",Ly,"_L=",L,"_J=",J,"_Jprime=",Jprime,"_D=",D);
	#if defined(USING_SU2)
	base += make_string("_S=",S);
	#endif
	#if defined(USING_U1)
	base += make_string("_M=",M);
	#endif
	base += make_string("_F=",F);
	lout << base << endl;
	lout.set(make_string(base,"_Mlimit=",Mlimit,".log"),wd+"log");
	
	bool FULLMMAX_FILENAME = args.get<bool>("FULLMMAX_FILENAME",false);
	if (FULLMMAX_FILENAME == false)
	{
		base += make_string("_Mlimit=",Mlimit);
	}
	
	string obsfile = make_string(wd,"obs/",base,".h5");
	string statefile = make_string(wd,"state/","state_",base);
	
	lout << args.info() << endl;
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.Minit = args.get<size_t>("Minit",500ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam.Mlimit = Mlimit;
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",20ul);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",20ul);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-7);
	GlobParam.tol_state = args.get<double>("tol_state",1e-5);
	GlobParam.CALC_S_ON_EXIT = false;
	GlobParam.savePeriod = args.get<size_t>("savePeriod",2ul);
	GlobParam.saveName = args.get<string>("saveName",statefile);
	//GlobParam.FULLMMAX_FILENAME = FULLMMAX_FILENAME;
	
	DMRG::CONTROL::DYN  DynParam;
	double eps_svd = args.get<double>("eps_svd",0.);
	DynParam.eps_svd = [eps_svd] (size_t i) {return eps_svd;};
	
	size_t Mincr_per = args.get<size_t>("Mincr_per",2ul);
	DynParam.Mincr_per = [Mincr_per,LOAD] (size_t i) {return (i==0 and LOAD!="")? 0:Mincr_per;};
	
	size_t Mincr_abs = args.get<size_t>("Mincr_abs",Mlimit);
	DynParam.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
	
	size_t start_2site = args.get<size_t>("start_2site",0ul);
	size_t end_2site = args.get<size_t>("end_2site",4ul);
	DynParam.iteration = [start_2site,end_2site] (size_t i) {return (i>=start_2site and i<=end_2site)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	
	// alpha
	size_t start_alpha = args.get<size_t>("start_alpha",0);
	size_t end_alpha = args.get<size_t>("end_alpha",GlobParam.max_halfsweeps-4);
	double alpha = args.get<double>("alpha",100.);
	DynParam.max_alpha_rsvd = [start_alpha, end_alpha, alpha] (size_t i) {return (i>=start_alpha and i<end_alpha)? alpha:0.;};
	
	DMRG::CONTROL::LANCZOS LanczosParam;
	size_t dimK = args.get<size_t>("dimK",500ul);
	LanczosParam.dimK = dimK;
	
	// kagome sites
	vector<vector<int>> i_evn;
	vector<vector<int>> i_odd;
	split_kagomeYC_BAB(Lx, Ly, i_evn, i_odd);
	
	// params
	vector<Param> params;
	params.push_back({"D",D});
	params.push_back({"maxPower",1ul}); // calc H^2 later
	ArrayXXd Jfull = hopping_kagomeYC_BAB(Lx,Ly,false,true,J,VERBOSE_LATTICE);
	
	int Nbonds = abs(Jfull.sum())/(2*abs(J)); // number of bonds
	ArrayXXd Jorig = Jfull;
	CuthillMcKeeCompressor CMK(Jfull,true); // PRINT=true
	//SloanCompressor CMK(Jfull,true);
	CMK.apply_compression(Jfull);
	params.push_back({"Jfull",Jfull});
	
	if (L<=20) lout << "Jorig=" << endl << Jorig << endl;
	if (L<=20) lout << "Jfull=" << endl << Jfull << endl;
	
//	for (int i=0; i<Jorig.rows(); ++i)
//	for (int j=0; j<Jorig.cols(); ++j)
//	{
//		if (abs(Jorig(i,j)) != 0.)
//		{
//			int inew = CMK.get_transform()[i];
//			int jnew = CMK.get_transform()[j];
//			lout << "i=" << i << ", j=" << j << ", d=" << abs(j-i) << ", inew=" << inew << ", jnew=" << jnew << ", dnew=" << abs(jnew-inew) << endl;
//		}
//	}
	
	SaveData obs;
	obs.add_scalars({"E", "e", "var", "em", "Mlimit", "fullMmax"});
	obs.add_matrices(L, L, {"SdagS"});
	obs.add_strings({"history"});
	
	H = MODEL(L,params);
	lout << H.info() << endl;
	
	///////////////////////////////////////////
	/////////// chiral correlations ///////////
	vector<std::pair<int,int>> bond1_diag, bond1_vert, bond1_horiz;
	
	// diagonal, evn x
	for (int x=0; x<i_evn.size()-1; ++x)
	for (int i=0; i<i_evn[x].size(); ++i)
	{
		int k = i_evn[x][i];
		int l = i_odd[x][(2*i+2)%Ly];
		if (VERBOSE) lout << "diagonal1 evn bond: " << min(k,l) << ", " << max(k,l) << endl;
		pair<int,int> b(min(k,l),max(k,l));
		bond1_diag.push_back(b);
	}
	
	// vertical, odd x
	for (int x=0; x<i_odd.size(); ++x)
	for (int i=0; i<i_odd[x].size(); i+=2)
	{
		int k = i_odd[x][i];
		int l = i_odd[x][(i+1)%Ly];
		if (VERBOSE) lout << "vertical1 odd bond: " << min(k,l) << ", " << max(k,l) << endl;
		pair<int,int> b(min(k,l),max(k,l));
		bond1_vert.push_back(b);
	}
	
	// horizontal, odd x
	for (int x=0; x<i_odd.size(); ++x)
	for (int i=1; i<i_odd[x].size(); i+=2)
	{
		int k = i_odd[x][i];
		int l = i_evn[x+1][(i-1)/2];
		if (VERBOSE) lout << "horizontal1 odd bond: " << min(k,l) << ", " << max(k,l) << endl;
		pair<int,int> b(min(k,l),max(k,l));
		bond1_horiz.push_back(b);
	}
	lout << endl;
	
	obs.add_vectors(bond1_horiz.size(), {"C"});
	
	vector<std::pair<int,int>> bond2_horiz, bond2_diag, bond2_vert;
	
	// diagonal, odd x
	for (int x=0; x<i_odd.size(); ++x)
	for (int i=0; i<i_odd[x].size(); i+=2)
	{
		int k = i_odd[x][i];
		int l = i_evn[x+1][i/2];
		if (VERBOSE) lout << "diagonal2 odd bond: " << min(k,l) << ", " << max(k,l) << endl;
		pair<int,int> b(min(k,l),max(k,l));
		bond2_diag.push_back(b);
	}
	
	// horizontal, evn x
	for (int x=0; x<i_evn.size()-1; ++x)
	for (int i=0; i<i_evn[x].size(); i+=1)
	{
		int k = i_evn[x][i];
		int l = i_odd[x][2*i+1];
		if (VERBOSE) lout << "horizontal2 evn bond: " << min(k,l) << ", " << max(k,l) << endl;
		pair<int,int> b(min(k,l),max(k,l));
		bond2_horiz.push_back(b);
	}
	
	// vertical, odd x
	for (int x=0; x<i_odd.size(); ++x)
	for (int i=1; i<i_odd[x].size(); i+=2)
	{
		int k = i_odd[x][i];
		int l = i_odd[x][(i+1)%Ly];
		if (VERBOSE) lout << "vertical2 odd bond: " << min(k,l) << ", " << max(k,l) << endl;
		pair<int,int> b(min(k,l),max(k,l));
		bond2_vert.push_back(b);
	}
	
	////
	
	OPERATOR Field;
	for (int b=0; b<Ly/2; ++b)
	{
		lout << "b=" << b << endl;
		int inew, jnew;
		
		inew = CMK.get_transform()[bond1_diag[b].first];
		jnew = CMK.get_transform()[bond1_diag[b].second];
		OPERATOR Pdiag = Projector(inew,jnew);
		
		inew = CMK.get_transform()[bond1_vert[b].first];
		jnew = CMK.get_transform()[bond1_vert[b].second];
		OPERATOR Pvert = Projector(inew,jnew);
		
		inew = CMK.get_transform()[bond1_horiz[b].first];
		jnew = CMK.get_transform()[bond1_horiz[b].second];
		OPERATOR Phoriz = Projector(inew,jnew);
		
		OPERATOR Term = diff(prod(Pdiag,Pvert),prod(Pdiag,Phoriz));
		
		if (b==0)
		{
			Field = Term;
		}
		else
		{
			Field = sum(Field,Term);
		}
	}
	lout << "Field=" << endl << Field.info() << endl;
	/////////// chiral correlations ///////////
	///////////////////////////////////////////
	
	Eigenstate<MODEL::StateXd> g;
	if (LOAD!="")
	{
		g.state.load(LOAD,g.energy);
		lout << "LOADED=" << g.state.info() << endl;
		lout << "energy=" << g.energy << endl;
	}
	
	MODEL::Solver DMRG(VERB);
	DMRG.GlobParam = GlobParam;
	DMRG.DynParam = DynParam;
	DMRG.LanczosParam = LanczosParam;
	DMRG.userSetGlobParam();
	DMRG.userSetDynParam();
	DMRG.userSetLanczosParam();
	obs.scal["Mlimit"] = Mlimit;
	
	if (CALC_GS)
	{
		if (abs(F) != 0.)
		{
			OPERATOR Hmpo = H;
			Field.scale(F);
			Hmpo = sum(Hmpo,Field);
			
			vector<Param> dummy;
			dummy.push_back({"D",D});
			dummy.push_back({"maxPower",1ul});
			dummy.push_back({"J",0.});
			H = MODEL(Hmpo,dummy);
			lout << H.info() << endl;
		}
		DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
		g.state.save(statefile,"",g.energy);
		
		//////////// excited states ////////////
		////////////////////////////////////////
		int Nexc = args.get<int>("Nexc",0);
		double Epenalty = args.get<double>("Epenalty",1e2);
		vector<Eigenstate<MODEL::StateXd>> excited(Nexc);
		if (Nexc>0)
		{
			lout << endl << termcolor::blue << "CALC_GAP" << termcolor::reset << endl;
			for (int n=0; n<Nexc; ++n)
			{
				lout << "------ n=" << n << " ------" << endl;
				GlobParam.saveName = make_string(wd,MODEL::FAMILY,"_n=",n,"_",base);
				MODEL::Solver DMRGe(VERB);
				DMRGe.Epenalty = Epenalty;
				lout << "Epenalty=" << DMRGe.Epenalty << endl;
				DMRGe.userSetGlobParam();
				DMRGe.userSetDynParam();
				DMRGe.userSetLanczosParam();
				DMRGe.GlobParam = GlobParam;
				DMRGe.DynParam = DynParam;
				DMRGe.LanczosParam = LanczosParam;
				DMRGe.push_back(g.state);
				for (int m=0; m<n; ++m)
				{
					DMRGe.push_back(excited[m].state);
				}
				
//				VectorXd overlaps(n+1);
				
				DMRGe.edgeState(H, excited[n], Q, LANCZOS::EDGE::GROUND);
				
//				overlaps(0) = dot(g.state,excited[n].state);
//				for (int m=0; m<n; ++m) overlaps(m+1) = dot(excited[m].state,excited[n].state);
//				
//				lout << endl;
//				lout << "excited[" << n << "].energy=" << setprecision(16) << excited[n].energy << endl;
//				
//				overlaps(0) = dot(g.state,excited[n].state);
//				for (int m=0; m<n; ++m) overlaps(m+1) = dot(excited[m].state,excited[n].state);
//				lout << "final overlap=" << overlaps.transpose() << endl;
			}
		}
		//////////// excited states ////////////
		////////////////////////////////////////
	}
	
	obs.scal["E"] = g.energy;
	obs.scal["e"] = g.energy/L;
	obs.scal["fullMmax"] = g.state.calc_fullMmax();
	
	if (CALC_CORR)
	{
		Stopwatch<> Timer;
		
		double SdagSm = 0.;
		int Nmbonds = 0; // number of bonds in the middle (x>Lx/4 and x<3*Lx/4)
		obs.mat["SdagS"].setZero();
		
		for (int j=0; j<L; ++j)
		{
			Stopwatch<> jTimer;
			for (int i=0; i<j; ++i)
			{
				if (abs(Jorig(i,j)) > 1e-8)
				{
					int inew = CMK.get_transform()[i];
					int jnew = CMK.get_transform()[j];
					
					obs.mat["SdagS"](i,j) = avg(g.state, H.SdagS(inew,jnew), g.state);
					//obs.mat["SdagS"](j,i) = obs.mat["SdagS"](i,j);
					
					int xi = find_x_kagomeYC(i,L,i_evn,i_odd);
					int xj = find_x_kagomeYC(j,L,i_evn,i_odd);
					if (xi>Lx/4 and xi<3*Lx/4)
					{
						SdagSm += obs.mat["SdagS"](i,j);
						Nmbonds += 1;
					}
					
					lout << i << "(x=" << xi << ")" << "\t" << j << "(x=" << xj << ")" << "\tSdagS=" << obs.mat["SdagS"](i,j) << endl;
				}
			}
			lout << Timer.info(make_string("SdagS j=",j)) << endl;
		}
		SdagSm = SdagSm/Nmbonds * 2.; // Nbonds/L -> 2
		lout << "SdagS(middle)=" << setprecision(16) << SdagSm << ", Nmbonds=" << Nmbonds << ", Nbonds=" << Nbonds << ", Nbonds/L=" << Nbonds/double(L) << endl;
		lout << "test: E=sum(SdagS)/L=" << obs.mat["SdagS"].sum()/L << endl;
		
		obs.scal["em"] = SdagSm;
		
		for (int b=0; b<bond1_horiz.size(); ++b)
		{
			int inew, jnew;
			OPERATOR Pdiag, Phoriz, Pvert;
			
			inew = CMK.get_transform()[bond1_diag[b].first];
			jnew = CMK.get_transform()[bond1_diag[b].second];
			Pdiag = Projector(inew,jnew);
			if (VERBOSE) lout << "Pdiag1: " << bond1_diag[b].first << ", " << bond1_diag[b].second << endl;
			
			inew = CMK.get_transform()[bond1_horiz[b].first];
			jnew = CMK.get_transform()[bond1_horiz[b].second];
			Phoriz = Projector(inew,jnew);
			if (VERBOSE) lout << "Phoriz1: " << bond1_horiz[b].first << ", " << bond1_horiz[b].second << endl;
			
			inew = CMK.get_transform()[bond1_vert[b].first];
			jnew = CMK.get_transform()[bond1_vert[b].second];
			Pvert = Projector(inew,jnew);
			if (VERBOSE) lout << "Pvert1: " << bond1_vert[b].first << ", " << bond1_vert[b].second << endl;
			
			obs.vec["C"](b) = avg(g.state, Pdiag, Pvert, g.state)-avg(g.state, Pdiag, Phoriz, g.state);
			
			inew = CMK.get_transform()[bond2_diag[b].first];
			jnew = CMK.get_transform()[bond2_diag[b].second];
			Pdiag = Projector(inew,jnew);
			if (VERBOSE) lout << "Pdiag2: " << bond2_diag[b].first << ", " << bond2_diag[b].second << endl;
			
			inew = CMK.get_transform()[bond2_horiz[b].first];
			jnew = CMK.get_transform()[bond2_horiz[b].second];
			Phoriz = Projector(inew,jnew);
			if (VERBOSE) lout << "Phoriz2: " << bond2_horiz[b].first << ", " << bond2_horiz[b].second << endl;
			
			inew = CMK.get_transform()[bond2_vert[b].first];
			jnew = CMK.get_transform()[bond2_vert[b].second];
			Pvert = Projector(inew,jnew);
			if (VERBOSE) lout << "Pvert2: " << bond2_vert[b].first << ", " << bond2_vert[b].second << endl;
			
			obs.vec["C"](b) += avg(g.state, Pdiag, Pvert, g.state)-avg(g.state, Pdiag, Phoriz, g.state);
			
			lout << "b=" << b << "\t<C>=" << obs.vec["C"](b) << endl;
		}
		lout << Timer.info("corr") << endl;
	}
	
	if (CALC_VAR)
	{
		Stopwatch<> Timer;
		double var;
		if (maxPower == 2)
		{
			auto Hsq = prod(H,H);
			lout << Hsq.info() << endl;
			var = abs(avg(g.state,Hsq,g.state)-pow(g.energy,2))/L;
		}
		else
		{
			var = abs(avg(g.state,H,H,g.state)-pow(g.energy,2))/L;
		}
		obs.scal["var"] = var;
		lout << Timer.info("H^2 & varE") << endl;
		lout << termcolor::blue << "varE=" << var << termcolor::reset << endl;
	}
	
	obs.str["history"] = lout.get_history();
	//obs.save(make_string(wd,"obs/",base));
	
	HDF5Interface target;
	target = HDF5Interface(obsfile,WRITE);
	obs.save(target,"");
	
	lout << "saved to: " << obsfile << endl;
}
