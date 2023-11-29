#include "SaveData.h"
#include <filesystem>
#include <boost/asio/ip/host_name.hpp>

map<string,int> make_Lmap()
{
	map<string,int> m;
	//
	m["CHAIN"] = 0; // chain
	m["RING"] = 0; // ring
	m["CLUMP"] = 0; // clump = all interact with each other
	// Platonic solids:
	m["P04"] = 4; // tetrahedron
	m["P05"] = 5; // bipyramid
	m["P06"] = 6; // octahedron
	m["P08"] = 8; // cube
	m["P12"] = 12; // icosahedron
	m["P20"] = 20; // dodecahedron
	// Triangles:
	m["T03"] = 3;
	m["T04"] = 4;
	m["T05"] = 5;
	m["T06"] = 6;
	// Triangulene:
	m["HEX04"] = 4;
	m["HEX13"] = 13;
	m["HEX22"] = 22;
	// Coronene:
	m["COR24"] = 24;
	// Corannulene:
	m["COR20"] = 20;
	// Fullerenes:
	m["C12"] = 12; // =truncated tetrahedron ATT
	m["C20"] = 20; // =dodecahedron P20
	m["C24"] = 24;
	m["C26"] = 26;
	m["C28"] = 28;
	m["C30"] = 30;
	m["C36"] = 36;
	m["C40"] = 40;
	m["C40T"] = 40;
	m["C60"] = 60;
	// Archimedean solids:
	m["ATT"] = 12; // truncated tetrahedron
	m["ACO"] = 12; // cuboctahedron
	m["ATO"] = 24; // truncated octahedron
	m["ATC"] = 24; // truncated cube
	m["AID"] = 30; // icosidodecahedron
	m["ASD"] = 60; // snub dodecahedron
	// sodalite cages:
	m["SOD15"] = 15; // NOT IMPLEMENTED
	m["SOD16"] = 16; // SOD60 pole
	m["SOD28"] = 28; // SOD60 equator
	m["SOD20"] = 20; // cuboctahedron decorated with P04
	m["SOD32"] = 32; // NOT IMPLEMENTED
	m["SOD50"] = 50; // icosidodecahedron decorated with P04
	m["SOD60"] = 60; // rectified truncated octahedron decorated with P04
	// square plaquette:
	m["SQR16"] = 16;
	m["SQR20"] = 20;
	
	m["Mn32"] = 32;
	
	m["fcc14"] = 14;
	m["fcc38"] = 38;
	m["fcc68"] = 68;
	m["fcc92"] = 92;
	m["fcc116"] = 116;
	
	m["fccPBC64"] = 64;
	
	return m;
}

map<string,string> make_vertexMap()
{
	map<string,string> m;
	
	m["ATT"] = "3.6^2";
	m["ACO"] = "3.4.3.4";
	m["ATO"] = "4.6^2";
	m["AID"] = "3.5.3.5";
	m["ASD"] = "3^4.5";
	m["ATC"] = "3.8^2";
	
	return m;
}

SaveData obs;

///////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	int L = args.get<int>("L",2);
	double tPerp = args.get<double>("tPerp",1.);
	double U = args.get<double>("U",1.);
	bool UPH = args.get<bool>("UPH",false);
	double S = args.get<double>("S",0);
	int D = 2*S+1;
	string wd = args.get<string>("wd","./");
	int VARIANT = args.get<int>("VARIANT",0); // to try different enumeration variants
	map<string,int> Lmap = make_Lmap();
	map<string,string> Vmap = make_vertexMap();
	string MOL = args.get<string>("MOL","CHAIN");
	if (MOL!="CHAIN" and MOL!="RING" and MOL!="LADDER") L = Lmap[MOL]; // for linear chain, include chain length using -L
	int N = args.get<double>("N",L);
	size_t Mlimit = args.get<size_t>("Mlimit",2000ul);
	string LOAD = args.get<string>("LOAD","");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	ArrayXXd hopping;
	
	if (MOL=="RING")
	{
		bool COMPRESSED = args.get<bool>("COMPRESSED",false);
		hopping = create_1D_PBC(L,1.,0.,COMPRESSED); // Heisenberg ring for testing
	}
	else if (MOL=="CHAIN")
	{
		hopping = create_1D_OBC(L,1.,0.); // Heisenberg chain for testing
	}
	else if (MOL == "LADDER")
	{
		hopping = hopping_ladder(L,1.,tPerp,0.,true);
	}
	else if (MOL.at(0) == 'P')
	{
		hopping = hopping_Platonic(L,VARIANT);
	}
	else if (MOL.at(0) == 'A')
	{
		hopping = hopping_Archimedean(Vmap[MOL],VARIANT);
	}
	else if (MOL.substr(0,3) == "SOD")
	{
		hopping = hopping_sodaliteCage(L,VARIANT);
	}
	else if (MOL.at(0)=='C' and MOL.at(1)!='O')
	{
		if (MOL == "C40T")
		{
			hopping = hopping_fullerene_C40Td(VARIANT);
		}
		else
		{
			hopping = hopping_fullerene(L,VARIANT);
		}
	}
	else if (MOL.at(0)=='T')
	{
		hopping = hopping_triangular(L,VARIANT);
	}
//	else if (MOL.at(0)=='H' and MOL.at(1)=='E' and MOL.at(2)=='X')
//	{
//		hopping = hopping_triangulene(L,VARIANT);
//	}
	else if (MOL.at(0)=='C' and MOL.at(1)=='O' and MOL.at(2)=='R')
	{
		if (L==20)
		{
			hopping = hopping_corannulene(L,VARIANT);
		}
		else
		{
			hopping = hopping_coronene(L,VARIANT);
		}
	}
	else if (MOL.at(0) == 'f' and MOL.at(1) == 'c' and MOL.at(2) == 'c')
	{
		if (MOL.at(3) == 'P' and MOL.at(4) == 'B' and MOL.at(5) == 'C')
		{
			lout << make_string("fccPBC",L,"_d=1.dat") << endl;
			ArrayXXd hopping1 = loadMatrix(make_string("fccPBC",L,"_d=1.dat"));
			ArrayXXd hopping2 = loadMatrix(make_string("fccPBC",L,"_d=2.dat"));
			hopping = hopping1.matrix();//+tPrime*hopping2.matrix();
			auto res = compress_CuthillMcKee(hopping,true);
			hopping = res;
		}
		else
		{
			ArrayXXd hopping1 = loadMatrix(make_string("fcc",L,"_d=1.dat"));
			ArrayXXd hopping2 = loadMatrix(make_string("fcc",L,"_d=2.dat"));
			hopping = hopping1.matrix();//+Jprime*hopping2.matrix();
			auto res = compress_CuthillMcKee(hopping,true);
			hopping = res;
		}
	}
	else
	{
		lout << "Unknown molecule!" << endl;
		throw;
	}
	
	string base = make_string("MOL=",MOL);
	if (MOL == "CHAIN" or MOL == "RING" or MOL=="LADDER")
	{
		base += make_string("_L=",L);
		if (MOL=="LADDER")
		{
			base += make_string("_tPerp=",tPerp);
		}
	}
	#ifndef USING_ED
	base += make_string("_Mlimit=",Mlimit);
	#endif
	base += make_string("_U=",U,"_S=",S,"_N=",N);
	
	if (wd.back() != '/') {wd += "/";}
	lout.set(base+".log",wd+"log");
	lout << boost::asio::ip::host_name() << endl;
	lout << args.info() << endl;
	
	cout << hopping << endl;
//	for (int i=0; i<hopping.rows(); ++i)
//	for (int j=i; j<hopping.rows(); ++j)
//	{
//		if (abs(hopping(i,j)) != 0.) lout << i << ", " << j << endl;
//	}
	
	// free fermions
	bool PRINT_FREE = args.get<bool>("PRINT_FREE",true);
	if (PRINT_FREE)
	{
		SelfAdjointEigenSolver<MatrixXd> Eugen(-1.*hopping.matrix());
		lout << Eugen.eigenvalues().transpose() << endl;
		VectorXd occ = Eugen.eigenvalues().head(N/2);
		VectorXd unocc = Eugen.eigenvalues().tail(L-N/2);
		lout << "orbital energies occupied:" << endl << occ.transpose()  << endl;
		lout << "orbital energies unoccupied:" << endl << unocc.transpose()  << endl << endl;
		double E0 = 2.*occ.sum();
		lout << setprecision(16) << "non-interacting fermions: E0=" << E0 << ", E0/L=" << E0/(L) << setprecision(6) << endl << endl;
	}
	
	#ifndef USING_ED
	DMRG::CONTROL::DYN  DynParam;
	int max_Nrich = args.get<int>("max_Nrich",-1);
	DynParam.max_Nrich = [max_Nrich] (size_t i) {return max_Nrich;};
	
	size_t Mincr_per = args.get<size_t>("Mincr_per",2ul);
	DynParam.Mincr_per = [Mincr_per,LOAD] (size_t i) {return (i==0 and LOAD!="")? 0:Mincr_per;};
	
	size_t Mincr_abs = args.get<size_t>("Mincr_abs",120ul);
	DynParam.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
	
	// glob. params
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.Minit = args.get<size_t>("Minit",100ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",100ul);
	GlobParam.Mlimit = Mlimit;
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",44ul);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",44ul);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-12);
	GlobParam.tol_state = args.get<double>("tol_state",1e-10);
	GlobParam.savePeriod = args.get<size_t>("savePeriod",0);
	GlobParam.CALC_S_ON_EXIT = false;
	GlobParam.INITDIR = static_cast<DMRG::DIRECTION::OPTION>(args.get<int>("INITDIR",1)); // 1=left->right, 0=right->left
	GlobParam.falphamin = args.get<double>("falphamin",0.1);
	size_t maxPower = args.get<size_t>("maxPower",2ul);
	
	size_t start_2site = args.get<size_t>("start_2site",0ul);
	size_t end_2site = args.get<size_t>("end_2site",6ul); //GlobParam.max_halfsweeps-3
	size_t period_2site = args.get<size_t>("period_2site",1ul);
	DynParam.iteration = [start_2site,end_2site,period_2site] (size_t i) {return (i>=start_2site and i<end_2site and i%period_2site==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	
	// alpha
	size_t start_alpha = args.get<size_t>("start_alpha",0);
	size_t end_alpha = args.get<size_t>("end_alpha",GlobParam.max_halfsweeps-8);
	double alpha = args.get<double>("alpha",100.);
	DynParam.max_alpha_rsvd = [start_alpha, end_alpha, alpha] (size_t i) {return (i>=start_alpha and i<end_alpha)? alpha:0.;};
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	#endif
	
	double Umin = args.get<double>("Umin",0.);
	double Umax = args.get<double>("Umax",10.);
	int Upoints = args.get<int>("Upoints",101);
	IntervalIterator Uit(Umin,Umax,Upoints);
	MatrixXd data(Upoints,1+10); data.setZero();
	
	bool CALC_EB = args.get<bool>("CALC_EB",true);
	bool CALC_GS = args.get<bool>("CALC_GS",true);
	
	if (CALC_EB)
	{
		#ifdef USING_ED
		for (Uit=Uit.begin(); Uit!=Uit.end(); ++Uit)
		{
			double U = *Uit;
			data(Uit.index(),0) = U;
			vector<Param> params;
			params.push_back({"U",U});
			//ArrayXd t0(L); t0.setConstant(-0.5*U);
			//params.push_back({"t0",t0});
			params.push_back({"tFull",hopping});
			
			//Stopwatch<> Watch;
			vector<pair<int,int>> NMpairs = {make_pair(L,0), // half-filling, singlet
				                             make_pair(L,2), // half-filling, triplet
				                             //make_pair(L,4), // half-filling, quintet
//				                             
//				                             make_pair(L+2,0), // N+2, singlet
//				                             make_pair(L+2,2), // N+2, triplet
//				                             
//				                             make_pair(L-2,0), // N+2, singlet
//				                             make_pair(L-2,2), // N+2, triplet
//				                             
//				                             make_pair(L+1,1), // N+1, doublet
//				                             make_pair(L+1,3), // N+1, quintuplet
//				                             
//				                             make_pair(L-1,1), // N-2, doublet
//				                             make_pair(L-1,3), // N+1, quintuplet
				                            };
			
			#pragma omp parallel for
			for (int i=0; i<NMpairs.size(); ++i)
			{
				int N = NMpairs[i].first; // Nup+Ndn
				int M = NMpairs[i].second; // Nup-Ndn
				int Nup = (N+M)/2;
				int Ndn = (N-M)/2;
				#pragma omp critical
				{
					lout << endl;
					lout << "===== U=" << U << ", Nup=" << Nup << ", Ndn=" << Ndn << " =====" << endl;
				}
				
				Eigenstate<VectorXd> g;
				
				if (N>2*L or Nup<0 or Ndn<0 or Nup>L or Ndn>L)
				{
					g.energy = std::nan("0");
				}
				else
				{
					#ifdef USING_ED
					MODEL H(L,Nup,Ndn,params);
//					SparseMatrixXd Hmatrix = H.Hmatrix();
//					if (UPH)
//					{
//						Hmatrix += U*H.ntot();
//						Hmatrix += ED::SparseId(H.dim(),U);
//					}
//					cout << H.ntot() << endl;
					#pragma omp critical
					{
						lout << H.info() << endl;
					}
					#endif
					
					#ifdef USING_ED
					if (H.dim() <= 200 and abs(U)>1e-5)
					{
						g.energy = H.eigenvalues()(0);
						g.state = H.eigenvectors().col(0);
//						MatrixXd Hdense = Hmatrix;
//						SelfAdjointEigenSolver<MatrixXd> Eugen(Hdense);
//						g.energy = Eugen.eigenvalues()(0);
						#pragma omp critical
						{
							lout << "U=" << U << ", N=" << N << ", M=" << M << ", E=" << setprecision(16) << g.energy << endl;
							lout << H.eigenvalues() << endl;
						}
					}
					else
					#endif
					{
						if (abs(U) <= 1e-10)
						{
							MatrixXd T0 = -1.*hopping;
							//T0.diagonal() = t0;
							SelfAdjointEigenSolver<MatrixXd> Eugen(T0);
							g.energy = Eugen.eigenvalues().head(Nup).sum()+Eugen.eigenvalues().head(Ndn).sum();
						}
						else
						{
							LanczosSolver<ED::SpinfulFermions,VectorXd,double> Lutz(LANCZOS::REORTHO::FULL);
							Lutz.edgeState(H, g, LANCZOS::EDGE::GROUND, 1e-7, 1e-4);
							//LanczosSolver<SparseMatrixXd,VectorXd,double> Lutz(LANCZOS::REORTHO::FULL);
							//Lutz.edgeState(Hmatrix, g, LANCZOS::EDGE::GROUND, 1e-7, 1e-4);
							
							#pragma omp critical
							{
								lout << Lutz.info() << endl;
								lout << setprecision(16) << "E=" << g.energy << ", E/L=" << g.energy/L << endl;
							}
							double davg = 0.;
							double navg = 0.;
							for (int l=0; l<L; ++l)
							{
								davg += ED::avg(g.state, H.d(l), g.state);
								navg += ED::avg(g.state, H.n(l), g.state);
							}
							davg /= L;
							navg /= L;
							double eavg = 1.-navg+davg;
							lout << "U=" << U << ", N=" << N << ", M=" << M << ", d=" << davg << ", e=" << eavg << ", nh=" << davg+eavg << endl;
						}
					}
					
					MatrixXd SdagS(L,L); SdagS.setZero();
					for (int i=0; i<L; ++i)
					for (int j=0; j<L; ++j)
					{
						SparseMatrixXd Op;
						Op = 0.25*(H.n<UP>(i)-H.n<DN>(i))*(H.n<UP>(j)-H.n<DN>(j));
						SdagS(i,j) += ED::avg(g.state, Op, g.state);
						if (i!=j)
						{
							Op = -0.5*H.cdagc<UP>(i,j)*H.cdagc<DN>(j,i);
							SdagS(i,j) += ED::avg(g.state, Op, g.state);
							Op = -0.5*H.cdagc<DN>(i,j)*H.cdagc<UP>(j,i);
							SdagS(i,j) += ED::avg(g.state, Op, g.state);
						}
						else
						{
							Op = 0.5*H.n(i)-H.d(i);
							SdagS(i,j) += ED::avg(g.state, Op, g.state);
						}
					}
					#pragma omp critical
					{
						lout << "U=" << U << ", N=" << N << ", M=" << M << ", SdagS.sum()=" << SdagS.sum() << endl;
					}
				}
				
				data(Uit.index(),i+1) = g.energy;
			}
			cout << endl;
			saveMatrix(data, make_string(base,".dat"));
		}
		#else
		obs.add_scalars({"energy", "var", "Mmax", "fullMmax"});
		obs.add_strings({"history"});
		
		vector<Param> params;
		params.push_back({"U",U});
		params.push_back({"tFull",hopping});
		params.push_back({"maxPower",1ul});
		
		MODEL H(L,params);
		lout << H.info() << endl;
		
		Eigenstate<MODEL::StateXd> g;
		if (LOAD!="")
		{
			g.state.load(LOAD,g.energy);
			lout << termcolor::blue << "LOADED=" << g.state.info() << termcolor::reset << endl;
			lout << termcolor::blue << "energy=" << setprecision(16) << g.energy << termcolor::reset << setprecision(6) << endl;
		}
		
		if (CALC_GS)
		{
			MODEL::Solver Lutz(VERB);
			Lutz.userSetGlobParam();
			Lutz.userSetDynParam();
			Lutz.GlobParam = GlobParam;
			Lutz.DynParam = DynParam;
			qarray<MODEL::Symmetry::Nq> Qc = {D,N};
			Lutz.edgeState(H, g, Qc, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
			
			string statefile = make_string(wd+"state/","state_",base);
			lout << "Saving state to: " << statefile << endl;
			g.state.save(statefile, base, g.energy);
		}
		
		bool CALC_VAR = args.get<bool>("CALC_VAR",true);
		if (CALC_VAR)
		{
			Stopwatch<> Timer;
			double var;
			if (maxPower == 2)
			{
				lout << "Computing H^2..." << endl;
				auto Hsq = prod(H,H);
				lout << Hsq.info() << endl;
				var = abs(avg(g.state,Hsq,g.state,1ul,DMRG::DIRECTION::RIGHT,VERB)-pow(g.energy,2))/L;
			}
			else
			{
				var = abs(avg(g.state,H,H,g.state)-pow(g.energy,2))/L;
			}
			obs.scal["var"] = var;
			lout << Timer.info("H^2 & varE") << endl;
			lout << termcolor::blue << "varE=" << var << termcolor::reset << endl;
		}
		
		obs.scal["energy"] = g.energy;
		obs.scal["Mmax"] = g.state.calc_Mmax();
		obs.scal["fullMmax"] = g.state.calc_fullMmax();
		obs.str["history"] = lout.get_history();
		
		HDF5Interface target;
		string obsfile = wd+"obs/"+base+".h5";
		lout << "Saving data to: " << obsfile << endl;
		target = HDF5Interface(obsfile,WRITE);
		obs.save(target);
		target.close();
		
		#endif
	}
	
	//MatrixXd T0 = -1.*hopping;
	//SelfAdjointEigenSolver<MatrixXd> Eugen(T0);
	//lout << endl << "single-particle energies=" << endl << Eugen.eigenvalues().head(L/2) << endl << "===" << endl << Eugen.eigenvalues().tail(L/2) << endl << endl;
	//double E0 = 2*Eugen.eigenvalues().head(L/2).sum();
	//lout << setprecision(16) << "E(N=L,M=0)=" << E0 << ", E/L=" << E0/L << endl;
}
