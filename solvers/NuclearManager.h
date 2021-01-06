#ifndef NUCLEAR_MANAGER
#define NUCLEAR_MANAGER

#include <boost/algorithm/string.hpp>

#include <Eigen/SparseCore>
#include <boost/rational.hpp>

#include "models/HubbardU1.h"
#include "solvers/DmrgSolver.h"
#include "HDF5Interface.h"

class NuclearManager
{
public:
	
	typedef VMPS::HubbardU1 MODEL;
	
	NuclearManager(){};
	
	/**
	sigma: Gaussian width
	*/
	NuclearManager (int Nclosed_input, int Nsingle_input, int Z_input, const ArrayXi &deg_input, const vector<string> &labels_input, 
	                const ArrayXd &eps0_input, const ArrayXXd &V0_input, double G0_input=1., double sigma_input=-1.,
	                bool REF_input=false, string PARAM_input="Seminole")
	:Nclosed(Nclosed_input), Nsingle(Nsingle_input), Z(Z_input), deg(deg_input), labels(labels_input), eps0(eps0_input), V0(V0_input), G0(G0_input), sigma(sigma_input), REF(REF_input), PARAM(PARAM_input)
	{
		Nlev = deg.rows();
		L = deg.sum();
		lout << "Nlev=" << Nlev << ", L=" << L << endl;
		V0.triangularView<Lower>() = V0.transpose();
		levelstring = str(labels_input);
//		replace_halves(levelstring);
		
		construct();
	};
	
	NuclearManager (int Nclosed_input, int Nsingle_input, int Z_input,const string &filename, double G0_input, bool REF_input=false)
	:Nclosed(Nclosed_input), Nsingle(Nsingle_input), Z(Z_input), G0(G0_input), REF(REF_input)
	{
		HDF5Interface source(filename+".h5", READ);
		
		source.load_vector<int>(deg, "deg", "");
		source.load_vector<double>(eps0, "eps0", "");
		source.load_matrix<double>(V0, "V0", "");
		
		Nlev = deg.rows();
		L = deg.sum();
		lout << "Nlev=" << Nlev << ", L=" << L << endl;
		V0.triangularView<Lower>() = V0.transpose();
		
		labels.resize(Nlev);
		for (int i=0; i<Nlev; ++i)
		{
			source.load_char(make_string("lev",i).c_str(), labels[i]);
		}
		levelstring = str(labels);
//		replace_halves(levelstring);
		
		construct();
	};
	
	void construct();
	
	void make_Hamiltonian (bool LOAD=false, bool SAVE=false, string wd="./");
	Eigenstate<MODEL::StateXd> calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE=LANCZOS::EDGE::GROUND) const;
	void compute (bool LOAD=false, bool SAVE=false, string wd="./");
	
	ArrayXd get_occ() const;
	inline ArrayXd get_eps0() const {return eps0;}
	inline ArrayXi get_deg() const {return deg;}
	inline ArrayXi get_offset() const {return offset;}
	inline size_t length() const {return L;}
	inline ArrayXd get_sign() {return sign;}
	inline ArrayXd get_levelsign (int level) {return sign_per_level[level];}
	inline VectorXd get_onsite_free() const {return onsite_free;}
	inline MODEL get_H() const {return H;}
	
	double Sn_Eref (int N) const;
	ArrayXd Sn_nref14() const;
	ArrayXd Sn_Nref14() const;
	ArrayXd Sn_Pref14() const;
	ArrayXd Sn_Sref14() const;
	
	const Eigenstate<VMPS::HubbardU1::StateXd> &get_g (int Nshell) const {return g[Nshell];}
	
private:
	
	bool REF = false;
	string PARAM = "Seminole";
	
	int Nclosed, Nsingle, Z;
	int Nlev;
	size_t L;
	
	string levelstring;
	vector<string> labels;
	VectorXi deg, offset;
	VectorXd eps0;
	ArrayXd sign;
	vector<ArrayXd> sign_per_level;
	
	MatrixXd V0, Ghop;
	double G0, sigma;
	VectorXd onsite, Gloc, onsite_free;
	
	VMPS::HubbardU1 H;
	vector<Eigenstate<MODEL::StateXd>> g;
	vector<VectorXd> n, avgN;
};

void NuclearManager::
construct()
{
	// offsets
	offset.resize(Nlev);
	offset(0) = 0;
	for (int i=1; i<Nlev; ++i)
	{
		offset(i) = deg.head(i).sum();
	}
	
	// G hopping
	Ghop.resize(L,L); Ghop.setZero();
	for (int i=0; i<Nlev; ++i)
	for (int j=0; j<Nlev; ++j)
	{
		if (sigma<0.)
		{
			Ghop.block(offset(i),offset(j),deg(i),deg(j)).setConstant(-G0*V0(i,j));
		}
		else
		{
			Ghop.block(offset(i),offset(j),deg(i),deg(j)).setConstant( -G0*V0(i,j)*exp( -0.5*pow(eps0(i)-eps0(j),2)/pow(sigma,2) ) );
		}
	}
	
	if (sigma > 0.)
	{
		MatrixXd Gfull = -G0*V0;
		for (int i=0; i<Nlev; ++i)
		for (int j=0; j<Nlev; ++j)
		{
			Gfull(i,j) *= exp(-0.5*pow(eps0(i)-eps0(j),2)/pow(sigma,2));
		}
		lout << "-G0*V0*exp()=" << endl << Gfull << endl;
	}
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Ghop(i,j) *= pow(-1,i+j); // cancels sign
	}
	
	ArrayXXd sign2d(L,L);
	sign2d.setZero();
	for (int j1=0; j1<Nlev; ++j1)
	for (int j2=0; j2<Nlev; ++j2)
	{
		boost::rational<int> J1 = deg(j1)-boost::rational<int>(1,2);
		boost::rational<int> J2 = deg(j2)-boost::rational<int>(1,2);
		
		ArrayXXd signblock(deg(j1),deg(j2));
		
		for (int m1=0; m1<deg(j1); ++m1)
		for (int m2=0; m2<deg(j2); ++m2)
		{
			boost::rational<int> M1 = J1-m1;
			boost::rational<int> M2 = J2-m2;
			assert((J1+J2+M1+M2).denominator() == 1);
			signblock(m1,m2) = pow(-1,(J1+J2-M1-M2).numerator());
		}
		
		sign2d.block(offset(j1),offset(j2),deg(j1),deg(j2)) = signblock;
	}
	
	sign.resize(L);
	sign_per_level.resize(Nlev);
	sign.setZero();
	for (int j=0; j<Nlev; ++j)
	{
		boost::rational<int> J = deg(j)-boost::rational<int>(1,2);
		
		ArrayXd signblock(deg(j));
		
		for (int m=0; m<deg(j); ++m)
		{
			boost::rational<int> M = J-m;
			signblock(m) = pow(-1,(J-M).numerator());
		}
		
		sign_per_level[j] = signblock;
		lout << signblock.transpose() << endl;
		
		sign.segment(offset(j),deg(j)) = signblock;
	}
//	lout << sign.transpose() << endl;
	
	Ghop = (Ghop.array()*sign2d).matrix();
//	lout << "Ghop=" << endl << Ghop << endl;
	
	// G local
	Gloc = Ghop.diagonal();
	Ghop.diagonal().setZero();
	
	// onsite
	onsite.resize(L); onsite.setZero();
	for (int i=0; i<Nlev; ++i)
	{
		onsite.segment(offset(i),deg(i)).setConstant(eps0(i));
	}
	
	onsite_free.resize(2*L);
	for (int i=0; i<L; ++i)
	{
		onsite_free(2*i) = onsite(i);
		onsite_free(2*i+1) = onsite(i);
	}
	
	/*PermutationMatrix<Dynamic,Dynamic> P(L);
	P.setIdentity();
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(P.indices().data(), P.indices().data()+P.indices().size(), g);
	Ghop = P.inverse()*Ghop*P;
	onsite = P.inverse()*onsite;
	Gloc = P.inverse()*Gloc;*/
}

ArrayXd NuclearManager::
get_occ() const
{
	ArrayXd res(Nlev);
	for (int i=0; i<deg.rows(); ++i)
	{
		res(i) = 2.*deg(i);
	}
	return res;
}

double NuclearManager::
Sn_Eref (int N) const
{
	double res = std::nan("0");
	if (N==12)
	{
		res = -75.831;
	}
	else if (N==14)
	{
		res = -86.308;
	}
	else if (N==16)
	{
		res = -95.942;
	}
	return res;
}

ArrayXd NuclearManager::
Sn_nref14() const
{
	ArrayXd res(5);
	res(0) = 0.870;
	res(1) = 0.744;
	res(2) = 0.157;
	res(3) = 0.178;
	res(4) = 0.133;
	return res;
}

ArrayXd NuclearManager::
Sn_Nref14() const
{
	ArrayXd res(5);
	res(0) = 6.96;
	res(1) = 4.46;
	res(2) = 0.627;
	res(3) = 0.356;
	res(4) = 1.6;
	return res;
}

ArrayXd NuclearManager::
Sn_Pref14() const
{
	ArrayXd res(5);
	res(0) = 0.81;
	res(1) = 0.93;
	res(2) = 0.524;
	res(3) = 0.396;
	res(4) = 0.845;
	return res;
}

ArrayXd NuclearManager::
Sn_Sref14() const
{
	ArrayXd res(5);
	res(0) = 6.86;
	res(1) = 6.55;
	res(2) = 7.25;
	res(3) = 6.98;
	res(4) = 7.12;
	return res;
}

void NuclearManager::
make_Hamiltonian (bool LOAD, bool SAVE, string wd)
{
	string filename = make_string(wd,"H_Z=",Z,"_Nclosed=",Nclosed,"_Nsingle=",Nsingle,"_G0=",G0);
	if (sigma>0.) filename += make_string("_sigma=",sigma);
	filename += make_string("_j=",levelstring);
	if (PARAM!="Seminole") filename += make_string("_PARAM=",PARAM);
	filename += ".h5";
	
	if (LOAD)
	{
		H = MODEL(L,{{"maxPower",1ul}});
		H.load(filename);
		lout << H.info() << endl;
	}
	else
	{
		vector<Param> params;
		params.push_back({"t",0.});
		for (size_t l=0; l<L; ++l)
		{
			params.push_back({"t0",onsite(l),l});
			params.push_back({"U",Gloc(l),l});
		}
		params.push_back({"Vxyfull",ArrayXXd(2.*Ghop)}); // 2 compensates 0.5 in our definition
		params.push_back({"maxPower",1ul});
		
		H = MODEL(L,params);
		lout << H.info() << endl;
		if (SAVE) H.save(filename);
	}
}

Eigenstate<VMPS::HubbardU1::StateXd> NuclearManager::
calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE) const
{
	qarray<MODEL::Symmetry::Nq> Q = {Nshell};
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = 12ul;
	GlobParam.max_halfsweeps = 100ul;
	GlobParam.Minit = 100ul;
	GlobParam.Qinit = 100ul;
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ;
	GlobParam.CALC_S_ON_EXIT = false;
	GlobParam.Mlimit = 500ul;
	
	DMRG::CONTROL::DYN  DynParam;
	size_t lim2site = 6ul;
	DynParam.iteration = [lim2site] (size_t i) {return (i<lim2site)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	DynParam.max_alpha_rsvd = [] (size_t i) {return (i<16)? 1e4:0.;};
	
	Stopwatch<> Timer;
	MODEL::Solver DMRG1(DMRG::VERBOSITY::ON_EXIT);
	MODEL::Solver DMRG2(DMRG::VERBOSITY::ON_EXIT);
	MODEL::Solver DMRG3(DMRG::VERBOSITY::ON_EXIT);
	MODEL::Solver DMRG4(DMRG::VERBOSITY::ON_EXIT);
	MODEL::Solver DMRG5(DMRG::VERBOSITY::ON_EXIT);
	MODEL::Solver DMRG6(DMRG::VERBOSITY::ON_EXIT);
	
	Eigenstate<MODEL::StateXd> g1,g2,g3,g4,g5,g6;
	
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			DMRG1.userSetGlobParam();
			DMRG1.userSetDynParam();
			DMRG1.GlobParam = GlobParam;
			DMRG1.DynParam = DynParam;
			DMRG1.edgeState(H, g1, Q, EDGE);
		}
		#pragma omp section
		{
			DMRG2.userSetGlobParam();
			DMRG2.userSetDynParam();
			DMRG2.GlobParam = GlobParam;
			DMRG2.GlobParam.INITDIR = DMRG::DIRECTION::LEFT;
			DMRG2.DynParam = DynParam;
			DMRG2.edgeState(H, g2, Q, EDGE);
		}
	}
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			DMRG3.userSetGlobParam();
			DMRG3.userSetDynParam();
			DMRG3.GlobParam = GlobParam;
			DMRG3.DynParam = DynParam;
			DMRG3.push_back(g1.state);
			DMRG3.push_back(g2.state);
			if (EDGE == LANCZOS::EDGE::ROOF) DMRG3.Epenalty = -1e4;
			DMRG3.edgeState(H, g3, Q, EDGE);
		}
		#pragma omp section
		{
			DMRG4.userSetGlobParam();
			DMRG4.userSetDynParam();
			DMRG4.GlobParam = GlobParam;
			DMRG4.GlobParam.INITDIR = DMRG::DIRECTION::LEFT;
			DMRG4.DynParam = DynParam;
			DMRG4.push_back(g1.state);
			DMRG4.push_back(g2.state);
			if (EDGE == LANCZOS::EDGE::ROOF) DMRG4.Epenalty = -1e4;
			DMRG4.edgeState(H, g4, Q, EDGE);
		}
	}
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			DMRG5.userSetGlobParam();
			DMRG5.userSetDynParam();
			DMRG5.GlobParam = GlobParam;
			DMRG5.DynParam = DynParam;
			DMRG5.push_back(g1.state);
			DMRG5.push_back(g2.state);
			DMRG5.push_back(g3.state);
			DMRG5.push_back(g4.state);
			if (EDGE == LANCZOS::EDGE::ROOF) DMRG5.Epenalty = -1e4;
			DMRG5.edgeState(H, g5, Q, EDGE);
		}
		#pragma omp section
		{
			DMRG6.userSetGlobParam();
			DMRG6.userSetDynParam();
			DMRG6.GlobParam = GlobParam;
			DMRG6.GlobParam.INITDIR = DMRG::DIRECTION::LEFT;
			DMRG6.DynParam = DynParam;
			DMRG6.push_back(g1.state);
			DMRG6.push_back(g2.state);
			DMRG6.push_back(g3.state);
			DMRG6.push_back(g4.state);
			if (EDGE == LANCZOS::EDGE::ROOF) DMRG6.Epenalty = -1e4;
			DMRG6.edgeState(H, g6, Q, EDGE);
		}
	}
	lout << g1.energy << "\t" << g2.energy << endl;
	lout << g3.energy << "\t" << g4.energy << endl;
	lout << g5.energy << "\t" << g6.energy << endl;
	bool PROJECTION_WAS_USEFUL = false;
	if (EDGE == LANCZOS::EDGE::GROUND)
	{
		if (g3.energy < g1.energy) PROJECTION_WAS_USEFUL = true;
		if (g4.energy < g2.energy) PROJECTION_WAS_USEFUL = true;
		if (g5.energy < min(g1.energy,g2.energy)) PROJECTION_WAS_USEFUL = true;
		if (g6.energy < min(g1.energy,g2.energy)) PROJECTION_WAS_USEFUL = true;
	}
	else
	{
		if (g3.energy > g1.energy) PROJECTION_WAS_USEFUL = true;
		if (g4.energy > g2.energy) PROJECTION_WAS_USEFUL = true;
		if (g5.energy > min(g1.energy,g2.energy)) PROJECTION_WAS_USEFUL = true;
		if (g6.energy > min(g1.energy,g2.energy)) PROJECTION_WAS_USEFUL = true;
	}
	if (PROJECTION_WAS_USEFUL) lout << termcolor::red << "PROJECTION_WAS_USEFUL=" << boolalpha << PROJECTION_WAS_USEFUL << termcolor::reset << endl;
	
	Eigenstate<MODEL::StateXd> res;
	
	if (EDGE == LANCZOS::EDGE::GROUND)
	{
		res = (g1.energy<=g2.energy)? g1:g2;
		if (g3.energy < res.energy) res = g3;
		if (g4.energy < res.energy) res = g4;
		if (g5.energy < res.energy) res = g5;
		if (g6.energy < res.energy) res = g6;
	}
	else
	{
		res = (g1.energy>=g2.energy)? g1:g2;
		if (g3.energy > res.energy) res = g3;
		if (g4.energy > res.energy) res = g4;
		if (g5.energy > res.energy) res = g5;
		if (g6.energy > res.energy) res = g6;
	}
	lout << Timer.info("edge state") << endl;
	
	return res;
}

void NuclearManager::
compute (bool LOAD, bool SAVE, string wd)
{
	make_Hamiltonian(LOAD,SAVE,wd);
	
	g.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	
	string filename = make_string(wd+"PairingResult_Z=",Z,"_Nclosed=",Nclosed,"_Nsingle=",Nsingle,"_G0=",G0);
	if (sigma>0.) filename += make_string("_sigma=",sigma);
	filename += make_string("_j=",levelstring);
	if (PARAM!="Seminole") filename += make_string("_PARAM=",PARAM);
	filename += ".h5";
	cout << "filename=" << filename << endl;
	
	HDF5Interface target(filename, WRITE);
	target.save_vector(eps0,"eps0","");
	target.save_vector(deg,"deg","");
	
	VectorXd energies(2*L+1);
	VectorXd energies_free(2*L+1);
	
	for (int Nshell=0; Nshell<=2*L; ++Nshell)
	{
		int A = Z+Nclosed+Nshell;
		lout << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << ", progress=" << round(Nshell*1./2*L*100,1) << "%" << endl;
		
		g[Nshell] = calc_gs(Nshell);
		
		energies(Nshell) = g[Nshell].energy;
		energies_free(Nshell) = onsite_free.head(Nshell).sum();
		lout << "noninteracting energy=" << energies_free(Nshell) << ", interacting energy=" << energies(Nshell) << endl;
		if (Nshell > 1 and Nshell<2*L-1) assert(g[Nshell].energy < energies_free(Nshell));
		
		if (Z==50 and Nclosed==50 and REF and (Nshell==12 or Nshell==14 or Nshell==16))
		{
			double diff = abs(g[Nshell].energy-Sn_Eref(Nshell));
			if (diff<1e-2)
			{
				lout << termcolor::green;
			}
			else
			{
				lout << termcolor::red;
			}
			lout << "ref=" << Sn_Eref(Nshell) << endl;
			lout << termcolor::reset;
		}
//		lout << "E_B/A=" << abs(g[Nshell].energy)/Nshell << endl;
		
		n[Nshell].resize(Nlev);
		avgN[Nshell].resize(Nlev);
		
//		ofstream Filer(make_string("./Z=",Z,"/avgN_Z=",Z,"_N=",N,"_Nshell=",Nshell,".dat"));
		
		int i0 = 0;
		for (int j=0; j<Nlev; ++j)
		{
			avgN[Nshell](j) = 0.;
			for (int i=0; i<deg(j); ++i)
			{
				avgN[Nshell](j) += avg(g[Nshell].state, H.n(i0+i), g[Nshell].state);
			}
			
			n[Nshell](j) = avgN[Nshell](j);
			n[Nshell](j) /= deg(j);
			n[Nshell](j) *= 0.5;
			
			double resP = 0.;
			if (Nshell>=2)
			{
				for (int i=0; i<deg(j); ++i)
				{
					double Pcontrib = sign(i0+i) * avg(g[Nshell-2].state, H.cc(i0+i), g[Nshell].state);
//					lout << "i=" << i << ", Pcontrib=" << Pcontrib << endl;
					resP += Pcontrib;
				}
			}
			resP /= sqrt(deg(j));
			resP = abs(resP);
			
//			double S = 0.;
//			if (Nshell>=1)
//			{
//				S = g[Nshell-1].energy-g[Nshell].energy;
//			}
			
			i0 += deg(j);
			
			double Nplot = (avgN[Nshell](j)<1e-14)? 0 : avgN[Nshell](j);
			double nplot = (n[Nshell](j)<1e-14)?    0 : n[Nshell](j);
			lout << "N(" << labels[j] << ")=" << Nplot << "/" << 2*deg(j) << ", " << "n=" << nplot << ", " << "P=" << resP;
//			Filer << eps0(j) << "\t" << avgN[Nshell](j) << "\t" << 2*deg(j) << "\t" << n[Nshell](j) << endl;
			if (Z==50 and Nclosed==50 and REF and Nshell==14)
			{
				lout << ", ref(N)=";
				if (abs(Sn_Nref14()(j)-avgN[Nshell](j))<1e-1)
				{
					lout << termcolor::green;
				}
				else
				{
					lout << termcolor::red;
				}
				lout << Sn_Nref14()(j) << termcolor::reset;
				
				lout << ", ref(n)=";
				if (abs(Sn_nref14()(j)-n[Nshell](j))<1e-1)
				{
					lout << termcolor::green;
				}
				else
				{
					lout << termcolor::red;
				}
				lout << Sn_nref14()(j) << termcolor::reset;
				
				lout << ", ref(P)=";
				if (abs(Sn_Pref14()(j)-resP)<1e-1)
				{
					lout << termcolor::green;
				}
				else
				{
					lout << termcolor::red;
				}
				lout << Sn_Pref14()(j) << termcolor::reset;
				
//				lout << ", ref(S)=";
//				if (abs(Sn_Sref14()(j)-S)<1e-1)
//				{
//					lout << termcolor::green;
//				}
//				else
//				{
//					lout << termcolor::red;
//				}
//				lout << Sn_Sref14()(j) << termcolor::reset;
			}
			lout << endl;
		}
		target.create_group(make_string(Nshell));
		target.save_scalar(g[Nshell].energy,"E0",make_string(Nshell));
		target.save_scalar(energies_free(Nshell),"E0free",make_string(Nshell));
		target.save_vector(avgN[Nshell],"avgN",make_string(Nshell));
		target.save_vector(n[Nshell],"n",make_string(Nshell));
		lout << endl;
//			Filer.close();
	}
	
	MatrixXd Delta3(0,2);
	for (int Nshell=1, i=0; Nshell<=2*L-1; ++Nshell, ++i)
	{
		double DeltaVal = 0.5*(2.*energies(Nshell)-energies(Nshell-1)-energies(Nshell+1));
		
//		lout << "Nn=" << Nshell+Nclosed << ", Delta3=" << DeltaVal << endl;
		Delta3.conservativeResize(Delta3.rows()+1,Delta3.cols());
		Delta3(i,0) = Nshell;
		Delta3(i,1) = DeltaVal;
	}
	target.save_matrix(Delta3,"Delta3");
	
	MatrixXd Delta3free(0,2);
	for (int Nshell=1, i=0; Nshell<=2*L-1; ++Nshell, ++i)
	{
		double DeltaVal = 0.5*(2.*energies_free(Nshell)-energies_free(Nshell-1)-energies_free(Nshell+1));
		
		Delta3free.conservativeResize(Delta3free.rows()+1,Delta3free.cols());
		Delta3free(i,0) = Nshell;
		Delta3free(i,1) = DeltaVal;
	}
	target.save_matrix(Delta3free,"Delta3free");
	
	MatrixXd Delta3b(0,2);
	for (int Nshell=2, i=0; Nshell<=2*L; ++Nshell, ++i)
	{
		double DeltaVal = 0.5*(energies(Nshell)-2.*energies(Nshell-1)+energies(Nshell-2));
		
//		lout << "Nn=" << Nshell+Nclosed << ", Delta3b=" << DeltaVal << endl;
		Delta3b.conservativeResize(Delta3b.rows()+1,Delta3b.cols());
		Delta3b(i,0) = Nshell;
		Delta3b(i,1) = DeltaVal;
	}
	target.save_matrix(Delta3b,"Delta3b");
	
	MatrixXd Delta4(0,2);
	for (int Nshell=2, i=0; Nshell<=2*L-1; ++Nshell, ++i)
	{
		double DeltaVal = 0.25*(energies(Nshell-2)-3.*energies(Nshell-1)+3.*energies(Nshell)-energies(Nshell+1));
		
//		lout << "Nn=" << Nshell+Nclosed << ", Delta4=" << DeltaVal << endl;
		Delta4.conservativeResize(Delta4.rows()+1,Delta4.cols());
		Delta4(i,0) = Nshell;
		Delta4(i,1) = DeltaVal;
	}
	target.save_matrix(Delta4,"Delta4");
	
	MatrixXd Delta5(0,2);
	for (int Nshell=2, i=0; Nshell<=2*L-2; ++Nshell, ++i)
	{
		double DeltaVal = -0.125*(6.*energies(Nshell-2)-4.*energies(Nshell+1)-4.*energies(Nshell-1)+energies(Nshell+2)+energies(Nshell-2));
		
//		lout << "Nn=" << Nshell+Nclosed << ", Delta5=" << DeltaVal << endl;
		Delta5.conservativeResize(Delta5.rows()+1,Delta5.cols());
		Delta5(i,0) = Nshell;
		Delta5(i,1) = DeltaVal;
	}
	target.save_matrix(Delta5,"Delta5");
	
	lout << "saved: " << filename << endl;
	
	target.close();
}

#endif
