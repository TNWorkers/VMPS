#ifndef NUCLEAR_MANAGER
#define NUCLEAR_MANAGER

#include <boost/algorithm/string.hpp>

#include <Eigen/SparseCore>
#include <boost/rational.hpp>

#include "solvers/DmrgSolver.h"
#include "HDF5Interface.h"
#include "ParamHandler.h"
#include "DmrgLinearAlgebra.h"

template<typename VectorType>
struct jEigenstate
{
	Eigenstate<VectorType> eigenstate;
	string label;
};

template<typename MODEL>
class NuclearManager
{
public:
	
//	typedef VMPS::HubbardU1 MODEL;
	
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
		
		construct();
	};
	
	void construct();
	
	void make_Hamiltonian (bool LOAD=false, bool SAVE=false, string wd="./");
	Eigenstate<typename MODEL::StateXd> calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE=LANCZOS::EDGE::GROUND, int Nruns=3, bool CALC_VAR=true, DMRG::VERBOSITY::OPTION VERB = DMRG::VERBOSITY::ON_EXIT);
	void compute (bool LOAD=false, bool SAVE=false, string wd="./", int Nruns=3, int Nshellmin=0, int Nshellmax=-1);
	void compute_parallel (bool LOAD=false, bool SAVE=false, string wd="./", int Nruns=3, int Nshellmin=0, int Nshellmax=-1);
	
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
	ArrayXd Sn_nref (int Nshell) const;
	ArrayXd Sn_Nref (int Nshell) const;
	ArrayXd Sn_Pref (int Nshell) const;
	ArrayXd Sn_Sref14() const;
	
	const Eigenstate<typename MODEL::StateXd> &get_g (int Nshell) const {return g[Nshell];}
	
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
	
	MODEL H;
	vector<MODEL> Hadiabatic;
	vector<double> adiabatic_sweeps;
	vector<Eigenstate<typename MODEL::StateXd>> g;
	vector<double> var;
	vector<int> Mmax;
	vector<VectorXd> n, avgN;
	vector<string> Jgs;
	vector<Mpo<typename MODEL::Symmetry>> cdagj;
};

template<typename MODEL>
void NuclearManager<MODEL>::
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
	
	if constexpr (MODEL::FAMILY == HUBBARD)
	{
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
	}
	
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

template<typename MODEL>
ArrayXd NuclearManager<MODEL>::
get_occ() const
{
	ArrayXd res(Nlev);
	for (int i=0; i<deg.rows(); ++i)
	{
		res(i) = 2.*deg(i);
	}
	return res;
}

template<typename MODEL>
double NuclearManager<MODEL>::
Sn_Eref (int Nshell) const
{
	double res = std::nan("0");
	if (Nshell == 12)
	{
		res = -75.831;
	}
	else if (Nshell == 14)
	{
		res = -86.308;
	}
	else if (Nshell == 16)
	{
		res = -95.942;
	}
	return res;
}

template<typename MODEL>
ArrayXd NuclearManager<MODEL>::
Sn_nref (int Nshell) const
{
	ArrayXd res = Sn_Nref(Nshell);
	for (int i=0; i<5; ++i) res(i) /= deg(i)*2;
	/*res(0) = 0.Z70;
	res(1) = 0.744;
	res(2) = 0.157;
	res(3) = 0.178;
	res(4) = 0.133;*/
	return res;
}

template<typename MODEL>
ArrayXd NuclearManager<MODEL>::
Sn_Nref (int Nshell) const
{
	ArrayXd res(5);
	if (Nshell == 12)
	{
		res(0) = 6.4551;
		res(1) = 3.6458;
		res(2) = 0.4757;
		res(3) = 0.2358;
		res(4) = 1.1877;
	}
	else if (Nshell == 14)
	{
		res(0) = 6.9562;
		res(1) = 4.4630;
		res(2) = 0.6265;
		res(3) = 0.3558;
		res(4) = 1.5985;
	}
	else if (Nshell == 16)
	{
		res(0) = 7.1413;
		res(1) = 4.7697;
		res(2) = 0.9280;
		res(3) = 0.6463;
		res(4) = 2.5147;
	}
	else if (Nshell == 18)
	{
		res(0) = 7.2727;
		res(1) = 4.9743;
		res(2) = 1.2526;
		res(3) = 0.9171;
		res(4) = 3.5833;
	}
	return res;
}

template<typename MODEL>
ArrayXd NuclearManager<MODEL>::
Sn_Pref (int Nshell) const
{
	ArrayXd res(5);
	if (Nshell == 12)
	{
		res(0) = 1.8870;
		res(1) = 1.7040;
		res(2) = 0.6569;
		res(3) = 0.3287;
		res(4) = 1.8089;
	}
	else if (Nshell == 14)
	{
		res(0) = 1.6197;
		res(1) = 1.6107;
		res(2) = 0.7411;
		res(3) = 0.3958;
		res(4) = 2.0687;
	}
	else if (Nshell == 16)
	{
		res(0) = 1.3592;
		res(1) = 1.3494;
		res(2) = 0.8722;
		res(3) = 0.5138;
		res(4) = 2.5256;
	}
	else if (Nshell == 18)
	{
		res(0) = 1.2466;
		res(1) = 1.2336;
		res(2) = 0.9720;
		res(3) = 0.5563;
		res(4) = 2.8951;
	}
	for (int i=0; i<5; ++i) res(i) /= sqrt(deg(i));
	/*res(0) = 0.81;
	res(1) = 0.93;
	res(2) = 0.524;
	res(3) = 0.396;
	res(4) = 0.845;*/
	return res;
}

template<typename MODEL>
ArrayXd NuclearManager<MODEL>::
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

template<typename MODEL>
void NuclearManager<MODEL>::
make_Hamiltonian (bool LOAD, bool SAVE, string wd)
{
	string filename = make_string(wd,"H_Z=",Z,"_Nclosed=",Nclosed,"_Nsingle=",Nsingle,"_G0=",G0);
	if (sigma>0.) filename += make_string("_sigma=",sigma);
	filename += make_string("_j=",levelstring);
	if (PARAM!="Seminole") filename += make_string("_PARAM=",PARAM);
	
	if (LOAD)
	{
		MODEL Hres(L,{{"maxPower",1ul}});
		Hres.load(filename);
		H = Hres;
		lout << H.info() << endl;
	}
	else
	{
		ArrayXXd Ghopx2 = 2.*Ghop; // 2 compensates 0.5 in our definition
		
		vector<Param> params_loc;
		params_loc.push_back({"t",0.});
		params_loc.push_back({"J",0.});
		params_loc.push_back({"maxPower",1ul});
		for (size_t l=0; l<L; ++l)
		{
			if constexpr (MODEL::FAMILY == HUBBARD)
			{
				params_loc.push_back({"t0",onsite(l),l});
				params_loc.push_back({"U",Gloc(l),l});
			}
			else if constexpr (MODEL::FAMILY == HEISENBERG)
			{
				params_loc.push_back({"nu",2.*onsite(l)+Gloc(l),l});
			}
		}
		
		MODEL Hloc(L,params_loc,BC::OPEN,DMRG::VERBOSITY::SILENT);
		MpoTerms<typename MODEL::Symmetry,double> Terms = Hloc;
		
		Stopwatch<> CompressionTimer;
		for (int i=0; i<Ghopx2.rows(); ++i)
		{
			cout << "compressing: site " << i+1 << "/" << Ghopx2.rows() << endl;
			vector<Param> params_tmp;
			params_tmp.push_back({"t",0.});
			params_tmp.push_back({"J",0.});
			params_tmp.push_back({"maxPower",1ul});
			ArrayXXd Ghopx2(Ghop.rows(),Ghop.cols());
			Ghopx2.setZero();
			Ghopx2.matrix().row(i) = 2.*Ghop.matrix().row(i);
			if constexpr (MODEL::FAMILY == HUBBARD)
			{
				params_tmp.push_back({"Vxyfull",Ghopx2});
			}
			else if constexpr (MODEL::FAMILY == HEISENBERG)
			{
				params_tmp.push_back({"Jxyfull",Ghopx2});
			}
			
			MODEL Hterm = MODEL(L, params_tmp, BC::OPEN, DMRG::VERBOSITY::SILENT);
			Terms.set_verbosity(DMRG::VERBOSITY::SILENT);
			Hterm.set_verbosity(DMRG::VERBOSITY::SILENT);
			Terms = MODEL::sum(Terms,Hterm);
		}
		
		vector<Param> params_full = params_loc;
		if constexpr (MODEL::FAMILY == HUBBARD)
		{
			params_full.push_back({"Vxyfull",Ghopx2});
		}
		else if constexpr (MODEL::FAMILY == HEISENBERG)
		{
			params_full.push_back({"Jxyfull",Ghopx2});
		}
		lout << CompressionTimer.info("MPO compression") << endl;
		
		Mpo<typename MODEL::Symmetry,double> Hmpo(Terms);
		H = MODEL(Hmpo,params_full);
//		H.calc(2ul); // takes too long
		lout << H.info() << endl;
		
		if (SAVE) H.save(filename);
	}
	
	// good:
	vector<double> tvals = {2., 1., 0.5, 0.25, 0.12, 0.05};
	adiabatic_sweeps = {4, 4, 4, 4 ,4, 4};
	// bad:
//	vector<double> tvals = {4., 3., 2., 1., 0.5, 0.25, 0.125, 0.05, 0.01};
//	adiabatic_sweeps = {2, 2, 2, 2, 2, 2, 2, 2, 2};
	Hadiabatic.resize(tvals.size());
	for (int i=0; i<tvals.size(); ++i)
	{
		MODEL Hperturb(L,{{"t",tvals[i]},{"Bx",tvals[i]},{"maxPower",1ul}}, BC::OPEN, DMRG::VERBOSITY::SILENT);
		Hperturb = sum<MODEL,double>(H, Hperturb, DMRG::VERBOSITY::SILENT);
		Hadiabatic[i] = Hperturb;
	}
	
	cdagj.resize(Nlev);
	for (int io=0; io<Nlev; ++io)
	{
		for (int id=0; id<deg[io]; ++id)
		{
			auto cdagj1 = H.template cdag<UP>(offset[io]+id);
			auto cdagj2 = H.template cdag<DN>(offset[io]+id);
			if (id==0)
			{
				cdagj[io] = sum(cdagj1,cdagj2);
			}
			else
			{
				cdagj[io] = sum(cdagj[io],cdagj1);
				cdagj[io] = sum(cdagj[io],cdagj2);
			}
		}
//		lout << "j-index=" << io << ", cdag[j]: " << cdagj[io].info() << endl;
	}
}

//template<typename MODEL>
//Eigenstate<typename MODEL::StateXd> NuclearManager<MODEL>::
//calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE, int Nruns, bool CALC_VAR) const
//{
//	qarray<MODEL::Symmetry::Nq> Q;
//	if constexpr (MODEL::FAMILY == HUBBARD)
//	{
//		Q = MODEL::singlet(Nshell);
//	}
//	else if constexpr (MODEL::FAMILY == HEISENBERG)
//	{
//		Q = {Nshell-static_cast<int>(L)};
//	}
//	
//	DMRG::CONTROL::GLOB GlobParam;
//	GlobParam.min_halfsweeps = 12ul;
//	GlobParam.max_halfsweeps = 36ul;
//	GlobParam.Minit = 100ul;
//	GlobParam.Qinit = 100ul;
//	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ;
//	GlobParam.CALC_S_ON_EXIT = false;
//	GlobParam.Mlimit = 500ul;
////	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
////	GlobParam.tol_eigval = 1e-12;
////	GlobParam.tol_state = 1e-10;
//	
//	DMRG::CONTROL::DYN  DynParam;
//	size_t lim2site = 24ul;
//	DynParam.iteration = [lim2site] (size_t i) {return (i<lim2site and i%2==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
//	DynParam.max_alpha_rsvd = [lim2site] (size_t i) {return (i<=lim2site+4ul)? 1e4:0.;};
//	
//	Stopwatch<> Timer;
//	
//	vector<Eigenstate<typename MODEL::StateXd>> gL;
//	vector<Eigenstate<typename MODEL::StateXd>> gR;
//	ArrayXd energiesL;
//	ArrayXd energiesR;
//	
//	int Nruns_adjusted =  (Nshell==0 or Nshell==2*L)? 1:Nruns;
//	
//	for (int r=0; r<Nruns_adjusted; ++r)
//	{
//		lout << "------r=" << r << "------" << endl;
//		#pragma omp parallel sections
//		{
//			#pragma omp section
//			{
//				Eigenstate<typename MODEL::StateXd> g;
//				typename MODEL::Solver DMRG1(DMRG::VERBOSITY::ON_EXIT);
//				DMRG1.userSetGlobParam();
//				DMRG1.userSetDynParam();
//				DMRG1.GlobParam = GlobParam;
//				DMRG1.DynParam = DynParam;
//				for (int s=0; s<gL.size(); ++s) DMRG1.push_back(gL[s].state);
//				DMRG1.Epenalty = (EDGE==LANCZOS::EDGE::GROUND)? 1e4:-1e4;
//				DMRG1.edgeState(H, g, Q, EDGE);
//				gL.push_back(g);
//				
//				energiesL.conservativeResize(energiesL.rows()+1);
//				energiesL(energiesL.rows()-1) = g.energy;
//			}
//			#pragma omp section
//			{
//				Eigenstate<typename MODEL::StateXd> g;
//				typename MODEL::Solver DMRG2(DMRG::VERBOSITY::ON_EXIT);
//				DMRG2.userSetGlobParam();
//				DMRG2.userSetDynParam();
//				DMRG2.GlobParam = GlobParam;
//				DMRG2.GlobParam.INITDIR = DMRG::DIRECTION::LEFT;
//				DMRG2.DynParam = DynParam;
//				DMRG2.Epenalty = (EDGE==LANCZOS::EDGE::GROUND)? 1e4:-1e4;
//				for (int s=0; s<gR.size(); ++s) DMRG2.push_back(gR[s].state);
//				DMRG2.edgeState(H, g, Q, EDGE);
//				gR.push_back(g);
//				
//				energiesR.conservativeResize(energiesR.rows()+1);
//				energiesR(energiesR.rows()-1) = g.energy;
//			}
//		}
//	}
//	for (int r=0; r<Nruns; ++r)
//	{
//		lout << gL[r].energy << "\t" << gR[r].energy << endl;
//	}
//	
//	double valL, valR;
//	int indexL, indexR;
//	string edgeStateLabel;
//	if (EDGE == LANCZOS::EDGE::GROUND)
//	{
//		valL = energiesL.minCoeff(&indexL);
//		valR = energiesR.minCoeff(&indexR);
//		edgeStateLabel = "ground";
//	}
//	else
//	{
//		valL = energiesL.maxCoeff(&indexL);
//		valR = energiesR.maxCoeff(&indexR);
//		edgeStateLabel = "roof";
//	}
//	if (indexL == 1 and valL<valR)
//	{
//		lout << termcolor::yellow << "Found " << edgeStateLabel << " state at r=" << indexL << " (L->R)" << termcolor::reset << endl;
//	}
//	else if (indexL > 1 and valL<valR)
//	{
//		lout << termcolor::red << "Found " << edgeStateLabel << " state at r=" << indexL << " (L->R)" << termcolor::reset << endl;
//	}
//	if (indexR == 1 and valL>valR)
//	{
//		lout << termcolor::yellow << "Found " << edgeStateLabel << " state at r=" << indexR << " (R->L)" << termcolor::reset << endl;
//	}
//	else if (indexR > 1  and valL>valR)
//	{
//		lout << termcolor::red << "Found " << edgeStateLabel << " state at r=" << indexR << " (R->L)" << termcolor::reset << endl;
//	}
//	
//	Eigenstate<typename MODEL::StateXd> res;
//	
//	if (EDGE == LANCZOS::EDGE::GROUND)
//	{
//		res = (gL[0].energy<=gR[0].energy)? gL[0]:gR[0];
//		for (int r=1; r<Nruns; ++r)
//		{
//			if (gL[r].energy < res.energy) res = gL[r];
//			if (gR[r].energy < res.energy) res = gR[r];
//		}
//	}
//	else
//	{
//		res = (gL[0].energy>=gR[0].energy)? gL[0]:gR[0];
//		for (int r=1; r<Nruns; ++r)
//		{
//			if (gL[r].energy > res.energy) res = gL[r];
//			if (gR[r].energy > res.energy) res = gR[r];
//		}
//	}
//	
//	if (CALC_VAR)
//	{
//		double var = abs(avg(res.state,H,H,res.state)-pow(res.energy,2))/L;
//		lout << "var=" << var << endl;
//	}
//	
//	lout << Timer.info("edge state") << endl;
//	
//	return res;
//}

template<typename MODEL>
Eigenstate<typename MODEL::StateXd> NuclearManager<MODEL>::
calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE, int Nruns, bool CALC_VAR, DMRG::VERBOSITY::OPTION VERB)
{
	qarray<MODEL::Symmetry::Nq> Q;
	if constexpr (MODEL::FAMILY == HUBBARD)
	{
		Q = MODEL::singlet(Nshell);
	}
	else if constexpr (MODEL::FAMILY == HEISENBERG)
	{
		Q = {Nshell-static_cast<int>(L)};
	}
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = 12ul;
	GlobParam.max_halfsweeps = 36ul;
	GlobParam.Minit = 100ul;
	GlobParam.Qinit = 100ul;
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ;
	GlobParam.CALC_S_ON_EXIT = false;
	GlobParam.Mlimit = 500ul;
//	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
	GlobParam.tol_eigval = 1e-9;
	GlobParam.tol_state = 1e-7;
	
	DMRG::CONTROL::DYN  DynParam;
	size_t lim2site = 24ul;
	DynParam.iteration = [lim2site] (size_t i) {return (i<lim2site and i%2==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	DynParam.max_alpha_rsvd = [lim2site] (size_t i) {return (i<=lim2site+4ul)? 1e4:0.;};
	size_t Mincr_per = 2ul;
	DynParam.Mincr_per = [Mincr_per] (size_t i) {return Mincr_per;};
	
	Stopwatch<> Timer;
	
//	int Nruns_adjusted = (Nshell%2==0)? 1:Nruns; //(Nshell==0 or Nshell==2*L)? 1:Nruns;
//	int Nruns_adjusted = (Nshell==0 or Nshell==2*L)? 1:Nruns;
//	int Nruns_adjusted = (Nshell%2==0)? 1:Nruns;
	int Nruns_adjusted = (Nshell%2==0)? Nruns:1;
	if (Nshell==0 or Nshell==2*L) Nruns_adjusted = 1;
	if (Nshell==42 and Nlev==13)  Nruns_adjusted = 10;
	
	vector<Eigenstate<typename MODEL::StateXd>> gL(Nruns_adjusted);
	vector<Eigenstate<typename MODEL::StateXd>> gR(Nruns_adjusted);
	ArrayXd energiesL(Nruns_adjusted);
	ArrayXd energiesR(Nruns_adjusted);
	string JgsL, JgsR;
	
	vector<jEigenstate<typename MODEL::StateXd>> ginit(Nlev);
	if (Nshell%2==1)
	{
		auto gtest = g[Nshell-1].state;
		double testenergy = 1e4;
		for (int io=0; io<Nlev; ++io)
		{
			typename MODEL::StateXd gtmp;
			OxV_exact(cdagj[io], gtest, gtmp, 2., DMRG::VERBOSITY::SILENT);
			gtmp /= sqrt(dot(gtmp,gtmp));
			double etmp = avg(gtmp, H, gtmp);
			ginit[io].eigenstate.state = gtmp;
			ginit[io].eigenstate.energy = etmp;
			ginit[io].label = labels[io];
		}
		
		sort(ginit.begin(), ginit.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
		
		if (VERB > DMRG::VERBOSITY::SILENT)
		for (int io=0; io<Nlev; ++io)
		{
			lout << "E(" << ginit[io].label << ")=" << ginit[io].eigenstate.energy << endl;
		}
	}
	
	for (int r=0; r<Nruns_adjusted; ++r)
	{
		if (VERB > DMRG::VERBOSITY::SILENT) lout << "------r=" << r << "------" << endl;
		
		std::array<typename MODEL::Solver,2> DMRG_LR;
		for (int i=0; i<2; ++i)
		{
			DMRG_LR[i] = typename MODEL::Solver(DMRG::VERBOSITY::SILENT);
			DMRG_LR[i].userSetGlobParam();
			DMRG_LR[i].userSetDynParam();
			DMRG_LR[i].GlobParam = GlobParam;
			DMRG_LR[i].DynParam = DynParam;
			DMRG_LR[i].Epenalty = (EDGE==LANCZOS::EDGE::GROUND)? 1e4:-1e4;
		}
		
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				for (int s=0; s<r; ++s)
				{
					DMRG_LR[0].push_back(gL[s].state);
					DMRG_LR[0].push_back(gR[s].state);
				}
				
				if (Nshell%2 == 0)
				{
					for (int i=0; i<Hadiabatic.size(); ++i)
					{
						DMRG_LR[0].GlobParam.min_halfsweeps = adiabatic_sweeps[i];
						DMRG_LR[0].GlobParam.max_halfsweeps = adiabatic_sweeps[i];
						DMRG_LR[0].edgeState(Hadiabatic[i], gL[r], Q, EDGE, (i==0)?false:true);
					}
					DMRG_LR[0].GlobParam.min_halfsweeps = 10ul;
					DMRG_LR[0].GlobParam.max_halfsweeps = 32ul;
					DMRG_LR[0].set_verbosity(VERB);
					DMRG_LR[0].edgeState(H, gL[r], Q, EDGE, true);
				}
				else
				{
					DMRG_LR[0].GlobParam.min_halfsweeps = 12ul;
					DMRG_LR[0].GlobParam.max_halfsweeps = 36ul;
					vector<jEigenstate<typename MODEL::StateXd>> gfinal = ginit;
					for (int io=0; io<min(gfinal.size(),4ul); ++io)
					{
						DMRG_LR[0].edgeState(H, gfinal[io].eigenstate, Q, EDGE, true);
					}
					sort(gfinal.begin(), gfinal.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
					gL[r] = gfinal[0].eigenstate;
					JgsL = gfinal[0].label;
				}
				
				energiesL(r) = gL[r].energy;
			}
			#pragma omp section
			{
				DMRG_LR[1].GlobParam.INITDIR = DMRG::DIRECTION::LEFT;
				for (int s=0; s<r; ++s)
				{
					DMRG_LR[1].push_back(gL[s].state);
					DMRG_LR[1].push_back(gR[s].state);
				}
				
				if (Nshell%2 == 0)
				{
					for (int i=0; i<Hadiabatic.size(); ++i)
					{
						DMRG_LR[1].GlobParam.min_halfsweeps = adiabatic_sweeps[i];
						DMRG_LR[1].GlobParam.max_halfsweeps = adiabatic_sweeps[i];
						DMRG_LR[1].edgeState(Hadiabatic[i], gR[r], Q, EDGE, (i==0)?false:true);
					}
					DMRG_LR[1].GlobParam.min_halfsweeps = 10ul;
					DMRG_LR[1].GlobParam.max_halfsweeps = 32ul;
					DMRG_LR[1].set_verbosity(VERB);
					DMRG_LR[1].edgeState(H, gR[r], Q, EDGE, true);
				}
				else
				{
					DMRG_LR[1].GlobParam.min_halfsweeps = 12ul;
					DMRG_LR[1].GlobParam.max_halfsweeps = 36ul;
					vector<jEigenstate<typename MODEL::StateXd>> gfinal = ginit;
					for (int io=0; io<min(gfinal.size(),4ul); ++io)
					{
						DMRG_LR[1].edgeState(H, gfinal[io].eigenstate, Q, EDGE, true);
					}
					sort(gfinal.begin(), gfinal.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
					gR[r] = gfinal[0].eigenstate;
					JgsR = gfinal[0].label;
				}
				
				energiesR(r) = gR[r].energy;
			}
		}
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT)
	for (int r=0; r<Nruns_adjusted; ++r)
	{
		lout << termcolor::blue << setprecision(9) << gL[r].energy << "\t" << gR[r].energy << setprecision(6) << termcolor::reset << endl;
	}
	
	double valL, valR;
	int indexL, indexR;
	string edgeStateLabel;
	if (EDGE == LANCZOS::EDGE::GROUND)
	{
		valL = energiesL.minCoeff(&indexL);
		valR = energiesR.minCoeff(&indexR);
		edgeStateLabel = "ground";
	}
	else
	{
		valL = energiesL.maxCoeff(&indexL);
		valR = energiesR.maxCoeff(&indexR);
		edgeStateLabel = "roof";
	}
	if (VERB > DMRG::VERBOSITY::SILENT)
	{
		if (indexL == 1 and valL<valR)
		{
			lout << termcolor::yellow << "Found " << edgeStateLabel << " state at r=" << indexL << " (L->R)" << termcolor::reset << endl;
		}
		else if (indexL > 1 and valL<valR)
		{
			lout << termcolor::red << "Found " << edgeStateLabel << " state at r=" << indexL << " (L->R)" << termcolor::reset << endl;
		}
		if (indexR == 1 and valL>valR)
		{
			lout << termcolor::yellow << "Found " << edgeStateLabel << " state at r=" << indexR << " (R->L)" << termcolor::reset << endl;
		}
		else if (indexR > 1  and valL>valR)
		{
			lout << termcolor::red << "Found " << edgeStateLabel << " state at r=" << indexR << " (R->L)" << termcolor::reset << endl;
		}
	}
	
	Eigenstate<typename MODEL::StateXd> res;
	
	if (EDGE == LANCZOS::EDGE::GROUND)
	{
		res = (gL[0].energy<=gR[0].energy)? gL[0]:gR[0];
		if (Nshell%2 == 1)
		{
			Jgs[Nshell] = (gL[0].energy<=gR[0].energy)? JgsL:JgsR;
		}
		else
		{
			Jgs[Nshell] = "0";
		}
		for (int r=1; r<Nruns_adjusted; ++r)
		{
			if (gL[r].energy < res.energy) res = gL[r];
			if (gR[r].energy < res.energy) res = gR[r];
		}
	}
	else
	{
		res = (gL[0].energy>=gR[0].energy)? gL[0]:gR[0];
		if (Nshell%2 == 1)
		{
			Jgs[Nshell] = (gL[0].energy>=gR[0].energy)? JgsL:JgsR;
		}
		else
		{
			Jgs[Nshell] = "0";
		}
		for (int r=1; r<Nruns_adjusted; ++r)
		{
			if (gL[r].energy > res.energy) res = gL[r];
			if (gR[r].energy > res.energy) res = gR[r];
		}
	}
	
	if (CALC_VAR)
	{
		double var = abs(avg(res.state,H,H,res.state)-pow(res.energy,2))/L;
		if (VERB > DMRG::VERBOSITY::SILENT) lout << "var=" << var << endl;
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT) lout << Timer.info("edge state") << endl;
	
	return res;
}

template<typename MODEL>
void NuclearManager<MODEL>::
compute (bool LOAD, bool SAVE, string wd, int Nruns, int Nshellmin, int Nshellmax)
{
	make_Hamiltonian(LOAD,SAVE,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jgs.resize(2*L+1);
	
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
	
	int Nshelllast = Nshellmax;
	if (Nshellmax==-1) Nshelllast = 2*L;
	
//	#pragma omp parallel for
	for (int Nshell=Nshellmin; Nshell<=Nshelllast; ++Nshell)
	{
		int A = Z+Nclosed+Nshell;
//		#pragma omp critical
		{
			lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << ", progress=" << round(Nshell*1./(2*L)*100,1) << "%" << termcolor::reset << endl;
		}
		
		g[Nshell] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, Nruns, false);
		
		lout << "j of ground state: " << Jgs[Nshell] << endl;
		
		if (Nshell != 0 and Nshell != 2*L)
		{
			var[Nshell] = abs(avg(g[Nshell].state,H,H,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
		}
		else
		{
			var[Nshell] = 0.;
		}
//		#pragma omp critical
		{
			lout << termcolor::blue << "var=" << var[Nshell] << termcolor::reset << endl;
		}
		
		Mmax[Nshell] = g[Nshell].state.calc_Mmax();
		
		energies(Nshell) = g[Nshell].energy;
		energies_free(Nshell) = onsite_free.head(Nshell).sum();
//		#pragma omp critical
		{
			lout << "noninteracting energy=" << energies_free(Nshell) << ", interacting energy=" << energies(Nshell) << endl;
		}
		if (Nshell > 1 and Nshell<2*L-1) assert(g[Nshell].energy < energies_free(Nshell));
		
		if (Z==50 and Nclosed==50 and REF and (Nshell==12 or Nshell==14 or Nshell==16))
		{
			double diff = abs(g[Nshell].energy-Sn_Eref(Nshell));
			if (diff<1e-2)
			{
				//#pragma omp critical
				{
					lout << termcolor::green;
				}
			}
			else
			{
				//#pragma omp critical
				{
					lout << termcolor::red;
				}
			}
			lout << "ref=" << Sn_Eref(Nshell) << endl;
			lout << termcolor::reset;
		}
		lout << "E_B/A=" << abs(g[Nshell].energy)/Nshell << " MeV" << endl;
		
		n[Nshell].resize(Nlev);
		avgN[Nshell].resize(Nlev);
		
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
			
			i0 += deg(j);
			
			double Nplot = (avgN[Nshell](j)<1e-14)? 0 : avgN[Nshell](j);
			double nplot = (n[Nshell](j)<1e-14)?    0 : n[Nshell](j);
			lout << "N(" << labels[j] << ")=" << Nplot << "/" << 2*deg(j) << ", " << "n=" << nplot;
			//<< ", " << "P=" << resP;
			
			if (Z==50 and Nclosed==50 and REF and (Nshell==12 or Nshell==14 or Nshell==16 or Nshell==18))
			{
				auto Sn_nref_ = Sn_nref(Nshell);
				auto Sn_Nref_ = Sn_Nref(Nshell);
				auto Sn_Pref_ = Sn_Pref(Nshell);
				
				lout << ", ref(N)=";
				if (abs(Sn_Nref_(j)-avgN[Nshell](j))<1e-3)
				{
					lout << termcolor::green;
				}
				else
				{
					lout << termcolor::red;
				}
				lout << Sn_Nref_(j) << termcolor::reset;
				
				lout << ", ref(n)=";
				if (abs(Sn_nref_(j)-n[Nshell](j))<1e-3)
				{
					lout << termcolor::green;
				}
				else
				{
					lout << termcolor::red;
				}
				lout << Sn_nref_(j) << termcolor::reset;
			}
			lout << endl;
		}
		target.create_group(make_string(Nshell));
		target.save_scalar(g[Nshell].energy,"E0",make_string(Nshell));
		target.save_scalar(energies_free(Nshell),"E0free",make_string(Nshell));
		target.save_vector(avgN[Nshell],"avgN",make_string(Nshell));
		target.save_vector(n[Nshell],"n",make_string(Nshell));
		target.save_scalar(var[Nshell],"var",make_string(Nshell));
		target.save_scalar(Mmax[Nshell],"Mmax",make_string(Nshell));
		target.save_scalar(Jgs[Nshell],"J",make_string(Nshell));
		lout << endl;
	}
	
	// calc P
//	double resP = 0.;
//	if (Nshell>=Nshellmin+2)
//	{
//		for (int i=0; i<deg(j); ++i)
//		{
//			double Pcontrib = sign(i0+i) * avg(g[Nshell-2].state, H.cc(i0+i), g[Nshell].state);
////					lout << "i=" << i << ", Pcontrib=" << Pcontrib << endl;
//			resP += Pcontrib;
//		}
//	}
//	resP /= sqrt(deg(j));
//	resP = abs(resP);
	
	// Pref
//	lout << ", ref(P)=";
//	if (abs(Sn_Pref_(j)-resP)<1e-3)
//	{
//		lout << termcolor::green;
//	}
//	else
//	{
//		lout << termcolor::red;
//	}
//	lout << Sn_Pref_(j) << termcolor::reset;
	
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

template<typename MODEL>
void NuclearManager<MODEL>::
compute_parallel (bool LOAD, bool SAVE, string wd, int Nruns, int Nshellmin, int Nshellmax)
{
	make_Hamiltonian(LOAD,SAVE,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jgs.resize(2*L+1);
	
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
	
	int Nshelllast = Nshellmax;
	if (Nshellmax==-1) Nshelllast = 2*L;
	
	#pragma omp parallel for
	for (int iNshell=Nshellmin; iNshell<=Nshelllast; iNshell+=2)
	{
		int jNshellmax = (iNshell+1<=Nshelllast)? 1:0;
		for (int jNshell=0; jNshell<=jNshellmax; ++jNshell)
		{
			int Nshell = iNshell+jNshell;
			int A = Z+Nclosed+Nshell;
			
			#pragma omp critical
			{
				lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << termcolor::reset << endl;
			}
			
			g[Nshell] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, Nruns, false, DMRG::VERBOSITY::SILENT);
			
			if (Nshell != 0 and Nshell != 2*L)
			{
				var[Nshell] = abs(avg(g[Nshell].state,H,H,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
			}
			else
			{
				var[Nshell] = 0.;
			}
			
			Mmax[Nshell] = g[Nshell].state.calc_Mmax();
			
			energies(Nshell) = g[Nshell].energy;
			energies_free(Nshell) = onsite_free.head(Nshell).sum();
			if (Nshell > 1 and Nshell<2*L-1) assert(g[Nshell].energy < energies_free(Nshell));
			
			n[Nshell].resize(Nlev);
			avgN[Nshell].resize(Nlev);
			
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
				
				i0 += deg(j);
				
				double Nplot = (avgN[Nshell](j)<1e-14)? 0 : avgN[Nshell](j);
				double nplot = (n[Nshell](j)<1e-14)?    0 : n[Nshell](j);
				//<< ", " << "P=" << resP;
			}
			target.create_group(make_string(Nshell));
			target.save_scalar(g[Nshell].energy,"E0",make_string(Nshell));
			target.save_scalar(energies_free(Nshell),"E0free",make_string(Nshell));
			target.save_vector(avgN[Nshell],"avgN",make_string(Nshell));
			target.save_vector(n[Nshell],"n",make_string(Nshell));
			target.save_scalar(var[Nshell],"var",make_string(Nshell));
			target.save_scalar(Mmax[Nshell],"Mmax",make_string(Nshell));
			target.save_scalar(Jgs[Nshell],"J",make_string(Nshell));
		}
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
