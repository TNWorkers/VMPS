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
	int index;
};

struct orbital
{
	orbital (const int &r)
	{
		l = r/2;
		s = (r%2==0)? UP:DN;
	}
	
	bool operator == (const orbital &b) const
	{
		return (l==b.l and s==b.s)? true:false;
	}
	
	int l;
	SPIN_INDEX s;
};

struct orbinfo
{
	bool operator == (const orbinfo &b) const
	{
		return (lev==b.lev and jx2==b.jx2 and label==b.label)? true:false;
	}
	
	std::array<int,4> lev;
	std::array<int,4> jx2;
	std::array<string,4> label;
};

double cgc (int j1x2, int j2x2, int Jx2, int m1x2, int m2x2, int Mx2)
{
	double Wigner3j = gsl_sf_coupling_3j(j1x2,j2x2,Jx2,m1x2,m2x2,-Mx2);
//	return pow(-1,-j1x2/2+j2x2/2-Mx2/2) * sqrt(Jx2+1.) * Wigner3j;
	return pow(-1,+j1x2/2-j2x2/2+Mx2/2) * sqrt(Jx2+1.) * Wigner3j;
}

struct NuclearInfo
{
	string info() const
	{
		stringstream ss;
		ss << "el=" << el << ", Zsingle=" << Zsingle << ", Nsingle=" << Nsingle << ", Nclosed=" << Nclosed << ", Nmax=" << Nmax << endl;
		return ss.str();
	}
	
	string el;
	int Nclosed;
	int Nmax;
	int Nsingle;
	int Zsingle;
	Nuclear::PARTICLE P;
	string PARAM;
	vector<string> levels;
	string levelstring;
	int Nlev;
};

void load_nuclearData (string el, string set, string rootfolder, NuclearInfo &info, double &V0, MatrixXd &Vij, vector<MatrixXd> &Vijnocgc, VectorXi &deg, VectorXd &eps0, bool &REF)
{
	REF = false;
	
	ifstream DataLoader(rootfolder+"data_nuclear/sets.sav");
	string line;
	while (getline(DataLoader, line))
	{
		if (line[0] == '#') continue;
		
		vector<string> split; 
		boost::split(split, line, boost::is_any_of("\t"));
		if (split[0] == el and split[1] == set)
		{
			//info.Z = Nuclear::Ztable(split[0]);
			info.el = split[0];
			info.Nclosed = stoi(split[2]);
			info.Nmax = stoi(split[3]);
			info.Nsingle = stoi(split[4]);
			info.Zsingle = stoi(split[5]);
			info.P = static_cast<Nuclear::PARTICLE>(stoi(split[9]));
			info.PARAM = split[10];
			lout << info.info() << endl;
			
			info.levelstring = split[6];
			boost::split(info.levels, info.levelstring, boost::is_any_of(","));
			info.Nlev = info.levels.size();
			Nuclear::replace_halves(info.levelstring);
			break;
		}
	}
	
	if (el == "Sn" and set == "benchmark")
	{
		// benchmark Sn values
		V0 = 1.;
		info.Nclosed = 50;
		info.Zsingle = 50;
		
		info.Nlev = 5;
		deg.resize(info.Nlev); deg << 4, 3, 2, 1, 6;
		
		eps0.resize(info.Nlev); eps0 << -6.121, -5.508, -3.749, -3.891, -3.778;
		
		Vij.resize(info.Nlev,info.Nlev);
		Vij.setZero();
		Vij(0,0) = 0.9850;
		Vij(0,1) = 0.5711;
		Vij(0,2) = 0.5184;
		Vij(0,3) = 0.2920;
		Vij(0,4) = 1.1454;
		Vij(1,1) = 0.7063;
		Vij(1,2) = 0.9056;
		Vij(1,3) = 0.3456;
		Vij(1,4) = 0.9546;
		Vij(2,2) = 0.4063;
		Vij(2,3) = 0.3515;
		Vij(2,4) = 0.6102;
		Vij(3,3) = 0.7244;
		Vij(3,4) = 0.4265;
		Vij(4,4) = 1.0599;
		
		for (int i=0; i<info.Nlev; ++i)
		for (int j=0; j<i; ++j)
		{
			Vij(i,j) = Vij(j,i);
		}
		
		int Jmax = 12;
		Vijnocgc.resize(Jmax/2+1);
		for (int J=0; J<=Jmax; J+=2)
		{
			Vijnocgc[J/2] = Vij;
		}
		
		for (int i=0; i<info.Nlev; ++i)
		for (int j=0; j<info.Nlev; ++j)
		{
			Vij(i,j) /= sqrt(deg(i)*deg(j));
		}
		
		REF = true;
	}
	else if (el == "h11shell" and set == "benchmark")
	{
		V0 = 1.;
		info.Nclosed = 0;
		info.Zsingle = 0;
		
		info.Nlev = 1;
		deg.resize(info.Nlev); deg << 6;
		
		eps0.resize(info.Nlev); eps0 << 0.;
		
		Vij.resize(info.Nlev,info.Nlev);
		Vij.setConstant(0.25);
		
		int Jmax = 10;
		Vijnocgc.resize(Jmax/2+1);
		for (int J=0; J<=Jmax; J+=2)
		{
			Vijnocgc[J/2] = Vij*6.;
		}
	}
	else if (el == "Richardson50" and set == "benchmark")
	{
		V0 = 3.;
		info.Nclosed = 0;
		info.Zsingle = 0;
		
		info.Nlev = 50;
		deg.resize(info.Nlev); deg.setConstant(1);
		
		eps0.resize(info.Nlev);
		for (int j=0; j<info.Nlev; ++j)
		{
			eps0[j] = j;
		}
		
		Vij.resize(info.Nlev,info.Nlev);
		Vij.setConstant(1.);
		
		int Jmax = 0;
		Vijnocgc.resize(Jmax/2+1);
		for (int J=0; J<=Jmax; J+=2)
		{
			Vijnocgc[J/2] = Vij;
		}
	}
	// From: Zelevinsky, Volya: Physics of Atomic Nuclei (2017), pp. 218-220
	else if (el == "f7shell" and set == "benchmark")
	{
		V0 = 2.535;
		std::array<double,4> V_J = {1., 0.981/V0, -0.140/V0, -0.664/V0};
		info.Nclosed = 0;
		info.Zsingle = 0;
		
		info.Nlev = 1;
		deg.resize(info.Nlev); deg << 4;
		
		eps0.resize(info.Nlev); eps0 << -9.626;
		
		Vij.resize(info.Nlev,info.Nlev);
		Vij.setConstant(1.);
		
		int Jmax = 6;
		Vijnocgc.resize(Jmax/2+1);
		for (int J=0; J<=Jmax; J+=2)
		{
			Vijnocgc[J/2] = Vij * V_J[J/2];
		}
	}
	else if (el == "testshell" and set == "benchmark")
	{
		V0 = 1.;
		std::array<double,3> V_J = {1., 0.5, 0.25};
		info.Nclosed = 0;
		info.Zsingle = 0;
		
		info.Nlev = 3;
		deg.resize(info.Nlev); deg << 1, 2, 3;
		
		eps0.resize(info.Nlev); eps0 << 0., 0., 0.;
		
		double lower_bound = 0.;
		double upper_bound = 1.;
		std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
		std::default_random_engine re;
		
		Vij.resize(info.Nlev,info.Nlev);
		Vij.setRandom();
		for (int i=0; i<info.Nlev; ++i)
		for (int j=0; j<=i; ++j)
		{
			Vij(i,j) = unif(re);
			Vij(j,i) = Vij(i,j);
		}
		
		int Jmax = 4;
		Vijnocgc.resize(Jmax/2+1);
		for (int J=0; J<=Jmax; J+=2)
		{
			Vijnocgc[J/2] = Vij * V_J[J/2];
		}
	}
	
	if (set != "benchmark")
	{
		HDF5Interface target(make_string(rootfolder+"NuclearLevels",info.PARAM,"/hdf5/NuclearData_Z=",info.Zsingle,"_N=",info.Nsingle,"_P=",info.P,"_j=",info.levelstring,".h5"), READ);
		target.load_matrix(Vij,"Vij","");
		target.load_vector(eps0,"eps0","");
		target.load_vector(deg,"deg","");
		
		int Jmax;
		target.load_scalar(Jmax,"Jmax","");
		Vijnocgc.resize(Jmax/2+1);
		for (int J=0; J<=Jmax; J+=2)
		{
			target.load_matrix(Vijnocgc[J/2],"Vijnocgc",make_string("J=",J));
		}
		
//		if (RANDOM)
//		{
//			static thread_local mt19937 generatorUniformReal(random_device{}());
//			uniform_real_distribution<double> distribution(0.0015, 0.015);
//			
//			for (int i=0; i<Vij.rows(); ++i)
//			for (int j=0; j<=i; ++j)
//			{
//				Vij(i,j) = distribution(generatorUniformReal);
//				Vij(j,i) = Vij(i,j);
//			}
//			lout << "Vij random=" << endl << Vij << endl;
//		}
	}
}

template<typename MODEL>
class NuclearManager
{
public:
	
	NuclearManager(){};
	
	NuclearManager (int Nclosed_input, int Nsingle_input, int Z_input, const ArrayXi &deg_input, const vector<string> &labels_input, 
	                const ArrayXd &eps0_input, const ArrayXXd &Vij_input, double V0_input=1.,
	                bool REF_input=false, string PARAM_input="Seminole")
	:Nclosed(Nclosed_input), Nsingle(Nsingle_input), Z(Z_input), deg(deg_input), labels(labels_input), eps0(eps0_input), Vij(Vij_input), V0(V0_input), REF(REF_input), PARAM(PARAM_input)
	{
		Nlev = deg.rows();
		L = deg.sum();
		lout << "Nlev=" << Nlev << ", L=" << L << endl;
		Vij.triangularView<Lower>() = Vij.transpose();
		levelstring = str(labels_input);
		
		construct();
	};
	
	NuclearManager (int Nclosed_input, int Nsingle_input, int Z_input,const string &filename, double V0_input, bool REF_input=false)
	:Nclosed(Nclosed_input), Nsingle(Nsingle_input), Z(Z_input), V0(V0_input), REF(REF_input)
	{
		HDF5Interface source(filename+".h5", READ);
		
		source.load_vector<int>(deg, "deg", "");
		source.load_vector<double>(eps0, "eps0", "");
		source.load_matrix<double>(Vij, "Vij", "");
		
		Nlev = deg.rows();
		L = deg.sum();
		lout << "Nlev=" << Nlev << ", L=" << L << endl;
		Vij.triangularView<Lower>() = Vij.transpose();
		
		labels.resize(Nlev);
		for (int i=0; i<Nlev; ++i)
		{
			source.load_char(make_string("lev",i).c_str(), labels[i]);
		}
		levelstring = str(labels);
		
		construct();
	};
	
	void construct();
	
	void make_Hamiltonian (bool LOAD=false, bool SAVE=false, string wd="./", bool ODD=true);
	void make_fullHamiltonian (bool LOAD=false, bool SAVE=false, string wd="./");
	
	tuple<Eigenstate<typename MODEL::StateXd>,string,int>
	calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE=LANCZOS::EDGE::GROUND, bool CALC_VAR=true, DMRG::VERBOSITY::OPTION VERB = DMRG::VERBOSITY::ON_EXIT) const;
	
	Eigenstate<typename MODEL::StateXd>
	calc_gs_full (int Nshell, LANCZOS::EDGE::OPTION EDGE=LANCZOS::EDGE::GROUND, bool CALC_VAR=true, DMRG::VERBOSITY::OPTION VERB = DMRG::VERBOSITY::ON_EXIT) const;
	
	void compute (bool LOAD_MPO=false, bool SAVE_MPO=false, string wd="./", int minNshell=0, int maxNshell=-1, bool SAVE=true);
	void compute_parallel (bool LOADv=false, bool SAVE_MPO=false, string wd="./", int minNshell=0, int maxNshell=-1, bool SAVE=true);
	
	ArrayXd get_occ() const;
	inline ArrayXd get_eps0() const {return eps0;}
	inline ArrayXi get_deg() const {return deg;}
	inline ArrayXi get_offset() const {return offset;}
	inline size_t length() const {return L;}
	inline ArrayXd get_sign() {return sign;}
	inline ArrayXd get_levelsign (int level) {return sign_per_level[level];}
	inline VectorXd get_onsite_free() const {return onsite_free;}
	inline MODEL get_H() const {return H;}
	inline MODEL get_Hfull() const {return Hfull;}
	
	double Sn_Eref (int N) const;
	ArrayXd Sn_nref (int Nshell) const;
	ArrayXd Sn_Nref (int Nshell) const;
	ArrayXd Sn_Pref (int Nshell) const;
	ArrayXd Sn_Sref14() const;
	
	const Eigenstate<typename MODEL::StateXd> &get_g (int Nshell) const {return g[Nshell];}
	MatrixXd get_Ghop() const {return Ghop;}
	VectorXd get_onsite() const {return onsite;}
	VectorXd get_Gloc() const {return Gloc;}
	VectorXi get_mfactor() const {return mfactor;}
	std::array<MatrixXd,2> get_FlipChart() const {return FlipChart;}
	ArrayXXd get_sign2d() const {return sign2d;}
	
	vector<tuple<orbital,orbital,orbital,orbital,double,orbinfo>> generate_couplingList (int J) const;
	
private:
	
	void save (string wd, int Nfrst=0, int Nlast=-1, int dNshell=1) const;
	
	bool REF = false;
	string PARAM = "Seminole";
	
	int Nclosed, Nsingle, Z;
	int Nlev;
	size_t L;
	
	string levelstring;
	vector<string> labels;
	VectorXi deg, offset, mfactor;
	VectorXd eps0;
	ArrayXd sign;
	std::array<MatrixXd,2> FlipChart;
	vector<boost::rational<int>> mlist, jlist;
	vector<int> orblist;
	ArrayXXd sign2d;
	
	vector<ArrayXd> sign_per_level;
	
	MatrixXd Vij, Ghop;
	double V0;
	VectorXd onsite, Gloc, onsite_free;
	VectorXd energies, energies_free;
	
	string outfile;
	
	vector<Eigenstate<typename MODEL::StateXd>> g;
	vector<double> var;
	vector<int> Mmax;
	vector<VectorXd> n, avgN;
	vector<string> Jlabel;
	vector<int> Jindex;
	
	MODEL H;
	vector<MODEL> Hodd;
	MODEL Hfull;
	
	vector<MODEL> Hperturb;
	vector<double> perturb_sweeps;
	vector<Mpo<typename MODEL::Symmetry>> make_cdagj (const MODEL &H_input) const;
	Mpo<typename MODEL::Symmetry> make_cdagj (int j, const MODEL &H_input) const;
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
	lout << "offset=" << offset.transpose() << endl;
	
	// G hopping
	Ghop.resize(L,L); Ghop.setZero();
	for (int i=0; i<Nlev; ++i)
	for (int j=0; j<Nlev; ++j)
	{
		Ghop.block(offset(i),offset(j),deg(i),deg(j)).setConstant(-V0*Vij(i,j));
	}
	
	if constexpr (MODEL::FAMILY == HUBBARD)
	{
		for (int i=0; i<L; ++i)
		for (int j=0; j<L; ++j)
		{
			Ghop(i,j) *= pow(-1,i+j); // cancels sign, so that one can use Vxy
		}
		
		// Convention: start with highest projection:
		// ...5/2,3/2,1/2
		// Careful when changing!
		sign2d.resize(L,L);
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
//		lout << "sign2d=" << endl << sign2d << endl;
		
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
	
	outfile = make_string("PairingResult_Z=",Z,"_Nclosed=",Nclosed,"_Nsingle=",Nsingle,"_V0=",V0);
	outfile += make_string("_j=",levelstring);
	if (PARAM!="Seminole") outfile += make_string("_PARAM=",PARAM);
	outfile += ".h5";
	lout << "outfile=" << outfile << endl;
	
	/*PermutationMatrix<Dynamic,Dynamic> P(L);
	P.setIdentity();
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(P.indices().data(), P.indices().data()+P.indices().size(), g);
	Ghop = P.inverse()*Ghop*P;
	onsite = P.inverse()*onsite;
	Gloc = P.inverse()*Gloc;*/
	
	mfactor.resize(L); mfactor.setZero();
	int l = 0;
	for (int j=0; j<Nlev; ++j)
	{
		boost::rational<int> jval = deg(j)-boost::rational<int>(1,2);
		
		// Convention: start with highest projection:
		// ...5/2,3/2,1/2
		// Careful when changing!
		for (boost::rational<int> mval=jval; mval>0; --mval)
		{
			mfactor(l) = (boost::rational<int>(2,1)*mval).numerator();
			mlist.push_back(+mval);
			mlist.push_back(-mval);
			jlist.push_back(jval);
			jlist.push_back(jval);
			orblist.push_back(j);
			orblist.push_back(j);
			++l;
		}
	}
//	lout << "mfactor=" << mfactor.transpose() << endl;
//	for (int i=0; i<mlist.size(); ++i)
//	{
//		lout << "m=" << mlist[i] << ", j=" << jlist[i] << ", orb=" << orblist[i] << endl;
//	}
	
	FlipChart[UP].resize(2*L,2*L); FlipChart[UP].setZero();
	FlipChart[DN].resize(2*L,2*L); FlipChart[DN].setZero();
	for (int i=0; i<mlist.size(); ++i)
	for (int j=0; j<mlist.size(); ++j)
	{
		double jval = boost::rational_cast<double>(jlist[i]);
		double m1   = boost::rational_cast<double>(mlist[i]);
		double m2   = boost::rational_cast<double>(mlist[j]);
		
		if (mlist[j] == mlist[i]+boost::rational<int>(1,1) and jlist[i]==jlist[j])
		{
			FlipChart[DN](i,j) = sqrt(jval*(jval+1.)-m1*m2);
		}
		if (mlist[j] == mlist[i]-boost::rational<int>(1,1) and jlist[i]==jlist[j])
		{
			FlipChart[UP](i,j) = sqrt(jval*(jval+1.)-m1*m2);
		}
	}
//	cout << "UP=" << endl << FlipChart[UP] << endl;
//	cout << "DN=" << endl << FlipChart[DN] << endl;
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
	/*if (Nshell == 12)
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
	}*/
	vector<double> EDenergies =
	{
		-0, //0
		-6.121, //1
		-14.0158183065649, //2
		-19.81090128197716, //3
		-27.49864249287803, //4
		-32.94931684903962, //5
		-40.44176691839276, //6
		-45.5231949893755, //7
		-52.83210266861026, //8
		-57.52194523311658, //9
		-64.64401299541129, //10
		-69.01055649707473, //11
		-75.83057632660048, //12
		-79.75643390536675, //13
		-86.30836166475844, //14
		-89.69722801960962, //15
		-95.94172568796624, //16
		-99.06316035101028, //17 -99.0631603509355 // 10 digits
		-104.9348539406987, //18
		-107.8672420394976, //19
		-113.3725141279236, //20
		-116.108976116819, //21
		-121.3002558685274, //22
		-123.8366779467813, //23
		-128.7462271575389, //24
		-131.0794704088455, //25
		-135.7291949708218, //26
		-137.8562329243708, //27
		-142.2621883681673, //28
		-144.1794965107825, //29
		-148.3543512502746, //30
		-150.0597499999999, //31
		-154.0119 //32
	};
	return EDenergies[Nshell];
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
make_Hamiltonian (bool LOAD, bool SAVE, string wd, bool ODD)
{
	string filename = make_string(wd,"H_Z=",Z,"_Nclosed=",Nclosed,"_Nsingle=",Nsingle,"_V0=",V0);
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
		if constexpr (MODEL::FAMILY == HUBBARD) params_loc.push_back({"REMOVE_SINGLE",true});
		
		for (size_t l=0; l<L; ++l)
		{
			if constexpr (MODEL::FAMILY == HUBBARD)
			{
				params_loc.push_back({"t0",onsite(l),l});
				params_loc.push_back({"U",Gloc(l),l});
				params_loc.push_back({"mfactor",mfactor(l),l});
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
//			cout << "compressing: site " << i+1 << "/" << Ghopx2.rows() << endl;
			vector<Param> params_tmp;
			params_tmp.push_back({"t",0.});
			params_tmp.push_back({"J",0.});
			params_tmp.push_back({"maxPower",1ul});
			if constexpr (MODEL::FAMILY == HUBBARD) params_tmp.push_back({"REMOVE_SINGLE",true});
			
			ArrayXXd Ghopx2row(Ghop.rows(),Ghop.cols());
			Ghopx2row.setZero();
			Ghopx2row.matrix().row(i) = Ghopx2.matrix().row(i);
			if constexpr (MODEL::FAMILY == HUBBARD)
			{
				params_tmp.push_back({"Vxyfull",Ghopx2row});
				for (size_t l=0; l<L; ++l) params_tmp.push_back({"mfactor",mfactor(l),l});
			}
			else if constexpr (MODEL::FAMILY == HEISENBERG)
			{
				params_tmp.push_back({"Jxyfull",Ghopx2row});
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
		
//		MODEL Hfull(L,params_full);
//		lout << "Hfull:" << Hfull.info() << endl;
		
		Mpo<typename MODEL::Symmetry,double> Hmpo(Terms);
		H = MODEL(Hmpo,params_full);
//		H.calc(2ul); // takes too long
		lout << H.info() << endl;
		lout << CompressionTimer.info("MPO compression even") << endl;
		
		if (SAVE) H.save(filename);
		
		// ODD
		if constexpr (MODEL::FAMILY == HUBBARD)
		{
			if (ODD)
			{
				Hodd.resize(Nlev);
				
				Stopwatch<> CompressionTimerOdd;
				#pragma omp parallel for
				for (int j=0; j<Nlev; ++j)
				{
					vector<Param> params_loc_odd;
					params_loc_odd.push_back({"t",0.});
					params_loc_odd.push_back({"J",0.});
					params_loc_odd.push_back({"maxPower",1ul});
					for (size_t l=0; l<L; ++l)
					{
						if (l<offset(j) or l>=offset(j)+deg(j)) params_loc_odd.push_back({"REMOVE_SINGLE",true,l});
						else                                    params_loc_odd.push_back({"REMOVE_SINGLE",false,l});
						
						params_loc_odd.push_back({"t0",onsite(l),l});
						params_loc_odd.push_back({"U",Gloc(l),l});
						params_loc_odd.push_back({"mfactor",mfactor(l),l});
					}
					
					MODEL Hloc_odd(L,params_loc_odd,BC::OPEN,DMRG::VERBOSITY::SILENT);
					MpoTerms<typename MODEL::Symmetry,double> Terms_odd = Hloc_odd;
					
					for (int i=0; i<Ghopx2.rows(); ++i)
					{
						vector<Param> params_tmp_odd;
						params_tmp_odd.push_back({"t",0.});
						params_tmp_odd.push_back({"J",0.});
						params_tmp_odd.push_back({"maxPower",1ul});
						for (size_t l=0; l<L; ++l)
						{
							if (l<offset(j) or l>=offset(j)+deg(j)) params_tmp_odd.push_back({"REMOVE_SINGLE",true,l});
							else                                    params_tmp_odd.push_back({"REMOVE_SINGLE",false,l});
							
							params_tmp_odd.push_back({"mfactor",mfactor(l),l});
						}
						
						ArrayXXd Ghopx2row(Ghop.rows(),Ghop.cols());
						Ghopx2row.setZero();
						Ghopx2row.matrix().row(i) = Ghopx2.row(i);
						params_tmp_odd.push_back({"Vxyfull",Ghopx2row});
						
						MODEL Hterm = MODEL(L, params_tmp_odd, BC::OPEN, DMRG::VERBOSITY::SILENT);
						Terms_odd.set_verbosity(DMRG::VERBOSITY::SILENT);
						Hterm.set_verbosity(DMRG::VERBOSITY::SILENT);
						Terms_odd = MODEL::sum(Terms_odd,Hterm);
					}
					
					vector<Param> params_full_odd = params_loc_odd;
					params_loc_odd.push_back({"Vxyfull",Ghopx2});
					
					Mpo<typename MODEL::Symmetry,double> Hmpo(Terms_odd);
					Hodd[j] = MODEL(Hmpo,params_full_odd);
				}
				lout << CompressionTimerOdd.info("MPO compression odd") << endl;
			}
		}
	}
	
	/*if constexpr (MODEL::FAMILY == HUBBARD)
	{
		// good:
		vector<double> tvals = {1., 0.5, 0.25, 0.125, 0.05};
		perturb_sweeps = {4, 4, 4, 4, 4};
		// bad:
//		vector<double> tvals = {4., 3., 2., 1., 0.5, 0.25, 0.125, 0.05, 0.01};
//		perturb_sweeps = {2, 2, 2, 2, 2, 2, 2, 2, 2};
		Hperturb.resize(tvals.size());
		for (int i=0; i<tvals.size(); ++i)
		{
			MODEL Hperturb(L,{{"t",tvals[i]},{"Bx",tvals[i]},{"maxPower",1ul}}, BC::OPEN, DMRG::VERBOSITY::SILENT);
			Hperturb = sum<MODEL,double>(H, Hperturb, DMRG::VERBOSITY::SILENT);
			Hperturb[i] = Hperturb;
		}
	}*/
	
//	cdagj = make_cdagj(H);
}

template<typename MODEL>
void NuclearManager<MODEL>::
make_fullHamiltonian (bool LOAD, bool SAVE, string wd)
{
	string filename = make_string(wd,"H_Z=",Z,"_Nclosed=",Nclosed,"_Nsingle=",Nsingle,"_V0=",V0);
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
				params_loc.push_back({"mfactor",mfactor(l),l});
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
//			cout << "compressing: site " << i+1 << "/" << Ghopx2.rows() << endl;
			vector<Param> params_tmp;
			params_tmp.push_back({"t",0.});
			params_tmp.push_back({"J",0.});
			params_tmp.push_back({"maxPower",1ul});
			
			ArrayXXd Ghopx2row(Ghop.rows(),Ghop.cols());
			Ghopx2row.setZero();
			Ghopx2row.matrix().row(i) = Ghopx2.matrix().row(i);
			if constexpr (MODEL::FAMILY == HUBBARD)
			{
				params_tmp.push_back({"Vxyfull",Ghopx2row});
				for (size_t l=0; l<L; ++l) params_tmp.push_back({"mfactor",mfactor(l),l});
			}
			else if constexpr (MODEL::FAMILY == HEISENBERG)
			{
				params_tmp.push_back({"Jxyfull",Ghopx2row});
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
		
		Mpo<typename MODEL::Symmetry,double> Hmpo(Terms);
		Hfull = MODEL(Hmpo,params_full);
		lout << CompressionTimer.info("MPO compression full") << endl;
		
		if (SAVE) H.save(filename);
	}
}

template<> 
vector<Mpo<typename VMPS::HubbardU1::Symmetry>> NuclearManager<VMPS::HubbardU1>::
make_cdagj (const VMPS::HubbardU1 &H_input) const
{
	vector<Mpo<typename VMPS::HubbardU1::Symmetry>> res;
	res.resize(Nlev);
	for (int j=0; j<Nlev; ++j)
	{
		for (int i=0; i<deg[j]; ++i)
		{
			auto cdagj1 = H_input.template cdag<UP>(offset(j)+i);
			auto cdagj2 = H_input.template cdag<DN>(offset(j)+i);
			if (i==0)
			{
				res[j] = sum(cdagj1,cdagj2);
			}
			else
			{
				res[j] = sum(res[j],cdagj1);
				res[j] = sum(res[j],cdagj2);
			}
		}
	}
	return res;
}

template<> 
Mpo<typename VMPS::HubbardU1::Symmetry> NuclearManager<VMPS::HubbardU1>::
make_cdagj (int j, const VMPS::HubbardU1 &H_input) const
{
	Mpo<typename VMPS::HubbardU1::Symmetry> res;
	for (int i=0; i<deg[j]; ++i)
	{
		auto cdagj1 = H_input.template cdag<UP>(offset(j)+i);
		auto cdagj2 = H_input.template cdag<DN>(offset(j)+i);
		if (i==0)
		{
			res = sum(cdagj1,cdagj2);
		}
		else
		{
			res = sum(res,cdagj1);
			res = sum(res,cdagj2);
		}
	}
	return res;
}

template<> 
vector<Mpo<typename VMPS::HubbardU1xU1::Symmetry>> NuclearManager<VMPS::HubbardU1xU1>::
make_cdagj (const VMPS::HubbardU1xU1 &H_input) const
{
	vector<Mpo<typename VMPS::HubbardU1xU1::Symmetry>> res;
	res.resize(Nlev);
	for (int j=0; j<Nlev; ++j)
	{
		for (int i=0; i<deg[j]; ++i)
		{
			auto cdagj1 = H_input.template cdag<UP>(offset(j)+i);
//			auto cdagj2 = H_input.template cdag<DN>(offset(j)+i);
			if (i==0)
			{
				res[j] = cdagj1; //sum(cdagj1,cdagj2);
			}
			else
			{
				res[j] = sum(res[j],cdagj1);
//				res[j] = sum(res[j],cdagj2);
			}
		}
	}
	return res;
}

template<> 
Mpo<typename VMPS::HubbardU1xU1::Symmetry> NuclearManager<VMPS::HubbardU1xU1>::
make_cdagj (int j, const VMPS::HubbardU1xU1 &H_input) const
{
	Mpo<typename VMPS::HubbardU1xU1::Symmetry> res;
	for (int i=0; i<deg[j]; ++i)
	{
		auto cdagj1 = H_input.template cdag<UP>(offset(j)+i);
//		auto cdagj2 = H_input.template cdag<DN>(offset(j)+i);
		if (i==0)
		{
			res = cdagj1; //sum(cdagj1,cdagj2);
		}
		else
		{
			res = sum(res,cdagj1);
//			res = sum(res,cdagj2);
		}
	}
	return res;
}

template<>
vector<Mpo<typename VMPS::HeisenbergU1::Symmetry>> NuclearManager<VMPS::HeisenbergU1>::
make_cdagj (const VMPS::HeisenbergU1 &H_input) const
{
	vector<Mpo<typename VMPS::HeisenbergU1::Symmetry>> res;
	return res;
}

/*template<>
Mpo<typename VMPS::HeisenbergU1::Symmetry> NuclearManager<VMPS::HeisenbergU1>::
make_cdagj (int j, const VMPS::HeisenbergU1 &H_input) const
{
	Mpo<typename VMPS::HeisenbergU1::Symmetry> res;
	return res;
}*/

template<typename MODEL>
vector<tuple<orbital,orbital,orbital,orbital,double,orbinfo>> NuclearManager<MODEL>::
generate_couplingList (int J) const
{
	vector<tuple<orbital,orbital,orbital,orbital,double,orbinfo>> res;
	
	for (int r1=0; r1<2*L; ++r1)
	for (int r2=0; r2<2*L; ++r2)
	for (int r3=0; r3<2*L; ++r3)
	for (int r4=0; r4<2*L; ++r4)
	{
		orbital o1(r1);
		orbital o2(r2);
		orbital o3(r3);
		orbital o4(r4);
		
		int j1x2 = (2*jlist[r1]).numerator();
		int j2x2 = (2*jlist[r2]).numerator();
		int j3x2 = (2*jlist[r3]).numerator();
		int j4x2 = (2*jlist[r4]).numerator();
		
		int m1x2 = (2*mlist[r1]).numerator();
		int m2x2 = (2*mlist[r2]).numerator();
		int m3x2 = (2*mlist[r3]).numerator();
		int m4x2 = (2*mlist[r4]).numerator();
		
		int lev1 = orblist[r1];
		int lev2 = orblist[r2];
		int lev3 = orblist[r3];
		int lev4 = orblist[r4];
		
		orbinfo oinfo;
		oinfo.jx2[0] = j1x2;
		oinfo.jx2[1] = j2x2;
		oinfo.jx2[2] = j3x2;
		oinfo.jx2[3] = j4x2;
		oinfo.lev[0] = lev1;
		oinfo.lev[1] = lev2;
		oinfo.lev[2] = lev3;
		oinfo.lev[3] = lev4;
		oinfo.label[0] = labels[lev1];
		oinfo.label[1] = labels[lev2];
		oinfo.label[2] = labels[lev3];
		oinfo.label[3] = labels[lev4];
		
		double factor = 0.;
		if (lev1==lev2 and lev3==lev4)
		//if (lev1==lev2 and lev3==lev4 and m1x2==-m2x2 and m3x2==-m4x2)
		{
			for (int M=-J; M<=J; ++M)
			{
				factor += cgc(j1x2,j2x2,2*J, m1x2,m2x2,2*M) * cgc(j3x2,j4x2,2*J, m3x2,m4x2,2*M);
			}
			//factor += cgc(j1x2,j2x2,2*J, m1x2,m2x2,2*0) * cgc(j3x2,j4x2,2*J, m3x2,m4x2,2*0);
		}
		if (factor != 0.)
		{
			bool ADDED = false;
			for (int i=0; i<res.size(); ++i)
			{
				if (
					(get<0>(res[i])==o2 and get<1>(res[i])==o1 and // exchanged -> minus
					 get<2>(res[i])==o4 and get<3>(res[i])==o3 and 
					 get<5>(res[i])==oinfo)
						or 
					(get<0>(res[i])==o1 and get<1>(res[i])==o2 and 
					 get<2>(res[i])==o3 and get<3>(res[i])==o4 and // exchanged -> minus
					 get<5>(res[i])==oinfo)
				   )
				{
					get<4>(res[i]) -= factor;
					ADDED = true;
				}
				else if (get<0>(res[i])==o2 and get<1>(res[i])==o1 and // exchanged -> minus
					     get<2>(res[i])==o3 and get<3>(res[i])==o4 and // doubly exchanged -> plus
					     get<5>(res[i])==oinfo)
				{
					get<4>(res[i]) += factor;
					ADDED = true;
				}
				if (ADDED) break;
			}
			if (!ADDED and abs(factor)>1e-8)
			{
				// permute o3 & o4 -> minus
				// a†(jm)a†(jm)a(j'n')a(j'm') → -a†(jm)a†(jn)a(j'm')a(j'n')
				// in order to have ordering a†(↑)a†(↓) * a(↓)a(↑) = L^+ * L^-
				res.push_back(make_tuple(o1,o2,o3,o4, -factor, oinfo));
			}
		}
	}
	return res;
}

template<>
tuple<Eigenstate<typename VMPS::HubbardU1::StateXd>,string,int> NuclearManager<VMPS::HubbardU1>::
calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE, bool CALC_VAR, DMRG::VERBOSITY::OPTION VERB) const
{
	qarray<VMPS::HubbardU1::Symmetry::Nq> Q = VMPS::HubbardU1::singlet(Nshell);
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = 12ul;
	GlobParam.max_halfsweeps = 24ul;
	GlobParam.Minit = 100ul;
	GlobParam.Qinit = 100ul;
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ;
	GlobParam.CALC_S_ON_EXIT = false;
	GlobParam.Mlimit = 500ul;
//	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
	GlobParam.tol_eigval = 1e-10;
	GlobParam.tol_state = 1e-8;
	
	DMRG::CONTROL::DYN  DynParam;
	size_t lim2site = 24ul;
	DynParam.iteration = [lim2site] (size_t i) {return (i<lim2site and i%2==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	DynParam.max_alpha_rsvd = [lim2site] (size_t i) {return (i<=lim2site+4ul)? 1e4:0.;};
	size_t Mincr_per = 2ul;
	DynParam.Mincr_per = [Mincr_per] (size_t i) {return Mincr_per;};
//	size_t min_Nsv_val = (Nshell<=4 or Nshell>=min(2*static_cast<int>(L)-4,0))? 1ul:0ul;
//	DynParam.min_Nsv = [min_Nsv_val] (size_t i) {return min_Nsv_val;};
	
	Stopwatch<> Timer;
	
	string Jlabel_res = "0";
	int Jindex_res = -1;
	Eigenstate<typename VMPS::HubbardU1::StateXd> gres;
	size_t Nguess = 4;
	vector<jEigenstate<typename VMPS::HubbardU1::StateXd>> godd;
	vector<jEigenstate<typename VMPS::HubbardU1::StateXd>> ginit;
	
	typename VMPS::HubbardU1::Solver DMRGs;
	DMRGs.userSetGlobParam();
	DMRGs.userSetDynParam();
	DMRGs.GlobParam = GlobParam;
	DMRGs.DynParam = DynParam;
	
	if (Nshell%2 == 0)
	{
		DMRGs.edgeState(H, gres, Q, EDGE);
	}
	else
	{
		//#pragma omp parallel for
		for (int j=0; j<Nlev; ++j)
		{
			double Ediff = 1.;
			int Niter = 0;
			Eigenstate<typename VMPS::HubbardU1::StateXd> gprev;
			while (Ediff > 1e-6)
			{
				auto DMRGt = DMRGs;
				DMRGt.GlobParam.min_halfsweeps = 1ul;
				DMRGt.set_verbosity(DMRG::VERBOSITY::SILENT);
				DMRGt.GlobParam.INITDIR = (Niter%2==1)? DMRG::DIRECTION::LEFT:DMRG::DIRECTION::RIGHT;
				DMRGt.edgeState(Hodd[j], gprev, {Nshell-1}, EDGE);
				Ediff = abs(gprev.energy-g[Nshell-1].energy);
				if (VERB > DMRG::VERBOSITY::SILENT) lout << "Ediff=" << Ediff << endl;
				++Niter;
				if (Niter == 4) break;
			}
			jEigenstate<typename VMPS::HubbardU1::StateXd> gtmp;
			bool INCLUDE = true;
			/*if (abs(gprev.energy)>1e-7 and Nshell>L)
			{
				INCLUDE = false;
			}*/
			if (2.*deg(j)-avgN[Nshell-1](j) > 0.01 and INCLUDE)
			{
				double compr_tol = (Nshell<=10 or Nshell>=max(2*static_cast<int>(L)-10,0))? 2.:1e-8;
				OxV_exact(make_cdagj(j,Hodd[j]), gprev.state, gtmp.eigenstate.state, compr_tol, DMRG::VERBOSITY::SILENT);
				gtmp.eigenstate.state /= sqrt(dot(gtmp.eigenstate.state,gtmp.eigenstate.state));
				gtmp.eigenstate.energy = avg(gtmp.eigenstate.state, Hodd[j], gtmp.eigenstate.state);
				gtmp.label = labels[j];
				gtmp.index = j;
				//#pragma omp critical
				{
					ginit.push_back(gtmp);
				}
			}
			if (VERB > DMRG::VERBOSITY::SILENT)
			{
				#pragma omp critical
				{
					lout << "guess: j=" << j 
					     << ", label=" << labels[j] 
					     << ", energy=" << gtmp.eigenstate.energy 
					     << ", Niter=" << Niter << endl;
				}
			}
		}
		
		assert(ginit.size()>0);
		
		sort(ginit.begin(), ginit.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
		
		// Calculate the ground state for the lowest guesses
		//#pragma omp parallel for
		for (int i=0; i<min(Nguess,ginit.size()); ++i)
		{
			int j = ginit[i].index;
			godd.push_back(ginit[i]);
			
			double var = 1.;
			int Niter = 0;
			while (var > 1e-5)
			{
				if (Niter>0 and VERB > DMRG::VERBOSITY::SILENT)
				{
					#pragma omp critical
					{
						lout << termcolor::yellow << "Nshell=" << Nshell 
						     << ": must restart j=" << j << " with var=" << var << termcolor::reset << endl;
					}
				}
				
				auto DMRGr = DMRGs;
				DMRGr.GlobParam.min_halfsweeps = 2ul;
				DMRGr.set_verbosity(DMRG::VERBOSITY::SILENT);
				DMRGr.edgeState(Hodd[j], godd[i].eigenstate, Q, EDGE, (Niter==0)?true:false);
				var = abs(avg(godd[i].eigenstate.state,Hodd[j],Hodd[j],godd[i].eigenstate.state)-pow(godd[i].eigenstate.energy,2))/L;
				if (Niter>0 and var<1e-5)
				{
					#pragma omp critical
					{
						lout << "Nshell=" << Nshell << ", j=" << j << ", new try brought: var=" << var << endl;
					}
				}
				++Niter;
				if (Niter == 21) break;
			}
		}
		
		sort(godd.begin(), godd.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
		
		gres = godd[0].eigenstate;
		Jlabel_res = godd[0].label;
		Jindex_res = godd[0].index;
		
		if (VERB > DMRG::VERBOSITY::SILENT)
		{
			for (int i=0; i<min(Nguess,ginit.size()); ++i)
			{
				lout << "j=" << godd[i].index << ", label=" << godd[i].label << ", energy=" << godd[i].eigenstate.energy << endl;
//				godd[i].eigenstate.state.graph(make_string("j=",godd[i].index));
			}
		}
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::blue << setprecision(16) << gres.energy << setprecision(6) << termcolor::reset << endl;
	}
	
	if (CALC_VAR)
	{
		double var;
		if (Nshell%2 == 0)
		{
			var = abs(avg(gres.state,H,H,gres.state)-pow(gres.energy,2))/L;
		}
		else
		{
			var = abs(avg(gres.state,Hodd[Jindex_res],Hodd[Jindex_res],gres.state)-pow(gres.energy,2))/L;
		}
		if (VERB > DMRG::VERBOSITY::SILENT) lout << "var=" << var << endl;
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT) lout << Timer.info("edge state") << endl;
	
	return {gres, Jlabel_res, Jindex_res};
}

template<>
tuple<Eigenstate<typename VMPS::HubbardU1xU1::StateXd>,string,int> NuclearManager<VMPS::HubbardU1xU1>::
calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE, bool CALC_VAR, DMRG::VERBOSITY::OPTION VERB) const
{
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = 12ul;
	GlobParam.max_halfsweeps = 24ul;
	GlobParam.Minit = 100ul;
	GlobParam.Qinit = 100ul;
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ;
	GlobParam.CALC_S_ON_EXIT = false;
	GlobParam.Mlimit = 500ul;
//	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
	GlobParam.tol_eigval = 1e-10;
	GlobParam.tol_state = 1e-8;
	
	DMRG::CONTROL::DYN  DynParam;
	size_t lim2site = 24ul;
	DynParam.iteration = [lim2site] (size_t i) {return (i<lim2site and i%2==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	DynParam.max_alpha_rsvd = [lim2site] (size_t i) {return (i<=lim2site+4ul)? 1e4:0.;};
	size_t Mincr_per = 2ul;
	DynParam.Mincr_per = [Mincr_per] (size_t i) {return Mincr_per;};
//	size_t min_Nsv_val = (Nshell<=4 or Nshell>=min(2*static_cast<int>(L)-4,0))? 1ul:0ul;
//	DynParam.min_Nsv = [min_Nsv_val] (size_t i) {return min_Nsv_val;};
	
	Stopwatch<> Timer;
	
	string Jlabel_res = "0";
	int Jindex_res = -1;
	Eigenstate<typename VMPS::HubbardU1xU1::StateXd> gres;
	vector<jEigenstate<typename VMPS::HubbardU1xU1::StateXd>> godd;
	
	typename VMPS::HubbardU1xU1::Solver DMRGs;
	DMRGs.userSetGlobParam();
	DMRGs.userSetDynParam();
	DMRGs.GlobParam = GlobParam;
	DMRGs.DynParam = DynParam;
	
	if (Nshell%2 == 0)
	{
		qarray<VMPS::HubbardU1xU1::Symmetry::Nq> Q = VMPS::HubbardU1xU1::singlet(Nshell);
		DMRGs.edgeState(H, gres, Q, EDGE);
	}
	else
	{
		//#pragma omp parallel for
		for (int j=0; j<Nlev; ++j)
		{
			qarray<VMPS::HubbardU1xU1::Symmetry::Nq> Q = {2*deg(j)-1,Nshell};
			
			typename VMPS::HubbardU1xU1::Solver DMRGr;
			DMRGr.userSetGlobParam();
			DMRGr.userSetDynParam();
			DMRGr.GlobParam = GlobParam;
			DMRGr.DynParam = DynParam;
			jEigenstate<typename VMPS::HubbardU1xU1::StateXd> gtry;
			gtry.label = labels[j];
			gtry.index = j;
			DMRGr.edgeState(Hfull, gtry.eigenstate, Q, EDGE);
			godd.push_back(gtry);
			cout << "j=" << j << ", label=" << labels[j] << ", energy=" << setprecision(16) << gtry.eigenstate.energy << setprecision(6) << ", Q=" << Q << endl;
		}
		gres = godd[0].eigenstate;
		Jlabel_res = godd[0].label;
		Jindex_res = godd[0].index;
		for (int j=1; j<Nlev; ++j)
		{
			if (godd[j].eigenstate.energy < gres.energy)
			{
				gres = godd[j].eigenstate;
				Jlabel_res = godd[j].label;
				Jindex_res = godd[j].index;
			}
		}
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::blue << setprecision(16) << gres.energy << setprecision(6) << termcolor::reset << endl;
	}
	
	if (CALC_VAR)
	{
		double var;
		if (Nshell%2 == 0)
		{
			var = abs(avg(gres.state,H,H,gres.state)-pow(gres.energy,2))/L;
		}
		else
		{
//			var = abs(avg(gres.state,Hodd[Jindex_res],Hodd[Jindex_res],gres.state)-pow(gres.energy,2))/L;
			var = abs(avg(gres.state,Hfull,Hfull,gres.state)-pow(gres.energy,2))/L;
		}
		if (VERB > DMRG::VERBOSITY::SILENT) lout << "var=" << var << endl;
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT) lout << Timer.info("edge state") << endl;
	
	return {gres, Jlabel_res, Jindex_res};
}

template<typename MODEL>
Eigenstate<typename MODEL::StateXd> NuclearManager<MODEL>::
calc_gs_full (int Nshell, LANCZOS::EDGE::OPTION EDGE, bool CALC_VAR, DMRG::VERBOSITY::OPTION VERB) const
{
	assert(Nshell%2 == 0);
	assert(Hfull.length() != 0 and "Calculate full Hamiltonian first!");
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = 12ul;
	GlobParam.max_halfsweeps = 24ul;
	GlobParam.Minit = 100ul;
	GlobParam.Qinit = 100ul;
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ;
	GlobParam.CALC_S_ON_EXIT = false;
	GlobParam.Mlimit = 500ul;
//	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
	GlobParam.tol_eigval = 1e-10;
	GlobParam.tol_state = 1e-8;
	
	DMRG::CONTROL::DYN  DynParam;
	size_t lim2site = 24ul;
	DynParam.iteration = [lim2site] (size_t i) {return (i<lim2site and i%2==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	DynParam.max_alpha_rsvd = [lim2site] (size_t i) {return (i<=lim2site+4ul)? 1e4:0.;};
	size_t Mincr_per = 2ul;
	DynParam.Mincr_per = [Mincr_per] (size_t i) {return Mincr_per;};
//	size_t min_Nsv_val = (Nshell<=4 or Nshell>=min(2*static_cast<int>(L)-4,0))? 1ul:0ul;
//	DynParam.min_Nsv = [min_Nsv_val] (size_t i) {return min_Nsv_val;};
	
	Eigenstate<typename MODEL::StateXd> gres;
	
	typename MODEL::Solver DMRGs;
	DMRGs.userSetGlobParam();
	DMRGs.userSetDynParam();
	DMRGs.GlobParam = GlobParam;
	DMRGs.DynParam = DynParam;
	
	if (Nshell%2 == 0)
	{
		qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(Nshell);
		DMRGs.edgeState(Hfull, gres, Q, EDGE);
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::blue << setprecision(16) << gres.energy << setprecision(6) << termcolor::reset << endl;
	}
	
	return gres;
}

/*template<>
Eigenstate<typename VMPS::HubbardU1::StateXd> NuclearManager<VMPS::HubbardU1>::
calc_gs_full (int Nshell, LANCZOS::EDGE::OPTION EDGE, int Nruns, bool CALC_VAR, DMRG::VERBOSITY::OPTION VERB) const
{
	qarray<VMPS::HubbardU1::Symmetry::Nq> Q = VMPS::HubbardU1::singlet(Nshell);
	
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
	
	int Nruns_adjusted = (Nshell%2==0)? Nruns:1;
	if (Nshell==0 or Nshell==2*L) Nruns_adjusted = 1;
//	if (Nshell==42 and Nlev==13)  Nruns_adjusted = 10;
	
	vector<Eigenstate<typename VMPS::HubbardU1::StateXd>> gL(Nruns_adjusted);
	vector<Eigenstate<typename VMPS::HubbardU1::StateXd>> gR(Nruns_adjusted);
	ArrayXd energiesL(Nruns_adjusted);
	ArrayXd energiesR(Nruns_adjusted);
	string JgsL, JgsR;
	
	vector<jEigenstate<typename VMPS::HubbardU1::StateXd>> ginit;
	if (Nshell%2==1)
	{
		jEigenstate<typename VMPS::HubbardU1::StateXd> gtmp;
		for (int j=0; j<Nlev; ++j)
		{
//			#pragma omp critical
//			{
//				lout << "j=" << j << ", label=" << labels[j] << ", free space=" << 2.*deg(j)-avgN[Nshell-1](j) << endl;
//			}
			if (2.*deg(j)-avgN[Nshell-1](j) > 0.01)
			{
				double compr_tol = (Nshell<=10 or Nshell>=max(2*static_cast<int>(L)-10,0))? 2.:1e-8;
//				double compr_tol = 2.;
				OxV_exact(cdagj[j], g[Nshell-1].state, gtmp.eigenstate.state, compr_tol, DMRG::VERBOSITY::SILENT);
				gtmp.eigenstate.state /= sqrt(dot(gtmp.eigenstate.state,gtmp.eigenstate.state));
				gtmp.eigenstate.energy = avg(gtmp.eigenstate.state, H, gtmp.eigenstate.state);
				ginit.push_back(gtmp);
				ginit[ginit.size()-1].label = labels[j];
			}
		}
		
		sort(ginit.begin(), ginit.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
		
		if (VERB > DMRG::VERBOSITY::SILENT)
		{
			for (int j=0; j<ginit.size(); ++j)
			{
				lout << "E(" << ginit[j].label << ")=" << ginit[j].eigenstate.energy << endl;
			}
		}
	}
	
	for (int r=0; r<Nruns_adjusted; ++r)
	{
		if (VERB > DMRG::VERBOSITY::SILENT) lout << "------r=" << r << "------" << endl;
		
		std::array<typename VMPS::HubbardU1::Solver,2> DMRG_LR;
		for (int i=0; i<2; ++i)
		{
			DMRG_LR[i] = typename VMPS::HubbardU1::Solver(DMRG::VERBOSITY::SILENT);
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
					for (int i=0; i<Hperturb.size(); ++i)
					{
						DMRG_LR[0].GlobParam.min_halfsweeps = perturb_sweeps[i];
						DMRG_LR[0].GlobParam.max_halfsweeps = perturb_sweeps[i];
						DMRG_LR[0].edgeState(Hperturb[i], gL[r], Q, EDGE, (i==0)?false:true);
					}
					DMRG_LR[0].GlobParam.min_halfsweeps = 10ul;
					DMRG_LR[0].GlobParam.max_halfsweeps = 32ul;
					DMRG_LR[0].set_verbosity(VERB);
					DMRG_LR[0].edgeState(H, gL[r], Q, EDGE, true);
					DMRG_LR[0].edgeState(H, gL[r], Q, EDGE);
				}
				else
				{
					DMRG_LR[0].GlobParam.min_halfsweeps = 12ul;
					DMRG_LR[0].GlobParam.max_halfsweeps = 36ul;
					vector<jEigenstate<typename VMPS::HubbardU1::StateXd>> gfinalL = ginit;
					for (int j=0; j<min(gfinalL.size(),4ul); ++j)
					{
						DMRG_LR[0].edgeState(H, gfinalL[j].eigenstate, Q, EDGE, true);
					}
					sort(gfinalL.begin(), gfinalL.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
					gL[r] = gfinalL[0].eigenstate;
					JgsL = gfinalL[0].label;
				}
				
				energiesL® = gL[r].energy;
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
					for (int i=0; i<Hperturb.size(); ++i)
					{
						DMRG_LR[1].GlobParam.min_halfsweeps = perturb_sweeps[i];
						DMRG_LR[1].GlobParam.max_halfsweeps = perturb_sweeps[i];
						DMRG_LR[1].edgeState(Hperturb[i], gR[r], Q, EDGE, (i==0)?false:true);
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
					vector<jEigenstate<typename VMPS::HubbardU1::StateXd>> gfinalR = ginit;
					for (int j=0; j<min(gfinalR.size(),4ul); ++j)
					{
						DMRG_LR[1].edgeState(H, gfinalR[j].eigenstate, Q, EDGE, true);
					}
					sort(gfinalR.begin(), gfinalR.end(), [] (const auto& lhs, const auto& rhs) {return lhs.eigenstate.energy < rhs.eigenstate.energy;});
					gR[r] = gfinalR[0].eigenstate;
					JgsR = gfinalR[0].label;
				}
				
				energiesR® = gR[r].energy;
			}
		}
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT)
	{
		for (int r=0; r<Nruns_adjusted; ++r)
		{
			lout << termcolor::blue << setprecision(9) << gL[r].energy << "\t" << gR[r].energy << setprecision(6) << termcolor::reset << endl;
		}
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
			lout << termcolor::yellow << "Found " << edgeStateLabel << " state at r=" << indexL << " (L→R)" << termcolor::reset << endl;
		}
		else if (indexL > 1 and valL<valR)
		{
			lout << termcolor::red << "Found " << edgeStateLabel << " state at r=" << indexL << " (L→R)" << termcolor::reset << endl;
		}
		if (indexR == 1 and valL>valR)
		{
			lout << termcolor::yellow << "Found " << edgeStateLabel << " state at r=" << indexR << " (R→L)" << termcolor::reset << endl;
		}
		else if (indexR > 1  and valL>valR)
		{
			lout << termcolor::red << "Found " << edgeStateLabel << " state at r=" << indexR << " (R→L)" << termcolor::reset << endl;
		}
	}
	
	Eigenstate<typename VMPS::HubbardU1::StateXd> res;
	string Jres;
	
	if (EDGE == LANCZOS::EDGE::GROUND)
	{
		res = (gL[0].energy<=gR[0].energy)? gL[0]:gR[0];
		if (Nshell%2 == 1)
		{
			Jres = (gL[0].energy<=gR[0].energy)? JgsL:JgsR;
		}
		else
		{
			Jres = "0";
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
			Jres = (gL[0].energy>=gR[0].energy)? JgsL:JgsR;
		}
		else
		{
			Jres = "0";
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
}*/

template<>
tuple<Eigenstate<typename VMPS::HeisenbergU1::StateXd>,string,int> NuclearManager<VMPS::HeisenbergU1>::
calc_gs (int Nshell, LANCZOS::EDGE::OPTION EDGE, bool CALC_VAR, DMRG::VERBOSITY::OPTION VERB) const
{
	qarray<VMPS::HeisenbergU1::Symmetry::Nq> Q = {Nshell-static_cast<int>(L)};
	
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = 16ul;
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
//	size_t min_Nsv_val = (Nshell<=4 or Nshell>=min(2*static_cast<int>(L)-4,0))? 1ul:0ul;
//	DynParam.min_Nsv = [min_Nsv_val] (size_t i) {return min_Nsv_val;};
	
	Stopwatch<> Timer;
	
	Eigenstate<typename VMPS::HeisenbergU1::StateXd> gres;
	
	typename VMPS::HeisenbergU1::Solver DMRGs;
	DMRGs.userSetGlobParam();
	DMRGs.userSetDynParam();
	DMRGs.GlobParam = GlobParam;
	DMRGs.DynParam = DynParam;
	
	DMRGs.edgeState(H, gres, Q, EDGE);
	
	if (VERB > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::blue << setprecision(16) << gres.energy << setprecision(6) << termcolor::reset << endl;
	}
	
	if (CALC_VAR)
	{
		double var = abs(avg(gres.state,H,H,gres.state)-pow(gres.energy,2))/L;
		if (VERB > DMRG::VERBOSITY::SILENT) lout << "var=" << var << endl;
	}
	
	if (VERB > DMRG::VERBOSITY::SILENT) lout << Timer.info("edge state") << endl;
	
	return {gres,"0",-1};
}

template<>
void NuclearManager<VMPS::HubbardU1xU1>::
compute (bool LOAD_MPO, bool SAVE_MPO, string wd, int minNshell, int maxNshell, bool SAVE)
{
	make_Hamiltonian(LOAD_MPO,SAVE_MPO,wd,false);
	make_fullHamiltonian(LOAD_MPO,SAVE_MPO,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jlabel.resize(2*L+1);
	Jindex.resize(2*L+1);
	energies.resize(2*L+1);
	energies_free.resize(2*L+1);
	
	int Nshelllast = (maxNshell==-1)? 2*L : maxNshell;
	
	for (int Nshell=minNshell; Nshell<=Nshelllast; Nshell+=1)
	{
		int A = Z+Nclosed+Nshell;
		lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << ", progress=" << round(Nshell*1./(2*L)*100,1) << "%" << termcolor::reset << endl;
		
		auto [gres, Jlabel_res, Jindex_res] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, false);
		g[Nshell] = gres;
		Jlabel[Nshell] = Jlabel_res;
		Jindex[Nshell] = Jindex_res;
		
		lout << g[Nshell].state.info() << endl;
		
		lout << "J of ground state: " << Jlabel[Nshell] << ", index=" << Jindex[Nshell] << endl;
		
		if (Nshell != 0 and Nshell != 2*L)
		{
			if (Nshell%2 == 0)
			{
				var[Nshell] = abs(avg(g[Nshell].state,H,H,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
			}
			else
			{
//				var[Nshell] = abs(avg(g[Nshell].state,Hodd[Jindex[Nshell]],Hodd[Jindex[Nshell]],g[Nshell].state)-pow(g[Nshell].energy,2))/L;
				var[Nshell] = abs(avg(g[Nshell].state,Hfull,Hfull,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
			}
		}
		else
		{
			var[Nshell] = 0.;
		}
		lout << termcolor::blue << "var=" << var[Nshell] << termcolor::reset << endl;
		
		Mmax[Nshell] = g[Nshell].state.calc_Mmax();
		
		energies(Nshell) = g[Nshell].energy;
		energies_free(Nshell) = onsite_free.head(Nshell).sum();
		lout << "noninteracting energy=" << energies_free(Nshell) << ", interacting energy=" << energies(Nshell) << endl;
		if (Nshell > 1 and Nshell<2*L-1) assert(g[Nshell].energy < energies_free(Nshell));
		
		if (Z==50 and Nclosed==50 and REF)
		{
			double diff = abs(g[Nshell].energy-Sn_Eref(Nshell));
			if (diff<1e-2) lout << termcolor::green;
			else           lout << termcolor::red;
			lout << "ref=" << Sn_Eref(Nshell) << ", diff=" << diff << endl;
			lout << termcolor::reset;
		}
		lout << "E_B/A=" << abs(g[Nshell].energy)/Nshell << " MeV" << endl;
		
		n[Nshell].resize(Nlev);
		avgN[Nshell].resize(Nlev);
		
		for (int j=0; j<Nlev; ++j)
		{
			avgN[Nshell](j) = 0.;
			for (int i=0; i<deg(j); ++i)
			{
				if (Nshell%2==0)
				{
					avgN[Nshell](j) += avg(g[Nshell].state, H.n(offset(j)+i), g[Nshell].state);
				}
				else
				{
//					avgN[Nshell](j) += avg(g[Nshell].state, Hodd[Jindex[Nshell]].n(offset(j)+i), g[Nshell].state);
					avgN[Nshell](j) += avg(g[Nshell].state, Hfull.n(offset(j)+i), g[Nshell].state);
				}
			}
			n[Nshell](j) = avgN[Nshell](j)/(2.*deg(j));
			
			double Nplot = (avgN[Nshell](j)<1e-14)? 0 : avgN[Nshell](j);
			double nplot = (n[Nshell](j)   <1e-14)? 0 : n[Nshell](j);
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
		lout << endl;
	}
	
	if (SAVE) save(wd, minNshell, Nshelllast);
}

template<>
void NuclearManager<VMPS::HubbardU1>::
compute (bool LOAD_MPO, bool SAVE_MPO, string wd, int minNshell, int maxNshell, bool SAVE)
{
	make_Hamiltonian(LOAD_MPO,SAVE_MPO,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jlabel.resize(2*L+1);
	Jindex.resize(2*L+1);
	energies.resize(2*L+1);
	energies_free.resize(2*L+1);
	
	int Nshelllast = (maxNshell==-1)? 2*L : maxNshell;
	
	for (int Nshell=minNshell; Nshell<=Nshelllast; ++Nshell)
	{
		int A = Z+Nclosed+Nshell;
		lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << ", progress=" << round(Nshell*1./(2*L)*100,1) << "%" << termcolor::reset << endl;
		
		auto [gres, Jlabel_res, Jindex_res] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, false);
		g[Nshell] = gres;
		Jlabel[Nshell] = Jlabel_res;
		Jindex[Nshell] = Jindex_res;
		
		lout << g[Nshell].state.info() << endl;
		
		lout << "J of ground state: " << Jlabel[Nshell] << ", index=" << Jindex[Nshell] << endl;
		
		if (Nshell != 0 and Nshell != 2*L)
		{
			if (Nshell%2 == 0)
			{
				var[Nshell] = abs(avg(g[Nshell].state,H,H,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
			}
			else
			{
				var[Nshell] = abs(avg(g[Nshell].state,Hodd[Jindex[Nshell]],Hodd[Jindex[Nshell]],g[Nshell].state)-pow(g[Nshell].energy,2))/L;
			}
		}
		else
		{
			var[Nshell] = 0.;
		}
		lout << termcolor::blue << "var=" << var[Nshell] << termcolor::reset << endl;
		
		Mmax[Nshell] = g[Nshell].state.calc_Mmax();
		
		energies(Nshell) = g[Nshell].energy;
		energies_free(Nshell) = onsite_free.head(Nshell).sum();
		lout << "noninteracting energy=" << energies_free(Nshell) << ", interacting energy=" << energies(Nshell) << endl;
		if (Nshell > 1 and Nshell<2*L-1) assert(g[Nshell].energy < energies_free(Nshell));
		
		if (Z==50 and Nclosed==50 and REF)
		{
			double diff = abs(g[Nshell].energy-Sn_Eref(Nshell));
			if (diff<1e-2) lout << termcolor::green;
			else           lout << termcolor::red;
			lout << "ref=" << Sn_Eref(Nshell) << ", diff=" << diff << endl;
			lout << termcolor::reset;
		}
		lout << "E_B/A=" << abs(g[Nshell].energy)/Nshell << " MeV" << endl;
		
		n[Nshell].resize(Nlev);
		avgN[Nshell].resize(Nlev);
		
		for (int j=0; j<Nlev; ++j)
		{
			avgN[Nshell](j) = 0.;
			for (int i=0; i<deg(j); ++i)
			{
				if (Nshell%2==0)
				{
					avgN[Nshell](j) += avg(g[Nshell].state, H.n(offset(j)+i), g[Nshell].state);
				}
				else
				{
					avgN[Nshell](j) += avg(g[Nshell].state, Hodd[Jindex[Nshell]].n(offset(j)+i), g[Nshell].state);
				}
			}
			n[Nshell](j) = avgN[Nshell](j)/(2.*deg(j));
			
			double Nplot = (avgN[Nshell](j)<1e-14)? 0 : avgN[Nshell](j);
			double nplot = (n[Nshell](j)   <1e-14)? 0 : n[Nshell](j);
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
		lout << endl;
	}
	
	// calc P
//	double resP = 0.;
//	if (Nshell>=minNshell+2)
//	{
//		for (int i=0; i<deg(j); ++i)
//		{
//			double Pcontrib = sign(offset(j)+i) * avg(g[Nshell-2].state, H.cc(offset(j)+i), g[Nshell].state);
////					lout << "i=" << i << ", Pcontrib=" << Pcontrib << endl;
//			resP += Pcontrib;
//		}
//	}
//	resP /= sqrt(deg(j));
//	resP = abs(resP);
	
	// Pref
//	lout << ", ref℗=";
//	if (abs(Sn_Pref_(j)-resP)<1e-3)
//	{
//		lout << termcolor::green;
//	}
//	else
//	{
//		lout << termcolor::red;
//	}
//	lout << Sn_Pref_(j) << termcolor::reset;
	
	if (SAVE) save(wd, minNshell, Nshelllast);
}

template<>
void NuclearManager<VMPS::HeisenbergU1>::
compute (bool LOAD_MPO, bool SAVE_MPO, string wd, int minNshell, int maxNshell, bool SAVE)
{
	make_Hamiltonian(LOAD_MPO,SAVE_MPO,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jlabel.resize(2*L+1);
	Jindex.resize(2*L+1);
	energies.resize(2*L+1);
	energies_free.resize(2*L+1);
	
	int Nshelllast = (maxNshell==-1)? 2*L : maxNshell;
	
	for (int Nshell=minNshell; Nshell<=Nshelllast; Nshell+=2)
	{
		int A = Z+Nclosed+Nshell;
		lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << ", progress=" << round(Nshell*1./(2*L)*100,1) << "%" << termcolor::reset << endl;
		
		auto [gres, Jlabel_res, Jindex_res] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, false);
		g[Nshell] = gres;
		Jlabel[Nshell] = Jlabel_res;
		Jindex[Nshell] = Jindex_res;
		
		lout << g[Nshell].state.info() << endl;
		
		if (Nshell != 0 and Nshell != 2*L)
		{
			var[Nshell] = abs(avg(g[Nshell].state,H,H,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
		}
		else
		{
			var[Nshell] = 0.;
		}
		lout << termcolor::blue << "var=" << var[Nshell] << termcolor::reset << endl;
		
		Mmax[Nshell] = g[Nshell].state.calc_Mmax();
		
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
		lout << "E_B/A=" << abs(g[Nshell].energy)/Nshell << " MeV" << endl;
		
		n[Nshell].resize(Nlev);
		avgN[Nshell].resize(Nlev);
		
		for (int j=0; j<Nlev; ++j)
		{
			avgN[Nshell](j) = 0.;
			for (int i=0; i<deg(j); ++i)
			{
				avgN[Nshell](j) += 2.*avg(g[Nshell].state, H.Sz(offset(j)+i), g[Nshell].state)+1.;
			}
			n[Nshell](j) = avgN[Nshell](j)/(2.*deg(j));
			
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
		lout << endl;
	}
	
	if (SAVE) save(wd, minNshell, Nshelllast, 2);
}

template<>
void NuclearManager<VMPS::HubbardU1>::
compute_parallel (bool LOAD_MPO, bool SAVE_MPO, string wd, int minNshell, int maxNshell, bool SAVE)
{
	make_Hamiltonian(LOAD_MPO,SAVE_MPO,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jlabel.resize(2*L+1);
	Jindex.resize(2*L+1);
	energies.resize(2*L+1);
	energies_free.resize(2*L+1);
	
	int Nshelllast = (maxNshell==-1)? 2*L : maxNshell;
	
	#pragma omp parallel for schedule(dynamic)
	for (int iNshell=minNshell; iNshell<=Nshelllast; iNshell+=2)
	{
		int jmaxNshell = (iNshell==Nshelllast)? 0:1;
		for (int jNshell=0; jNshell<=jmaxNshell; ++jNshell)
		{
			int Nshell = iNshell+jNshell;
			int A = Z+Nclosed+Nshell;
			
			#pragma omp critical
			{
				lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << termcolor::reset << endl;
			}
			
			auto [gres, Jlabel_res, Jindex_res] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, false, DMRG::VERBOSITY::SILENT);
			g[Nshell] = gres;
			Jlabel[Nshell] = Jlabel_res;
			Jindex[Nshell] = Jindex_res;
			
			#pragma omp critical
			{
				lout << "Nshell=" << Nshell << " done!, E0=" << g[Nshell].energy << endl;
			}
			
			if (Nshell != 0 and Nshell != 2*L)
			{
				if (Nshell%2 == 0)
				{
					var[Nshell] = abs(avg(g[Nshell].state,H,H,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
				}
				else
				{
					var[Nshell] = abs(avg(g[Nshell].state,Hodd[Jindex[Nshell]],Hodd[Jindex[Nshell]],g[Nshell].state)-pow(g[Nshell].energy,2))/L;
				}
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
			
			for (int j=0; j<Nlev; ++j)
			{
				avgN[Nshell](j) = 0.;
				for (int i=0; i<deg(j); ++i)
				{
					if (Nshell%2==0)
					{
						avgN[Nshell](j) += avg(g[Nshell].state, H.n(offset(j)+i), g[Nshell].state);
					}
					else
					{
						avgN[Nshell](j) += avg(g[Nshell].state, Hodd[Jindex[Nshell]].n(offset(j)+i), g[Nshell].state);
					}
				}
				n[Nshell](j) = avgN[Nshell](j)/(2.*deg(j));
			}
		}
	}
	
	if (SAVE) save(wd, minNshell, Nshelllast);
}

template<>
void NuclearManager<VMPS::HubbardU1xU1>::
compute_parallel (bool LOAD_MPO, bool SAVE_MPO, string wd, int minNshell, int maxNshell, bool SAVE)
{
	make_Hamiltonian(LOAD_MPO,SAVE_MPO,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jlabel.resize(2*L+1);
	Jindex.resize(2*L+1);
	energies.resize(2*L+1);
	energies_free.resize(2*L+1);
	
	int Nshelllast = (maxNshell==-1)? 2*L : maxNshell;
	
	#pragma omp parallel for schedule(dynamic)
	for (int iNshell=minNshell; iNshell<=Nshelllast; iNshell+=2)
	{
		int jmaxNshell = (iNshell==Nshelllast)? 0:1;
		for (int jNshell=0; jNshell<=jmaxNshell; ++jNshell)
		{
			int Nshell = iNshell+jNshell;
			int A = Z+Nclosed+Nshell;
			
			#pragma omp critical
			{
				lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << termcolor::reset << endl;
			}
			
			auto [gres, Jlabel_res, Jindex_res] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, false, DMRG::VERBOSITY::SILENT);
			g[Nshell] = gres;
			Jlabel[Nshell] = Jlabel_res;
			Jindex[Nshell] = Jindex_res;
			
			#pragma omp critical
			{
				lout << "Nshell=" << Nshell << " done!, E0=" << g[Nshell].energy << endl;
			}
			
			if (Nshell != 0 and Nshell != 2*L)
			{
				if (Nshell%2 == 0)
				{
					var[Nshell] = abs(avg(g[Nshell].state,H,H,g[Nshell].state)-pow(g[Nshell].energy,2))/L;
				}
				else
				{
					var[Nshell] = abs(avg(g[Nshell].state,Hodd[Jindex[Nshell]],Hodd[Jindex[Nshell]],g[Nshell].state)-pow(g[Nshell].energy,2))/L;
				}
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
			
			for (int j=0; j<Nlev; ++j)
			{
				avgN[Nshell](j) = 0.;
				for (int i=0; i<deg(j); ++i)
				{
					if (Nshell%2==0)
					{
						avgN[Nshell](j) += avg(g[Nshell].state, H.n(offset(j)+i), g[Nshell].state);
					}
					else
					{
						avgN[Nshell](j) += avg(g[Nshell].state, Hodd[Jindex[Nshell]].n(offset(j)+i), g[Nshell].state);
					}
				}
				n[Nshell](j) = avgN[Nshell](j)/(2.*deg(j));
			}
		}
	}
	
	if (SAVE) save(wd, minNshell, Nshelllast);
}

template<>
void NuclearManager<VMPS::HeisenbergU1>::
compute_parallel (bool LOAD_MPO, bool SAVE_MPO, string wd, int minNshell, int maxNshell, bool SAVE)
{
	make_Hamiltonian(LOAD_MPO,SAVE_MPO,wd);
	
	g.resize(2*L+1);
	var.resize(2*L+1);
	Mmax.resize(2*L+1);
	avgN.resize(2*L+1);
	n.resize(2*L+1);
	Jlabel.resize(2*L+1);
	Jindex.resize(2*L+1);
	energies.resize(2*L+1);
	energies_free.resize(2*L+1);
	
	int Nshelllast = (maxNshell==-1)? 2*L : maxNshell;
	
	#pragma omp parallel for schedule(dynamic)
	for (int Nshell=minNshell; Nshell<=Nshelllast; Nshell+=2)
	{
		int A = Z+Nclosed+Nshell;
		
		#pragma omp critical
		{
			lout << termcolor::bold << "A=" << A << ", Z=" << Z << ", N=" << Nclosed+Nshell << ", Nshell=" << Nshell << "/" << 2*L << termcolor::reset << endl;
		}
		
		auto [gres, Jlabel_res, Jindex_res] = calc_gs(Nshell, LANCZOS::EDGE::GROUND, false, DMRG::VERBOSITY::SILENT);
		g[Nshell] = gres;
		Jlabel[Nshell] = Jlabel_res;
		Jindex[Nshell] = Jindex_res;
		
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
		
		for (int j=0; j<Nlev; ++j)
		{
			avgN[Nshell](j) = 0.;
			for (int i=0; i<deg(j); ++i)
			{
				avgN[Nshell](j) += 2.*avg(g[Nshell].state, H.Sz(offset(j)+i), g[Nshell].state)+1.;
			}
			n[Nshell](j) = avgN[Nshell](j)/(2.*deg(j));
		}
	}
	
	if (SAVE) save(wd, minNshell, Nshelllast, 2);
}

template<typename MODEL>
void NuclearManager<MODEL>::
save (string wd, int Nfrst, int Nlast, int dNshell) const
{
	HDF5Interface target(wd+outfile, WRITE);
	target.save_vector(eps0,"eps0","");
	target.save_vector(deg,"deg","");
	
	for (int Nshell=Nfrst; Nshell<=Nlast; Nshell+=dNshell)
	{
		target.create_group(make_string(Nshell));
		target.save_scalar(g[Nshell].energy,"E0",make_string(Nshell));
		target.save_scalar(energies_free(Nshell),"E0free",make_string(Nshell));
		target.save_vector(avgN[Nshell],"avgN",make_string(Nshell));
		target.save_vector(n[Nshell],"n",make_string(Nshell));
		target.save_scalar(var[Nshell],"var",make_string(Nshell));
		target.save_scalar(Mmax[Nshell],"Mmax",make_string(Nshell));
		target.save_scalar(Jlabel[Nshell],"Jlabel",make_string(Nshell));
		target.save_scalar(Jindex[Nshell],"Jindex",make_string(Nshell));
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
	
	lout << termcolor::green << "saved: " << wd+outfile << termcolor::reset << endl;
	
	target.close();
}

#endif
