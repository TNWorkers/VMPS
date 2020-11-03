#ifndef NUCLEAR_MANAGER
#define NUCLEAR_MANAGER

#include "models/HubbardU1.h"
#include "solvers/DmrgSolver.h"
#include "HDF5Interface.h"

class NuclearManager
{
public:
	
	typedef VMPS::HubbardU1 MODEL;
	
	NuclearManager(){};
	
	NuclearManager (int N_input, int Z_input, const ArrayXi &deg_input, const vector<string> &labels_input, const ArrayXd &eps0_input, const ArrayXXd &V0_input, double G0_input=1.)
	:N(N_input), Z(Z_input), deg(deg_input), labels(labels_input), eps0(eps0_input), V0(V0_input), G0(G0_input)
	{
		Nlev = deg.rows();
		L = deg.sum();
		V0.triangularView<Lower>() = V0.transpose();
		construct();
	};
	
	NuclearManager (int N_input, int Z_input,const string &filename, double G0_input)
	:N(N_input), Z(Z_input), G0(G0_input)
	{
		HDF5Interface source(filename+".h5", READ);
		
		source.load_vector<int>(deg, "deg", "");
		source.load_vector<double>(eps0, "eps0", "");
		source.load_matrix<double>(V0, "V0", "");
		
		Nlev = deg.rows();
		L = deg.sum();
		V0.triangularView<Lower>() = V0.transpose();
		
		labels.resize(Nlev);
		for (int i=0; i<Nlev; ++i)
		{
			source.load_char(make_string("lev",i).c_str(), labels[i]);
		}
		
		construct();
	};
	
	void construct()
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
//			double lambda = V0(i,j)/sqrt(deg(i)*deg(j));
			Ghop.block(offset(i),offset(j),deg(i),deg(j)).setConstant(-G0*V0(i,j));
//			lout << "block: " << offset(i) << ", " << offset(j) << ": " << deg(i) << "x" << deg(j) << "\t" << Ghop(i,j) << endl;
		}
		
		for (int i=0; i<L; ++i)
		for (int j=0; j<L; ++j)
		{
			Ghop(i,j) *= pow(-1,i+j); // cancels sign
		}
		
		ArrayXXd sign(L,L);
		sign.setZero();
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
				lout << "J1=" << J1 << ", M1=" << M1 << ", J2=" << J2 << ", M2=" << M2 << ": " << (J1+J2+M1+M2).numerator() << endl;
				signblock(m1,m2) = pow(-1,(J1+J2+M1+M2).numerator());
			}
			
			sign.block(offset(j1),offset(j2),deg(j1),deg(j2)) = signblock;
		}
		lout << sign << endl;
		
//		for (int j1=0; j1<Nlev; ++j1)
//		for (int m1=0; m1<deg(j1); ++m1)
//		{
//			boost::rational<int> J1 = deg(j1)-boost::rational<int>(1,2);
//			boost::rational<int> M1 = J1-m1;
//			lout << "J1=" << J1 << ", M1=" << M1 << endl;
//		}
		
		Ghop = (Ghop.array()*sign).matrix();
		
		lout << Ghop << endl;
		
		// G local
//		Gloc.resize(L); Gloc.setZero();
//		for (int i=0; i<Nlev; ++i)
//		{
//			Gloc.segment(offset(i),deg(i)).setConstant(-G0*V0(i,i));
//		}
		Gloc = Ghop.diagonal();
		Ghop.diagonal().setZero();
//		lout << "Gloc=" << Gloc.transpose() << endl;
		
		// onsite
		onsite.resize(L); onsite.setZero();
		for (int i=0; i<Nlev; ++i)
		{
			onsite.segment(offset(i),deg(i)).setConstant(eps0(i));
		}
		lout << "onsite=" << onsite.transpose() << endl;
	}
	
	void compute()
	{
		// model
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
		
		DMRG::CONTROL::GLOB GlobParam;
		GlobParam.min_halfsweeps = 10ul;
		GlobParam.max_halfsweeps = 100ul;
		GlobParam.Minit = 100ul;
		GlobParam.Qinit = 100ul;
		GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ
		GlobParam.CALC_S_ON_EXIT = false;
		GlobParam.Mlimit = 500ul;
		
		DMRG::CONTROL::DYN  DynParam;
		size_t lim2site = 30ul;
		DynParam.iteration = [lim2site] (size_t i) {return (i<lim2site)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
		
		g.resize(2*L+1);
		avgN.resize(2*L+1);
		n.resize(2*L+1);
		
		for (int Nshell=0; Nshell<=2*L; ++Nshell)
		{
			int A = N+Z+Nshell;
			lout << "A=" << A << ", Z=" << Z << ", N=" << N+Nshell << ", Nshell=" << Nshell << endl;
			qarray<MODEL::Symmetry::Nq> Q = {Nshell};
			MODEL::Solver DMRG(DMRG::VERBOSITY::ON_EXIT);
			DMRG.userSetGlobParam();
			DMRG.userSetDynParam();
			DMRG.GlobParam = GlobParam;
			DMRG.DynParam = DynParam;
			DMRG.edgeState(H, g[Nshell], Q, LANCZOS::EDGE::GROUND);
			if (Z==50 and N==50 and (Nshell==12 or Nshell==14 or Nshell==16))
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
			lout << "E_B/A=" << abs(g[Nshell].energy)/Nshell << endl;
			
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
				i0 += deg(j);
				
				n[Nshell](j) = avgN[Nshell](j);
				n[Nshell](j) /= deg(j);
				n[Nshell](j) *= 0.5;
				
				lout << "N(" << labels[j] << ")=" << avgN[Nshell](j) << "/" << 2*deg(j) << ", n=" << n[Nshell](j);
				if (Z==50 and N==50 and Nshell==14)
				{
					lout << ", ref(N)=" << Sn_Nref14()(j) << ", ref(n)=" << Sn_nref14()(j);
				}
				lout << endl;
			}
			lout << endl;
		}
		
//		ofstream Filer(make_string("E.dat"));
//		for (int N=0; N<=Nmax; N=N+1)
//		{
//			Filer << Nshell+N << "\t" << energies(N) << endl;
//		}
//		Filer.close();
//		
//		ofstream Delta1Filer(make_string("Delta1.dat"));
//		for (int N=1; N<=Nmax-1; N=N+1)
//		{
//			double DeltaVal1 = 0.5*(2.*energies(N)-energies(N-1)-energies(N+1));
//			
//			lout << "Nn=" << Nshell+N << ", Delta=" << DeltaVal1 << endl;
//			Delta1Filer << Nshell+N << "\t" << DeltaVal1 << endl;
//		}
//		Delta1Filer.close();
//		
//		ofstream Delta2Filer(make_string("Delta2.dat"));
//		for (int N=1; N<=Nmax-1; N=N+2)
//		{
//			double DeltaVal2 = -0.5*(energies(N-1)+energies(N+1)-2.*energies(N));
//			
//			lout << "Nn=" << Nshell+N << ", Delta=" << DeltaVal2 << endl;
//			Delta2Filer << Nshell+N << "\t" << DeltaVal2 << endl;
//		}
//		Delta2Filer.close();
//		
//		ofstream Delta3Filer(make_string("Delta3.dat"));
//		for (int N=2; N<=Nmax; N=N+1)
//		{
//			double DeltaVal3 = -pow(-1,N) * (energies(N)-2.*energies(N-1)+energies(N-2));
//			
//			lout << "Nn=" << Nshell+N << ", Delta=" << DeltaVal3 << endl;
//			Delta3Filer << Nshell+N << "\t" << DeltaVal3 << endl;
//		}
//		Delta3Filer.close();
	}
	
	ArrayXd get_occ() const;
	inline ArrayXd get_eps0() const {return eps0;}
	inline ArrayXi get_deg() const {return deg;}
	inline ArrayXi get_offset() const {return offset;}
	
	double Sn_Eref (int N) const;
	ArrayXd Sn_nref14() const;
	ArrayXd Sn_Nref14() const;
	
private:
	
	int N, Z;
	int Nlev;
	size_t L;
	
	vector<string> labels;
	VectorXi deg, offset;
	VectorXd eps0;
	
	MatrixXd V0, Ghop;
	double G0;
	VectorXd onsite, Gloc;
	
	VMPS::HubbardU1 H;
	vector<Eigenstate<MODEL::StateXd>> g;
	vector<VectorXd> n, avgN;
};

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


#endif
