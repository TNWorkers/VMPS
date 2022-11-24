#ifndef HUBBARD_KSPACE
#define HUBBARD_KSPACE

struct KspaceRawTerms
{
	// 2-site:
	vector<tuple<int,int,double> > spin_exchange;
	vector<tuple<int,int,double> > corr_hopping;
	vector<tuple<int,int,double> > pair_hopping;
	
	// 3-site:
	vector<tuple<int,int,int,double> > nonlocal_spin;
	vector<tuple<int,int,int,double> > corr_hopping3;
	vector<tuple<int,int,int,double> > doublon_decay;
	
	// 4-site
	vector<tuple<int,int,int,int,double> > foursite;
	
	map<tuple<int,int,int,int>,double> Umap;
};

struct KspaceMpoTerms
{
	typedef VMPS::HubbardSU2xU1 MODEL;
	typedef Mpo<MODEL::Symmetry,MODEL::Scalar_> OPERATOR;
	
	// 1-site:
	vector<tuple<size_t,double> > HubbardU_kspace;
	
	// 2-site:
	vector<OPERATOR> Hmpo_spin_exchange;
	vector<OPERATOR> Hmpo_pair_hopping;
	vector<OPERATOR> Hmpo_corr_hopping;
	
	// 3-site:
	vector<OPERATOR> Hmpo_corr_hopping3;
	vector<OPERATOR> Hmpo_nonlocal_spin;
	vector<OPERATOR> Hmpo_doublon_decay;
	
	// 4-site:
	vector<OPERATOR> Hmpo_foursite;
};

struct KspaceHTerms
{
	typedef VMPS::HubbardSU2xU1 MODEL;
	typedef Mpo<MODEL::Symmetry,MODEL::Scalar_> OPERATOR;
	
	// 1-site:
	vector<tuple<size_t,double> > HubbardU_kspace;
	
	// 2-site:
	vector<MODEL> H2_spin_exchange;
	vector<MODEL> H2_pair_hopping;
	vector<MODEL> H2_corr_hopping;
	
	// 3-site:
	vector<MODEL> H3_corr_hopping3;
	vector<MODEL> H3_nonlocal_spin;
	vector<MODEL> H3_doublon_decay;
	
	// 4-site:
	vector<MODEL> H4;
};

class HubbardKspace
{
	typedef VMPS::HubbardSU2xU1 MODEL;
	typedef Mpo<MODEL::Symmetry,MODEL::Scalar_> OPERATOR;
	
public:
	
	// UU: unitary transformation of the hopping matrix
	// U: Hubbard-U in real space
	HubbardKspace (const MatrixXd &UU_input, double U_input)
	:UU(UU_input), U(U_input)
	{
		assert(UU.rows() == UU.cols());
		L = static_cast<size_t>(UU.rows());
		Umap.clear();
		compute();
	};
	
	string info() const;
	
	void compute();
	
	KspaceHTerms get_Hterms() const {return Hterms;};
	
private:
	
	size_t L;
	MatrixXd UU;
	double U;
	map<tuple<int,int,int,int>,double> Umap;
	
	KspaceRawTerms Raw;
	KspaceMpoTerms Terms;
	KspaceHTerms Hterms;
};

string HubbardKspace::
info() const
{
	stringstream ss;
	ss << "HubbardKspace:" << endl;
	ss << "#spin_exchange: " << Raw.spin_exchange.size() << endl;
	ss << "#corr_hopping: " << Raw.corr_hopping.size() << endl;
	ss << "#pair_hopping: " << Raw.pair_hopping.size() << endl;
	ss << "#nonlocal_spin: " << Raw.nonlocal_spin.size() << endl;
	ss << "#corr_hopping3: " << Raw.corr_hopping3.size() << endl;
	ss << "#doublon_decay: " << Raw.doublon_decay.size() << endl;
	ss << "#foursite: " << Raw.foursite.size() << endl;
	ss << "#total: " << Raw.spin_exchange.size()+Raw.corr_hopping.size()+Raw.pair_hopping.size()+Raw.nonlocal_spin.size()+Raw.corr_hopping3.size()+Raw.doublon_decay.size()+Raw.foursite.size() <<endl;
	return ss.str();
}

void HubbardKspace::
compute()
{
	vector<tuple<int,int,int,int> > terms_1site;
	vector<tuple<int,int,int,int> > terms_2site;
	vector<tuple<int,int,int,int> > terms_3site;
	vector<tuple<int,int,int,int> > terms_4site;
	
	for (int k=0; k<L; ++k)
	for (int l=0; l<L; ++l)
	for (int m=0; m<L; ++m)
	for (int n=0; n<L; ++n)
	{
		std::set<int> s;
		s.insert(k);
		s.insert(l);
		s.insert(m);
		s.insert(n);
		
		double Uelement = 0.;
		for (int i=0; i<L; ++i)
		{
			Uelement += UU(i,k)*UU(i,l)*UU(i,m)*UU(i,n);
		}
		if (abs(Uelement) > 1e-8)
		{
			auto Vklmn = make_tuple(k,l,m,n);
			Umap[Vklmn] = U*Uelement;
			
			if (s.size() == 1)
			{
				terms_1site.push_back(Vklmn);
			}
			else if (s.size() == 2)
			{
				terms_2site.push_back(Vklmn);
			}
			else if (s.size() == 3)
			{
				terms_3site.push_back(Vklmn);
			}
			else if (s.size() == 4)
			{
				terms_4site.push_back(Vklmn);
			}
		}
	}
	
	/////////// 1-SITE ///////////
	for (int t=0; t<terms_1site.size(); ++t)
	{
		size_t i = get<0>(terms_1site[t]);
		Terms.HubbardU_kspace.push_back(make_tuple(i,Umap[make_tuple(i,i,i,i)]));
	}
	Hterms.HubbardU_kspace = Terms.HubbardU_kspace;
	
	/////////// 2-SITE ///////////
	while (terms_2site.size() != 0)
	{
		for (int t=0; t<terms_2site.size(); ++t)
		{
			int k1 = get<0>(terms_2site[t]);
			int k2 = get<1>(terms_2site[t]);
			int k3 = get<2>(terms_2site[t]);
			int k4 = get<3>(terms_2site[t]);
		
			if (k1==k4 and k2==k3 and k1!=k2)
			{
				int i = k1;
				int j = k2;
			
				auto Vijji = make_tuple(i,j,j,i);
				auto Vjiij = make_tuple(j,i,i,j);
				auto Viijj = make_tuple(i,i,j,j);
				auto Vjjii = make_tuple(j,j,i,i);
				
				Raw.spin_exchange.push_back(make_tuple(i,j,Umap[Vijji]));
				//lout << "i=" << i << ", j=" << j << ", spin_exchange=" << Umap[Vijji] << ", " << Umap[Vjiij] << ", " << Umap[Viijj] << ", " << Umap[Vjjii] << endl;
				/*if (i==L/2 or j==L/2)
				{
					lout << "i=" << i << ", j=" << j << ", spin_exchange=" << Umap[Vijji] << ", " << Umap[Vjiij] << ", " << Umap[Viijj] << ", " << Umap[Vjjii] << endl;
				}*/
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vijji), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vjiij), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Viijj), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vjjii), terms_2site.end());
				break;
			}
			else if (k1==k2 and k2==k3 and k1!=k4)
			{
				int i = k1;
				int j = k4;
				
				auto Vjiii = make_tuple(j,i,i,i);
				auto Vijii = make_tuple(i,j,i,i);
				auto Viiji = make_tuple(i,i,j,i);
				auto Viiij = make_tuple(i,i,i,j);
				
				Raw.corr_hopping.push_back(make_tuple(i,j,Umap[Vjiii]));
				//lout << "i=" << i << ", j=" << j << ", corr_hopping=" << Umap[Vjiii] << ", " << Umap[Vijii] << ", " << Umap[Viiji] << ", " << Umap[Viiij] << endl;
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vjiii), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vijii), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Viiji), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Viiij), terms_2site.end());
				break;
			}
			else if (k1==k3 and k2==k4 and k1!=k2)
			{
				int i = k1;
				int j = k2;
				
				auto Vijij = make_tuple(i,j,i,j);
				auto Vjiji = make_tuple(j,i,j,i);
				
				Raw.pair_hopping.push_back(make_tuple(i,j,Umap[Vijij]));
				//lout << "i=" << i << ", j=" << j << ", pair_hopping=" << Umap[Vijij] << ", " << Umap[Vjiji] << endl;
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vijij), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vjiji), terms_2site.end());
				break;
			}
		}
	}
	
	/////////// 3-SITE ///////////
	while (terms_3site.size() != 0)
	{
		for (int t=0; t<terms_3site.size(); ++t)
		{
			int k1 = get<0>(terms_3site[t]);
			int k2 = get<1>(terms_3site[t]);
			int k3 = get<2>(terms_3site[t]);
			int k4 = get<3>(terms_3site[t]);
		
			if (k1==k2)
			{
				int i = k1;
				int j = k3;
				int k = k4;
				
				auto Viijk = make_tuple(i,i,j,k);
				auto Viikj = make_tuple(i,i,k,j);
				auto Vjkii = make_tuple(j,k,i,i);
				auto Vkjii = make_tuple(k,j,i,i);
				
				Raw.corr_hopping3.push_back(make_tuple(i,j,k,Umap[Viijk]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", corr_hopping3=" << Umap[Viijk] << ", " << Umap[Viikj] << ", " << Umap[Vjkii] << ", " << Umap[Vkjii] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Viijk), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Viikj), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vjkii), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vkjii), terms_3site.end());
				break;
			}
			else if (k1==k3)
			{
				int i = k1;
				int j = k2;
				int k = k4;
				
				auto Vijik = make_tuple(i,j,i,k);
				auto Vikij = make_tuple(i,k,i,j);
				auto Vjiki = make_tuple(j,i,k,i);
				auto Vkiji = make_tuple(k,i,j,i);
				
				Raw.doublon_decay.push_back(make_tuple(i,j,k,Umap[Vijik]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", doublon_decay=" << Umap[Vijik] << ", " << Umap[Vikij] << ", " << Umap[Vjiki] << ", " << Umap[Vkiji] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vijik), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vikij), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vjiki), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vkiji), terms_3site.end());
				break;
			}
			else if (k1==k4)
			{
				int i = k1;
				int j = k2;
				int k = k3;
				
				auto Vijki = make_tuple(i,j,k,i);
				auto Vkiij = make_tuple(k,i,i,j);
				auto Vjiik = make_tuple(j,i,i,k);
				auto Vikji = make_tuple(i,k,j,i);
				
				Raw.nonlocal_spin.push_back(make_tuple(i,j,k,Umap[Vijki]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", nonlocal_spin=" << Umap[Vijki] << ", " << Umap[Vkiij] << ", " << Umap[Vjiik] << ", " << Umap[Vikji] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vijki), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vkiij), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vjiik), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vikji), terms_3site.end());
				break;
			}
		}
	}
	
	/////////// 4-SITE ///////////
	while (terms_4site.size() != 0)
	{
		for (int t=0; t<terms_4site.size(); ++t)
		{
			int k1 = get<0>(terms_4site[t]);
			int k3 = get<1>(terms_4site[t]);
			int k2 = get<2>(terms_4site[t]);
			int k4 = get<3>(terms_4site[t]);
			
			int i = k1;
			int j = k2;
			int k = k3;
			int l = k4;
			
			auto Vikjl = make_tuple(i,k,j,l);
			auto Vjlik = make_tuple(j,l,i,k);
			auto Vjkil = make_tuple(j,k,i,l);
			auto Viljk = make_tuple(i,l,j,k);
			
			auto Vkilj = make_tuple(k,i,l,j);
			auto Vljki = make_tuple(l,j,k,i);
			auto Vkjli = make_tuple(k,j,l,i);
			auto Vlikj = make_tuple(l,i,k,j);
			
			Raw.foursite.push_back(make_tuple(i,j,k,l,Umap[Vikjl]));
			/*lout << "i=" << i << ", j=" << j << ", k=" << k << ", l=" << l 
				 << ", foursite=" 
				 << Umap[Vikjl] << ", " << Umap[Vjlik] << ", " << Umap[Vjkil] << ", " << Umap[Viljk] << ", "
				 << Umap[Vkilj] << ", " << Umap[Vljki] << ", " << Umap[Vkjli] << ", " << Umap[Vlikj] 
				 << endl;*/
			
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vikjl), terms_4site.end());
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vjlik), terms_4site.end());
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vjkil), terms_4site.end());
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Viljk), terms_4site.end());
			
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vkilj), terms_4site.end());
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vljki), terms_4site.end());
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vkjli), terms_4site.end());
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vlikj), terms_4site.end());
			break;
		}
	}
	
	vector<Param> dummy;
	dummy.push_back({"t",0.});
	dummy.push_back({"maxPower",1ul});
	
	MODEL H1(L,dummy);
	
	int s;
	
	Terms.Hmpo_spin_exchange.resize(Raw.spin_exchange.size());
	Terms.Hmpo_pair_hopping.resize(Raw.pair_hopping.size());
	Terms.Hmpo_corr_hopping.resize(Raw.corr_hopping.size());
	
	Hterms.H2_spin_exchange.resize(Raw.spin_exchange.size());
	Hterms.H2_pair_hopping.resize(Raw.pair_hopping.size());
	Hterms.H2_corr_hopping.resize(Raw.corr_hopping.size());
	
	s = 0;
	for (int t=0; t<Raw.spin_exchange.size(); ++t)
	{
		int i = get<0>(Raw.spin_exchange[t]);
		int j = get<1>(Raw.spin_exchange[t]);
		double lambda = get<2>(Raw.spin_exchange[t]);
		
		auto SdagS = H1.SdagS(i,j); SdagS.scale(-2.*lambda);
		Terms.Hmpo_spin_exchange[s] = sum(Terms.Hmpo_spin_exchange[s],SdagS);
		
		auto nn = H1.nn(i,j); nn.scale(0.5*lambda);
		Terms.Hmpo_spin_exchange[s] = sum(Terms.Hmpo_spin_exchange[s],nn);
		Hterms.H2_spin_exchange[s] = MODEL(Terms.Hmpo_spin_exchange[s],dummy);
		s = (s+1)%Terms.Hmpo_spin_exchange.size();
	}
	
	s = 0;
	for (int t=0; t<Raw.corr_hopping.size(); ++t)
	{
		int i = get<0>(Raw.corr_hopping[t]);
		int j = get<1>(Raw.corr_hopping[t]);
		double lambda = get<2>(Raw.corr_hopping[t]);
	
		auto T1 = H1.cdag_nc(j,i);
		auto T2 = H1.cdagn_c(i,j);
		auto Term = sum(T1,T2);
		Term.scale(lambda);
		Terms.Hmpo_corr_hopping[s] = sum(Terms.Hmpo_corr_hopping[s],Term);
		Hterms.H2_corr_hopping[s] = MODEL(Terms.Hmpo_corr_hopping[s],dummy);
		s = (s+1)%Terms.Hmpo_corr_hopping.size();
	}
	
	s = 0;
	for (int t=0; t<Raw.pair_hopping.size(); ++t)
	{
		int i = get<0>(Raw.pair_hopping[t]);
		int j = get<1>(Raw.pair_hopping[t]);
		double lambda = get<2>(Raw.pair_hopping[t]);
	
		auto T1 = prod(H1.cdagcdag(i),H1.cc(j));
		auto T2 = prod(H1.cdagcdag(j),H1.cc(i));
		auto Term = sum(T1,T2);
		Term.scale(lambda);
		Terms.Hmpo_pair_hopping[s] = sum(Terms.Hmpo_pair_hopping[s],Term);
		Hterms.H2_pair_hopping[s] = MODEL(Terms.Hmpo_pair_hopping[s],dummy);
		s = (s+1)%Terms.Hmpo_pair_hopping.size();
	}
	
	Terms.Hmpo_corr_hopping3.resize(Raw.corr_hopping3.size());
	Terms.Hmpo_nonlocal_spin.resize(Raw.nonlocal_spin.size());
	Terms.Hmpo_doublon_decay.resize(Raw.doublon_decay.size());
	
	Hterms.H3_corr_hopping3.resize(Raw.corr_hopping3.size());
	Hterms.H3_nonlocal_spin.resize(Raw.nonlocal_spin.size());
	Hterms.H3_doublon_decay.resize(Raw.doublon_decay.size());
	
	s = 0;
	for (int t=0; t<Raw.corr_hopping3.size(); ++t)
	{
		int i = get<0>(Raw.corr_hopping3[t]);
		int j = get<1>(Raw.corr_hopping3[t]);
		int k = get<2>(Raw.corr_hopping3[t]);
		double lambda = get<3>(Raw.corr_hopping3[t]);
		
		auto T1 = prod(H1.n(i), H1.cdagc(j,k));
		auto T2 = prod(H1.cdagc(k,j), H1.n(i));
		auto Term = sum(T1,T2);
		Term.scale(0.5*lambda);
		Terms.Hmpo_corr_hopping3[s] = sum(Terms.Hmpo_corr_hopping3[s],Term);
		Hterms.H3_corr_hopping3[s] = MODEL(Terms.Hmpo_corr_hopping3[s],dummy);
		s = (s+1)%Terms.Hmpo_corr_hopping3.size();
	}
	
	s = 0;
	for (int t=0; t<Raw.nonlocal_spin.size(); ++t)
	{
		int i = get<0>(Raw.nonlocal_spin[t]);
		int j = get<1>(Raw.nonlocal_spin[t]);
		int k = get<2>(Raw.nonlocal_spin[t]);
		double lambda = get<3>(Raw.nonlocal_spin[t]);
		{
			double factor = sqrt(2)*sqrt(3)*sqrt(3);
			auto T1 = prod(H1.S(i,0,factor), H1.cdagc3(j,k));
			auto T2 = prod(H1.cdagc3(k,j), H1.Sdag(i,0,factor));
			auto Term = sum(T1,T2);
			Term.scale(lambda);
			Terms.Hmpo_nonlocal_spin[s] = sum(Terms.Hmpo_nonlocal_spin[s],Term);
			Hterms.H3_nonlocal_spin[s] = MODEL(Terms.Hmpo_nonlocal_spin[s],dummy);
		}
		// This is compensated by the 0.5 factor in corr_hopping3:
		/*{
			auto T1 = prod(H1.n(i), H1.cdagc(j,k));
			auto T2 = prod(H1.cdagc(k,j), H1.n(i));
			auto Term = sum(T1,T2);
			Term.scale(-0.5*lambda);
			Terms.Hmpo_nonlocal_spin[s] = sum(Terms.Hmpo_nonlocal_spin[s],Term);
		}*/
		s = (s+1)%Terms.Hmpo_nonlocal_spin.size();
	}
	
	s = 0;
	for (int t=0; t<Raw.doublon_decay.size(); ++t)
	{
		int i = get<0>(Raw.doublon_decay[t]);
		int j = get<1>(Raw.doublon_decay[t]);
		int k = get<2>(Raw.doublon_decay[t]);
		double lambda = get<3>(Raw.doublon_decay[t]);
		
		auto T1 = prod(H1.cdagcdag(i), H1.cc1(j,k));
		auto T2 = prod(H1.cdagcdag1(j,k), H1.cc(i));
		auto Term = diff(T2,T1);
		Term.scale(lambda);
		Terms.Hmpo_doublon_decay[s] = sum(Terms.Hmpo_doublon_decay[s],Term);
		Hterms.H3_doublon_decay[s] = MODEL(Terms.Hmpo_doublon_decay[s],dummy);
		s = (s+1)%Terms.Hmpo_doublon_decay.size();
	}
	
	Terms.Hmpo_foursite.resize(Raw.foursite.size());
	
	Hterms.H4.resize(Raw.foursite.size());
	
	s = 0;
	for (int t=0; t<Raw.foursite.size(); ++t)
	{
		int i = get<0>(Raw.foursite[t]);
		int j = get<1>(Raw.foursite[t]);
		int k = get<2>(Raw.foursite[t]);
		int l = get<3>(Raw.foursite[t]);
		double lambda = get<4>(Raw.foursite[t]);
	
		auto T1 = prod(H1.cdagcdag1(i,j), H1.cc1(k,l));
		auto T2 = prod(H1.cdagcdag1(l,k), H1.cc1(j,i));
		auto Term = sum(T1,T2);
		Term.scale(-lambda);
		Terms.Hmpo_foursite[s] = sum(Terms.Hmpo_foursite[s],Term);
		Hterms.H4[s] = MODEL(Terms.Hmpo_foursite[s],dummy);
		s = (s+1)%Terms.Hmpo_foursite.size();
	}
}

#endif
