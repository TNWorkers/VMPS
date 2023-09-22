#ifndef HUBBARD_KSPACE
#define HUBBARD_KSPACE

template<typename Scalar>
struct KspaceRawTerms
{
	// 2-site:
	vector<tuple<int,int,Scalar> > spin_exchange;
	vector<tuple<int,int,Scalar> > density_density; // only for U1xU1
	vector<tuple<int,int,Scalar> > corr_hopping;
	vector<tuple<int,int,Scalar> > corr_hoppingB;
	vector<tuple<int,int,Scalar> > pair_hopping;
	
	// 3-site:
	vector<tuple<int,int,int,Scalar> > nonlocal_spin;
	vector<tuple<int,int,int,Scalar> > nonlocal_spinB; // only for U1xU1
	vector<tuple<int,int,int,Scalar> > corr_hopping3;
	vector<tuple<int,int,int,Scalar> > corr_hopping3B;
	vector<tuple<int,int,int,Scalar> > doublon_decay;
	
	// 4-site
	vector<tuple<int,int,int,int,Scalar> > foursite;
	
	map<tuple<int,int,int,int>,Scalar> Umap;
};

template<typename MODEL>
struct KspaceMpoTerms
{
	// 1-site:
	vector<tuple<size_t,double> > HubbardU_kspace;
	
	// 2-site:
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_spin_exchange;
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_density_density;
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_pair_hopping;
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_corr_hopping;
	
	// 3-site:
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_corr_hopping3;
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_nonlocal_spin;
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_doublon_decay;
	
	// 4-site:
	vector<Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> > Hmpo_foursite;
};

template<typename MODEL>
struct KspaceHTerms
{
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

template<typename MODEL>
class HubbardKspace
{
public:
	
	typedef Mpo<typename MODEL::Symmetry, typename MODEL::Scalar_> OPERATOR;
	
	HubbardKspace(){};
	
	// UU: unitary transformation of the hopping matrix
	// U: Hubbard-U in real space
	HubbardKspace (const MatrixXcd &UU_input, double U_input, DMRG::VERBOSITY::OPTION VERB_input=DMRG::VERBOSITY::SILENT, bool VUMPS_input=false, vector<int> x_input={}, vector<int> y_input={})
	:UU(UU_input), U(U_input), VUMPS(VUMPS_input), x(x_input), y(y_input), VERB(VERB_input)
	{
		assert(UU.rows() == UU.cols());
		L = static_cast<size_t>(UU.rows());
		
		if (x.size() == 0)
		{
			x.resize(L);
			y.resize(L);
			for (int i=0; i<L; ++i)
			{
				x[i] = i;
				y[i] = 0;
			}
			Lred = L;
		}
		else
		{
			Lred = *std::max_element(x.begin(), x.end())+1;
		}
		
		dummy_params.push_back({"maxPower",1ul});
//		typename MODEL::Scalar_ zero = 0.;
//		dummy.push_back({"t",zero});
		
		Umap.clear();
		compute_raw();
		compute_MPO();
	};
	
	HubbardKspace (const MatrixXcd &UU_input, double U_input, const vector<Param> &params, DMRG::VERBOSITY::OPTION VERB_input=DMRG::VERBOSITY::SILENT, bool VUMPS_input=false, vector<int> x_input={}, vector<int> y_input={})
	:UU(UU_input), U(U_input), dummy_params(params), VUMPS(VUMPS_input), x(x_input), y(y_input), VERB(VERB_input)
	{
		assert(UU.rows() == UU.cols());
		L = static_cast<size_t>(UU.rows());
		
		if (x.size() == 0)
		{
			x.resize(L);
			y.resize(L);
			for (int i=0; i<L; ++i)
			{
				x[i] = i;
				y[i] = 0;
			}
			Lred = L;
		}
		else
		{
			Lred = *std::max_element(x.begin(), x.end())+1;
			lout << "Lred=" << Lred << endl;
		}
		
		Umap.clear();
		compute_raw();
		compute_MPO();
	};
	
	string info() const;
	
	template<class Dummy = typename MODEL::Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),void>::type compute_raw();
	
	template<class Dummy = typename MODEL::Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),void>::type compute_raw();
	
	template<class Dummy = typename MODEL::Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),void>::type compute_MPO();
	
	template<class Dummy = typename MODEL::Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),void>::type compute_MPO();
	
	KspaceHTerms<MODEL> get_Hterms() const {return Hterms;};
	
	MODEL sum_all() const;
	MODEL sum_all(const ArrayXXcd &hopping) const;
	MODEL sum_2site() const;
	MODEL sum_3site() const;
	MODEL sum_4site() const;
	Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> sum_all_mpo() const;
	Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> sum_2site_mpo() const;
	
private:
	
	vector<Param> dummy_params;
	
	DMRG::VERBOSITY::OPTION VERB;
	
	size_t L;
	size_t Lred;
	MatrixXcd UU;
	double U;
	map<tuple<int,int,int,int>,typename MODEL::Scalar_> Umap;
	
	bool VUMPS;
	
	// If sites are blocked up, translate into x & y
	vector<int> x;
	vector<int> y;
	
	KspaceRawTerms<typename MODEL::Scalar_> Raw;
	KspaceMpoTerms<MODEL> Terms;
	KspaceHTerms<MODEL> Hterms;
};

template<typename MODEL>
string HubbardKspace<MODEL>::
info() const
{
	stringstream ss;
	ss << "HubbardKspace:" << endl;
	ss << "#spin_exchange: " << Raw.spin_exchange.size()+Raw.density_density.size() << endl;
	ss << "#corr_hopping: " << Raw.corr_hopping.size()+Raw.corr_hoppingB.size() << endl;
	ss << "#pair_hopping: " << Raw.pair_hopping.size() << endl;
	ss << "#nonlocal_spin: " << Raw.nonlocal_spin.size()+Raw.nonlocal_spinB.size() << endl;
	ss << "#corr_hopping3: " << Raw.corr_hopping3.size()+Raw.corr_hopping3B.size() << endl;
	ss << "#doublon_decay: " << Raw.doublon_decay.size() << endl;
	ss << "#foursite: " << Raw.foursite.size() << endl;
	ss << "#total: " << Raw.spin_exchange.size()+Raw.density_density.size()+Raw.corr_hopping.size()+Raw.corr_hoppingB.size()+Raw.pair_hopping.size()+Raw.nonlocal_spin.size()+Raw.nonlocal_spinB.size()+Raw.corr_hopping3.size()+Raw.corr_hopping3B.size()+Raw.doublon_decay.size()+Raw.foursite.size() <<endl;
	return ss.str();
}

template<typename MODEL>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),void>::type HubbardKspace<MODEL>::
compute_raw()
{
	vector<tuple<int,int,int,int> > terms_1site;
	vector<tuple<int,int,int,int> > terms_2site;
	vector<tuple<int,int,int,int> > terms_3site;
	vector<tuple<int,int,int,int> > terms_4site;
	
	// Carry out sum over local space, fill Umap and sort terms according to 1/2/3/4 site
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
		
		complex<double> Uelement_ = 0.;
		for (int i=0; i<L; ++i)
		{
			Uelement_ += conj(UU(i,k))*UU(i,l)*conj(UU(i,m))*UU(i,n);
		}
		typename MODEL::Scalar_ Uelement;
		if constexpr(is_same<typename MODEL::Scalar_,double>::value)
		{
			if (abs(Uelement_.imag()) > 1e-10)
			{
				lout << termcolor::red << "Warning: Non-zero imaginary part of transformed matrix element=" << Uelement_.imag() << termcolor::reset << endl;
			}
			Uelement = Uelement_.real();
		}
		else
		{
			Uelement = Uelement_;
		}
		if (abs(Uelement) > 1e-8)
		{
			auto Vklmn = make_tuple(k,l,m,n);
			//if (k!=m and l!=n)
			//{
			//	lout << "k=" << k << ", l=" << l << ", m=" << m << ", n=" << n << ", U=" << Uelement << endl;
			//}
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
	
	// Start filling Raw (not yet MPOs):
	// Groups the 1/2/3/4-site contributions according to the various MPO terms
	
	/////////// 1-SITE ///////////
	for (int t=0; t<terms_1site.size(); ++t)
	{
		size_t i = get<0>(terms_1site[t]);
		Terms.HubbardU_kspace.push_back(make_tuple(i,real(Umap[make_tuple(i,i,i,i)])));
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
				//lout << "i=" << i << ", j=" << j << ", spin_exchange: Vijji=" << Umap[Vijji] << ", Vjiij=" << Umap[Vjiij] << ", Viijj=" << Umap[Viijj] << ", Vjjii=" << Umap[Vjjii] << endl;
				
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
				
				auto Viiij = make_tuple(i,i,i,j);
				auto Vijii = make_tuple(i,j,i,i);
				
				auto Viiji = make_tuple(i,i,j,i);
				auto Vjiii = make_tuple(j,i,i,i);
				
				Raw.corr_hopping.push_back(make_tuple(i,j,Umap[Viiij]));
				//lout << "i=" << i << ", j=" << j << ", corr_hopping: Viiij=" << Umap[Viiij] << ", Vijii=" << Umap[Vijii] << ", Viiji" << Umap[Viiji] << ", Vjiii" << Umap[Vjiii] << endl;
				
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
				//lout << "i=" << i << ", j=" << j << ", pair_hopping: Vijij=" << Umap[Vijij] << ", Vjiji=" << Umap[Vjiji] << endl;
				
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
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", corr_hopping3: Viijk=" << Umap[Viijk] << ", Viikj=" << Umap[Viikj] << ", Vjkii=" << Umap[Vjkii] << ", Vkjii=" << Umap[Vkjii] << endl;
				
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
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", doublon_decay: Vijik=" << Umap[Vijik] << ", Vijik=" << Umap[Vikij] << ", Vjiki=" << Umap[Vjiki] << ", Vkiji=" << Umap[Vkiji] << endl;
				
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
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", nonlocal_spin: Vijki=" << Umap[Vijki] << ", Vkiij=" << Umap[Vkiij] << ", Vjiik=" << Umap[Vjiik] << ", Vikji=" << Umap[Vikji] << endl;
				
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
				 << ", 4s: " 
				 << "Vikjl=" << Umap[Vikjl] << ", Vjlik=" << Umap[Vjlik] << ", Vjkil=" << Umap[Vjkil] << ", Viljk=" << Umap[Viljk]
				 << ", Vkilj=" << Umap[Vkilj] << ", Vljki=" << Umap[Vljki] << ", Vkjli=" << Umap[Vkjli] << ", Vkjli=" << Umap[Vkjli] 
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
}

template<typename MODEL>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),void>::type HubbardKspace<MODEL>::
compute_MPO()
{
	MODEL Hdummy(Lred,dummy_params,BC::OPEN,DMRG::VERBOSITY::SILENT);
	//cout << Hdummy.info() << endl;
	
	int s; // s controls how many terms are summed into bigger MPOs; default is: no additional summation
	
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
		double lambda = real(get<2>(Raw.spin_exchange[t]));
		
		if (VERB>0) lout << "spin exchange: " << i << ", " << j << ", λ=" << lambda << endl;
		
		//auto SdagS = Hdummy.SdagS(i,j); SdagS.scale(-2.*lambda);
		auto SdagS = Hdummy.SdagS(x[i],x[j],y[i],y[j]); SdagS.scale(-2.*lambda);
		Terms.Hmpo_spin_exchange[s] = sum(Terms.Hmpo_spin_exchange[s],SdagS);
		
		//auto nn = Hdummy.nn(i,j); nn.scale(0.5*lambda);
		auto nn = Hdummy.nn(x[i],x[j],y[i],y[j]); nn.scale(0.5*lambda);
		Terms.Hmpo_spin_exchange[s] = sum(Terms.Hmpo_spin_exchange[s],nn);
		Hterms.H2_spin_exchange[s] = MODEL(Terms.Hmpo_spin_exchange[s],dummy_params);
		s = (s+1)%Terms.Hmpo_spin_exchange.size();
	}
	
	s = 0;
	for (int t=0; t<Raw.corr_hopping.size(); ++t)
	{
		int i = get<0>(Raw.corr_hopping[t]);
		int j = get<1>(Raw.corr_hopping[t]);
		typename MODEL::Scalar_ lambda = get<2>(Raw.corr_hopping[t]);
		
		//auto T1 = Hdummy.cdagn_c(i,j); T1.scale(lambda);
		//auto T2 = Hdummy.cdag_nc(j,i); T2.scale(conjIfcomplex(lambda));
		auto T1 = Hdummy.cdagn_c(x[i],x[j],y[i],y[j]); T1.scale(lambda);
		auto T2 = Hdummy.cdag_nc(x[j],x[i],y[j],y[i]); T2.scale(conjIfcomplex(lambda));
		auto Term = sum(T1,T2);
		
		if (VERB>0) lout << "corr_hopping: " << i << ", " << j << ", lambda=" << lambda << endl;
		
		Terms.Hmpo_corr_hopping[s] = sum(Terms.Hmpo_corr_hopping[s],Term);
		Hterms.H2_corr_hopping[s] = MODEL(Terms.Hmpo_corr_hopping[s],dummy_params);
		s = (s+1)%Terms.Hmpo_corr_hopping.size();
	}
	
	s = 0;
	for (int t=0; t<Raw.pair_hopping.size(); ++t)
	{
		int i = get<0>(Raw.pair_hopping[t]);
		int j = get<1>(Raw.pair_hopping[t]);
		typename MODEL::Scalar_ lambda = get<2>(Raw.pair_hopping[t]);
		
		if (VERB>0)
		{
			lout << "pair hopping: " << i << ", " << j << ", λ=" << lambda << ", Q=" 
				 << Hdummy.cdagcdag(x[i],y[i]).Qtarget() << ", " 
				 << Hdummy.cc(x[j],y[j]).Qtarget()
				 << endl;
		}
		
		//auto T1 = prod(Hdummy.cdagcdag(i),Hdummy.cc(j)); T1.scale(lambda);
		//auto T2 = prod(Hdummy.cdagcdag(j),Hdummy.cc(i)); T2.scale(conjIfcomplex(lambda));
		auto T1 = prod(Hdummy.cdagcdag(x[i],y[i]),Hdummy.cc(x[j],y[j])); T1.scale(lambda);
		auto T2 = prod(Hdummy.cdagcdag(x[j],y[j]),Hdummy.cc(x[i],y[i])); T2.scale(conjIfcomplex(lambda));
		auto Term = sum(T1,T2);
		
		Terms.Hmpo_pair_hopping[s] = sum(Terms.Hmpo_pair_hopping[s],Term);
		Hterms.H2_pair_hopping[s] = MODEL(Terms.Hmpo_pair_hopping[s],dummy_params);
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
		typename MODEL::Scalar_ lambda = get<3>(Raw.corr_hopping3[t]);
		
		//auto T1 = prod(Hdummy.n(i), Hdummy.cdagc(j,k)); T1.scale(0.5*lambda);
		//auto T2 = prod(Hdummy.cdagc(k,j), Hdummy.n(i)); T2.scale(0.5*conjIfcomplex(lambda));
		auto T1 = prod(Hdummy.n(x[i],y[i]), Hdummy.cdagc(x[j],x[k],y[j],y[k])); T1.scale(0.5*lambda);
		auto T2 = prod(Hdummy.cdagc(x[k],x[j],y[k],y[j]), Hdummy.n(x[i],y[i])); T2.scale(0.5*conjIfcomplex(lambda));
		auto Term = sum(T1,T2);
		Terms.Hmpo_corr_hopping3[s] = sum(Terms.Hmpo_corr_hopping3[s],Term);
		Hterms.H3_corr_hopping3[s] = MODEL(Terms.Hmpo_corr_hopping3[s],dummy_params);
		s = (s+1)%Terms.Hmpo_corr_hopping3.size();
	}
	
	s = 0;
	for (int t=0; t<Raw.nonlocal_spin.size(); ++t)
	{
		int i = get<0>(Raw.nonlocal_spin[t]);
		int j = get<1>(Raw.nonlocal_spin[t]);
		int k = get<2>(Raw.nonlocal_spin[t]);
		typename MODEL::Scalar_ lambda = get<3>(Raw.nonlocal_spin[t]);
		{
			double factor = sqrt(2)*sqrt(3)*sqrt(3);
			//auto T1 = prod(Hdummy.S(i,0,factor), Hdummy.cdagc3(j,k)); T1.scale(lambda);
			//auto T2 = prod(Hdummy.cdagc3(k,j), Hdummy.Sdag(i,0,factor)); T2.scale(conjIfcomplex(lambda));
			auto T1 = prod(Hdummy.S(x[i],y[i],factor), Hdummy.cdagc3(x[j],x[k],y[j],y[k])); T1.scale(lambda);
			auto T2 = prod(Hdummy.cdagc3(x[k],x[j],y[k],y[j]), Hdummy.Sdag(x[i],y[i],factor)); T2.scale(conjIfcomplex(lambda));
			auto Term = sum(T1,T2);
			Terms.Hmpo_nonlocal_spin[s] = sum(Terms.Hmpo_nonlocal_spin[s],Term);
			Hterms.H3_nonlocal_spin[s] = MODEL(Terms.Hmpo_nonlocal_spin[s],dummy_params);
		}
		// This is compensated by the 0.5 factor in corr_hopping3:
		/*{
			auto T1 = prod(Hdummy.n(x[i],y[i]), Hdummy.cdagc(x[j],x[k],y[j],y[k]));
			auto T2 = prod(Hdummy.cdagc(x[k],x[j],y[k],y[j]), Hdummy.n(x[i],y[i]));
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
		typename MODEL::Scalar_ lambda = get<3>(Raw.doublon_decay[t]);
		
		if (VERB>0)
		{
			lout << "doublon decay: cdagcdag_" << i << ", cc1_" << j << "_" << k << ", λ=" << lambda << ", Q=" 
				   << Hdummy.cdagcdag(x[i],y[i]).Qtarget() << ", " 
				   << Hdummy.cc1(x[j],x[k],y[j],y[k]).Qtarget()
				   << endl;
		}
		
		//auto T1 = prod(Hdummy.cdagcdag(i), Hdummy.cc1(j,k)); T1.scale(lambda);
		//auto T2 = prod(Hdummy.cdagcdag1(j,k), Hdummy.cc(i)); T2.scale(conjIfcomplex(lambda));
		auto T1 = prod(Hdummy.cdagcdag(x[i],y[i]), Hdummy.cc1(x[j],x[k],y[j],y[k])); T1.scale(lambda);
		auto T2 = prod(Hdummy.cdagcdag1(x[j],x[k],y[j],y[k]), Hdummy.cc(x[i],y[i])); T2.scale(conjIfcomplex(lambda));
		auto Term = diff(T2,T1);
		Terms.Hmpo_doublon_decay[s] = sum(Terms.Hmpo_doublon_decay[s],Term);
		Hterms.H3_doublon_decay[s] = MODEL(Terms.Hmpo_doublon_decay[s],dummy_params);
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
		typename MODEL::Scalar_ lambda = get<4>(Raw.foursite[t]);
		
		if (VERB>0)
		{
			lout << "foursite: " << i << ", " << j << ", " << k << ", " << l << ", λ=" << lambda << ", Q=" 
				 << Hdummy.cdagcdag1(x[i],x[j],y[i],y[j]).Qtarget() << ", " 
				 << Hdummy.cc1(x[k],x[l],y[k],y[l]).Qtarget() << ", " 
				 << Hdummy.cdagcdag1(x[l],x[k],y[l],y[k]).Qtarget() << ", " 
				 << Hdummy.cc1(x[j],x[i],y[j],y[i]).Qtarget() 
				 << endl;
		}
		
		//auto T1 = prod(Hdummy.cdagcdag1(i,j), Hdummy.cc1(k,l)); T1.scale(-lambda);
		//auto T2 = prod(Hdummy.cdagcdag1(l,k), Hdummy.cc1(j,i)); T2.scale(-conjIfcomplex(lambda));
		auto T1 = prod(Hdummy.cdagcdag1(x[i],x[j],y[i],y[j]), Hdummy.cc1(x[k],x[l],y[k],y[l])); T1.scale(-lambda);
		auto T2 = prod(Hdummy.cdagcdag1(x[l],x[k],y[l],y[k]), Hdummy.cc1(x[j],x[i],y[j],y[i])); T2.scale(-conjIfcomplex(lambda));
		auto Term = sum(T1,T2);
		//Term.scale(-lambda);
		Terms.Hmpo_foursite[s] = sum(Terms.Hmpo_foursite[s],Term);
		Hterms.H4[s] = MODEL(Terms.Hmpo_foursite[s],dummy_params);
		s = (s+1)%Terms.Hmpo_foursite.size();
	}
}

template<typename MODEL>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),void>::type HubbardKspace<MODEL>::
compute_raw()
{
	vector<tuple<int,int,int,int> > terms_1site;
	vector<tuple<int,int,int,int> > terms_2site;
	vector<tuple<int,int,int,int> > terms_3site;
	vector<tuple<int,int,int,int> > terms_4site;
	
	// Carry out sum over local space, fill Umap and sort terms according to 1/2/3/4 site
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
		
		complex<double> Uelement_ = 0.;
		for (int i=0; i<L; ++i)
		{
			Uelement_ += conj(UU(i,k))*UU(i,l)*conj(UU(i,m))*UU(i,n);
		}
		typename MODEL::Scalar_ Uelement;
		if constexpr(is_same<typename MODEL::Scalar_,double>::value)
		{
			if (abs(Uelement_.imag()) > 1e-10)
			{
				lout << termcolor::red << "Warning: Non-zero imaginary part of transformed matrix element=" << Uelement_.imag() << termcolor::reset << endl;
			}
			Uelement = Uelement_.real();
		}
		else
		{
			Uelement = Uelement_;
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
	
	// Start filling Raw (not yet MPOs):
	// Groups the 1/2/3/4-site contributions according to the various MPO terms
	
	/////////// 1-SITE ///////////
	for (int t=0; t<terms_1site.size(); ++t)
	{
		size_t i = get<0>(terms_1site[t]);
		Terms.HubbardU_kspace.push_back(make_tuple(i,real(Umap[make_tuple(i,i,i,i)])));
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
			
			if (k1==k4 and k2==k3)
			{
				int i = k1;
				int j = k2;
				
				auto Vijji = make_tuple(i,j,j,i);
				auto Vjiij = make_tuple(j,i,i,j);
				
				Raw.spin_exchange.push_back(make_tuple(i,j,Umap[Vijji]));
				//lout << "i=" << i << ", j=" << j << ", spin_exchange: Vijji=" << Umap[Vijji] << ", Vjiij=" << Umap[Vjiij] << endl;
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vijji), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vjiij), terms_2site.end());
				break;
			}
			else if (k1==k2 and k3==k4)
			{
				int i = k1;
				int j = k3;
				
				auto Viijj = make_tuple(i,i,j,j);
				
				Raw.density_density.push_back(make_tuple(i,j,Umap[Viijj]));
				//lout << "i=" << i << ", j=" << j << ", density_density: Viijj=" << Umap[Viijj] << endl;
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Viijj), terms_2site.end());
				break;
			}
			else if (k1==k2 and k2==k3)
			{
				int i = k1;
				int j = k4;
				
				auto Viiij = make_tuple(i,i,i,j);
				auto Viiji = make_tuple(i,i,j,i);
				
				Raw.corr_hopping.push_back(make_tuple(i,j,Umap[Viiij]));
				//lout << "i=" << i << ", j=" << j << ", corr_hopping: Viiij=" << Umap[Viiij] << ", Viiji" << Umap[Viiji] << endl;
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Viiij), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Viiji), terms_2site.end());
				break;
			}
			else if (k1==k3 and k3==k4)
			{
				int i = k1;
				int j = k2;
				
				auto Vijii = make_tuple(i,j,i,i);
				auto Vjiii = make_tuple(j,i,i,i);
				
				Raw.corr_hoppingB.push_back(make_tuple(i,j,Umap[Vijii]));
				//lout << "i=" << i << ", j=" << j << ", corr_hoppingB: Vijii=" << Umap[Vijii] << ", Vjiii" << Umap[Vjiii] << endl;
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vijii), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vjiii), terms_2site.end());
				break;
			}
			else if (k1==k3 and k2==k4)
			{
				int i = k1;
				int j = k2;
				
				auto Vijij = make_tuple(i,j,i,j);
				auto Vjiji = make_tuple(j,i,j,i);
				
				Raw.pair_hopping.push_back(make_tuple(i,j,Umap[Vijij]));
				//lout << "i=" << i << ", j=" << j << ", pair_hopping: Vijij=" << Umap[Vijij] << ", Vjiji=" << Umap[Vjiji] << endl;
				
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vijij), terms_2site.end());
				terms_2site.erase(std::remove(terms_2site.begin(), terms_2site.end(), Vjiji), terms_2site.end());
				break;
			}
//			else
//			{
//				lout << "2site other: k1=" << k1 << ", k2=" << k2 << ", k3=" << k3 << ", k4=" << k4 << endl;
//			}
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
			//lout << "k1=" << k1 << ", k2=" << k2 << ", k3=" << k3 << ", k4=" << k4 << endl;
			
			if (k1==k2)
			{
				int i = k1;
				int j = k3;
				int k = k4;
				
				auto Viijk = make_tuple(i,i,j,k);
				auto Viikj = make_tuple(i,i,k,j);
				
				Raw.corr_hopping3.push_back(make_tuple(i,j,k,Umap[Viijk]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", corr_hopping3: Viijk=" << Umap[Viijk] << ", Viikj=" << Umap[Viikj] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Viijk), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Viikj), terms_3site.end());
				break;
			}
			else if (k3==k4)
			{
				int i = k1;
				int j = k2;
				int k = k3;
				
				auto Vijkk = make_tuple(i,j,k,k);
				auto Vjikk = make_tuple(j,i,k,k);
				
				Raw.corr_hopping3B.push_back(make_tuple(i,j,k,Umap[Vijkk]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", corr_hopping3B: Vjkii=" << Umap[Vjkii] << ", Vkjii=" << Umap[Vkjii] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vijkk), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vjikk), terms_3site.end());
				break;
			}
			else if (k1==k4)
			{
				int i = k1;
				int j = k2;
				int k = k3;
				
				auto Vijki = make_tuple(i,j,k,i);
				auto Vjiik = make_tuple(j,i,i,k);
				
				Raw.nonlocal_spin.push_back(make_tuple(i,j,k,Umap[Vijki]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", nonlocal_spin: Vijki=" << Umap[Vijki] << ", Vjiik=" << Umap[Vjiik] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vijki), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vjiik), terms_3site.end());
				break;
			}
			else if (k2==k3)
			{
				int i = k1;
				int j = k2;
				int k = k4;
				
				auto Vijjk = make_tuple(i,j,j,k);
				auto Vjikj = make_tuple(j,i,k,j);
				
				Raw.nonlocal_spinB.push_back(make_tuple(i,j,k,Umap[Vijjk]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", nonlocal_spinB: Vijjk=" << Umap[Vijjk] << ", Vjikj=" << Umap[Vjikj] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vijjk), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vjikj), terms_3site.end());
				break;
			}
			else if (k1==k3)
			{
				int i = k1;
				int j = k2;
				int k = k4;
				
				auto Vijik = make_tuple(i,j,i,k);
				auto Vjiki = make_tuple(j,i,k,i);
				
				Raw.doublon_decay.push_back(make_tuple(i,j,k,Umap[Vijik]));
				//lout << "i=" << i << ", j=" << j << ", k=" << k << ", doublon_decay: Vijik=" << Umap[Vijik] << ", Vjiki=" << Umap[Vjiki] << endl;
				
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vijik), terms_3site.end());
				terms_3site.erase(std::remove(terms_3site.begin(), terms_3site.end(), Vjiki), terms_3site.end());
				break;
			}
//			else
//			{
//				lout << "3site other: k1=" << k1 << ", k2=" << k2 << ", k3=" << k3 << ", k4=" << k4 << endl;
//			}
		}
	}
	
	/////////// 4-SITE ///////////
	while (terms_4site.size() != 0)
	{
		for (int t=0; t<terms_4site.size(); ++t)
		{
			int k1 = get<0>(terms_4site[t]);
			int k2 = get<1>(terms_4site[t]);
			int k3 = get<2>(terms_4site[t]);
			int k4 = get<3>(terms_4site[t]);
			
			int i = k1;
			int j = k2;
			int k = k3;
			int l = k4;
			
			auto Vijkl = make_tuple(i,j,k,l);
			auto Vjilk = make_tuple(j,i,l,k);
			
			Raw.foursite.push_back(make_tuple(i,j,k,l,Umap[Vijkl]));
			lout << "i=" << i << ", j=" << j << ", k=" << k << ", l=" << l << ", 4s: " << "Vikjl=" << Umap[Vijkl] << ", Vjilk=" << Umap[Vjilk] << endl;
			
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vijkl), terms_4site.end());
			terms_4site.erase(std::remove(terms_4site.begin(), terms_4site.end(), Vjilk), terms_4site.end());
			break;
		}
	}
}

template<typename MODEL>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),void>::type HubbardKspace<MODEL>::
compute_MPO()
{
	MODEL Hdummy(Lred,dummy_params,BC::OPEN,DMRG::VERBOSITY::SILENT);
	
	/////////// 2-SITE ///////////
	for (int t=0; t<Raw.spin_exchange.size(); ++t)
	{
		int i = get<0>(Raw.spin_exchange[t]);
		int j = get<1>(Raw.spin_exchange[t]);
		typename MODEL::Scalar_ lambda = get<2>(Raw.spin_exchange[t]); // Vijji
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template cdagc<DN,DN>(j,i)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template cdagc<DN,DN>(i,j)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_spin_exchange.push_back(hop12);
		Hterms.H2_spin_exchange.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.density_density.size(); ++t)
	{
		int i = get<0>(Raw.density_density[t]);
		int j = get<1>(Raw.density_density[t]);
		double lambda = real(get<2>(Raw.density_density[t])); // Viijj
		
		OPERATOR hop12 = prod(Hdummy.template n<UP>(i),Hdummy.template n<DN>(j)); hop12.scale(lambda);
		
		Terms.Hmpo_spin_exchange.push_back(hop12);
		Hterms.H2_spin_exchange.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.corr_hopping.size(); ++t)
	{
		int i = get<0>(Raw.corr_hopping[t]);
		int j = get<1>(Raw.corr_hopping[t]);
		typename MODEL::Scalar_ lambda = get<2>(Raw.corr_hopping[t]); // Viiij
		
		OPERATOR hop1 = prod(Hdummy.template n<UP>(i),Hdummy.template cdagc<DN,DN>(i,j)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template n<UP>(i),Hdummy.template cdagc<DN,DN>(j,i)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_corr_hopping.push_back(hop12);
		Hterms.H2_corr_hopping.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.corr_hoppingB.size(); ++t)
	{
		int i = get<0>(Raw.corr_hoppingB[t]);
		int j = get<1>(Raw.corr_hoppingB[t]);
		typename MODEL::Scalar_ lambda = get<2>(Raw.corr_hoppingB[t]); // Vijii
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template n<DN>(i)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template n<DN>(i)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_corr_hopping.push_back(hop12);
		Hterms.H2_corr_hopping.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.pair_hopping.size(); ++t)
	{
		int i = get<0>(Raw.pair_hopping[t]);
		int j = get<1>(Raw.pair_hopping[t]);
		typename MODEL::Scalar_ lambda = get<2>(Raw.pair_hopping[t]); // Vijij
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template cdagc<DN,DN>(i,j)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template cdagc<DN,DN>(j,i)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_pair_hopping.push_back(hop12);
		Hterms.H2_pair_hopping.push_back(MODEL(hop12,dummy_params));
	}
	/////////// 3-SITE ///////////
	for (int t=0; t<Raw.corr_hopping3.size(); ++t)
	{
		int i = get<0>(Raw.corr_hopping3[t]);
		int j = get<1>(Raw.corr_hopping3[t]);
		int k = get<2>(Raw.corr_hopping3[t]);
		typename MODEL::Scalar_ lambda = get<3>(Raw.corr_hopping3[t]); // Viijk
		
		OPERATOR hop1 = prod(Hdummy.template n<UP>(i),Hdummy.template cdagc<DN,DN>(j,k)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template n<UP>(i),Hdummy.template cdagc<DN,DN>(k,j)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_corr_hopping3.push_back(hop12);
		Hterms.H3_corr_hopping3.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.corr_hopping3B.size(); ++t)
	{
		int i = get<0>(Raw.corr_hopping3B[t]);
		int j = get<1>(Raw.corr_hopping3B[t]);
		int k = get<2>(Raw.corr_hopping3B[t]);
		typename MODEL::Scalar_ lambda = get<3>(Raw.corr_hopping3B[t]); // Vijkk
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template n<DN>(k)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template n<DN>(k)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_corr_hopping3.push_back(hop12);
		Hterms.H3_corr_hopping3.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.nonlocal_spin.size(); ++t)
	{
		int i = get<0>(Raw.nonlocal_spin[t]);
		int j = get<1>(Raw.nonlocal_spin[t]);
		int k = get<2>(Raw.nonlocal_spin[t]);
		typename MODEL::Scalar_ lambda = get<3>(Raw.nonlocal_spin[t]); // Vijki
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template cdagc<DN,DN>(k,i)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template cdagc<DN,DN>(i,k)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_nonlocal_spin.push_back(hop12);
		Hterms.H3_nonlocal_spin.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.nonlocal_spinB.size(); ++t)
	{
		int i = get<0>(Raw.nonlocal_spinB[t]);
		int j = get<1>(Raw.nonlocal_spinB[t]);
		int k = get<2>(Raw.nonlocal_spinB[t]);
		typename MODEL::Scalar_ lambda = get<3>(Raw.nonlocal_spinB[t]); // Vijjk
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template cdagc<DN,DN>(j,k)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template cdagc<DN,DN>(k,j)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_nonlocal_spin.push_back(hop12);
		Hterms.H3_nonlocal_spin.push_back(MODEL(hop12,dummy_params));
	}
	for (int t=0; t<Raw.doublon_decay.size(); ++t)
	{
		int i = get<0>(Raw.doublon_decay[t]);
		int j = get<1>(Raw.doublon_decay[t]);
		int k = get<2>(Raw.doublon_decay[t]);
		typename MODEL::Scalar_ lambda = get<3>(Raw.doublon_decay[t]); // Vijik
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template cdagc<DN,DN>(i,k)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template cdagc<DN,DN>(k,i)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_doublon_decay.push_back(hop12);
		Hterms.H3_doublon_decay.push_back(MODEL(hop12,dummy_params));
	}
	/////////// 4-SITE ///////////
	for (int t=0; t<Raw.foursite.size(); ++t)
	{
		int i = get<0>(Raw.foursite[t]);
		int j = get<1>(Raw.foursite[t]);
		int k = get<2>(Raw.foursite[t]);
		int l = get<3>(Raw.foursite[t]);
		typename MODEL::Scalar_ lambda = get<4>(Raw.foursite[t]); // Vijkl
		
		OPERATOR hop1 = prod(Hdummy.template cdagc<UP,UP>(i,j),Hdummy.template cdagc<DN,DN>(k,l)); hop1.scale(lambda);
		OPERATOR hop2 = prod(Hdummy.template cdagc<UP,UP>(j,i),Hdummy.template cdagc<DN,DN>(l,k)); hop2.scale(conj(lambda));
		OPERATOR hop12 = sum(hop1,hop2);
		
		Terms.Hmpo_foursite.push_back(hop12);
		Hterms.H4.push_back(MODEL(hop12,dummy_params));
	}
	
//	for (int k=0; k<L; ++k)
//	for (int l=0; l<L; ++l)
//	for (int m=0; m<L; ++m)
//	for (int n=0; n<L; ++n)
//	{
//		std::set<int> s;
//		s.insert(k);
//		s.insert(l);
//		s.insert(m);
//		s.insert(n);
//		
//		typename MODEL::Scalar_ Uelement = 0.;
//		for (int i=0; i<L; ++i)
//		{
//			Uelement += conj(UU(i,k))*UU(i,l)*conj(UU(i,m))*UU(i,n);
//		}
//		if (abs(Uelement) <= 1e-8) continue;
//		
//		auto Vklmn = make_tuple(k,l,m,n);
//		Umap[Vklmn] = U*Uelement;
//		
//		bool LOCAL = false;
//		bool CORRHOP = false;
//		bool SPINEXC = false;
//		bool PAIRHOP = false;
//		bool NONLOCSPIN = false;
//		bool CORRHOP3 = false;
//		bool DOUBLON = false;
//		bool FOURSITE = false;
//		
//		if (s.size() == 1)
//		{
//			//lout << termcolor::blue << "Viiii(local):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//			LOCAL = true;
//		}
//		else if (s.size() == 2)
//		{
//			if (l==m and m==n and l==n and k!=l)
//			{
//				//lout << termcolor::magenta << "Vjiii(corrhopA):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//				CORRHOP = true;
//			}
//			else if (k==m and m==n and k==n and k!=l)
//			{
//				//lout << termcolor::magenta << "Vijii(corrhopB):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//				CORRHOP = true;
//			}
//			else if (k==l and l==n and k==n and k!=m)
//			{
//				//lout << termcolor::magenta << "Viiji(corrhopC):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//				CORRHOP = true;
//			}
//			else if (k==l and l==m and k==m and k!=n)
//			{
//				//lout << termcolor::magenta << "Viiij(corrhopD):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//				CORRHOP = true;
//			}
//			else if (k==m and l==n and k!=l)
//			{
//				//lout << termcolor::red << "Viijj(pairhop):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//				PAIRHOP = true;
//			}
//			else if (k==n and l==m and k!=l)
//			{
//				//lout << termcolor::green << "Vijij(spinflip):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//				SPINEXC = true;
//			}
//			else if (k==l and m==n and k!=m)
//			{
//				//lout << termcolor::green << "Viijj(Ising):\t" << k << "\t" << l << "\t" << m << "\t" << n << ":\t" << U*Uelement << ", s.size()=" << s.size() << termcolor::reset << endl;
//				SPINEXC = true;
//			}
//		}
//		else if (s.size() == 3)
//		{
//			if (k==l or m==n)
//			{
//				CORRHOP3 = true;
//			}
//			else if (k==m or l==n)
//			{
//				DOUBLON = true;
//			}
//			else if (k==n or l==m)
//			{
//				NONLOCSPIN = true;
//			}
//		}
//		else if (s.size() == 4)
//		{
//			FOURSITE = true;
//		}
//		
//		if (s.size() == 1)
//		{
//			Terms.HubbardU_kspace.push_back(make_tuple(k,real(U*Uelement)));
//		}
//		else
//		{
//			OPERATOR hop12 = prod(Hdummy.template cdagc<UP,UP>(k,l),Hdummy.template cdagc<DN,DN>(m,n)); hop12.scale(U*Uelement);
//			
//			if (SPINEXC)
//			{
//				Terms.Hmpo_spin_exchange.push_back(hop12);
//				Hterms.H2_spin_exchange.push_back(MODEL(hop12,dummy));
//			}
//			else if (CORRHOP)
//			{
//				Terms.Hmpo_corr_hopping.push_back(hop12);
//				Hterms.H2_corr_hopping.push_back(MODEL(hop12,dummy));
//			}
//			else if (PAIRHOP)
//			{
//				Terms.Hmpo_pair_hopping.push_back(hop12);
//				Hterms.H2_pair_hopping.push_back(MODEL(hop12,dummy));
//			}
//			else if (NONLOCSPIN)
//			{
//				Terms.Hmpo_nonlocal_spin.push_back(hop12);
//				Hterms.H3_nonlocal_spin.push_back(MODEL(hop12,dummy));
//			}
//			else if (CORRHOP3)
//			{
//				Terms.Hmpo_corr_hopping3.push_back(hop12);
//				Hterms.H3_corr_hopping3.push_back(MODEL(hop12,dummy));
//			}
//			else if (DOUBLON)
//			{
//				Terms.Hmpo_doublon_decay.push_back(hop12);
//				Hterms.H3_doublon_decay.push_back(MODEL(hop12,dummy));
//			}
//			else if (FOURSITE)
//			{
//				Terms.Hmpo_foursite.push_back(hop12);
//				Hterms.H4.push_back(MODEL(hop12,dummy));
//			}
//		}
//	}
	Hterms.HubbardU_kspace = Terms.HubbardU_kspace;
}

template<typename MODEL>
MODEL HubbardKspace<MODEL>::
sum_all() const
{
	auto res = Terms.Hmpo_spin_exchange[0];
	
	for (int s=1; s<Terms.Hmpo_spin_exchange.size(); ++s) res = sum(res,Terms.Hmpo_spin_exchange[s]);
	for (int s=0; s<Terms.Hmpo_density_density.size(); ++s) res = sum(res,Terms.Hmpo_density_density[s]);
	for (int s=0; s<Terms.Hmpo_pair_hopping.size(); ++s) res = sum(res,Terms.Hmpo_pair_hopping[s]);
	for (int s=0; s<Terms.Hmpo_corr_hopping.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping[s]);
	for (int s=0; s<Terms.Hmpo_corr_hopping3.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping3[s]);
	for (int s=0; s<Terms.Hmpo_nonlocal_spin.size(); ++s) res = sum(res,Terms.Hmpo_nonlocal_spin[s]);
	for (int s=0; s<Terms.Hmpo_doublon_decay.size(); ++s) res = sum(res,Terms.Hmpo_doublon_decay[s]);
	for (int s=0; s<Terms.Hmpo_foursite.size(); ++s) res = sum(res,Terms.Hmpo_foursite[s]);
	
	return MODEL(res,dummy_params);
}

template<typename MODEL>
MODEL HubbardKspace<MODEL>::
sum_all (const ArrayXXcd &hoppingRealSpace) const
{
	ArrayXXcd hopping = (UU.adjoint() * hoppingRealSpace.matrix() * UU).array();
	
	MODEL Hdummy(Lred,dummy_params,BC::OPEN,DMRG::VERBOSITY::SILENT);
	auto [i,Uval] = Terms.HubbardU_kspace[0];
	OPERATOR dtmp = Hdummy.d(0);
	dtmp.scale(Uval);
	auto res = dtmp;
	
	for (int t=1; t<Terms.HubbardU_kspace.size(); ++t)
	{
		auto [i,Uval] = Terms.HubbardU_kspace[t];
		OPERATOR dtmp = Hdummy.d(t);
		dtmp.scale(Uval);
		res = sum(res,dtmp);
	}
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		if (abs(hopping(i,j)) > 1e-10)
		{
			if (i==j)
			{
				OPERATOR ntmp = Hdummy.n(i);
				ntmp.scale(-hopping(i,i));
				res = sum(res,ntmp);
				//lout << "i=" << i << ", t0val=" << t0val << endl;
			}
			else
			{
				OPERATOR htmp;
				htmp = Hdummy.cdagc(x[i], x[j], y[i], y[j]); htmp.scale(-hopping(i,j));
				res = sum(res,htmp);
	//			htmp = Hdummy.cdagc(x[j], x[i], y[j], y[i]); htmp.scale(conj(hopping(i,j)));
	//			res = sum(res,htmp);
				//lout << "i=" << i << ", j=" << j << ", hopval=" << -hopping(i,j) << endl;
			}
		}
	}
	
	for (int s=0; s<Terms.Hmpo_spin_exchange.size(); ++s) res = sum(res,Terms.Hmpo_spin_exchange[s]);
	for (int s=0; s<Terms.Hmpo_density_density.size(); ++s) res = sum(res,Terms.Hmpo_density_density[s]);
	for (int s=0; s<Terms.Hmpo_pair_hopping.size(); ++s) res = sum(res,Terms.Hmpo_pair_hopping[s]);
	for (int s=0; s<Terms.Hmpo_corr_hopping.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping[s]);
	for (int s=0; s<Terms.Hmpo_corr_hopping3.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping3[s]);
	for (int s=0; s<Terms.Hmpo_nonlocal_spin.size(); ++s) res = sum(res,Terms.Hmpo_nonlocal_spin[s]);
	for (int s=0; s<Terms.Hmpo_doublon_decay.size(); ++s) res = sum(res,Terms.Hmpo_doublon_decay[s]);
	for (int s=0; s<Terms.Hmpo_foursite.size(); ++s) res = sum(res,Terms.Hmpo_foursite[s]);
	
	return MODEL(res,dummy_params);
}

template<typename MODEL>
Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> HubbardKspace<MODEL>::
sum_all_mpo() const
{
	auto res = Terms.Hmpo_spin_exchange[0];
	
	for (int s=1; s<Terms.Hmpo_spin_exchange.size(); ++s) res = sum(res,Terms.Hmpo_spin_exchange[s]);
	for (int s=0; s<Terms.Hmpo_density_density.size(); ++s) res = sum(res,Terms.Hmpo_density_density[s]);
	for (int s=0; s<Terms.Hmpo_pair_hopping.size(); ++s) res = sum(res,Terms.Hmpo_pair_hopping[s]);
	for (int s=0; s<Terms.Hmpo_corr_hopping.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping[s]);
	
	for (int s=0; s<Terms.Hmpo_corr_hopping3.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping3[s]);
	for (int s=0; s<Terms.Hmpo_nonlocal_spin.size(); ++s) res = sum(res,Terms.Hmpo_nonlocal_spin[s]);
	for (int s=0; s<Terms.Hmpo_doublon_decay.size(); ++s) res = sum(res,Terms.Hmpo_doublon_decay[s]);
	
	for (int s=0; s<Terms.Hmpo_foursite.size(); ++s) res = sum(res,Terms.Hmpo_foursite[s]);
	
	return res;
}

template<typename MODEL>
Mpo<typename MODEL::Symmetry,typename MODEL::Scalar_> HubbardKspace<MODEL>::
sum_2site_mpo() const
{
	auto res = Terms.Hmpo_spin_exchange[0];
	
	for (int s=1; s<Terms.Hmpo_spin_exchange.size(); ++s) res = sum(res,Terms.Hmpo_spin_exchange[s]);
	for (int s=0; s<Terms.Hmpo_density_density.size(); ++s) res = sum(res,Terms.Hmpo_density_density[s]);
	for (int s=0; s<Terms.Hmpo_pair_hopping.size(); ++s) res = sum(res,Terms.Hmpo_pair_hopping[s]);
	for (int s=0; s<Terms.Hmpo_corr_hopping.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping[s]);
	
	return res;
}

template<typename MODEL>
MODEL HubbardKspace<MODEL>::
sum_2site() const
{
	auto res = Terms.Hmpo_spin_exchange[0];
	
	for (int s=1; s<Terms.Hmpo_spin_exchange.size(); ++s) res = sum(res,Terms.Hmpo_spin_exchange[s]);
	for (int s=0; s<Terms.Hmpo_density_density.size(); ++s) res = sum(res,Terms.Hmpo_density_density[s]);
	for (int s=0; s<Terms.Hmpo_pair_hopping.size(); ++s) res = sum(res,Terms.Hmpo_pair_hopping[s]);
	for (int s=0; s<Terms.Hmpo_corr_hopping.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping[s]);
	
	return MODEL(res,dummy_params);
}

template<typename MODEL>
MODEL HubbardKspace<MODEL>::
sum_3site() const
{
	auto res = Terms.Hmpo_spin_exchange[0];
	
	for (int s=0; s<Terms.Hmpo_corr_hopping3.size(); ++s) res = sum(res,Terms.Hmpo_corr_hopping3[s]);
	for (int s=0; s<Terms.Hmpo_nonlocal_spin.size(); ++s) res = sum(res,Terms.Hmpo_nonlocal_spin[s]);
	for (int s=0; s<Terms.Hmpo_doublon_decay.size(); ++s) res = sum(res,Terms.Hmpo_doublon_decay[s]);
	
	return MODEL(res,dummy_params);
}

template<typename MODEL>
MODEL HubbardKspace<MODEL>::
sum_4site() const
{
	auto res = Terms.Hmpo_spin_exchange[0];
	
	for (int s=0; s<Terms.Hmpo_foursite.size(); ++s) res = sum(res,Terms.Hmpo_foursite[s]);
	
	return MODEL(res,dummy_params);
}

#endif
