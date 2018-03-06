#ifndef DMRG_HAMILTONIAN_TERMS
#define DMRG_HAMILTONIAN_TERMS

#include <vector>
#include <tuple>
#include <string>
#include "numeric_limits.h" // from ALGS

template<typename Symmetry, typename Scalar> using LocalTerms = 
std::vector<std::tuple<Scalar, SiteOperator<Symmetry,Scalar> > >;

template<typename Symmetry, typename Scalar> using TightTerms = 
std::vector<std::tuple<Scalar, SiteOperator<Symmetry,Scalar>, SiteOperator<Symmetry,Scalar> > >;

template<typename Symmetry, typename Scalar> using NextnTerms = 
std::vector<std::tuple<Scalar, SiteOperator<Symmetry,Scalar>, SiteOperator<Symmetry,Scalar>, SiteOperator<Symmetry,Scalar> > >;

template<typename Symmetry, typename Scalar>
struct HamiltonianTerms
{
	/**local terms of Hamiltonian, format: coupling, operator*/
	LocalTerms<Symmetry,Scalar> local;
	
	/**nearest-neighbour terms of Hamiltonian, format: coupling, operator 1, operator 2*/
	TightTerms<Symmetry,Scalar> tight;
	
	/**next-nearest-neighbour terms of Hamiltonian, format: coupling, operator 1, operator 2, transfer operator*/
	NextnTerms<Symmetry,Scalar> nextn;
	
	inline size_t auxdim() {return 2+tight.size()+2*nextn.size();}
	
	std::string name="";
	std::vector<std::string> info;
	
	std::string get_info() const
	{
		std::stringstream ss;
		std::copy(info.begin(), info.end()-1, std::ostream_iterator<std::string>(ss,","));
		ss << info.back();
		
		std::string res = ss.str();
		
		while (res.find("perp") != std::string::npos) res.replace(res.find("perp"), 4, "⟂");
		while (res.find("para") != std::string::npos) res.replace(res.find("para"), 4, "∥");
		while (res.find("para") != std::string::npos) res.replace(res.find("prime"), 4, "'");
		while (res.find("perp") != std::string::npos) res.replace(res.find("Perp"), 4, "⟂");
		while (res.find("para") != std::string::npos) res.replace(res.find("Para"), 4, "∥");
		while (res.find("para") != std::string::npos) res.replace(res.find("Prime"), 4, "'");
		
		return res;
	}
	
	size_t D() const
	{
		assert(local.size()!=0 or tight.size()!=0 or nextn.size()!=0);
		
		size_t res;
		if (local.size()!=0)
		{
			res = std::get<1>(local[0]).data.rows();
		}
		else if (tight.size()!=0)
		{
			return std::get<1>(tight[0]).data.rows();
		}
		else if (nextn.size()!=0)
		{
			return std::get<1>(nextn[0]).data.rows();
		}
		return res;
	}
	
	void scale (double factor, double offset=0.)
	{
		if (abs(factor-1.) > ::mynumeric_limits<double>::epsilon())
		{
			for (size_t i=0; i<local.size(); ++i)
			{
				std::get<0>(local[i]) *= factor;
			}
			for (size_t i=0; i<tight.size(); ++i)
			{
				std::get<0>(tight[i]) *= factor;
			}
			for (size_t i=0; i<nextn.size(); ++i)
			{
				std::get<0>(nextn[i]) *= factor;
			}
		}
		
		if (abs(offset) > ::mynumeric_limits<double>::epsilon())
		{
			SiteOperator<Symmetry,Scalar> Id;
			Id.data = Matrix<Scalar,Dynamic,Dynamic>::Identity(D(),D()).sparseView();
			local.push_back(std::make_tuple(offset,Id));
		}
	}
	
	template<typename OtherScalar>
	HamiltonianTerms<Symmetry,OtherScalar> cast() const
	{
		HamiltonianTerms<Symmetry,OtherScalar> Tout;
		Tout.name = name;
		Tout.info = info;
		
		for (size_t i=0; i<local.size(); ++i)
		{
			Tout.local.push_back(std::make_tuple(std::get<0>(local[i]), (std::get<1>(local[i]).template cast<OtherScalar>() )));
		}
		for (size_t i=0; i<tight.size(); ++i)
		{
			Tout.local.push_back(std::make_tuple(std::get<0>(tight[i]), (std::get<1>(tight[i]).template cast<OtherScalar>() )));
		}
		for (size_t i=0; i<nextn.size(); ++i)
		{
			Tout.local.push_back(std::make_tuple(std::get<0>(nextn[i]), (std::get<1>(nextn[i]).template cast<OtherScalar>() )));
		}
		return Tout;
	}
};

template<typename Symmetry> using HamiltonianTermsXd  = HamiltonianTerms<Symmetry,double>;
template<typename Symmetry> using HamiltonianTermsXcd = HamiltonianTerms<Symmetry,std::complex<double> >;

#endif
