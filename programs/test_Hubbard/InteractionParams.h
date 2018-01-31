#ifndef INTERACTIONPARAMS
#define INTERACTIONPARAMS

#include <limits>
#include <gsl/gsl_math.h>
#include <Eigen/Dense>

struct InteractionParams
{
	//-----------<U>-----------------
	double U;
	vector<double> Uvec;
	bool COULOMB_CHECK;
	Eigen::VectorXd onsiteEnergies;
	bool ONSITE_CHECK;
	Eigen::VectorXd hoppings;
	double T2 = 0;
	
	inline void set_U (double U_input)
	{
		if (gsl_fcmp(U_input,0.,1e-10) != 0)
		{
			U = U_input;
			COULOMB_CHECK = true;
		}
		else
		{
			U = 0;
			COULOMB_CHECK = false;
		}
	}
	
	inline void set_U (vector<double> Uvec_input, double Uoffset=0)
	{
		U = std::numeric_limits<double>::quiet_NaN();
		Uvec = Uvec_input;
		transform(Uvec.begin(), Uvec.end(), Uvec.begin(), bind2nd(std::plus<double>(), Uoffset));
		COULOMB_CHECK = true;
	}
	
	inline void set_U (Eigen::ArrayXd Uvec_input, double Uoffset=0)
	{
		U = std::numeric_limits<double>::quiet_NaN();
		Uvec.resize(Uvec_input.rows());
		for (int i=0; i<Uvec.size(); ++i)
		{
			Uvec[i] = Uvec_input(i) + Uoffset;
		}
		COULOMB_CHECK = true;
	}
	
	inline void set_U_toZero() {U = 0; COULOMB_CHECK = false;}
	//-----------</U>-----------------

	//-----------<on-site energies>-----------------
	inline void set_onsiteEnergies (Eigen::VectorXd values)
	{
		onsiteEnergies = values;
		ONSITE_CHECK = true;
	}
	
	inline void set_onsiteEnergies (vector<double> values)
	{
		onsiteEnergies.resize(values.size());
		for (int i=0; i<values.size(); ++i)
		{
			onsiteEnergies(i) = values[i];
		}
		ONSITE_CHECK = true;
	}
	
	inline void set_onsiteEnergies (int volume, double value)
	{
		Eigen::VectorXd v(volume);
		v.setConstant(value);
		onsiteEnergies = v;
		ONSITE_CHECK = true;
	}
	
	inline void set_onsiteEnergies_toZero() {ONSITE_CHECK = false;}
	//-----------</on-site energies>-----------------
	
	//-----------<hoppings>-----------------
	inline void set_hoppings (std::initializer_list<double> hoppings_input)
	{
		hoppings.resize(hoppings_input.size());
		int i=0;
		for (auto o=hoppings_input.begin(); o!=hoppings_input.end(); ++o)
		{
			hoppings(i) = *o;
			++i;
		}
	}
	
	inline void set_hoppings (Eigen::VectorXd hoppings_input, double toffset=0)
	{
		hoppings = hoppings_input;
		hoppings.array() += toffset;
	}
	
	inline void set_hoppings_toNN()
	{
		hoppings.resize(1);
		hoppings(0) = -1.;
	}
	
	inline void set_hoppings_toZero()
	{
		hoppings.resize(1);
		hoppings(0) = -1e-15;
	}
	//-----------</hoppings>-----------------
};

#endif
