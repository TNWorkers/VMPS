#ifndef ENTROPY_OBSERVER
#define ENTROPY_OBSERVER

#include "EigenFiles.h" // from TOOLS

template<typename MpsType>
class EntropyObserver
{
public:
	
	EntropyObserver(){};
	EntropyObserver (size_t L_input, size_t tpoints_input, DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT, double DeltaS_input=1e-2)
	:L(L_input), tpoints(tpoints_input), DeltaS(DeltaS_input), CHOSEN_VERBOSITY(VERBOSITY)
	{
		data.resize(tpoints,L-1);
	}
	
	vector<bool> TWO_SITE (int it, const MpsType &Psi, double r=1.);
	
	void save (string filename);
	void save (int it, string filename);
	
private:
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	size_t L;
	size_t tpoints;
	double DeltaS;
	MatrixXd data;
};

template<typename MpsType>
vector<bool> EntropyObserver<MpsType>::
TWO_SITE (int it, const MpsType &Psi, double r)
{
	vector<bool> res(L-1);
	
	for (int b=0; b<L-1; ++b)
	{
		data(it,b) = Psi.entropy()(b);
		
		double DeltaSb = std::nan("0");
		
		if (it == 1)
		{
			// backward derivative using 1 point
			DeltaSb = (data(it,b)-data(it-1,b))/data(it-1,b);
		}
		else if (it > 1)
		{
			// backward derivative using 2 points
			// 0.5*(1.+r): heuristic correction factor for arbitrary timestep
			// r=dt(-2)/dt(-1)
			DeltaSb = 0.5*(-3.*data(it,b)+4.*data(it-1,b)-data(it-2,b)) * 0.5*(1.+r) /data(it-2,b);
		}
		
		if (it == 0)
		{
			res[b] = true;
		}
		else
		{
			res[b] = (abs(DeltaSb) > DeltaS)? true:false;
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "EntropyObserver: ";
		for (int b=0; b<L-1; ++b)
		{
			if (res[b])
			{
				lout << termcolor::red << 2 << termcolor::reset;
			}
			else
			{
				lout << termcolor::blue << 1 << termcolor::reset;
			}
		}
		int trues = std::count(res.begin(), res.end(), true);
		lout << " N_steps(2-site)=" << trues << " (" << trues*100./(L-1) << "%)" << endl;
	}
	
	if (res[0] == true and it>0 and CHOSEN_VERBOSITY > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::yellow << "Entropy increase at the left edge!" << termcolor::reset << endl;
	}
	else if (res[L-2] == true and it>0 and CHOSEN_VERBOSITY > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::yellow << "Entropy increase at the right edge!" << termcolor::reset << endl;
	}
	
	return res;
}

template<typename MpsType>
void EntropyObserver<MpsType>::
save (string filename)
{
	saveMatrix(data, filename);
}

template<typename MpsType>
void EntropyObserver<MpsType>::
save (int it, string filename)
{
	saveMatrix(data.topRows(it+1), filename);
}

#endif
