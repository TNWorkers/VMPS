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
		DeltaSb.resize(tpoints,L-1);
	}
	
	vector<bool> TWO_SITE (int it, const MpsType &Psi, double r=1., vector<size_t> true_overrides={}, vector<size_t> false_overrides={});
	
	void elongate (size_t Lleft=0, size_t Lright=0);
	
	void save (string filename) const;
	void save (int it, string filename) const;
	
	MatrixXd get_DeltaSb() const {return DeltaSb;};
	
private:
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	size_t L;
	size_t tpoints;
	double DeltaS;
	MatrixXd data;
	MatrixXd DeltaSb;
};

template<typename MpsType>
vector<bool> EntropyObserver<MpsType>::
TWO_SITE (int it, const MpsType &Psi, double r, vector<size_t> true_overrides, vector<size_t> false_overrides)
{
	vector<bool> res(L-1);
	
	for (int b=0; b<L-1; ++b)
	{
		data(it,b) = Psi.entropy()(b);
		
		DeltaSb(it,b) = std::nan("0");
		
		if (it == 1)
		{
			// backward derivative using 1 point
			DeltaSb(it,b) = (data(it,b)-data(it-1,b))/data(it-1,b);
		}
		else if (it > 1)
		{
			// backward derivative using 2 points
			// 0.5*(1.+r): heuristic correction factor for arbitrary timestep
			// r=dt(-2)/dt(-1)
			DeltaSb(it,b) = 0.5*(3.*data(it,b)-4.*data(it-1,b)+data(it-2,b)) * 0.5*(1.+r) /data(it-2,b);
		}
		
		if (it == 0)
		{
			res[b] = true;
		}
		else
		{
			res[b] = (abs(DeltaSb(it,b)) > DeltaS)? true:false;
		}
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "b=" << b << ", δS(b)=" << DeltaSb(it,b) << ", S=" << data(it,b) << endl;
		}
	}
	
	if (res[0] == true and it>0 and CHOSEN_VERBOSITY > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::yellow << "Entropy increase at the left edge!" << termcolor::reset << endl;
	}
	if (res[L-2] == true and it>0 and CHOSEN_VERBOSITY > DMRG::VERBOSITY::SILENT)
	{
		lout << termcolor::yellow << "Entropy increase at the right edge!" << termcolor::reset << endl;
	}
	
	for (int ib=0; ib<true_overrides.size(); ++ib)
	{
		res[true_overrides[ib]] = true;
	}
	for (int ib=0; ib<false_overrides.size(); ++ib)
	{
		res[false_overrides[ib]] = false;
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "EntropyObserver: ";
		for (int b=0; b<L-1; ++b)
		{
			if (res[b])
			{
				cout << termcolor::red;
				lout << 2;
			}
			else
			{
				cout << termcolor::blue;
				lout << 1;
			}
		}
		cout << termcolor::reset;
		int trues = std::count(res.begin(), res.end(), true);
		lout << " N_steps(2-site)=" << trues << " (" << round(trues*100./(L-1),1) << "%)" << endl;
	}
	
	return res;
}

template<typename MpsType>
void EntropyObserver<MpsType>::
elongate (size_t Lleft, size_t Lright)
{
	if (Lleft>0 or Lright>0)
	{
		int Lold = L;
		L += Lleft + Lright;
		MatrixXd data_new(tpoints,L-1+Lleft+Lright);
		data_new.setZero();
		data_new.block(Lleft,0,data.rows(),data.cols()) = data;
		data = data_new;
	}
}

template<typename MpsType>
void EntropyObserver<MpsType>::
save (string filename) const
{
	saveMatrix(data, filename);
}

template<typename MpsType>
void EntropyObserver<MpsType>::
save (int it, string filename) const
{
	saveMatrix(data.topRows(it+1), filename);
}

#endif
