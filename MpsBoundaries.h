#ifndef MPSBOUNDARIES
#define MPSBOUNDARIES

template<typename Symmetry, typename Scalar>
class MpsBoundaries
{
public:
	
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	
	MpsBoundaries(){};
	
	MpsBoundaries (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L_input,
	               const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R_input,
	               const vector<vector<Biped<Symmetry,MatrixType> > > &AL_input,
	               const vector<vector<Biped<Symmetry,MatrixType> > > &AR_input,
	               const vector<vector<qarray<Symmetry::Nq> > > &qloc_input)
	:L(L_input), R(R_input), qloc(qloc_input)
	{
		A[0] = AL_input;
		A[1] = AR_input;
		N_sites = qloc.size();
		assert(A[0].size() == N_sites and A[1].size() == N_sites);
		TRIVIAL_BOUNDARIES = false;
	};
	
	inline size_t length() const {return N_sites;}
	
	inline bool IS_TRIVIAL() const {return TRIVIAL_BOUNDARIES;}
	
	void set_open_bc (qarray<Symmetry::Nq> &Qtot)
	{
		if (TRIVIAL_BOUNDARIES)
		{
			L.clear();
			L.setVacuum();
			R.clear();
			R.setTarget(qarray3<Symmetry::Nq>{Qtot, Qtot, Symmetry::qvacuum()});
		}
	}
	
	template<typename OtherScalar>
	MpsBoundaries<Symmetry,OtherScalar> cast() const
	{
		MpsBoundaries<Symmetry,OtherScalar> Bout;
		
		Bout.qloc = qloc;
		
		Bout.L = L.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
		Bout.R = R.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
		
		Bout.Lsq = Lsq.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
		Bout.Rsq = Rsq.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
		
		for (size_t g=0; g<A.size(); ++g)
		{
			Bout.A[g].resize(A[g].size());
		}
		
		Bout.N_sites = N_sites;
		Bout.TRIVIAL_BOUNDARIES = TRIVIAL_BOUNDARIES;
		
		for (size_t g=0; g<A.size(); ++g)
		for (size_t l=0; l<A[g].size(); ++l)
		{
			Bout.A[g][l].resize(A[g][l].size());
			
			for (size_t s=0; s<qloc[l].size(); ++s)
			{
				Bout.A[g][l][s].in = A[g][l][s].in;
				Bout.A[g][l][s].out = A[g][l][s].out;
				Bout.A[g][l][s].dict = A[g][l][s].dict;
				Bout.A[g][l][s].dim = A[g][l][s].dim;
				
				Bout.A[g][l][s].block.resize(A[g][l][s].dim);
				for (size_t q=0; q<A[g][l][s].dim; ++q)
				{
					Bout.A[g][l][s].block[q] = A[g][l][s].block[q].template cast<OtherScalar>();
				}
			}
		}
		
		return Bout;
	}
	
	bool TRIVIAL_BOUNDARIES = true;
	
	size_t N_sites = 0;
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Lsq;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Rsq;
	
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> A;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
};

#endif
