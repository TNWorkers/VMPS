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
		L.clear();
		L.setVacuum();
		R.clear();
		R.setTarget(qarray3<Symmetry::Nq>{Qtot, Qtot, Symmetry::qvacuum()});
		TRIVIAL_BOUNDARIES = true;
	}
	
	template<typename OtherScalar>
	MpsBoundaries<Symmetry,OtherScalar> cast() const
	{
		MpsBoundaries<Symmetry,OtherScalar> Bout;
		
		Bout.qloc = qloc;
		
		Bout.L = L.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
		Bout.R = R.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
		
		Bout.A[0].resize(A[0].size());
		Bout.A[1].resize(A[1].size());
		
		Bout.N_sites = N_sites;
		Bout.TRIVIAL_BOUNDARIES = TRIVIAL_BOUNDARIES;
		
		for (size_t g=0; g<2; ++g)
		for (size_t l=0; l<A[g].size(); ++l)
		{
			Bout.A[g][l].resize(A[g][l].size());
			
			for (size_t s=0; s<qloc[l].size(); ++s)
			{
				Bout.A[g][l][s].in = A[g][l][s].in;
				Bout.A[g][l][s].out = A[g][l][s].out;
				Bout.A[g][l][s].block.resize(A[g][l][s].dim);
				Bout.A[g][l][s].dict = A[g][l][s].dict;
				Bout.A[g][l][s].dim = A[g][l][s].dim;
				
				for (size_t q=0; q<A[g][l][s].dim; ++q)
				{
					Bout.A[g][l][s].block[q] = A[g][l][s].block[q].template cast<OtherScalar>();
				}
			}
		}
		
//		Bout.qOp = qOp;
//		
//		Bout.W.resize(W.size());
//		for (int i1=0; i1<W.size(); ++i1)
//		{
//			Bout.W[i1].resize(W[i1].size());
//			for (int i2=0; i2<W[i1].size(); ++i2)
//			{
//				Bout.W[i1][i2].resize(W[i1][i2].size());
//				for (int i3=0; i3<W[i1][i2].size(); ++i3)
//				{
//					Bout.W[i1][i2][i3].resize(W[i1][i2][i3].size());
//				}
//			}
//		}
//		
//		for (int i1=0; i1<W.size(); ++i1)
//		for (int i2=0; i2<W[i1].size(); ++i2)
//		for (int i3=0; i3<W[i1][i2].size(); ++i3)
//		for (int i4=0; i4<W[i1][i2][i3].size(); ++i4)
//		{
//			Bout.W[i1][i2][i3][i4] = W[i1][i2][i3][i4].template cast<OtherScalar>();
//		}
		
		return Bout;
	}
	
	bool TRIVIAL_BOUNDARIES = true;
	
	size_t N_sites = 0;
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,2> A;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
};

#endif
