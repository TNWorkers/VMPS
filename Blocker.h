#ifndef BLOCKER_H_
#define BLOCKER_H_

template<typename Symmetry, typename Scalar=double>
class Blocker
{
private:
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	static constexpr size_t Nq = Symmetry::Nq;

public:
	Blocker() {};
	
	Blocker(vector<Biped<Symmetry,MatrixType> > &A_in, const vector<qarray<Nq> > &qloc_in, const Qbasis<Symmetry> &inbase_in, const Qbasis<Symmetry> &outbase_in)
		:A(A_in), qloc(qloc_in), inbase(inbase_in), outbase(outbase_in) {};

	void block(DMRG::DIRECTION::OPTION DIR);

	vector<Biped<Symmetry,MatrixType> > reblock(const Biped<Symmetry,MatrixType> &B, DMRG::DIRECTION::OPTION DIR);
	
	Biped<Symmetry,MatrixType> Aclump(DMRG::DIRECTION::OPTION DIR) { block(DIR); return Aclump_;}

	void FORCE_NEW_BLOCKING() {HAS_BLOCKED = false;}
	
private:
	vector<Biped<Symmetry,MatrixType> > &A;
	vector<qarray<Nq> > qloc;
	Qbasis<Symmetry> inbase, outbase;

	Biped<Symmetry,MatrixType> Aclump_;

	bool HAS_BLOCKED=false;

	void block_left();
	void block_right();

	vector<Biped<Symmetry,MatrixType> > reblock_left(const Biped<Symmetry,MatrixType> &B);
	vector<Biped<Symmetry,MatrixType> > reblock_right(const Biped<Symmetry,MatrixType> &B);
};

template<typename Symmetry, typename Scalar>
void Blocker<Symmetry,Scalar>::
block(DMRG::DIRECTION::OPTION DIR)
{
	if (HAS_BLOCKED) {return;}
	if (DIR == DMRG::DIRECTION::LEFT) {block_left();}
	else if (DIR == DMRG::DIRECTION::RIGHT) {block_right();}
	HAS_BLOCKED = true;
}

template<typename Symmetry, typename Scalar>
vector<Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > > Blocker<Symmetry,Scalar>::
reblock(const Biped<Symmetry,MatrixType> &B, DMRG::DIRECTION::OPTION DIR)
{
	assert (HAS_BLOCKED and "Only can reblock if the A-tensor got blocked before.");
	if (DIR == DMRG::DIRECTION::LEFT) {return reblock_left(B);}
	else if (DIR == DMRG::DIRECTION::RIGHT) {return reblock_right(B);}
	else {exit(1);} //avoid stupid warning for no return
}

template<typename Symmetry, typename Scalar>
void Blocker<Symmetry,Scalar>::
block_right()
{
	Aclump_.clear();
	for (size_t qin=0; qin<inbase.Nq(); ++qin)
	{
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Ncolsvec;
		for (size_t s=0; s<qloc.size(); ++s)
		for (size_t q=0; q<A[s].dim; ++q)
		{
			if (A[s].in[q] == inbase[qin])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Ncolsvec.push_back(A[s].block[q].cols());
			}
		}
		
		if (Ncolsvec.size() > 0)
		{
			// do the glue
			size_t Nrows = A[svec[0]].block[qvec[0]].rows();
			for (size_t i=1; i<svec.size(); ++i)
			{
				assert(A[svec[i]].block[qvec[i]].rows() == Nrows);
			}
			size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
			
			MatrixType Mtmp(Nrows,Ncols);
			Mtmp.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Mtmp.block(0,stitch, Nrows,Ncolsvec[i]) = A[svec[i]].block[qvec[i]]* Symmetry::coeff_leftSweep(A[svec[i]].out[qvec[i]],A[svec[i]].in[qvec[i]],qloc[svec[i]]);
				stitch += Ncolsvec[i];
			}
			Aclump_.push_back(inbase[qin], inbase[qin], Mtmp);
		}
	}
}

template<typename Symmetry, typename Scalar>
void Blocker<Symmetry,Scalar>::
block_left()
{
	Aclump_.clear();
	for (size_t qout=0; qout<outbase.Nq(); ++qout)
	{
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Nrowsvec;
		for (size_t s=0; s<qloc.size(); ++s)
		for (size_t q=0; q<A[s].dim; ++q)
		{
			if (A[s].out[q] == outbase[qout])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Nrowsvec.push_back(A[s].block[q].rows());
			}
		}
		
		if (Nrowsvec.size() > 0)
		{
			// do the glue
			size_t Ncols = A[svec[0]].block[qvec[0]].cols();
			for (size_t i=1; i<svec.size(); ++i) {assert(A[svec[i]].block[qvec[i]].cols() == Ncols);}
			size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
			
			MatrixType Mtmp(Nrows,Ncols);
			Mtmp.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Mtmp.block(stitch,0, Nrowsvec[i],Ncols) = A[svec[i]].block[qvec[i]];
				stitch += Nrowsvec[i];
			}
			Aclump_.push_back(outbase[qout], outbase[qout], Mtmp);
		}
	}
}

template<typename Symmetry, typename Scalar>
vector<Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > > Blocker<Symmetry,Scalar>::
reblock_left(const Biped<Symmetry,MatrixType> &B)
{
	vector<Biped<Symmetry,MatrixType> > Aout(qloc.size());
	
	for (size_t qout=0; qout<outbase.Nq(); ++qout)
	{
		auto it = B.dict.find({outbase[qout], outbase[qout]});
		assert(it != B.dict.end());
		size_t qB = it->second;

		// determine how many A's to glue together
		vector<size_t> svec, qvec, Nrowsvec;
		for (size_t s=0; s<qloc.size(); ++s)
		for (size_t q=0; q<A[s].dim; ++q)
		{
			if (A[s].out[q] == outbase[qout])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Nrowsvec.push_back(A[s].block[q].rows());
			}
		}

		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			MatrixType Mtmp;
			Mtmp = B.block[qB].block(stitch,0, Nrowsvec[i],B.block[qB].cols());
					
			if (Mtmp.size() != 0)
			{
				Aout[svec[i]].push_back(A[svec[i]].in[qvec[i]], A[svec[i]].out[qvec[i]], Mtmp);
			}
			stitch += Nrowsvec[i];
		}
	}
	return Aout;
}

template<typename Symmetry, typename Scalar>
vector<Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > > Blocker<Symmetry,Scalar>::
reblock_right(const Biped<Symmetry,MatrixType> &B)
{
	vector<Biped<Symmetry,MatrixType> > Aout(qloc.size());
	
	for (size_t qin=0; qin<inbase.Nq(); ++qin)
	{
		auto it = B.dict.find({inbase[qin], inbase[qin]});
		assert(it != B.dict.end());
		size_t qB = it->second;
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Ncolsvec;
		for (size_t s=0; s<qloc.size(); ++s)
		for (size_t q=0; q<A[s].dim; ++q)
		{
			if (A[s].in[q] == inbase[qin])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Ncolsvec.push_back(A[s].block[q].cols());
			}
		}

		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			MatrixType Mtmp;
			Mtmp = B.block[qB].block(0,stitch, B.block[qB].rows(),Ncolsvec[i])*Symmetry::coeff_sign(A[svec[i]].out[qvec[i]],A[svec[i]].in[qvec[i]],qloc[svec[i]]);
			
			if (Mtmp.size() != 0)
			{
				Aout[svec[i]].push_back(A[svec[i]].in[qvec[i]], A[svec[i]].out[qvec[i]], Mtmp);
			}
			stitch += Ncolsvec[i];
		}
	}
	return Aout;
}

#endif //BLOCKER_H_
