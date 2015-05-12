#ifndef VANILLA_DMRGCONTRACTIONS
#define VANILLA_DMRGCONTRACTIONS

template<size_t D, typename MatrixType>
void contract_L (const vector<MatrixType> &Lold, 
                 const std::array<MatrixType,D> &Abra, 
                 const std::array<std::array<SparseMatrixXd,D>,D> &W, 
                 const std::array<MatrixType,D> &Aket, 
                 vector<MatrixType> &Lnew)
{
	for (size_t i=0; i<Lnew.size(); ++i)
	{
		Lnew[i].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<W[s1][s2].outerSize(); ++k)
	for (SparseMatrixXd::InnerIterator iW(W[s1][s2],k); iW; ++iW)
	{
		size_t a1 = iW.row();
		size_t a2 = iW.col();
		
		if (Lold[a1].rows() != 0)
		{
			MatrixType Mtmp = iW.value() * (Abra[s1].adjoint() * Lold[a1] * Aket[s2]);
			
			if (Lnew[a2].rows() != Mtmp.rows() or 
				Lnew[a2].cols() != Mtmp.cols())
			{
				Lnew[a2] = Mtmp;
			}
			else
			{
				Lnew[a2] += Mtmp;
			}
		}
	}
}

template<size_t D, typename MatrixType>
void contract_R (const vector<MatrixType> &Rold,
                 const std::array<MatrixType,D> &Abra, 
                 const std::array<std::array<SparseMatrixXd,D>,D> &W, 
                 const std::array<MatrixType,D> &Aket, 
                 vector<MatrixType> &Rnew)
{
	for (size_t i=0; i<Rnew.size(); ++i)
	{
		Rnew[i].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<W[s1][s2].outerSize(); ++k)
	for (SparseMatrixXd::InnerIterator iW(W[s1][s2],k); iW; ++iW)
	{
		size_t a1 = iW.row();
		size_t a2 = iW.col();
		
		if (Rold[a2].rows() != 0)
		{
			MatrixType Mtmp = iW.value() * (Aket[s2] * Rold[a2] * Abra[s1].adjoint());
			
			if (Rnew[a1].rows() != Mtmp.rows() or 
				Rnew[a1].cols() != Mtmp.cols())
			{
				Rnew[a1] = Mtmp;
			}
			else
			{
				Rnew[a1] += Mtmp;
			}
		}
	}
}

template<size_t D, typename MatrixType>
void contract_L (const vector<vector<MatrixType> > &Lold, 
                 const std::array<MatrixType,D> &Abra, 
                 const std::array<std::array<SparseMatrixXd,D>,D> &Wbot, 
                 const std::array<std::array<SparseMatrixXd,D>,D> &Wtop, 
                 const std::array<MatrixType,D> &Aket, 
                 vector<vector<MatrixType> > &Lnew)
{
	for (size_t i=0; i<Lnew.size(); ++i)
	for (size_t j=0; j<Lnew[i].size(); ++j)
	{
		Lnew[i][j].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (int kbot=0; kbot<Wbot[s1][s2].outerSize(); ++kbot)
	for (SparseMatrixXd::InnerIterator iWbot(Wbot[s1][s2],kbot); iWbot; ++iWbot)
	for (int ktop=0; ktop<Wtop[s2][s3].outerSize(); ++ktop)
	for (SparseMatrixXd::InnerIterator iWtop(Wtop[s2][s3],ktop); iWtop; ++iWtop)
	{
		size_t br = iWbot.row();
		size_t bc = iWbot.col();
		size_t tr = iWtop.row();
		size_t tc = iWtop.col();
		
		if (Lold[br][tr].rows() != 0)
		{
			MatrixType Mtmp = (iWbot.value() * iWtop.value()) * (Abra[s1].adjoint() * Lold[br][tr] * Aket[s3]);
			
			if (Lnew[bc][tc].rows() != Mtmp.rows() or 
				Lnew[bc][tc].cols() != Mtmp.cols())
			{
				Lnew[bc][tc] = Mtmp;
			}
			else
			{
				Lnew[bc][tc] += Mtmp;
			}
		}
	}
}

#endif
