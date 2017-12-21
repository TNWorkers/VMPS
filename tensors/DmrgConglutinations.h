#ifndef STRAWBERRY_DMRGCONGLUTINATIONS
#define STRAWBERRY_DMRGCONGLUTINATIONS

/**Conglutinates two matrices by applying a direct sum.
\note Two template parameters are needed in order insert a scaled matrix, which makes a different type in Eigen.
@param Min : matrix to be added
@param Mout : matrix written to*/
template<typename MatrixType1, typename MatrixType2>
void addBottomRight (const MatrixType1 &Min, MatrixType2 &Mout)
{
	int r1 = Mout.rows();
	int c1 = Mout.cols();
	int r2 = Min.rows();
	int c2 = Min.cols();
	
	Mout.conservativeResize(r1+r2, c1+c2);
	
	Mout.bottomLeftCorner(r2,c1).setZero();
	Mout.topRightCorner(r1,c2).setZero();
	Mout.bottomRightCorner(r2,c2) = Min;
}

/**Conglutinates two matrices by adding to the right.
Rows must match or one of the two has to be empty.
\note Two template parameters are needed in order insert a scaled matrix, which makes a different type in Eigen.
@param Min : matrix to be added
@param Mout : matrix written to*/
template<typename MatrixType1, typename MatrixType2>
void addRight (const MatrixType1 &Min, MatrixType2 &Mout)
{
	int r1 = Mout.rows();
	int c1 = Mout.cols();
	int c2 = Min.cols();
	
	/**If Mout empty, set Min = Mout*/
	if (r1 == 0 and c2 != 0)
	{
		Mout = Min;
	}
	else
	{
		/**If Min also empty, do nothing.*/
		if (c2 != 0)
		{
			assert(Min.rows() == Mout.rows());
			Mout.conservativeResize(r1, c1+c2);
			Mout.rightCols(c2) = Min;
		}
	}
}

/**Conglutinates two matrices by adding to the bottom.
Columns must match or one of the two has to be empty.
\note Two template parameters are needed in order insert a scaled matrix, which makes a different type in Eigen.
@param Min : matrix to be added
@param Mout : matrix written to*/
template<typename MatrixType1, typename MatrixType2>
void addBottom (const MatrixType1 &Min, MatrixType2 &Mout)
{
	int r1 = Mout.rows();
	int c1 = Mout.cols();
	int r2 = Min.rows();
	
	/**If Mout empty, set Min = Mout*/
	if (r1 == 0 and r2 != 0)
	{
		Mout = Min;
	}
	else
	{
		/**If Min also empty, do nothing.*/
		if (r2 != 0)
		{
			assert(Min.cols() == Mout.cols());
			Mout.conservativeResize(r1+r2, c1);
			Mout.bottomRows(r2) = Min;
		}
	}
}

/**Removes a column of a matrix. Might be useful to remove zero columns before doing SVD and such.
@param i : column to be removed
@param M : apply to this matrix*/
template<typename MatrixType>
void remove_col (size_t i, MatrixType &M)
{
	size_t Mrows = M.rows();
	size_t Mcols = M.cols()-1;
	
	if (i<Mcols)
	{
		M.block(0,i,Mrows,Mcols-i) = M.block(0,i+1,Mrows,Mcols-i);
	}
	M.conservativeResize(Mrows,Mcols);
}

/**Removes a row of a matrix. Might be useful to remove zero rows before doing SVD and such.
@param i : row to be removed
@param M : apply to this matrix*/
template<typename MatrixType>
void remove_row (size_t i, MatrixType &M)
{
	size_t Mrows = M.rows()-1;
	size_t Mcols = M.cols();
	
	if (i<Mcols)
	{
		M.block(0,i,Mrows-i,Mcols) = M.block(i+1,0,Mrows-i,Mcols);
	}
	M.conservativeResize(Mrows,Mcols);
}

#endif
