void n (int isite, SPIN_INDEX sigma, SparseMatrixXd &Mout) const;
void n (int isite, SparseMatrixXd &Mout) const;
void Sz (int isite, SparseMatrixXd &Mout) const;
void SzSz (int isite, int jsite, SparseMatrixXd &Mout) const;
void Szsq (int isite, SparseMatrixXd &Mout) const;
void SzSzSum (SparseMatrixXd &Mout) const;
void SvecSvec (int i, int j, SparseMatrixXd &Mout) const;
void SvecSvecSum (SparseMatrixXd &Mout) const;
void Nop (SPIN_INDEX sigma, SparseMatrixXd &Mout) const;
void Nop (SparseMatrixXd &Mout) const;

SparseMatrixXd n (int isite, SPIN_INDEX sigma) const;
SparseMatrixXd n (SPIN_INDEX sigma, int isite) const;
SparseMatrixXd n (int isite) const;
SparseMatrixXd d (int isite) const; // double occupancy = doublon
SparseMatrixXd dsum() const;
SparseMatrixXd h (int isite) const; // no occupancy = holon
SparseMatrixXd hsum() const;
SparseMatrixXd hopping_element (int origin, int destination, SPIN_INDEX sigma) const;
SparseMatrixXd Sz (int isite) const;
SparseMatrixXd SzSz (int isite, int jsite) const;
SparseMatrixXd Szsq (int isite) const;
SparseMatrixXd SzSzSum() const;
SparseMatrixXd SvecSvec (int isite, int jsite) const;
SparseMatrixXd SvecSvecSum() const;
SparseMatrixXd Nop (SPIN_INDEX sigma) const;
SparseMatrixXd Nop() const;

//SparseMatrixXd n (OperatorLabel label);
//SparseMatrixXd SzSz (OperatorLabel label);
//SparseMatrixXd SzSzSum (OperatorLabel label);
//SparseMatrixXd SvecSvec (OperatorLabel label);
//SparseMatrixXd SvecSvecSum (OperatorLabel label);
