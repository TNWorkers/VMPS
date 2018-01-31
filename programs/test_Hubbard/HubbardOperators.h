void HubbardModel::
n (int isite, SPIN_INDEX sigma, SparseMatrixXd &Mout) const
{
	if (sigma == UP)
	{
		SparseMatrixXd Mtemp;
		spin[UP]->n(isite,Mtemp);
		MkronI(Mtemp, spin[DN]->dim(), Mout);
	}
	else if (sigma == DN)
	{
		SparseMatrixXd Mtemp;
		spin[DN]->n(isite,Mtemp);
		IkronM(spin[UP]->dim(), Mtemp, Mout);
	}
}

inline SparseMatrixXd HubbardModel::
n (int isite, SPIN_INDEX sigma) const
{
	SparseMatrixXd Mout;
	n(isite,sigma,Mout);
	return Mout;
}

inline SparseMatrixXd HubbardModel::
n (SPIN_INDEX sigma, int isite) const
{
	return n(isite,sigma);
}

inline void HubbardModel::
n (int isite, SparseMatrixXd &Mout) const
{
	n(isite,UP,Mout);
	SparseMatrixXd Mtemp;
	n(isite,DN,Mtemp);
	Mout += Mtemp;
}

inline SparseMatrixXd HubbardModel::
n (int isite) const
{
	SparseMatrixXd Mout;
	n(isite,Mout);
	return Mout;
}

inline SparseMatrixXd HubbardModel::
d (int isite) const
{
	return n(isite,UP)*n(isite,DN);
}

inline SparseMatrixXd HubbardModel::
dsum() const
{
	SparseMatrixXd Mout = n(0,UP)*n(0,DN);
	for (int iL=1; iL<N_sites; ++iL)
	{
		Mout += n(iL,UP)*n(iL,DN);
	}
	return Mout/N_sites;
}

inline SparseMatrixXd HubbardModel::
h (int isite) const
{
	SparseMatrixXd Id(N_states,N_states);
	for (int state=0; state<N_states; ++state)
	{
		Id.insert(state,state) = 1.;
	}
	return (Id-n(isite,UP))*(Id-n(isite,DN));
}

inline SparseMatrixXd HubbardModel::
hsum() const
{
	SparseMatrixXd Mout = h(0);
	for (int iL=1; iL<N_sites; ++iL)
	{
		Mout += h(iL);
	}
	return Mout/N_sites;
}

inline SparseMatrixXd HubbardModel::
hopping_element (int origin, int destination, SPIN_INDEX sigma) const
{
	SparseMatrixXd M;
	if (origin != destination)
	{
		M = spin[sigma]->hopping_element(origin,destination,1.,false);
	}
	else
	{
		M = spin[sigma]->n(origin);
	}
	
	SparseMatrixXd Id(spin[!sigma]->dim(), spin[!sigma]->dim());
	for (int state=0; state<spin[!sigma]->dim(); ++state)
	{
		Id.insert(state,state) = 1.;
	}
	
	return (sigma==UP)? kroneckerProduct(M,Id) : kroneckerProduct(Id,M);

}

//
//inline SparseMatrixXd HubbardModel::
//n (OperatorLabel label)
//{
//	int index = storedOperators.get_index(label);
//	if (index == -1)
//	{
//		if (label.spin = UPDN) {storedOperators.add(n(label.isite,UP)+n(label.isite,DN),label);}
//		else {storedOperators.add(n(label.isite,label.spin),label);}
//		return storedOperators[storedOperators.size()-1];
//	}
//	else
//	{
//		return storedOperators[index];
//	}
//	/*if (label.spin = UPDN)
//	{
//		return n(label.isite,UP)+n(label.isite,DN);
//	}
//	else
//	{
//		return n(label.isite,label.spin);
//	}*/
//}

inline void HubbardModel::
Nop (SparseMatrixXd &Mout) const
{
	Nop(UP,Mout);
	SparseMatrixXd Mtemp;
	Nop(DN,Mtemp);
	Mout += Mtemp;
}

inline SparseMatrixXd HubbardModel::
Nop() const
{
	SparseMatrixXd Mout;
	Nop(Mout);
	return Mout;
}

inline void HubbardModel::
Sz (int isite, SparseMatrixXd &Mout) const
{
	n(isite,UP,Mout);
	SparseMatrixXd Mtemp;
	n(isite,DN,Mtemp);
	Mout -= Mtemp;
	Mout *= 0.5;
}

inline SparseMatrixXd HubbardModel::
Sz (int isite) const
{
	return 0.5*(n(isite,UP)-n(isite,DN));
}

inline void HubbardModel::
SzSz (int isite, int jsite, SparseMatrixXd &Mout) const
{
	Sz(isite,Mout);
	SparseMatrixXd Mtemp;
	Sz(jsite,Mtemp);
	Mout = Mout*Mtemp;
}

inline SparseMatrixXd HubbardModel::
SzSz (int isite, int jsite) const
{
	return Sz(isite)*Sz(jsite);
}

//
//inline SparseMatrixXd HubbardModel::
//SzSz (OperatorLabel label)
//{
//	int index = storedOperators.get_index(label);
//	if (index == -1)
//	{
//		storedOperators.add(Sz(label.isite)*Sz(label.jsite),label);
//		return storedOperators[storedOperators.size()-1];
//	}
//	else
//	{
//		return storedOperators[index];
//	}
//	//return Sz(label.isite)*Sz(label.jsite);
//}

void HubbardModel::
SzSzSum (SparseMatrixXd &Mout) const
{
	SzSz(0,1,Mout);
	
	Geometry Euklid;
	MatrixXi next_neighbours = Euklid.neighbours(1, L_edge, spacedim, BOUNDARIES);
	SparseMatrixXd Mtemp;
	
	int N_bonds = 1;
	for (int isite=1; isite<N_sites; ++isite)
	for (int idim=0; idim<next_neighbours.cols()/2; ++idim)
	{
		if (next_neighbours(isite,idim) != -1)
		{
			SzSz(isite,next_neighbours(isite,idim),Mtemp);
			Mout += Mtemp;
			++N_bonds;
		}
	}
	Mout /= N_bonds;
}

inline SparseMatrixXd HubbardModel::
SzSzSum() const
{
	SparseMatrixXd Mout;
	SzSzSum(Mout);
	return Mout;
}

//
//inline SparseMatrixXd HubbardModel::
//SzSzSum (OperatorLabel label)
//{
//	int index = storedOperators.get_index(label);
//	if (index == -1)
//	{
//		storedOperators.add(SzSzSum(),label);
//		return storedOperators[storedOperators.size()-1];
//	}
//	else
//	{
//		return storedOperators[index];
//	}
//	//return SzSzSum();
//}

inline void HubbardModel::
Szsq (int isite, SparseMatrixXd &Mout) const
{
	n(isite,UP,Mout);
	SparseMatrixXd Mtemp;
	n(isite,DN,Mtemp);
	Mout = 0.25*(Mout+Mtemp-2.*Mout*Mtemp);
}

inline SparseMatrixXd HubbardModel::
Szsq (int isite) const
{
	return 0.25*(n(isite,UP)+n(isite,DN)-2.*n(isite,UP)*n(isite,DN));
}

void HubbardModel::
SvecSvec (int i, int j, SparseMatrixXd &Mout) const
{
	SparseMatrixXd Mtemp;
	if (i==j)
	{
		n(i,UP,Mout);
		n(i,DN,Mtemp);
		Mout = -1.5*Mout*Mtemp;
		n(i,Mtemp);
		Mout += 0.75*Mtemp;
	}
	else
	{
//		Mout = tensor_product(spin[UP]->hopping_element(j,i,1.), spin[DN]->hopping_element(i,j,1.)) 
//		+ tensor_product(spin[UP]->hopping_element(i,j,1.), spin[DN]->hopping_element(j,i,1.));
//		Mout *= -0.5;
//		Mout += Sz(i)*Sz(j);
		SparseMatrixXd MtempUP;
		SparseMatrixXd MtempDN;
		spin[UP]->hopping_element(j,i,1.,false,MtempUP);
		spin[DN]->hopping_element(i,j,1.,false,MtempDN);
		tensor_product(MtempUP,MtempDN,Mout);
		spin[UP]->hopping_element(i,j,1.,false,MtempUP);
		spin[DN]->hopping_element(j,i,1.,false,MtempDN);
		tensor_product(MtempUP,MtempDN,Mtemp);
		Mout += Mtemp;
		Mout *= -0.5;
		SzSz(i,j,Mtemp);
		Mout += Mtemp;
	}
}

SparseMatrixXd HubbardModel::
SvecSvec (int isite, int jsite) const
{
	SparseMatrixXd Mout;
	if (isite==jsite)
	{
		Mout = -1.5*n(isite,UP)*n(isite,DN)+0.75*n(isite);
	}
	else
	{
		Mout = -0.5*(tensor_product(spin[UP]->hopping_element(jsite,isite,1.,false), spin[DN]->hopping_element(isite,jsite,1.,false)) 
		            +tensor_product(spin[DN]->hopping_element(isite,jsite,1.,false), spin[UP]->hopping_element(jsite,isite,1.,false)));
		Mout += SzSz(isite,jsite);
	}
	return Mout;
}

//
//inline SparseMatrixXd HubbardModel::
//SvecSvec (OperatorLabel label)
//{
//	int index = storedOperators.get_index(label);
//	if (index == -1)
//	{
//		storedOperators.add(SvecSvec(label.isite,label.jsite),label);
//		return storedOperators[storedOperators.size()-1];
//	}
//	else
//	{
//		return storedOperators[index];
//	}
//	//return SvecSvec(label.isite,label.jsite);
//}

void HubbardModel::
SvecSvecSum (SparseMatrixXd &Mout) const
{
	SvecSvec(0,1,Mout);
	
	Geometry Euklid;
	MatrixXi next_neighbours = Euklid.neighbours(1, L_edge, spacedim, BOUNDARIES);
	SparseMatrixXd Mtemp;
	
	int N_bonds=1;
	for (int isite=1; isite<N_sites; ++isite)
	for (int idim=0; idim<next_neighbours.cols()/2; ++idim)
	{
		if (next_neighbours(isite,idim) != -1)
		{
			SvecSvec(isite,next_neighbours(isite,idim),Mtemp);
			Mout += Mtemp;
			++N_bonds;
		}
	}
	Mout /= N_bonds;
}

inline SparseMatrixXd HubbardModel::
SvecSvecSum() const
{
	SparseMatrixXd Mout;
	SvecSvecSum(Mout);
	return Mout;
}

//
//inline SparseMatrixXd HubbardModel::
//SvecSvecSum (OperatorLabel label)
//{
//	int index = storedOperators.get_index(label);
//	if (index == -1)
//	{
//		storedOperators.add(SvecSvecSum(),label);
//		return storedOperators[storedOperators.size()-1];
//	}
//	else
//	{
//		return storedOperators[index];
//	}
//	//return SvecSvecSum();
//}
