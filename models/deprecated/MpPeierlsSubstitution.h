#ifndef STRAWBERRY_PEIERLSSUBSTITUTION
#define STRAWBERRY_PEIERLSSUBSTITUTION

#include "MpHubbardModel.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}
\f$.
\note If the nnn-hopping is positive, the ground state energy is lowered.*/
class PeierlsSubstitution : public MpoQ<2,complex<double> >, public TimeDependence<complex<double> >
{
public:
	
	/**
	\param L_input : chain length
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	template<BC_CHOICE CHOICE> PeierlsSubstitution (BC<CHOICE> BC_input, double U_input, const vector<double> &onsite_input, double V_input=0.);
	
	HamiltonianTermsXcd set_operators (const FermionBase &F, double U, double onsite, complex<double> h, double V);
//	HamiltonianTermsXcd set_operators (const FermionBase &F, vector<double> Uvec, MatrixXcd tInter, complex<double> tIntra, double V);
//	HamiltonianTermsXcd set_operators (const FermionBase &F, double U, complex<double> tInter, complex<double> tIntra, double V);
	
	typedef MpsQ<2,complex<double> >                            StateXcd;
	typedef DmrgSolverQ<2,PeierlsSubstitution,complex<double> > Solver;
	
	void set_onsite (const vector<double> &onsite_input) {onsite=onsite_input;}
	
	/**Rebuilds the MPO with the given A(t).*/
	void update();
	
private:
	
	double U;
	double V = 0;
	vector<double> onsite;
	
	FermionBase FB;
	
	BC_CHOICE CHOSEN_BC;
};

HamiltonianTermsXcd PeierlsSubstitution::
set_operators (const FermionBase &FB, double U, double onsite, complex<double> h, double V)
{
	HamiltonianTermsXcd Terms;
	complex<double> Vc = V;
	complex<double> Uc = U;
	complex<double> onsitec = onsite;
	
	SparseMatrixXcd cUP = FB.c(UP).cast<complex<double> >();
	SparseMatrixXcd cDN = FB.c(DN).cast<complex<double> >();
	SparseMatrixXcd cdagUP = FB.cdag(UP).cast<complex<double> >();
	SparseMatrixXcd cdagDN = FB.cdag(DN).cast<complex<double> >();
	SparseMatrixXcd fsign = FB.sign().cast<complex<double> >();
	SparseMatrixXcd n = FB.n().cast<complex<double> >();
	SparseMatrixXcd d = FB.d().cast<complex<double> >();
	
	Terms.tight.push_back(make_tuple(-h, cdagUP, fsign * cUP));
	Terms.tight.push_back(make_tuple(-h, cdagDN, fsign * cDN));
	Terms.tight.push_back(make_tuple(+conj(h), cUP, fsign * cdagUP));
	Terms.tight.push_back(make_tuple(+conj(h), cDN, fsign * cdagDN));
	
	if (V != 0.)
	{
		Terms.tight.push_back(make_tuple(Vc, n, n));
	}
	
	Terms.local.push_back(make_tuple(Uc+onsitec, d));
//	Terms.local.push_back(make_tuple(onsitec, n));
	
	return Terms;
}

//HamiltonianTermsXcd PeierlsSubstitution::
//set_operators (const FermionBase &F, vector<double> Uvec, MatrixXcd tInter, complex<double> tIntra, double V)
//{
//	assert(Uvec.size() == 2);
//	HamiltonianTermsXcd Terms;
//	
//	for (int i=0; i<2; ++i)
//	{
//		if (tInter(i,i) != 0.)
//		{
//			SparseMatrixXcd fsign  = FB.sign().cast<complex<double> >();
//			SparseMatrixXcd cUP    = FB.c(UP,i).cast<complex<double> >();
//			SparseMatrixXcd cDN    = FB.c(DN,i).cast<complex<double> >();
//			SparseMatrixXcd cdagUP = FB.cdag(UP,i).cast<complex<double> >();
//			SparseMatrixXcd cdagDN = FB.cdag(DN,i).cast<complex<double> >();
//			
//			Terms.tight.push_back(make_tuple(-tInter(i,i), cdagUP, fsign * cUP));
//			Terms.tight.push_back(make_tuple(-tInter(i,i), cdagDN, fsign * cDN));
//			Terms.tight.push_back(make_tuple(+conj(tInter(i,i)), cUP, fsign * cdagUP));
//			Terms.tight.push_back(make_tuple(+conj(tInter(i,i)), cDN, fsign * cdagDN));
//		}
//		if (V != 0.)
//		{
//			complex<double> Vc = V;
//			Terms.tight.push_back(make_tuple(Vc, FB.n(i).cast<complex<double> >(), 
//			                                     FB.n(i).cast<complex<double> >()));
//		}
//	}
//	
//	Terms.local.push_back(make_tuple(complex<double>(1.,0.), FB.HubbardHamiltonian(Uvec,tIntra,V)));
//	
//	return Terms;
//}

//HamiltonianTermsXcd PeierlsSubstitution::
//set_operators (const FermionBase &F, double U, complex<double> tInter, complex<double> tIntra, double V)
//{
//	vector<double> Uvec(2);
//	fill(Uvec.begin(), Uvec.end(), U);
//	
//	MatrixXcd MtInter(2,2);
//	MtInter.setZero();
//	MtInter(0,0) = tInter;
//	MtInter(1,1) = conj(tInter);
//	cout << "inter 0->0: " << tInter << ", inter 1->1: " << conj(tInter) << endl;
//	return set_operators(F, Uvec, MtInter, tIntra, V);
//}

//PeierlsSubstitution::
//PeierlsSubstitution (size_t L_input, double U_input, double A_input, double V_input)
//:MpoQ<2,complex<double> > (L_input/2, 2, HubbardModel::qloc(1,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "PeierlsSubstitution"),
//U(U_input), A(A_input), V(V_input)
//{
//	stringstream ss;
//	ss << "(U=" << U << ",A=" << A << ",V=" << V << "," << BC_CHOICE::HAIRSLIDE << ")";
//	this->label += ss.str();
//	
//	F = FermionBase(N_legs,!isfinite(U));
//	
//	vector<SuperMatrix<complex<double> > > G(this->N_sites);
//	
//	HamiltonianTermsXcd Terms;
//	
//	for (size_t l=0; l<this->N_sites; ++l)
//	{
//		cout << "l=" << l << endl;
//		cout << "forw=" << exp(-1.i*A) << ", back=" << exp(+1.i*A) << endl;
//		if (l==0)
//		{
//			Terms = set_operators(F, U,exp(-1.i*A),exp(+1.i*A),V);
//			
//			this->Daux = Terms.auxdim();
//			G[l].setRowVector(Daux,FB.dim());
//			G[l] = Generator(Terms).row(Daux-1);
//		}
//		else if (l==this->N_sites-1)
//		{
//			Terms = set_operators(F, U,exp(-1.i*A),exp(-1.i*A),V);
//			
//			this->Daux = Terms.auxdim();
//			G[l].setColVector(Daux,FB.dim());
//			G[l] = Generator(Terms).col(0);
//		}
//		else
//		{
//			Terms = set_operators(F, U,exp(-1.i*A),0,V);
//			
//			this->Daux = Terms.auxdim();
//			G[l].setMatrix(Daux,FB.dim());
//			G[l] = Generator(Terms);
//		}
//	}
//	
//	this->construct(G, this->W, this->Gvec);
//}

template<BC_CHOICE CHOICE>
PeierlsSubstitution::
PeierlsSubstitution (BC<CHOICE> BC_input, double U_input, const vector<double> &onsite_input, double V_input)
:MpoQ<2,complex<double> > (BC_input.Lx, BC_input.Ly, HubbardModel::qloc(BC_input.Ly,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "PeierlsSubstitution"),
U(U_input), onsite(onsite_input), V(V_input), CHOSEN_BC(CHOICE)
{
//	static_assert(CHOSEN_BC==CHAIN or CHOSEN_BC==RING);
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << "," << CHOSEN_BC << ")";
	this->label += ss.str();
	
	FB = FermionBase(N_legs,!isfinite(U));
//	update(); // set t=0
}

void PeierlsSubstitution::
update()
{
	vector<SuperMatrix<complex<double> > > G(this->N_sites);
	
	calc_Fval();
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		HamiltonianTermsXcd Terms = set_operators(FB, U,onsite[l],Fval,V);
		this->Daux = Terms.auxdim()+Terms.tight.size();
		
		if (l==0)
		{
			if (CHOSEN_BC == RING)
			{
				G[l].setRowVector(Daux,FB.dim());
				SuperMatrix<complex<double> > G_PBC;
				G_PBC.setRowVector(Terms.tight.size(),FB.dim());
				for (int i=0; i<Terms.tight.size(); ++i)
				{
					// The conj() is important, therwise there will be wrong hopping on the last bond!
					G_PBC(0,i) = conj(get<0>(Terms.tight[i])) * get<1>(Terms.tight[i]);
				}
				G[l] = directSum(Generator(Terms).row(Terms.auxdim()-1),G_PBC);
			}
			else if (CHOSEN_BC == CHAIN)
			{
				this->Daux = Terms.auxdim();
				G[l].setRowVector(Daux,FB.dim());
				G[l] = Generator(Terms).row(Daux-1);
			}
		}
		else if (l==this->N_sites-1)
		{
			if (CHOSEN_BC == RING)
			{
				G[l].setColVector(Daux,FB.dim());
				SuperMatrix<complex<double> > G_PBC;
				G_PBC.setColVector(Terms.tight.size(),FB.dim());
				for (int i=0; i<Terms.tight.size(); ++i)
				{
					G_PBC(i,0) = get<2>(Terms.tight[i]);
				}
				G[l] = directSum(Generator(Terms).col(0),G_PBC);
			}
			else if (CHOSEN_BC == CHAIN)
			{
				this->Daux = Terms.auxdim();
				G[l].setColVector(Daux,FB.dim());
				G[l] = Generator(Terms).col(0);
			}
		}
		else
		{
			if (CHOSEN_BC == RING)
			{
				G[l].setMatrix(Daux,FB.dim());
				SuperMatrix<complex<double> > G_PBC;
				G_PBC.setMatrix(Terms.tight.size(),FB.dim());
				for (int i=0; i<Terms.tight.size(); ++i)
				{
					G_PBC(i,i) = FB.sign().cast<complex<double> >();
				}
				G[l] = directSum(Generator(Terms),G_PBC);
			}
			else if (CHOSEN_BC == CHAIN)
			{
				this->Daux = Terms.auxdim();
				G[l].setMatrix(Daux,FB.dim());
				G[l] = Generator(Terms);
			}
		}
	}
	
	this->construct(G, this->W, this->Gvec);
}

}

#endif
