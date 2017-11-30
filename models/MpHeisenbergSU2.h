#ifndef STRAWBERRY_HEISENBERGMODELSU2
#define STRAWBERRY_HEISENBERGMODELSU2

#include "symmetry/SU2.h"
#include "spins/BaseSU2.h"
#include "MpoQ.h"
#include "DmrgExternalQ.h"

namespace VMPS::models
{

/** \class HeisenbergSU2
  * \ingroup Models
  *
  * \brief Heisenberg Model
  *
  * MPO representation of 
  \f$
  H = -J \sum_{<ij>} \left(\mathbf{S_i}\mathbf{S_j}\right) -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right)
  \f$.
  *
  \note Take use of the Spin SU(2) symmetry.
  \note \f$J<0\f$ is antiferromagnetic
  */
class HeisenbergSU2 : public MpoQ<Sym::SU2<double>,double>
{
public:
	typedef Sym::SU2<double> Symmetry;
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::SparseMatrix<double> SparseMatrixType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;

	typedef SiteOperatorQ<Symmetry,MatrixType> Operator;

public:
	/** Does nothing. */
	HeisenbergSU2() : MpoQ<Symmetry,double>() {};

	/**
	\param Lx_input : chain length
	\param J_input : \f$J\f$ nn exchange coupling constant
	\param D_input : \f$2S+1\f$ Magnitude of the Spin operators
	\param Jprime_input : \f$J'\f$ nnn exchange coupling constant
	\param Ly_input : amount of legs in ladder
	*/
	HeisenbergSU2(std::size_t Lx_input, double J_input=-1., std::size_t D_input=2, double Jprime_input=0., std::size_t Ly_input=1);
		
	/**Creates the MPO generator matrix for the Heisenberg model (of any spin (\f$D=2S+1\f$)).*/
	static HamiltonianTermsXd<Symmetry> set_operators (const spins::BaseSU2<> &Spins, double J, double Jprime=0., double Jintra=0., bool PERIODIC=false);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const spins::BaseSU2<> &Spins, const Eigen::MatrixXd &Jinter, double Jprime=0., double Jintra=0.,
													   bool PERIODIC=false);

	//---label stuff---
	///@{
	/**Creates a label for this MpoQ to have a nice output.
	\param D : \f$2S+1\f$
	\param J : \f$J_z\f$
	\param Jprime : \f$J'\f$*/
	static string create_label (std::size_t D,  double J, double Jprime=0.)
	{
		auto S = frac(D-1,2);
		std::stringstream ss;
		ss << "Heisenberg(SU(2),S=" << S << ",J=" << J;
		if (Jprime != 0.) {ss << ",J'=" << Jprime;}
		ss << ")";
		return ss.str();
	}
	
	/**local basis: \f$\{ \left|\frac{1}{2}\right> \}\f$ if D=2 and N_legs=1*/
	static const std::vector<qType> getqloc ( const spins::BaseSU2<double>& Spins_in );
	static const std::vector<qType> getqOp ();
	static const std::vector<Index> getqlocDeg ( const spins::BaseSU2<double>& Spins_in );
	
	/**Labels the conserved quantum number as "S".*/
	static const std::array<string,1> Stotlabel;
	///@}
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                                StateXd;
	typedef MpsQ<Symmetry,complex<double> >                      StateXcd;
	typedef DmrgSolverQ<Symmetry,HeisenbergSU2,double>           Solver;
	typedef MpsQCompressor<Symmetry,double,double>               CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double>      CompressorXcd;
	typedef MpoQ<Symmetry,double>                                MpOperator;
	///@}
	
	/**Calculates the necessary auxiliary dimension.*/
	static std::size_t calc_Daux (double Jprime=0.)
	{
		std::size_t res;
		if( Jprime==0 ) { res = 3; }
		else { res = 4; }
		return res;
	}
	
	MpoQ<Symmetry,double> SSdag (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	
private:
	
	double J=-1;
	double Jprime=0.;
	std::size_t D=2;
	spins::BaseSU2<> Spins;
};

const std::array<string,1> HeisenbergSU2::Stotlabel{"S"};

const std::vector<Sym::SU2<double>::qType> HeisenbergSU2::
getqloc ( const spins::BaseSU2<double>& Spins_in )
{	
	return Spins_in.qloc();
};

const std::vector<Index> HeisenbergSU2::
getqlocDeg ( const spins::BaseSU2<double>& Spins_in )
{	
	return Spins_in.qlocDeg();
};

const std::vector<Sym::SU2<double>::qType> HeisenbergSU2::
getqOp ()
{	
	std::vector<qType> vout(2);
	vout[0] = {1};
	vout[1] = {3};
	return vout;
};

HamiltonianTermsXd<Sym::SU2<double> > HeisenbergSU2::
set_operators (const spins::BaseSU2<> &Spins, double J, double Jprime, double Jintra, bool PERIODIC)
{
	MatrixXd JInter(Spins.orbitals(),Spins.orbitals()); JInter.setIdentity(); JInter *= J;
	return set_operators(Spins, JInter, Jprime, Jintra, PERIODIC);
}

HamiltonianTermsXd<Sym::SU2<double> > HeisenbergSU2::
set_operators (const spins::BaseSU2<> &Spins, const MatrixXd &Jinter, double Jprime, double Jintra, bool PERIODIC)
{
	assert(Spins.orbitals() == Jinter.rows() and Spins.orbitals() == Jinter.cols() );
	
	HamiltonianTermsXd<Symmetry> Terms;
	// Terms.Id = Spins.Id();
	
	for (int leg1=0; leg1<Spins.orbitals(); ++leg1)
	for (int leg2=0; leg2<Spins.orbitals(); ++leg2)
	{
		if (Jinter(leg1,leg2) != 0.)
		{
			Terms.tight.push_back(make_tuple(-std::sqrt(3)*Jinter(leg1,leg2),
											 Spins.Sdag(leg1).plain<SparseMatrixType>(),
											 Spins.S(leg2).plain<SparseMatrixType>()));
		}
	}
	
	if (Jprime != 0.)
	{
		Terms.nextn.push_back(make_tuple(-std::sqrt(3)*Jprime, Spins.S(0).plain<SparseMatrixType>(),
										 Spins.S(0).plain<SparseMatrixType>(),
										 Spins.Id().plain<SparseMatrixType>()));
	}

	if( Spins.orbitals() > 1 )
	{
		Terms.local.push_back(make_tuple(1., Spins.HeisenbergHamiltonian(Jintra,PERIODIC).plain<SparseMatrixType>()));
	}
	
	return Terms;
}

HeisenbergSU2::
HeisenbergSU2 (std::size_t Lx_input, double J_input, std::size_t D_input, double Jprime_input, std::size_t Ly_input)
	:MpoQ<Sym::SU2<double> > (Lx_input, Ly_input)
{
	//assign stuff
	D=D_input;
	J=J_input;
	Jprime=Jprime_input;
	Spins=spins::BaseSU2<>(Ly_input,D);
	for (std::size_t l=0; l<N_sites; l++)
	{
		qloc__[l] = Spins.basis();
		qloc[l] = qloc__[l].qloc();
		qOp[l] = HeisenbergSU2::getqOp();		
	}
	Qtot = {1};
	qlabel = HeisenbergSU2::Stotlabel;
	label = HeisenbergSU2::create_label(D_input,J_input);

	HamiltonianTermsXd<Symmetry> Terms = set_operators(Spins, J, Jprime, J);
	Daux = Terms.auxdim();
	auto G = ::Generator(Terms);
	this->construct(G, this->W, this->Gvec);

	this->GOT_SQUARE = false;
	this->calc_auxBasis();
}

MpoQ<Sym::SU2<double> > HeisenbergSU2::
SSdag (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for (std::size_t l=0; l<N_sites; l++)
	{
		Mout.setLocBasis(Spins.basis(),l);
	}

	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HeisenbergSU2::Stotlabel;
	if(locx1 == locx2)
	{
		auto product = std::sqrt(3.)*Operator::prod(Spins.Sdag(locy1),Spins.S(locy2),Symmetry::qvacuum());
		Mout.setLocal(locx1,product.plain<SparseMatrixType>());
		return Mout;
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(std::sqrt(3.)*Spins.Sdag(locy1)).plain<SparseMatrixType>(), Spins.S(locy2).plain<SparseMatrixType>()});
		return Mout;
	}
}

} //end namespace VMPS::models

#endif
