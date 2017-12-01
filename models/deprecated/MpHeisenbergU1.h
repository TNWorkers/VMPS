#ifndef STRAWBERRY_HEISENBERGMODELU1
#define STRAWBERRY_HEISENBERGMODELU1

#include "symmetry/U1.h"
#include "spins/BaseU1.h"
#include "MpoQ.h"
#include "DmrgExternalQ.h"

namespace VMPS::models
{
/** \class HeisenbergU1
  * \ingroup Models
  *
  * \brief Heisenberg Model
  *
  * MPO representation of 
  \f$
  H = -J \sum_{<ij>} \left(\mathbf{S_i}\mathbf{S_j}\right) -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right) -B_z \sum_i S_i^z
  \f$.
  *
  \param D = 2S+1: spin quantum number.
  \note Take use of the Spin-Sz U(1) symmetry.
  \note \f$J<0\f$ is antiferromagnetic
  */
class HeisenbergU1 : public MpoQ<Sym::U1<double>,double>
{
public:
	typedef Sym::U1<double> Symmetry;
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	template<Index Rank> using TensorType = Eigen::Tensor<double,Rank,Eigen::ColMajor,Index>;
	typedef SiteOperator<Symmetry,double> Operator;
public:
	/** Does nothing. */
	HeisenbergU1() : MpoQ<Symmetry,double>() {};

	/**
	\param Lx_input : chain length
	\param Jz_input : \f$J_z\f$ nn exchange coupling constant (Z)
	\param Jxy_input : \f$J_{xy}\f$ nn exchange coupling constant (XY)
	\param D_input : \f$2S+1\f$ Magnitude of the Spin operators
	\param Bz_input : \f$B_z\f$ Magnitude of the magnetic field in z-direction
	\param Jprime_input : \f$J'\f$ nnn exchange coupling constant
	\param Ly_input : amount of legs in ladder
	*/
	HeisenbergU1(std::size_t Lx_input, double Jz_input=-1., double Jxy_input=std::numeric_limits<double>::infinity(), std::size_t D_input=2,
				 double Bz_input = 0., double Jprime_input=0., std::size_t Ly_input=1);
		
	/**Creates the MPO generator matrix for the Heisenberg model (of any spin (\f$D=2S+1\f$)).*/
	static HamiltonianTermsXd<Symmetry> set_operators (const spins::BaseU1<> &Spins, double Jz, double Jxy, double Bz_input = 0.,
													   double Jprime=0., double Jintra=0., bool PERIODIC=false);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const spins::BaseU1<> &Spins, const Eigen::MatrixXd &Jinter, const Eigen::MatrixXd &Jinter_xy,
													   double Bz_input = 0., double Jprime=0., double Jintra=0., bool PERIODIC=false);

	//---label stuff---
	///@{
	/**Creates a label for this MpoQ to have a nice output.
	\param D : \f$2S+1\f$
	\param J : \f$J\f$
	\param Bz : \f$B_z\f$
	\param Jprime : \f$J'\f$*/
	static string create_label (std::size_t D,  double Jz, double Jxy, double Bz=0., double Jprime=0.)
	{
		auto S = frac(D-1,2);
		std::stringstream ss;
		ss << "Heisenberg(U(1),S=" << S << ",J=" << Jz;
		if (Bz != 0.) {ss << ",Bz=" << Bz;}
		if (Jprime != 0.) {ss << ",J'=" << Jprime;}
		ss << ")";
		return ss.str();
	}
	
	/**local basis: \f$\{ \left|\frac{1}{2}\right> \}\f$ if D=2 and N_legs=1*/
	static const std::vector<qType> getqloc ( const spins::BaseU1<double>& Spins_in );
	static const std::vector<qType> getqOp ( double Jxy );
	
	/**Labels the conserved quantum number as "M".*/
	static const std::array<string,1> Mtotlabel;
	///@}
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                                StateXd;
	typedef MpsQ<Symmetry,complex<double> >                      StateXcd;
	typedef DmrgSolverQ<Symmetry,HeisenbergU1,double,false>      Solver;
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
	
	MpoQ<Symmetry,double> SzSz (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	
private:
	
	double Jz=-1;
	double Jxy=std::numeric_limits<double>::infinity();
	double Bz=0.;
	double Jprime=0.;
	std::size_t D=2;
	spins::BaseU1<> Spins;
};

const std::array<string,1> HeisenbergU1::Mtotlabel{"M"};

const std::vector<Sym::U1<double>::qType> HeisenbergU1::
getqloc ( const spins::BaseU1<double>& Spins_in )
{	
	return Spins_in.qloc();
};

const std::vector<Sym::U1<double>::qType> HeisenbergU1::
getqOp ( double Jxy )
{	
	std::vector<qType> vout;
	vout.push_back({0}); //Sz and Identity
	if( Jxy != 0. )
	{
		vout.push_back({2}); //Splus
		vout.push_back({-2}); //Sminus
	}
	return vout;
};

HamiltonianTermsXd<Sym::U1<double> > HeisenbergU1::
set_operators (const spins::BaseU1<> &Spins, double Jz, double Jxy, double Bz, double Jprime, double Jintra, bool PERIODIC)
{
	MatrixXd JInter(Spins.orbitals(),Spins.orbitals()); JInter.setIdentity(); JInter *= Jz;
	MatrixXd JInter_xy(Spins.orbitals(),Spins.orbitals()); JInter_xy.setIdentity(); JInter_xy *= Jxy;
	return set_operators(Spins, JInter, JInter_xy, Bz, Jprime, Jintra, PERIODIC);
}

HamiltonianTermsXd<Sym::U1<double> > HeisenbergU1::
set_operators (const spins::BaseU1<> &Spins, const MatrixXd &Jinter, const MatrixXd &Jinter_xy, double Bz, double Jprime, double Jintra, bool PERIODIC)
{
	assert(Spins.orbitals() == Jinter.rows() and Spins.orbitals() == Jinter.cols() );
	assert(Spins.orbitals() == Jinter_xy.rows() and Spins.orbitals() == Jinter_xy.cols() );
	
	HamiltonianTermsXd<Symmetry> Terms;
	Terms.Id = Spins.Id();
	
	for (int leg1=0; leg1<Spins.orbitals(); ++leg1)
	for (int leg2=0; leg2<Spins.orbitals(); ++leg2)
	{
		if (Jinter(leg1,leg2) != 0.)
		{
			Terms.tight.push_back(make_tuple(-Jinter(leg1,leg2), Spins.Sz(leg1), Spins.Sz(leg2)));
		}
		if (Jinter_xy(leg1,leg2) != 0. )
		{
			Terms.tight.push_back(make_tuple(-0.5*Jinter_xy(leg1,leg2), Spins.Splus(leg1), Spins.Sminus(leg2)));
			Terms.tight.push_back(make_tuple(-0.5*Jinter_xy(leg1,leg2), Spins.Sminus(leg1), Spins.Splus(leg2)));
		}
	}
	
	if (Jprime != 0.)
	{
		Terms.nextn.push_back(make_tuple(-Jprime, Spins.Sz(0), Spins.Sz(0), Spins.Id()));
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, Spins.Splus(0), Spins.Sminus(0), Spins.Id()));
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, Spins.Sminus(0), Spins.Splus(0), Spins.Id()));
	}

	if( Spins.orbitals() >= 1 and Bz != 0. )
	{
		Terms.local.push_back(make_tuple(1., Spins.HeisenbergHamiltonian(Jintra,Bz,PERIODIC)));
	}
	
	return Terms;
}

HeisenbergU1::
HeisenbergU1 (std::size_t Lx_input, double Jz_input, double Jxy_input, std::size_t D_input, double Bz_input, double Jprime_input, std::size_t Ly_input)
	:MpoQ<Sym::U1<double> > (Lx_input, Ly_input)
{
	//assign stuff
	D=D_input;
	Jz=Jz_input;
	Jxy=Jxy_input;
	if(Jxy == std::numeric_limits<double>::infinity()) {Jxy = Jz;}
	Bz=Bz_input;
	Jprime=Jprime_input;
	Spins=spins::BaseU1<>(Ly_input,D);
	for (std::size_t l=0; l<N_sites; l++)
	{
		qloc__[l] = Spins.basis();
		qloc[l] = qloc__[l].qloc();
		qOp[l] = HeisenbergU1::getqOp(Jxy);
	}
	Qtot = {0};
	qlabel = HeisenbergU1::Mtotlabel;
	label = HeisenbergU1::create_label(D,Jz,Jxy);
	
	HamiltonianTermsXd<Symmetry> Terms = set_operators(Spins, Jz, Jxy, Bz, Jprime, Jz);
	Daux = Terms.auxdim();
	auto G = ::Generator(Terms);
	this->construct(G);
}

MpoQ<Sym::U1<double> > HeisenbergU1::
SzSz (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for (std::size_t l=0; l<N_sites; l++)
	{
		Mout.setLocBasis(HeisenbergU1::getqloc(Spins),l);
	}

	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HeisenbergU1::Mtotlabel;
	if(locx1 == locx2)
	{
		auto product = Operator::prod(Spins.Sz(locy1),Spins.Sz(locy2),Symmetry::qvacuum());
		Mout.setLocal(locx1,product,Symmetry::qvacuum());
		return Mout;
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {Spins.Sz(locy1), Spins.Sz(locy2)}, {{0},{0}});
		return Mout;
	}
}

} //end namespace VMPS::models

#endif
