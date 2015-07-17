
#ifndef TEBD_PROPAGATOR
#define TEBD_PROPAGATOR

#include <cmath>

const static tuple<int,double> optTEBDstep (double tmax, double dtmax, double tol_accu=0.05)
{
	double dt3 = pow(tol_accu/tmax,0.5);
	double dt9 = pow(tol_accu/tmax,0.25);
	
	dt3 = min(dt3,dtmax);
	dt9 = min(dt9,dtmax);
//	cout << "dt3=" << dt3 << ", dt9=" << dt9 << endl;
	
	double cost3 = tmax/dt3*3;
	double cost9 = tmax/dt9*9;
//	cout << "cost3=" << cost3 << ", cost9=" << cost9 << endl;
	
	tuple<int,double> out;
	get<0>(out) = (cost9<cost3)? 9 : 3;
	get<1>(out) = (cost9<cost3)? dt9 : dt3;
	return out;
}

enum TEBD_EXCEPTION {NO_EXCEPTION, EXCEPT_FIRST, EXCEPT_LAST, DOUBLE_FIRST, DOUBLE_LAST};

template<typename Hamiltonian, size_t Nq, typename TimeScalar, typename VectorType>
class TEBDPropagator
{
public:
	
	TEBDPropagator(DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT);
	
	/**
	Performs a time evolution step \f$V_{out} \approx e^{H \cdot \delta t} V_{in}.\f$.
	\param H : Hamiltonian
	\param Vin : input MPS
	\param Vout : output MPS
	\param dt : timestep \f$\delta t\f$, \p complex<double> for real time, \p double for imaginary time
	\param tol : compression tolerance in MpsQCompressor
	\param Nexp : amount of exponentials to apply, implemented: 2,3,5,9,11 (see McLachlan, SIAM J.Sci.Comput. Vol. 16, No. 1, pp. 151-168, January 1995)
	 - \p Nexp = 2 : error \f$O(\delta t^2)\f$
	 - \p Nexp = 3 : error \f$O(\delta t^3)\f$, error constant : 0.070
	 - \p Nexp = 5 : error \f$O(\delta t^3)\f$, error constant : 0.026
	 - \p Nexp = 9 : error \f$O(\delta t^5)\f$, error constant : 0.014
	 - \p Nexp = 11: error \f$O(\delta t^5)\f$, error constant : 0.0046
	\param EX : set do not apply first or last gate
	*/
	void t_step (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, TimeScalar dt, double tol=1e-5, int Nexp=3, TEBD_EXCEPTION EX=NO_EXCEPTION);
	
private:
	
	TimeScalar tstep;
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	MpoQ<Nq,TimeScalar> GateEvn1;
	MpoQ<Nq,TimeScalar> GateEvn2;
	MpoQ<Nq,TimeScalar> GateEvn3;
	
	MpoQ<Nq,TimeScalar> GateOdd1;
	MpoQ<Nq,TimeScalar> GateOdd2;
	MpoQ<Nq,TimeScalar> GateOdd3;
	
	MpoQ<Nq,TimeScalar> GateEvn1Double;
	MpoQ<Nq,TimeScalar> GateOdd1Double;
};

template<typename Hamiltonian, size_t Nq, typename TimeScalar, typename VectorType>
TEBDPropagator<Hamiltonian,Nq,TimeScalar,VectorType>::
TEBDPropagator(DMRG::VERBOSITY::OPTION VERBOSITY)
:tstep(0.), CHOSEN_VERBOSITY(VERBOSITY)
{}

template<typename Hamiltonian, size_t Nq, typename TimeScalar, typename VectorType>
void TEBDPropagator<Hamiltonian,Nq,TimeScalar,VectorType>::
t_step (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, TimeScalar dt, double tol, int Nexp, TEBD_EXCEPTION EX)
{
	assert(Nexp==2 or Nexp==3 or Nexp==5 or Nexp==9 or Nexp==11 and "Implemented orders: 2,3,5,9,11");
	
	if (dt != tstep)
	{
		tstep = dt;
		if (Nexp == 2)
		{
			// error O(dt^2)
			GateEvn1 = H.BondPropagator(tstep,EVEN);
			GateOdd1 = H.BondPropagator(tstep,ODD);
			GateEvn1Double = H.BondPropagator(2.*tstep,EVEN);
			GateOdd1Double = H.BondPropagator(2.*tstep,ODD);
		}
		if (Nexp == 3)
		{
			// leapfrog, error O(dt^3) 0.070
			GateEvn1 = H.BondPropagator(0.5*tstep,EVEN);
			GateOdd1 = H.BondPropagator(tstep,ODD);
			GateEvn1Double = H.BondPropagator(tstep,EVEN);
		}
		else if (Nexp == 5)
		{
			// error O(dt^3) 0.026
			double xOdd1 = 0.5;
			double y = pow(2.*sqrt(326.)-36.,1./3.);
			double xEvn1 = (y*y+6.*y-2.)/(12.*y);
			double xEvn2 = 1.-2.*xEvn1;
			
			GateEvn1 = H.BondPropagator(xEvn1*tstep,EVEN);
			GateOdd1 = H.BondPropagator(xOdd1*tstep,ODD);
			GateEvn2 = H.BondPropagator(xEvn2*tstep,EVEN);
			GateEvn1Double = H.BondPropagator(2.*xEvn1*tstep,EVEN);
		}
		else if (Nexp == 9)
		{
			// error O(dt^5), 0.014
			double xEvn1 = (642.+sqrt(471.))/3924.;
			double xEvn2 = 121./3924.*(12.-sqrt(471.));
			double xEvn3 = 1.-2.*(xEvn1+xEvn2);
			double xOdd1 = 6./11.;
			double xOdd2 = -1./22.;
			
			GateEvn1 = H.BondPropagator(xEvn1*tstep,EVEN);
			GateOdd1 = H.BondPropagator(xOdd1*tstep,ODD);
			GateEvn2 = H.BondPropagator(xEvn2*tstep,EVEN);
			GateOdd2 = H.BondPropagator(xOdd2*tstep,ODD);
			GateEvn3 = H.BondPropagator(xEvn3*tstep,EVEN);
			GateEvn1Double = H.BondPropagator(2.*xEvn1*tstep,EVEN);
		}
		else if (Nexp == 11)
		{
			// error O(dt^5), 0.0046
//			double x1 = 0.25 * pow(1.-pow(cbrt(2.),-4.),-1.);
//			double x2 = 1.-4.*x1;
//			GateEvn1 = H.BondPropagator(0.5*x1*tstep,EVEN);
//			GateOdd1 = H.BondPropagator(x1*tstep,ODD);
//			GateEvn2 = H.BondPropagator(x1*tstep,EVEN);
//			GateEvn3 = H.BondPropagator((0.5*(x1+x2))*tstep,EVEN);
//			GateOdd2 = H.BondPropagator(x2*tstep,ODD);
			
			double xEvn1 = (14.-sqrt(19.))/108.;
			double xEvn2 = (20.-7.*sqrt(19.))/108.;
			double xEvn3 = 0.5-xEvn1-xEvn2;
			double xOdd1 = 0.4;
			double xOdd2 = -0.1;
			double xOdd3 = 0.4;
			
			GateEvn1 = H.BondPropagator(xEvn1*tstep,EVEN);
			GateOdd1 = H.BondPropagator(xOdd1*tstep,ODD);
			GateEvn2 = H.BondPropagator(xEvn2*tstep,EVEN);
			GateOdd2 = H.BondPropagator(xOdd2*tstep,ODD);
			GateEvn3 = H.BondPropagator(xEvn3*tstep,EVEN);
			GateOdd3 = H.BondPropagator(xOdd3*tstep,ODD);
			GateEvn1Double = H.BondPropagator(2.*xEvn1*tstep,EVEN);
		}
	}
	
	Stopwatch Chronos;
	
	MpsQCompressor<Nq,complex<double>,complex<double> > Compadre(CHOSEN_VERBOSITY);
	VectorType Vtmp1, Vtmp2;
	
	auto apply_gate = [this,&Compadre,&tol,&Nexp] (const MpoQ<Nq,TimeScalar> &Gate, const VectorType &Vin, VectorType &Vout, int i, PARITY P)
	{
		Compadre.varCompress(Gate, Vin, Vout,  Vin.calc_Dmax(), tol);
		if (CHOSEN_VERBOSITY != DMRG::VERBOSITY::SILENT)
		{
			string p = (P==EVEN)? "EVEN " : "ODD ";
			lout << p << i+1 << "/" << Nexp << ": " << Compadre.info() << endl;
		}
	};
	
	if (Nexp <= 2)
	{
		apply_gate(GateEvn1,Vin, Vtmp1, 0,EVEN);
		apply_gate(GateOdd1,Vtmp1,Vout, 1,ODD);
	}
	if (Nexp == 3)
	{
		if (EX == NO_EXCEPTION or EX == DOUBLE_FIRST or EX == DOUBLE_LAST)
		{
			if (EX == NO_EXCEPTION or EX == DOUBLE_LAST) {apply_gate(GateEvn1,Vin, Vtmp1, 0,EVEN);}
			else                                         {apply_gate(GateEvn1Double,Vin, Vtmp1, 0,EVEN);}
			apply_gate(GateOdd1,Vtmp1,Vtmp2, 1,ODD);
			if (EX == NO_EXCEPTION or EX == DOUBLE_FIRST) {apply_gate(GateEvn1,Vtmp2,Vout, 2,EVEN);}
			else                                          {apply_gate(GateEvn1Double,Vtmp2, Vout, 2,EVEN);}
		}
		else if (EX == EXCEPT_FIRST)
		{
			apply_gate(GateOdd1,Vin,Vtmp1,  1,ODD);
			apply_gate(GateEvn1,Vtmp1,Vout, 2,EVEN);
		}
		else if (EX == EXCEPT_LAST)
		{
			apply_gate(GateEvn1,Vin,  Vtmp1, 1,EVEN);
			apply_gate(GateOdd1,Vtmp1,Vout,  2,ODD);
		}
	}
	else if (Nexp == 5)
	{
		apply_gate(GateEvn1,Vin,  Vtmp1, 0,EVEN);
		apply_gate(GateOdd1,Vtmp1,Vtmp2, 1,ODD);
		apply_gate(GateEvn2,Vtmp2,Vtmp1, 2,EVEN);
		apply_gate(GateOdd1,Vtmp1,Vtmp2, 3,ODD);
		apply_gate(GateEvn1,Vtmp2,Vout,  4,EVEN);
	}
	else if (Nexp == 7)
	{
		apply_gate(GateEvn1,Vin,  Vtmp1, 0,EVEN);
		apply_gate(GateOdd1,Vtmp1,Vtmp2, 1,ODD);
		apply_gate(GateEvn2,Vtmp2,Vtmp1, 2,EVEN);
		apply_gate(GateOdd2,Vtmp1,Vtmp2, 3,ODD);
		apply_gate(GateEvn2,Vtmp2,Vtmp1, 4,EVEN);
		apply_gate(GateOdd1,Vtmp1,Vtmp2, 5,ODD);
		apply_gate(GateEvn1,Vtmp2,Vout,  6,EVEN);
	}
	else if (Nexp == 9)
	{
		if (EX == NO_EXCEPTION or EX == DOUBLE_FIRST or EX == DOUBLE_LAST)
		{
			if (EX == NO_EXCEPTION or EX == DOUBLE_LAST) {apply_gate(GateEvn1,Vin, Vtmp1, 0,EVEN);}
			else                                         {apply_gate(GateEvn1Double,Vin, Vtmp1, 0,EVEN);}
			apply_gate(GateOdd1,Vtmp1,Vtmp2, 1,ODD);
			apply_gate(GateEvn2,Vtmp2,Vtmp1, 2,EVEN);
			apply_gate(GateOdd2,Vtmp1,Vtmp2, 3,ODD);
			apply_gate(GateEvn3,Vtmp2,Vtmp1, 4,EVEN);
			apply_gate(GateOdd2,Vtmp1,Vtmp2, 5,ODD);
			apply_gate(GateEvn2,Vtmp2,Vtmp1, 6,EVEN);
			apply_gate(GateOdd1,Vtmp1,Vtmp2, 7,ODD);
			if (EX == NO_EXCEPTION or EX == DOUBLE_FIRST) {apply_gate(GateEvn1,Vtmp2,Vout, 8,EVEN);}
			else                                          {apply_gate(GateEvn1Double,Vtmp2, Vout, 8,EVEN);}
		}
		else if (EX == EXCEPT_FIRST)
		{
			apply_gate(GateOdd1,Vin,  Vtmp2, 1,ODD);
			apply_gate(GateEvn2,Vtmp2,Vtmp1, 2,EVEN);
			apply_gate(GateOdd2,Vtmp1,Vtmp2, 3,ODD);
			apply_gate(GateEvn3,Vtmp2,Vtmp1, 4,EVEN);
			apply_gate(GateOdd2,Vtmp1,Vtmp2, 5,ODD);
			apply_gate(GateEvn2,Vtmp2,Vtmp1, 6,EVEN);
			apply_gate(GateOdd1,Vtmp1,Vtmp2, 7,ODD);
			apply_gate(GateEvn1,Vtmp2,Vout,  8,EVEN);
		}
		else if (EX == EXCEPT_LAST)
		{
			apply_gate(GateEvn1,Vin,  Vtmp1, 1,EVEN);
			apply_gate(GateOdd1,Vtmp1,Vtmp2, 2,ODD);
			apply_gate(GateEvn2,Vtmp2,Vtmp1, 3,EVEN);
			apply_gate(GateOdd2,Vtmp1,Vtmp2, 4,ODD);
			apply_gate(GateEvn3,Vtmp2,Vtmp1, 5,EVEN);
			apply_gate(GateOdd2,Vtmp1,Vtmp2, 6,ODD);
			apply_gate(GateEvn2,Vtmp2,Vtmp1, 7,EVEN);
			apply_gate(GateOdd1,Vtmp1,Vout,  8,ODD);
		}

	}
	else if (Nexp == 11)
	{
//		apply_gate(GateEvn1,Vin,  Vtmp1, 0);
//		apply_gate(GateOdd1,Vtmp1,Vtmp2, 1);
//		apply_gate(GateEvn2,Vtmp2,Vtmp1, 2);
//		apply_gate(GateOdd1,Vtmp1,Vtmp2, 3);
//		apply_gate(GateEvn3,Vtmp2,Vtmp1, 4);
//		apply_gate(GateOdd2,Vtmp1,Vtmp2, 5);
//		apply_gate(GateEvn3,Vtmp2,Vtmp1, 6);
//		apply_gate(GateOdd1,Vtmp1,Vtmp2, 7);
//		apply_gate(GateEvn2,Vtmp2,Vtmp1, 8);
//		apply_gate(GateOdd1,Vtmp1,Vtmp2, 9);
//		apply_gate(GateEvn1,Vtmp2,Vout, 10);
		
		apply_gate(GateEvn1,Vin,  Vtmp1, 0,EVEN);
		apply_gate(GateOdd1,Vtmp1,Vtmp2, 1,ODD);
		apply_gate(GateEvn2,Vtmp2,Vtmp1, 2,EVEN);
		apply_gate(GateOdd2,Vtmp1,Vtmp2, 3,ODD);
		apply_gate(GateEvn3,Vtmp2,Vtmp1, 4,EVEN);
		apply_gate(GateOdd3,Vtmp1,Vtmp2, 5,ODD);
		apply_gate(GateEvn3,Vtmp2,Vtmp1, 6,EVEN);
		apply_gate(GateOdd2,Vtmp1,Vtmp2, 7,ODD);
		apply_gate(GateEvn2,Vtmp2,Vtmp1, 8,EVEN);
		apply_gate(GateOdd1,Vtmp1,Vtmp2, 9,ODD);
		apply_gate(GateEvn1,Vtmp2,Vout, 10,EVEN);
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << Chronos.info("exp(-i*H*dt)*V") << endl;
		lout << "Vout: " << Vout.info() << endl;
		lout << endl;
	}
}

#endif
