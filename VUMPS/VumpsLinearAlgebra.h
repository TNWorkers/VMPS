#ifndef VUMPSLINEARALGEBRA
#define VUMPSLINEARALGEBRA

#include "tensors/Multipede.h"

/**Calculates the matrix element between two Umps and an Mpo. Goes from the left and uses \f$A_C\f$ and \f$A_R\f$.*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const Mpo<Symmetry,MpoScalar> &O, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	size_t Ncells = 1; 
	auto Obs = O;
	
	if (Obs.length() != Vket.length() and Vket.Qtarget() != Symmetry::qvacuum())
	{
		assert(Obs.length()%Vket.length() == 0); //?
		Ncells = static_cast<size_t>(Obs.length()/Vket.length());
		qarray<Symmetry::Nq> transformed_Qtot = ::adjustQN<Symmetry>(Vket.Qtarget(),Ncells);
		Obs.transform_base(transformed_Qtot,false);
	}
	else
	{
		Obs.transform_base(Vket.Qtarget(),false);
	}
	
	auto Vbra_copy = Vbra;
	auto Vket_copy = Vket;
	
	if (Obs.length() != Vket.length() and Vket.Qtarget() != Symmetry::qvacuum())
	{
		Vbra_copy.adjustQN(Ncells);
		Vket_copy.adjustQN(Ncells);
	}
	
	B.setIdentity(Obs.inBasis(0).inner_dim(Symmetry::qvacuum()), 1, Vket_copy.inBasis(0));
	for (size_t l=0; l<Obs.length(); ++l)
	{
		GAUGE::OPTION g = (l==0)? GAUGE::C : GAUGE::R;
		contract_L(B,
				   Vbra_copy.A_at(g,l%Vket.length()), Obs.W_at(l),
				   Vket_copy.A_at(g,l%Vket.length()), Obs.locBasis(l), Obs.opBasis(l), Bnext);
		
		B.clear();
		B = Bnext;
		Bnext.clear();
	}

	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > IdR;
	IdR.setIdentity(Obs.outBasis(Obs.length()-1).inner_dim(Symmetry::qvacuum()), 1, Vket_copy.outBasis((Obs.length()-1)%Vket.length()));
	
	return contract_LR(B,IdR);
}

//template<typename Symmetry, typename MpoScalar, typename Scalar>
//complex<Scalar> avg (const vector<vector<Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic>>>> &Abra,
//            const Biped<Symmetry,Matrix<complex<double>,Dynamic,Dynamic>> &L,
//            const Mpo<Symmetry,MpoScalar> &O, 
//            const vector<vector<Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic>>>> &Aket,
//            const Biped<Symmetry,Matrix<complex<double>,Dynamic,Dynamic>> &R)
//{
//	Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Bnext;
//	Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > B(L);
//	Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Blast(R);
//	size_t Ncells = 1; 
//	auto Obs = O;
//	
////	if (Obs.length() != Aket.size() and Vket.Qtarget() != Symmetry::qvacuum())
////	{
////		assert(Obs.length()%Vket.length() == 0); //?
////		Ncells = static_cast<size_t>(Obs.length()/Vket.length());
////		qarray<Symmetry::Nq> transformed_Qtot = ::adjustQN<Symmetry>(Vket.Qtarget(),Ncells);
////		Obs.transform_base(transformed_Qtot,false);
////	}
////	else
////	{
////		Obs.transform_base(Vket.Qtarget(),false);
////	}
//	
//	for (size_t l=0; l<Obs.length(); ++l)
//	{
//		contract_L(B,
//				   Abra[l%Aket.size()], Obs.W_at(l),
//				   Aket[l%Aket.size()], Obs.locBasis(l), Obs.opBasis(l), Bnext);
//		
//		B.clear();
//		B = Bnext;
//		Bnext.clear();
//	}
//	
//	return contract_LR(B,Blast);
//}

template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const Mpo<Symmetry,MpoScalar> &O1, 
            const Mpo<Symmetry,MpoScalar> &O2, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	size_t Ncells = 1; 
	auto Obs1 = O1;
	auto Obs2 = O2;
	
	if (Obs1.length() != Vket.length() and Vket.Qtarget() != Symmetry::qvacuum())
	{
		assert(Obs1.length()%Vket.length() == 0); //?
		Ncells = static_cast<size_t>(Obs1.length()/Vket.length());
		qarray<Symmetry::Nq> transformed_Qtot = ::adjustQN<Symmetry>(Vket.Qtarget(),Ncells);
		Obs1.transform_base(transformed_Qtot,false);
		Obs2.transform_base(transformed_Qtot,false);
	}
	else
	{
		Obs1.transform_base(Vket.Qtarget(),false);
		Obs2.transform_base(Vket.Qtarget(),false);
	}
	
	auto Vbra_copy = Vbra;
	auto Vket_copy = Vket;
	
	if (Obs1.length() != Vket.length() and Vket.Qtarget() != Symmetry::qvacuum())
	{
		Vbra_copy.adjustQN(Ncells);
		Vket_copy.adjustQN(Ncells);
	}

	B.setIdentity(Obs1.outBasis(Obs1.length()-1).inner_dim(Symmetry::qvacuum()), 1, Vket_copy.outBasis((Obs1.length()-1)%Vket.length()));

	for (size_t l=O1.length()-1; l!=-1; --l)
	{
//		GAUGE::OPTION g = (l==O1.length()-1)? GAUGE::C : GAUGE::L;
		GAUGE::OPTION g = (l==0)? GAUGE::C : GAUGE::R;

		contract_R(B,
		           Vbra_copy.A_at(g,l%Vket.length()), Obs1.W_at(l), Obs2.W_at(l), Vket_copy.A_at(g,l%Vket.length()), 
		           Obs1.locBasis(l), Obs1.opBasis(l), Obs2.opBasis(l), 
		           Bnext);
		
		B.clear();
		B = Bnext;
		Bnext.clear();
//		cout << "l=" << l << ", B.dim=" << B.dim << endl;
//		if (l==O1.length()-1)
//		{
//			cout << B.print() << endl;
//		}
	}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > IdL;
	IdL.setIdentity(Obs1.inBasis(0).inner_dim(Symmetry::qvacuum()), 1, Vket_copy.inBasis(0));
	
//	cout << "res=" << contract_LR(IdL,B) << ", B.dim=" << B.dim << endl;
	
	return contract_LR(IdL,B);
}

/**Calculates the matrix element for a vector of MPOs, summing up the result.*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const vector<Mpo<Symmetry,MpoScalar> > &O, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Scalar out = 0;
	
	for (int t=0; t<O.size(); ++t)
	{
//		cout << "partial val=" << avg(Vbra,O[t],Vket) << endl;
		out += avg(Vbra,O[t],Vket);
	}
	return out;
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const vector<Mpo<Symmetry,MpoScalar> > &O1, 
            const vector<Mpo<Symmetry,MpoScalar> > &O2, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Scalar out = 0;
	
	for (int i=0; i<O1.size(); ++i)
	for (int j=0; j<O2.size(); ++j)
	{
		cout << "partial val=" << avg(Vbra, O1[i], O2[j], Vket) << endl;
		out += avg(Vbra, O1[i], O2[j], Vket);
	}
	return out;
}

#endif
