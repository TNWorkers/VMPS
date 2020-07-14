#ifndef TWO_SITE_GATE_H_
#define TWO_SITE_GATE_H_

#include "tensors/Qbasis.h"

template<typename Symmetry, typename Scalar>
class TwoSiteGate
{
public:
	TwoSiteGate() {};
	TwoSiteGate(const vector<vector<vector<vector<vector<Scalar> > > > > &data_in) : data(data_in){};
	TwoSiteGate(const Qbasis<Symmetry> &s1, const Qbasis<Symmetry> &s2);

	void print() const;

	//
	//        s1p  s2p
	//         |    |
	//  swap=  ****** = delta(s1,s2p)*delta(s2,s1p)
	//         |    |
	//        s1   s2
	//
	void setSwapGate(bool FERMIONIC=false);

	void setIdentity();
	
	Qbasis<Symmetry> midBasis() const {return Bmid;}
	Qbasis<Symmetry> leftBasis() const {return B1;}
	Qbasis<Symmetry> rightBasis() const {return B2;}
	
//private:
	//data[s1][s2][s1_p][s2_p][k]:
	//
	//   s1p  s2p          s1p  s2p
	//    |    |            \   /
	//    ^    ^             \ /
	//    |    |              ^
	//    ******              |
	//    *gate*     CGC:     k
	//    ******              ^
	//    |    |              |
	//    ^    ^             / \
	//    |    |            /   \
	//   s1   s2          s1     s2
	//
	vector<vector<vector<vector<vector<Scalar> > > > > data;
	Qbasis<Symmetry> B1, B2, Bmid;

	vector<qarray<Symmetry::Nq> > qloc1, qloc2, qmid;
	
	void resize();
};

template<typename Symmetry, typename Scalar>
TwoSiteGate<Symmetry,Scalar>::
TwoSiteGate (const Qbasis<Symmetry> &s1, const Qbasis<Symmetry> &s2)
	: B1(s1),B2(s2)
{
	//Bmid is the combined basis from B1 and B2.
	Bmid = B1.combine(B2);
	qloc1 = B1.qloc();
	qloc2 = B2.qloc();
	qmid = Bmid.qs();
	resize();
}

template<typename Symmetry, typename Scalar>
void TwoSiteGate<Symmetry,Scalar>::
resize()
{
	data.resize(qloc1.size());
	for (size_t s1=0; s1< qloc1.size(); s1++)
	{
		data[s1].resize(qloc2.size());
		for (size_t s2=0; s2<qloc2.size(); s2++)
		{
			data[s1][s2].resize(qloc1.size());
			for (size_t s1p=0; s1p<qloc1.size(); s1p++)
			{
				data[s1][s2][s1p].resize(qloc2.size());
				for (size_t s2p=0; s2p<qloc2.size(); s2p++)
				{
					data[s1][s2][s1p][s2p].resize(qmid.size());
				}
			}
		}
	}
}


template<typename Symmetry, typename Scalar>
void TwoSiteGate<Symmetry,Scalar>::
setSwapGate(bool FERMIONIC)
{
	assert(!FERMIONIC and "Fermionic sqap gates are not yet implemented.");
	for (size_t s1=0;  s1<qloc1.size();  s1++)
	for (size_t s2=0;  s2<qloc2.size();  s2++)
	for (size_t s1p=0; s1p<qloc1.size(); s1p++)
	for (size_t s2p=0; s2p<qloc2.size(); s2p++)
	for (size_t k=0;    k<qmid.size();   k++)
	{
		if (s1 == s2p and s2 == s1p)
		{
			data[s1][s2][s1p][s2p][k] = Symmetry::coeff_swapPhase(qloc1[s1],qloc2[s2])*Scalar(1.);
		}
		else {data[s1][s2][s1p][s2p][k] = Scalar(0.);}
	}
}

template<typename Symmetry, typename Scalar>
void TwoSiteGate<Symmetry,Scalar>::
setIdentity()
{
	for (size_t s1=0;  s1<qloc1.size();  s1++)
	for (size_t s2=0;  s2<qloc2.size();  s2++)
	for (size_t s1p=0; s1p<qloc1.size(); s1p++)
	for (size_t s2p=0; s2p<qloc2.size(); s2p++)
	for (size_t k=0;    k<qmid.size();   k++)
	{
		if (s1 == s1p and s2 == s2p)
		{
			data[s1][s2][s1p][s2p][k] = Scalar(1.);
		}
		else {data[s1][s2][s1p][s2p][k] = Scalar(0.);}
	}
}

template<typename Symmetry, typename Scalar>
void TwoSiteGate<Symmetry,Scalar>::
print() const
{
	for (size_t s1=0;  s1<qloc1.size();  s1++)
	for (size_t s2=0;  s2<qloc2.size();  s2++)
	for (size_t s1p=0; s1p<qloc1.size(); s1p++)
	for (size_t s2p=0; s2p<qloc2.size(); s2p++)
	for (size_t k=0;    k<qmid.size();   k++)
	{
		cout << "data[" << s1 << "][" << s2 << "][" << s1p << "][" << s2p << "][" << k << "]=" << data[s1][s2][s1p][s2p][k] << endl; 
	}
}


#endif
