#ifndef PARAMCOLLECTION
#define PARAMCOLLECTION

#include <set>

#include "ParamHandler.h"
#include "CuthillMcKeeCompressor.h" // from ALGS
#include <boost/rational.hpp>

ArrayXXd create_1D_OBC (size_t L, double lambda1=1., double lambda2=0.)
{
	ArrayXXd res(L,L); res.setZero();
	
	res.matrix().diagonal<1>().setConstant(lambda1);
	res.matrix().diagonal<-1>().setConstant(lambda1);
	
	res.matrix().diagonal<2>().setConstant(lambda2);
	res.matrix().diagonal<-2>().setConstant(lambda2);
	
	return res;
}

// Simple generation of periodic boundary conditions with NN and NNN coupling
// If COMPRESSED=true, the ring is flattened so that adjacent sites are as close together as possible
ArrayXXd create_1D_PBC (size_t L, double lambda1=1., double lambda2=0., bool COMPRESSED=false)
{
	ArrayXXd res(L,L);
	res.setZero();
	
	res = create_1D_OBC(L,lambda1,lambda2);
	
	res(0,L-1) = lambda1;
	res(L-1,0) = lambda1;
	
	res(0,L-2) = lambda2;
	res(L-2,0) = lambda2;
	res(1,L-1) = lambda2;
	res(L-1,1) = lambda2;
	
	if (COMPRESSED and lambda2 == 0.)
	{
		res.setZero();
		
		res(0,1) = lambda1; res(1,0) = lambda1;
		res(L-2,L-1) = lambda1; res(L-1,L-2) = lambda1;
		for (size_t l=0; l<L-2; l++)
		{
			res(l,l+2) = lambda1;
			res(l+2,l) = lambda1;
		}
	}
	else if (COMPRESSED and lambda2 != 0.)
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	return res;
}

ArrayXXd hopping_Archimedean (string vertex_conf, int VARIANT=0, double lambda1=1., double lambda2=1.)
{
	ArrayXXd res;
	
	if (vertex_conf == "3.5.3.5") // icosidodecahedron
	{
		int L=30;
		res.resize(L,L); res.setZero();
		
		if (VARIANT==1 or VARIANT==0) // my naive counting
		{
			res(0,1) = lambda1;
			res(1,2) = lambda1;
			res(2,3) = lambda1;
			res(3,4) = lambda1;
			res(0,4) = lambda1;
			
			res(0,5) = lambda1;
			res(0,6) = lambda1;
			res(1,6) = lambda1;
			res(1,7) = lambda1;
			res(2,7) = lambda1;
			res(2,8) = lambda1;
			res(3,8) = lambda1;
			res(3,9) = lambda1;
			res(4,5) = lambda1;
			res(4,9) = lambda1;
			
			res(5,11) = lambda1;
			res(5,12) = lambda1;
			res(6,13) = lambda1;
			res(6,14) = lambda1;
			res(7,15) = lambda1;
			res(7,16) = lambda1;
			res(8,17) = lambda1;
			res(8,18) = lambda1;
			res(9,10) = lambda1;
			res(9,19) = lambda1;
			
			res(12,13) = lambda1;
			res(13,14) = lambda1;
			res(14,15) = lambda1;
			res(15,16) = lambda1;
			res(16,17) = lambda1;
			res(17,18) = lambda1;
			res(18,19) = lambda1;
			res(10,19) = lambda1;
			res(10,11) = lambda1;
			res(11,12) = lambda1;
			
			res(12,22) = lambda1;
			res(13,22) = lambda1;
			res(14,23) = lambda1;
			res(15,23) = lambda1;
			res(16,24) = lambda1;
			res(17,24) = lambda1;
			res(18,20) = lambda1;
			res(19,20) = lambda1;
			res(10,21) = lambda1;
			res(11,21) = lambda1;
			
			res(20,28) = lambda1;
			res(20,29) = lambda1;
			res(21,27) = lambda1;
			res(21,28) = lambda1;
			res(22,26) = lambda1;
			res(22,27) = lambda1;
			res(23,25) = lambda1;
			res(23,26) = lambda1;
			res(24,25) = lambda1;
			res(24,29) = lambda1;
			
			res(25,26) = lambda1;
			res(26,27) = lambda1;
			res(27,28) = lambda1;
			res(28,29) = lambda1;
			res(25,29) = lambda1;
		}
		else if (VARIANT==2) // According to Exler, Schnack (2003)
		{
			res(1-1,3-1) = lambda1;
			res(1-1,4-1) = lambda1;
			res(1-1,28-1) = lambda1;
			res(1-1,29-1) = lambda1;
			
			res(2-1,5-1) = lambda1;
			res(2-1,8-1) = lambda1;
			res(2-1,26-1) = lambda1;
			res(2-1,29-1) = lambda1;
			
			res(3-1,4-1) = lambda1;
			res(3-1,6-1) = lambda1;
			res(3-1,30-1) = lambda1;
			
			res(4-1,5-1) = lambda1;
			res(4-1,7-1) = lambda1;
			
			res(5-1,7-1) = lambda1;
			res(5-1,8-1) = lambda1;
			
			res(6-1,30-1) = lambda1;
			res(6-1,9-1) = lambda1;
			res(6-1,12-1) = lambda1;
			
			res(7-1,9-1) = lambda1;
			res(7-1,10-1) = lambda1;
			
			res(8-1,11-1) = lambda1;
			res(8-1,14-1) = lambda1;
			
			res(9-1,10-1) = lambda1;
			res(9-1,12-1) = lambda1;
			
			res(10-1,11-1) = lambda1;
			res(10-1,13-1) = lambda1;
			
			res(11-1,13-1) = lambda1;
			res(11-1,14-1) = lambda1;
			
			res(12-1,15-1) = lambda1;
			res(12-1,18-1) = lambda1;
			
			res(13-1,15-1) = lambda1;
			res(13-1,16-1) = lambda1;
			
			res(14-1,17-1) = lambda1;
			res(14-1,20-1) = lambda1;
			
			res(15-1,16-1) = lambda1;
			res(15-1,18-1) = lambda1;
			
			res(16-1,17-1) = lambda1;
			res(16-1,19-1) = lambda1;
			
			res(17-1,19-1) = lambda1;
			res(17-1,20-1) = lambda1;
			
			res(18-1,21-1) = lambda1;
			res(18-1,24-1) = lambda1;
			
			res(19-1,21-1) = lambda1;
			res(19-1,22-1) = lambda1;
			
			res(20-1,23-1) = lambda1;
			res(20-1,26-1) = lambda1;
			
			res(21-1,22-1) = lambda1;
			res(21-1,24-1) = lambda1;
			
			res(22-1,23-1) = lambda1;
			res(22-1,25-1) = lambda1;
			
			res(23-1,25-1) = lambda1;
			res(23-1,26-1) = lambda1;
			
			res(24-1,27-1) = lambda1;
			res(24-1,30-1) = lambda1;
			
			res(25-1,27-1) = lambda1;
			res(25-1,28-1) = lambda1;
			
			res(26-1,29-1) = lambda1;
			
			res(27-1,28-1) = lambda1;
			res(27-1,30-1) = lambda1;
			
			res(28-1,29-1) = lambda1;
		}
	}
	else if (vertex_conf == "3^4.5") // snub dodecahedron
	{
		int L=60;
		res.resize(L,L); res.setZero();
		
		for (int i=0; i<=58; ++i) res(i,i+1) = lambda1;
		
		res( 0, 4) = lambda1;
		res( 0, 6) = lambda1;
		res( 0, 7) = lambda1;
		res( 0, 8) = lambda1;
		
		res( 1, 8) = lambda1;
		res( 1, 9) = lambda1;
		res( 1,10) = lambda1;
		
		res( 2,10) = lambda1;
		res( 2,11) = lambda1;
		res( 2,12) = lambda1;
		
		res( 3,12) = lambda1;
		res( 3,13) = lambda1;
		res( 3,14) = lambda1;
		
		res( 4, 6) = lambda1;
		res( 4,14) = lambda1;
		
		res( 5,14) = lambda1;
		res( 5,15) = lambda1;
		res( 5,16) = lambda1;
		
		res( 6,18) = lambda1;
		
		res( 7,18) = lambda1;
		res( 7,19) = lambda1;
		
		res( 8,21) = lambda1;
		
		res( 9,21) = lambda1;
		res( 9,22) = lambda1;
		
		res(10,24) = lambda1;
		
		res(11,24) = lambda1;
		res(11,25) = lambda1;
		
		res(12,27) = lambda1;
		
		res(13,27) = lambda1;
		res(13,28) = lambda1;
		
		res(15,29) = lambda1;
		res(15,30) = lambda1;
		
		res(16,30) = lambda1;
		res(16,31) = lambda1;
		
		res(17,31) = lambda1;
		res(17,32) = lambda1;
		res(17,33) = lambda1;
		
		res(18,33) = lambda1;
		
		res(19,33) = lambda1;
		res(19,34) = lambda1;
		
		res(20,34) = lambda1;
		res(20,35) = lambda1;
		res(20,36) = lambda1;
		
		res(21,36) = lambda1;
		
		res(22,36) = lambda1;
		res(22,37) = lambda1;
		
		res(23,37) = lambda1;
		res(23,38) = lambda1;
		res(23,39) = lambda1;
		
		res(24,39) = lambda1;
		
		res(25,39) = lambda1;
		res(25,40) = lambda1;
		
		res(26,40) = lambda1;
		res(26,41) = lambda1;
		res(26,42) = lambda1;
		
		res(27,42) = lambda1;
		
		res(28,42) = lambda1;
		res(28,43) = lambda1;
		
		res(29,43) = lambda1;
		res(29,44) = lambda1;
		
		res(30,44) = lambda1;
		
		res(31,46) = lambda1;
		
		res(32,46) = lambda1;
		res(32,47) = lambda1;
		
		res(34,48) = lambda1;
		
		res(35,48) = lambda1;
		res(35,49) = lambda1;
		
		res(37,50) = lambda1;
		
		res(38,50) = lambda1;
		res(38,51) = lambda1;
		
		res(40,52) = lambda1;
		
		res(41,52) = lambda1;
		res(41,53) = lambda1;
		
		res(43,54) = lambda1;
		
		res(44,54) = lambda1;
		
		res(45,54) = lambda1;
		res(45,55) = lambda1;
		res(45,56) = lambda1;
		
		res(46,56) = lambda1;
		
		res(47,56) = lambda1;
		res(47,57) = lambda1;
		
		res(48,57) = lambda1;
		
		res(49,57) = lambda1;
		res(49,58) = lambda1;
		
		res(50,58) = lambda1;
		
		res(51,58) = lambda1;
		res(51,59) = lambda1;
		
		res(52,59) = lambda1;
		
		res(53,55) = lambda1;
		res(53,59) = lambda1;
		
		res(55,59) = lambda1;
	}
	else if (vertex_conf == "4.6^2") // truncated octahedron
	{
		int L=24;
		res.resize(L,L); res.setZero();
		
		for (int i=0; i<=4; ++i) res(i,i+1) = lambda1;
		res(0,5) = lambda1;
		
		res(5,6) = lambda1;
		res(0,9) = lambda1;
		res(1,10) = lambda1;
		res(2,13) = lambda1;
		res(3,14) = lambda1;
		res(4,17) = lambda1;
		
		for (int i=6; i<=16; ++i) res(i,i+1) = lambda1;
		res(6,17) = lambda1;
		
		res(7,19) = lambda1;
		res(8,20) = lambda1;
		res(11,21) = lambda1;
		res(12,22) = lambda1;
		res(15,23) = lambda1;
		res(16,18) = lambda1;
		
		for (int i=18; i<=22; ++i) res(i,i+1) = lambda1;
		res(18,23) = lambda1;
	}
	else if (vertex_conf == "3.4.3.4") // cuboctahedron
	{
		int L=12;
		res.resize(L,L); res.setZero();
		
		for (int i=0; i<=2; ++i) res(i,i+1) = lambda1;
		res(0,3) = lambda1;
		
		res(0,4) = lambda1;
		res(3,4) = lambda1;
		res(0,5) = lambda1;
		res(1,5) = lambda1;
		res(1,6) = lambda1;
		res(2,6) = lambda1;
		res(2,7) = lambda1;
		res(3,7) = lambda1;
		
		res(4,8) = lambda1;
		res(7,8) = lambda1;
		res(4,9) = lambda1;
		res(5,9) = lambda1;
		res(5,10) = lambda1;
		res(6,10) = lambda1;
		res(6,11) = lambda1;
		res(7,11) = lambda1;
		
		for (int i=8; i<=10; ++i) res(i,i+1) = lambda1;
		res(8,11) = lambda1;
	}
	else if (vertex_conf == "3.6^2") // truncated tetrahedron = C12
	{
		int L=12;
		res.resize(L,L); res.setZero();
		
		res(0,1) = lambda1;
		res(1,4) = lambda2;
		res(4,8) = lambda1;
		res(5,8) = lambda2;
		res(2,5) = lambda1;
		res(0,2) = lambda2;
		
		res(0,3) = lambda1;
		res(1,3) = lambda1;
		
		res(4,9) = lambda1;
		res(8,9) = lambda1;
		
		res(5,6) = lambda1;
		res(2,6) = lambda1;
		
		res(3,7) = lambda2;
		res(6,10) = lambda2;
		res(9,11) = lambda2;
		
		res(7,10) = lambda1;
		res(10,11) = lambda1;
		res(7,11) = lambda1;
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0 and vertex_conf != "3.6^2" and vertex_conf != "3.4.3.4")
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	
	return res;
}

// reference: PRB 93, 165406 (2016), Appendix C
ArrayXXd hopping_fullerene (int L=60, int VARIANT=0, double lambda1=1., double lambda2=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	// lambda1: pentagon bond, lambda2: hexagon bond
	if (L== 60)
	{
		if (VARIANT==1) // inwards spiral
		{
			for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
			res(0,4) = lambda1;
			
			res(0,8) = lambda2; //1
			res(1,11) = lambda2; //2
			res(2,14) = lambda2; //3
			res(3,17) = lambda2; //4
			res(4,5) = lambda2; //5
			
			for (int i=5; i<=18; ++i) res(i,i+1) = lambda1;
				res(6,7) = lambda2; //6
				res(9,10) = lambda2; //7
				res(12,13) = lambda2; //8
				res(15,16) = lambda2; //9
				res(18,19) = lambda2; //10
			res(5,19) = lambda1;
			
			res(7,24) = lambda1;
			res(9,25) = lambda1;
			res(10,28) = lambda1;
			res(12,29) = lambda1;
			res(13,32) = lambda1;
			res(15,33) = lambda1;
			res(16,36) = lambda1;
			res(18,37) = lambda1;
			res(19,20) = lambda1;
			res(6,21) = lambda1;
			
			for (int i=20; i<=38; ++i) res(i,i+1) = lambda1;
				res(21,22) = lambda2; //11
				res(23,24) = lambda2; //12
				res(25,26) = lambda2; //13
				res(27,28) = lambda2; //14
				res(29,30) = lambda2; //15
				res(31,32) = lambda2; //16
				res(33,34) = lambda2; //17
				res(35,36) = lambda2; //18
				res(37,38) = lambda2; //19
			res(20,39) = lambda2; //20
			
			res(26,44) = lambda1;
			res(27,46) = lambda1;
			res(30,47) = lambda1;
			res(31,49) = lambda1;
			res(34,50) = lambda1;
			res(35,52) = lambda1;
			res(38,53) = lambda1;
			res(39,40) = lambda1;
			res(22,41) = lambda1;
			res(23,43) = lambda1;
			
			for (int i=40; i<=54; ++i) res(i,i+1) = lambda1;
				res(40,41) = lambda2; //21
				res(43,44) = lambda2; //22
				res(46,47) = lambda2; //23
				res(49,50) = lambda2; //24
				res(52,53) = lambda2; //25
			res(40,54) = lambda1;
			
			res(45,57) = lambda2; //26
			res(48,58) = lambda2; //27
			res(51,59) = lambda2; //28
			res(54,55) = lambda2; //29
			res(42,56) = lambda2; //30
			
			for (int i=55; i<=58; ++i) res(i,i+1) = lambda1;
			res(55,59) = lambda1;
		}
		// reference: https://www.qmul.ac.uk/sbcs/iupac/fullerene2/311.html
		// also in: Phys. Rev. B 72, 064453 (2005)
		else if (VARIANT==2 or VARIANT==0) // Kivelson
		{
			res(0,4) = lambda1;
			res(3,4) = lambda1;
			res(2,3) = lambda1;
			res(1,2) = lambda1;
			res(0,1) = lambda1;
			
			res(0,8) = lambda1;
			res(4,5) = lambda1;
			res(3,17) = lambda1;
			res(2,14) = lambda1;
			res(1,11) = lambda1;
			
			res(7,8) = lambda1;
			res(6,7) = lambda1;
			res(5,6) = lambda1;
			res(5,19) = lambda1;
			res(18,19) = lambda1;
			res(17,18) = lambda1;
			res(16,17) = lambda1;
			res(15,16) = lambda1;
			res(14,15) = lambda1;
			res(13,14) = lambda1;
			res(12,13) = lambda1;
			res(11,12) = lambda1;
			res(10,11) = lambda1;
			res(9,10) = lambda1;
			res(8,9) = lambda1;
			
			res(7,22) = lambda1;
			res(6,21) = lambda1;
			res(19,20) = lambda1;
			res(18,29) = lambda1;
			res(16,28) = lambda1;
			res(15,27) = lambda1;
			res(13,26) = lambda1;
			res(12,25) = lambda1;
			res(10,24) = lambda1;
			res(9,23) = lambda1;
			
			res(22,33) = lambda1;
			res(32,33) = lambda1;
			res(21,32) = lambda1;
			res(20,21) = lambda1;
			res(20,31) = lambda1;
			res(30,31) = lambda1;
			res(29,30) = lambda1;
			res(28,29) = lambda1;
			res(28,39) = lambda1;
			res(38,39) = lambda1;
			res(27,38) = lambda1;
			res(26,27) = lambda1;
			res(26,37) = lambda1;
			res(36,37) = lambda1;
			res(25,36) = lambda1;
			res(24,25) = lambda1;
			res(24,35) = lambda1;
			res(34,35) = lambda1;
			res(23,34) = lambda1;
			res(22,23) = lambda1;
			
			res(33,46) = lambda1;
			res(32,44) = lambda1;
			res(31,43) = lambda1;
			res(30,41) = lambda1;
			res(39,40) = lambda1;
			res(38,53) = lambda1;
			res(37,52) = lambda1;
			res(36,50) = lambda1;
			res(35,49) = lambda1;
			res(34,47) = lambda1;
			
			res(45,46) = lambda1;
			res(44,45) = lambda1;
			res(43,44) = lambda1;
			res(42,43) = lambda1;
			res(41,42) = lambda1;
			res(40,41) = lambda1;
			res(40,54) = lambda1;
			res(53,54) = lambda1;
			res(52,53) = lambda1;
			res(51,52) = lambda1;
			res(50,51) = lambda1;
			res(49,50) = lambda1;
			res(48,49) = lambda1;
			res(47,48) = lambda1;
			res(46,47) = lambda1;
			
			res(45,57) = lambda1;
			res(42,56) = lambda1;
			res(54,55) = lambda1;
			res(51,59) = lambda1;
			res(48,58) = lambda1;
			res(56,57) = lambda1;
			res(55,56) = lambda1;
			res(55,59) = lambda1;
			res(58,59) = lambda1;
			res(57,58) = lambda1;
		}
	}
	else if (L == 20)
	{
//		res(11,12) = lambda1;
//		res(12,13) = lambda1;
//		res(13,14) = lambda1;
//		res(5,14) = lambda1;
//		res(5,6) = lambda1;
//		res(6,7) = lambda1;
//		res(7,8) = lambda1;
//		res(8,9) = lambda1;
//		res(9,10) = lambda1;
//		res(10,11) = lambda1;
//		
//		res(0,1) = lambda1;
//		res(1,2) = lambda1;
//		res(2,3) = lambda1;
//		res(3,4) = lambda1;
//		res(0,4) = lambda1;
//		
//		res(2,11) = lambda1;
//		res(3,13) = lambda1;
//		res(4,5) = lambda1;
//		res(0,7) = lambda1;
//		res(1,9) = lambda1;
//		res(2,11) = lambda1;
//		
//		res(15,16) = lambda1;
//		res(16,17) = lambda1;
//		res(17,18) = lambda1;
//		res(18,19) = lambda1;
//		res(15,19) = lambda1;
//		
//		res(12,19) = lambda1;
//		res(14,15) = lambda1;
//		res(6,16) = lambda1;
//		res(8,17) = lambda1;
//		res(10,18) = lambda1;
		
		// better numbering (inwards spiral):
		
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(3,4) = lambda1;
		res(0,4) = lambda1;
		
		res(0,7) = lambda1;
		res(1,9) = lambda1;
		res(2,11) = lambda1;
		res(3,13) = lambda1;
		res(4,5) = lambda1;
		
		res(5,6) = lambda1;
		res(6,7) = lambda1;
		res(7,8) = lambda1;
		res(8,9) = lambda1;
		res(9,10) = lambda1;
		res(10,11) = lambda1;
		res(11,12) = lambda1;
		res(12,13) = lambda1;
		res(13,14) = lambda1;
		res(5,14) = lambda1;
		
		res(6,16) = lambda1;
		res(8,17) = lambda1;
		res(10,18) = lambda1;
		res(12,19) = lambda1;
		res(14,15) = lambda1;
		
		res(15,16) = lambda1;
		res(16,17) = lambda1;
		res(17,18) = lambda1;
		res(18,19) = lambda1;
		res(15,19) = lambda1;
	}
	else if (L==40)
	{
		for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
		res(0,4) = lambda1;
		
		res(0,8) = lambda1;
		res(1,11) = lambda1;
		res(2,14) = lambda1;
		res(3,17) = lambda1;
		res(4,5) = lambda1;
		
		for (int i=5; i<=18; ++i) res(i,i+1) = lambda1;
		res(5,19) = lambda1;
		
		res(6,21) = lambda1;
		res(7,23) = lambda1;
		res(9,24) = lambda1;
		res(10,26) = lambda1;
		res(12,27) = lambda1;
		res(13,29) = lambda1;
		res(15,30) = lambda1;
		res(16,32) = lambda1;
		res(18,33) = lambda1;
		res(19,20) = lambda1;
		
		for (int i=20; i<=33; ++i) res(i,i+1) = lambda1;
		res(20,34) = lambda1;
		
		res(22,36) = lambda1;
		res(25,37) = lambda1;
		res(28,38) = lambda1;
		res(31,39) = lambda1;
		res(34,35) = lambda1;
		
		for (int i=35; i<=38; ++i) res(i,i+1) = lambda1;
		res(35,39) = lambda1;
	}
	else if (L==30)
	{
		for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
		res(0,4) = lambda1;
		
		res(0,7) = lambda1;
		res(1,9) = lambda1;
		res(2,11) = lambda1;
		res(3,13) = lambda1;
		res(4,5) = lambda1;
		
		for (int i=5; i<=13; ++i) res(i,i+1) = lambda1;
		res(5,14) = lambda1;
		
		res(6,17) = lambda1;
		res(8,19) = lambda1;
		res(10,21) = lambda1;
		res(12,23) = lambda1;
		res(14,15) = lambda1;
		
		for (int i=15; i<=23; ++i) res(i,i+1) = lambda1;
		res(15,24) = lambda1;
		
		res(16,26) = lambda1;
		res(18,27) = lambda1;
		res(20,28) = lambda1;
		res(22,29) = lambda1;
		res(24,25) = lambda1;
		
		for (int i=25; i<=28; ++i) res(i,i+1) = lambda1;
		res(25,29) = lambda1;
	}
	else if (L==26)
	{
		for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
		res(0,4) = lambda1;
		
		res(0,7) = lambda1;
		res(1,10) = lambda1;
		res(2,12) = lambda1;
		res(3,14) = lambda1;
		res(4,5) = lambda1;
		
		for (int i=5; i<=14; ++i) res(i,i+1) = lambda1;
		res(5,15) = lambda1;
		
		res(6,18) = lambda1;
		res(8,19) = lambda1;
		res(9,21) = lambda1;
		res(11,22) = lambda1;
		res(13,25) = lambda1;
		res(15,16) = lambda1;
		
		for (int i=16; i<=23; ++i) res(i,i+1) = lambda1;
		
		res(16,25) = lambda1;
		res(17,24) = lambda1;
		res(20,24) = lambda1;
		res(23,25) = lambda1;
	}
	else if (L==24)
	{
		for (int i=0; i<=4; ++i) res(i,i+1) = lambda1;
		res(0,5) = lambda1;
		
		res(0,8) = lambda1;
		res(1,10) = lambda1;
		res(2,12) = lambda1;
		res(3,14) = lambda1;
		res(4,16) = lambda1;
		res(5,6) = lambda1;
		
		for (int i=6; i<=16; ++i) res(i,i+1) = lambda1;
		res(6,17) = lambda1;
		
		res(17,18) = lambda1;
		res(7,19) = lambda1;
		res(9,20) = lambda1;
		res(11,21) = lambda1;
		res(13,22) = lambda1;
		res(15,23) = lambda1;
		
		for (int i=18; i<=22; ++i) res(i,i+1) = lambda1;
		res(18,23) = lambda1;
	}
	else if (L==12)
	{
		return hopping_Archimedean("3.6^2",VARIANT,lambda1,lambda2);
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	
	return res;
}

ArrayXXd hopping_Platonic (int L, int VARIANT=0, double lambda1=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L == 4) // tetrahedron
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		res(0,3) = lambda1;
		res(1,3) = lambda1;
		res(2,3) = lambda1;
	}
	else if (L == 6) // octahedron
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(0,3) = lambda1;
		
		res(0,4) = lambda1;
		res(1,4) = lambda1;
		res(2,4) = lambda1;
		res(3,4) = lambda1;
		
		res(0,5) = lambda1;
		res(1,5) = lambda1;
		res(2,5) = lambda1;
		res(3,5) = lambda1;
	}
	// reference: Phys. Rev. B 72, 064453 (2005)
	else if (L == 12)
	{
//		res(0,1) = lambda1;
//		res(1,2) = lambda1;
//		res(0,2) = lambda1;
//		
//		res(1,5) = lambda1;
//		res(2,5) = lambda1;
//		res(2,6) = lambda1;
//		res(2,3) = lambda1;
//		res(0,3) = lambda1;
//		res(0,7) = lambda1;
//		res(0,4) = lambda1;
//		res(1,4) = lambda1;
//		res(1,8) = lambda1;
//		
//		res(4,8) = lambda1;
//		res(5,8) = lambda1;
//		res(5,6) = lambda1;
//		res(3,6) = lambda1;
//		res(3,7) = lambda1;
//		res(4,7) = lambda1;
//		
//		res(8,10) = lambda1;
//		res(5,10) = lambda1;
//		res(6,10) = lambda1;
//		res(6,11) = lambda1;
//		res(3,11) = lambda1;
//		res(7,11) = lambda1;
//		res(7,9) = lambda1;
//		res(4,9) = lambda1;
//		res(9,8) = lambda1;
//		
//		res(9,10) = lambda1;
//		res(10,11) = lambda1;
//		res(9,11) = lambda1;
		
		// better numbering (inwards spiral):
		
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		
		res(0,3) = lambda1;
		res(0,4) = lambda1;
		res(0,5) = lambda1;
		
		res(1,5) = lambda1;
		res(1,6) = lambda1;
		res(1,7) = lambda1;
		
		res(2,3) = lambda1;
		res(2,7) = lambda1;
		res(2,8) = lambda1;
		
		res(3,4) = lambda1;
		res(4,5) = lambda1;
		res(5,6) = lambda1;
		res(6,7) = lambda1;
		res(7,8) = lambda1;
		res(3,8) = lambda1;
		
		res(3,9) = lambda1;
		res(4,9) = lambda1;
		res(4,10) = lambda1;
		res(5,10) = lambda1;
		res(6,10) = lambda1;
		res(6,11) = lambda1;
		res(7,11) = lambda1;
		res(8,11) = lambda1;
		res(8,9) = lambda1;
		
		res(9,10) = lambda1;
		res(10,11) = lambda1;
		res(9,11) = lambda1;
	}
	else if (L==20) // dodecahedron
	{
		res = hopping_fullerene(L, VARIANT, lambda1, lambda1);
	}
	else if (L==8) // cube
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(0,3) = lambda1;
		
		res(0,5) = lambda1;
		res(1,6) = lambda1;
		res(2,7) = lambda1;
		res(3,4) = lambda1;
		
		res(4,5) = lambda1;
		res(5,6) = lambda1;
		res(6,7) = lambda1;
		res(4,7) = lambda1;
	}
	
	res += res.transpose().eval();
	
	// not required for small Platonic solids
//	if (VARIANT==0)
//	{
//		auto res_ = compress_CuthillMcKee(res);
//		res = res_;
//	}
	
	return res;
}

ArrayXXd hopping_triangular (int L, int VARIANT=0, double lambda1=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L==3)
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
	}
	else if (L==3)
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		
		res(2,3) = lambda1;
		res(2,4) = lambda1;
		res(3,4) = lambda1;
	}
	else if (L==6)
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		
		res(2,3) = lambda1;
		res(2,4) = lambda1;
		res(3,4) = lambda1;
		
		res(1,4) = lambda1;
		res(1,5) = lambda1;
		res(4,5) = lambda1;
	}
	
	res += res.transpose().eval();
	return res;
}

void add_tetrahedron (int i, int j, int k, int l, vector<pair<size_t,size_t>> &target)
{
//	std::cout << "ijkl=" << i << ", " << j << ", " << k << ", " << l << std::endl;
//	
	(i<j)? target.push_back(pair<size_t,size_t>(i,j)) : target.push_back(pair<size_t,size_t>(j,i));
	(i<k)? target.push_back(pair<size_t,size_t>(i,k)) : target.push_back(pair<size_t,size_t>(k,i));
	(i<l)? target.push_back(pair<size_t,size_t>(i,l)) : target.push_back(pair<size_t,size_t>(l,i));
	(j<k)? target.push_back(pair<size_t,size_t>(j,k)) : target.push_back(pair<size_t,size_t>(k,j));
	(j<l)? target.push_back(pair<size_t,size_t>(j,l)) : target.push_back(pair<size_t,size_t>(l,j));
	(k<l)? target.push_back(pair<size_t,size_t>(k,l)) : target.push_back(pair<size_t,size_t>(l,k));
//	
//	for (int n=0; n<6; ++n)
//	{
//		std::cout << "tetrahedron: " << target[target.size()-6+n].first << ", " << target[target.size()-6+n].second << std::endl;
//	}
//	std::cout << std::endl;
}

ArrayXXd hopping_sodaliteCage (int L=60, int VARIANT=0, double lambda1=1.)
{
	std::vector<std::pair<std::size_t, std::size_t>> edges;
	
	ArrayXXd res(L,L); res.setZero();
	
	if (L==60)
	{
		if (VARIANT==0)
		{
			// dmax=16 with Cuthill-McKee algorithm
			add_tetrahedron(0,1,3,4,edges); // dmax=4
			add_tetrahedron(0,2,5,6,edges); // dmax=6
			add_tetrahedron(5,13,14,15,edges); // dmax=10
			add_tetrahedron(14,20,27,28,edges); // dmax=14
			add_tetrahedron(8,19,20,21,edges); // dmax=13
			add_tetrahedron(3,7,8,9,edges); // dmax=6
			
			add_tetrahedron(9,11,22,23,edges); // dmax=14
			add_tetrahedron(4,10,11,12,edges); // dmax=8
			add_tetrahedron(6,16,17,18,edges); // dmax=12
			add_tetrahedron(15,18,29,30,edges); // dmax=15
			add_tetrahedron(28,35,42,43,edges); // dmax=15
			add_tetrahedron(21,33,34,35,edges); // dmax=14
			
			add_tetrahedron(34,37,49,50,edges); // dmax=16
			add_tetrahedron(23,36,37,38,edges); // dmax=15
			add_tetrahedron(12,24,25,26,edges); // dmax=14
			add_tetrahedron(17,25,31,32,edges); // dmax=15
			add_tetrahedron(30,44,45,46,edges); // dmax=16
			add_tetrahedron(43,45,53,54,edges); // dmax=11
			
			add_tetrahedron(54,56,58,59,edges); // dmax=5
			add_tetrahedron(50,52,57,58,edges); // dmax=8
			add_tetrahedron(38,41,51,52,edges); // dmax=14
			add_tetrahedron(26,39,40,41,edges); // dmax=15
			add_tetrahedron(32,40,47,48,edges); // dmax=16
			add_tetrahedron(46,48,55,56,edges); // dmax=10
		}
		else if (VARIANT==1)
		{
			// dmax=25 without optimization
			add_tetrahedron(0,5,6,13,edges); // dmax=13
			add_tetrahedron(0,1,7,14,edges); // dmax=14
			add_tetrahedron(1,2,8,15,edges); // dmax=14
			add_tetrahedron(2,3,9,16,edges); // dmax=14
			add_tetrahedron(3,4,10,17,edges); // dmax=14
			add_tetrahedron(4,5,11,12,edges); // dmax=8
			
			add_tetrahedron(12,20,21,33,edges); // dmax=21
			add_tetrahedron(13,21,22,34,edges); // dmax=21
			add_tetrahedron(14,24,25,37,edges); // dmax=23
			add_tetrahedron(15,25,26,38,edges); // dmax=23
			add_tetrahedron(16,28,29,41,edges); // dmax=25
			add_tetrahedron(17,18,29,30,edges); // dmax=13
			
			add_tetrahedron(18,19,31,43,edges); // dmax=25
			add_tetrahedron(19,20,32,44,edges); // dmax=25
			add_tetrahedron(22,23,35,45,edges); // dmax=23
			add_tetrahedron(23,24,36,46,edges); // dmax=23
			add_tetrahedron(26,27,39,47,edges); // dmax=21
			add_tetrahedron(27,28,40,42,edges); // dmax=15
			
			add_tetrahedron(42,57,58,59,edges); // dmax=17
			add_tetrahedron(43,55,56,57,edges); // dmax=14
			add_tetrahedron(44,53,54,55,edges); // dmax=11
			add_tetrahedron(45,51,52,53,edges); // dmax=8
			add_tetrahedron(46,49,50,51,edges); // dmax=5
			add_tetrahedron(47,48,49,59,edges); // dmax=12
		}
	}
	else if (L==50)
	{
		if (VARIANT==0)
		{
			add_tetrahedron(0,2,3,6,edges);
			add_tetrahedron(3,8,7,9,edges);
			add_tetrahedron(8,11,19,20,edges);
			add_tetrahedron(4,10,11,12,edges);
			add_tetrahedron(0,1,4,5,edges);
			
			add_tetrahedron(6,16,17,18,edges);
			add_tetrahedron(9,21,22,23,edges);
			add_tetrahedron(20,33,34,35,edges);
			add_tetrahedron(12,24,25,26,edges);
			add_tetrahedron(5,13,14,15,edges);
			
			add_tetrahedron(15,17,29,30,edges);
			add_tetrahedron(18,22,31,32,edges);
			add_tetrahedron(23,34,36,37,edges);
			add_tetrahedron(25,35,38,39,edges);
			add_tetrahedron(14,26,27,28,edges);
			
			add_tetrahedron(30,42,43,44,edges);
			add_tetrahedron(32,44,45,46,edges);
			add_tetrahedron(37,46,47,48,edges);
			add_tetrahedron(39,41,48,49,edges);
			add_tetrahedron(28,40,41,42,edges);
		}
		else if (VARIANT==1)
		{
			add_tetrahedron(0,1,6,12,edges);
			add_tetrahedron(1,2,7,13,edges);
			add_tetrahedron(2,3,8,14,edges);
			add_tetrahedron(3,4,9,10,edges);
			add_tetrahedron(0,4,5,11,edges);
			
			add_tetrahedron(12,21,31,32,edges);
			add_tetrahedron(13,23,33,34,edges);
			add_tetrahedron(14,15,25,26,edges);
			add_tetrahedron(10,17,27,28,edges);
			add_tetrahedron(11,19,29,30,edges);
			
			add_tetrahedron(20,30,31,38,edges);
			add_tetrahedron(22,32,33,39,edges);
			add_tetrahedron(24,25,34,35,edges);
			add_tetrahedron(16,26,27,36,edges);
			add_tetrahedron(18,28,29,37,edges);
			
			add_tetrahedron(38,44,45,49,edges);
			add_tetrahedron(39,40,45,46,edges);
			add_tetrahedron(35,41,46,47,edges);
			add_tetrahedron(36,42,47,48,edges);
			add_tetrahedron(37,43,48,49,edges);
		}
	}
	else if (L==20)
	{
		if (VARIANT==0)
		{
			add_tetrahedron(0,1,4,5,edges);
			add_tetrahedron(0,2,3,6,edges);
			add_tetrahedron(3,7,8,9,edges);
			add_tetrahedron(8,4,10,11,edges);
			
			add_tetrahedron(5,13,12,14,edges);
			add_tetrahedron(6,14,15,16,edges);
			add_tetrahedron(9,16,17,18,edges);
			add_tetrahedron(11,13,18,19,edges);
		}
		else if (VARIANT==1)
		{
			add_tetrahedron(0,3,4,12,edges);
			add_tetrahedron(0,1,5,13,edges);
			add_tetrahedron(1,2,6,14,edges);
			add_tetrahedron(2,3,7,15,edges);
			
			add_tetrahedron(4,8,9,16,edges);
			add_tetrahedron(5,9,10,17,edges);
			add_tetrahedron(6,10,11,18,edges);
			add_tetrahedron(7,8,11,19,edges);
		}
	}
	
	for (int e=0; e<edges.size(); ++e)
	{
		int i = edges[e].first;
		int j = edges[e].second;
		res(i,j) = lambda1;
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		compress_CuthillMcKee(res,true);
	}
	
	return res;
}

// Calculates distance matrix from adjacency matrix of a graph
// The distance matrix has 1 for nearest neighbours, 2 for next-nearest neighbours etc.
ArrayXXi calc_distanceMatrix (ArrayXXd adjacencyMatrix)
{
	int L = adjacencyMatrix.rows();
	assert(adjacencyMatrix.cols() == L);
	
	ArrayXXi dist0(L,L); dist0 = 0;
	dist0.matrix().diagonal().setConstant(1);
	
	ArrayXXi dist1(L,L); dist1 = 0;
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		if (abs(adjacencyMatrix(i,j)) > 0.)
		{
			dist1(i,j) = 1;
		}
	}
	
	ArrayXXi next = dist1;
	
	// each entry is a matrix with 1 for the given distance, 0 otherwise
	vector<ArrayXXi> dist;
	dist.push_back(dist0);
	dist.push_back(dist1);
	
	while (next.sum() != 0)
	{
		// calculate d-th power of adjacency matrix
		next = next.matrix()*dist1.matrix();
		
		// remove already known distances (multiply array-wise with 0)
		for (int d=0; d<dist.size(); ++d) next = next*(1-dist[d]);
		// set non-zero entries to 1
		for (int i=0; i<L; ++i) for (int j=0; j<L; ++j) if (next(i,j)>0) next(i,j) = 1;
		
		dist.push_back(next);
	}
	
	ArrayXXi res(L,L); res = 0;
	for (int d=0; d<dist.size(); ++d) res = res+d*dist[d];
	return res;
}

void push_back_KondoUnpacked (vector<Param> &params, size_t L, double J, double t, size_t D, bool START_WITH_SPIN=true)
{
	int SPIN_PARITY = (START_WITH_SPIN==true)? 0:1;
	for (size_t l=0; l<2*L; ++l)
	{
		// spin site
		if (l%2 == SPIN_PARITY)
		{
			params.push_back({"D",D,l});
			params.push_back({"LyF",0ul,l});
			if (START_WITH_SPIN==true)
			{
				params.push_back({"Inext",J,l});
				params.push_back({"Iprev",0.,l});
			}
			else
			{
				params.push_back({"Inext",0.,l});
				params.push_back({"Iprev",J,l});
			}
			params.push_back({"tPrime",0.,l});
		}
		// fermionic site
		else
		{
			params.push_back({"D",1ul,l});
			params.push_back({"LyF",1ul,l});
			params.push_back({"tPrime",t,l});
			params.push_back({"Inext",0.,l});
			params.push_back({"Iprev",0.,l});
		}
	}
}

//Eigen::ArrayXXd coupling2d (double coupl_x, double coupl_y, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	Eigen::ArrayXXd out(Lx*Ly,Lx*Ly); out.setZero();
//	for (int ix=0; ix<Lx; ix++)
//	for (int iy=0; iy<Ly; iy++)
//	for (int jx=0; jx<Lx; jx++)
//	for (int jy=0; jy<Ly; jy++)
//	{
//		if (abs(ix-jx) == 1 and (iy==jy))
//		{
//			out(ix*Ly+iy, jx*Ly+jy) += coupl_x;
//		}
//		else if (abs(iy-jy) == 1 and (ix==jx))
//		{
//			out(ix*Ly+iy, jx*Ly+jy) += coupl_y;
//		}
//		else if (abs(iy-jy) == Ly-1 and (ix==jx) and PERIODIC_Y == true)
//		{
//			out(ix*Ly+iy, jx*Ly+jy) += coupl_y;
//		}
//	}
//	return out;
//}

//Eigen::ArrayXXd coupling2d (double coupl, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	return coupling2d (coupl,coupl,Lx,Ly,PERIODIC_Y);
//}

//Eigen::ArrayXXd coupling2d_snake (double coupl_x, double coupl_y, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	Eigen::ArrayXXd out(Lx*Ly,Lx*Ly); out.setZero();
//	
//	// Mirrors the y coordinate to create a snake.
//	auto mirror = [&Ly] (int iy) -> int
//	{
//		vector<int> v(Ly);
//		iota (begin(v),end(v),0);
//		reverse(v.begin(),v.end());
//		return v[iy];
//	};
//	
//	for (int ix=0; ix<Lx; ix++)
//	for (int iy=0; iy<Ly; iy++)
//	for (int jx=0; jx<Lx; jx++)
//	for (int jy=0; jy<Ly; jy++)
//	{
//		// mirror even y only
//		int iy_ = (ix%2==0)? iy : mirror(iy);
//		int jy_ = (jx%2==0)? jy : mirror(jy);
//		int index_i = iy_+Ly*ix;
//		int index_j = jy_+Ly*jx;
//		
////		if (jx==0 and jy==0)
////		{
////			cout << ix << ", " << iy << " → index=" << index_i << endl;
////		}
//		
//		if (abs(ix-jx) == 1 and (iy==jy))
//		{
//			out(index_i,index_j) += coupl_x;
//		}
//		else if (abs(iy-jy) == 1 and (ix==jx))
//		{
//			out(index_i,index_j) += coupl_y;
//		}
//		else if (abs(iy-jy) == Ly-1 and (ix==jx) and PERIODIC_Y == true)
//		{
//			out(index_i,index_j) += coupl_y;
//		}
//	}
//	return out;
//}

//Eigen::ArrayXXd coupling2d_snake (double coupl, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	return coupling2d_snake (coupl,coupl,Lx,Ly,PERIODIC_Y);
//}

//// standard deviation for a hopping matrix
//double sigma_hop (const Eigen::ArrayXXd &thop)
//{
//	double res = 0.;
//	
//	Eigen::ArrayXd x(thop.rows()); x = 0;
//	
//	for (int i=0; i<thop.rows(); ++i)
//	for (int j=0; j<thop.cols(); ++j)
//	{
//		if (thop(i,j) != 0.) x(i) += abs(j-i);
//	}
//	
//	double avg = x.sum()/x.rows();
//	double var = ((x-avg)*(x-avg)).sum();
//	
//	return sqrt(var);
//}

vector<Param> Tinf_params_fermions (size_t Ly, size_t maxPower=1ul)
{
	vector<Param> res;
	res.push_back({"Ly",Ly});
	res.push_back({"maxPower",maxPower});
	res.push_back({"OPEN_BC",true});
	if (Ly == 2ul)
	{
		res.push_back({"t",0.});
		res.push_back({"tRung",1.});
	}
	else
	{
		res.push_back({"t",1.,0});
		res.push_back({"t",0.,1});
	}
	return res;
}

vector<Param> Tinf_params_spins (size_t L, size_t Ly, size_t maxPower=1ul, size_t D=2ul, bool SOFT=false)
{
	vector<Param> res;
	res.push_back({"Ly",Ly});
	res.push_back({"maxPower",maxPower});
	res.push_back({"OPEN_BC",true});
	if (SOFT and D==3)
	{
		if (Ly==2)
		{
			for (size_t l=1; l<L-1; ++l)
			{
				res.push_back({"D",D,l});
			}
			res.push_back({"D",2ul,0});
			res.push_back({"D",2ul,L-1});
		}
		else if (Ly==1)
		{
			for (size_t l=2; l<2*L-2; ++l)
			{
				res.push_back({"D",D,l});
			}
			res.push_back({"D",2ul,0});
			res.push_back({"D",2ul,1});
			res.push_back({"D",2ul,L-2});
			res.push_back({"D",2ul,L-1});
		}
	}
	else
	{
		res.push_back({"D",D});
	}
	if (Ly == 2ul)
	{
		res.push_back({"J",0.});
		res.push_back({"Jrung",1.});
	}
	else
	{
		res.push_back({"J",1.,0});
		res.push_back({"J",0.,1});
	}
	return res;
}

inline double conjIfImag (double x) {return x;}
inline std::complex<double> conjIfImag (std::complex<double> x) {return conj(x);}

template<typename Scalar>
Array<Scalar,Dynamic,Dynamic> hopping_PAM (int L, Scalar tfc, Scalar tcc, Scalar tff, Scalar tx, Scalar ty, bool PBC=false)
{
	Array<Scalar,Dynamic,Dynamic> t1site(2,2); t1site = 0;
	t1site(0,1) = tfc;
	
	// L: Anzahl der physikalischen fc-Sites
	Array<Scalar,Dynamic,Dynamic> res(2*L,2*L); res = 0;
	
	for (int l=0; l<L; ++l)
	{
		res.block(2*l,2*l, 2,2) = t1site;
	}
	
	for (int l=0; l<L-1; ++l)
	{
		res(2*l,   2*l+2) = tcc;
		res(2*l+1, 2*l+3) = tff;
		res(2*l+1, 2*l+2) = tx;
		res(2*l,   2*l+3) = conjIfImag(ty);
	}
	
//	cout << "OBC:" << endl;
//	cout << endl << res << endl << endl;
	if (PBC)
	{
		for (int l=0; l<2*L-4; l+=2)
		{
			// increase hopping range
			res(l,  l+4) = res(l,  l+2); // d=2 -> d=4
			res(l,  l+5) = res(l,  l+3); // d=3 -> d=5
			res(l+1,l+4) = res(l+1,l+2); // d=1 -> d=3
			res(l+1,l+5) = res(l+1,l+3); // d=2 -> d=4
			// delete NN hopping except for edges
			if (l>=2)
			{
				res(l,  l+2) = 0.;
				res(l,  l+3) = 0.;
				res(l+1,l+2) = 0.;
				res(l+1,l+3) = 0.;
			}
		}
//		res(0,2*L-2) = res(0,2);
//		res(0,2*L-1) = res(0,3);
//		res(1,2*L-2) = res(1,2);
//		res(1,2*L-1) = res(1,3);
	}
//	cout << "PBC:" << endl;
//	cout << endl << res << endl << endl;
	
	res.matrix() += res.matrix().adjoint().eval();
	
	return res;
}

template<typename Scalar>
Array<Scalar,Dynamic,Dynamic> hopping_PAM_T (int L, Scalar tfc, Scalar tcc, Scalar tff, Scalar tx, Scalar ty, bool ANCILLA_HOPPING=false, double bugfix=1e-7)
{
	Array<Scalar,Dynamic,Dynamic> res_tmp = hopping_PAM(L/2,tfc,tcc,tff,tx,ty);
	Array<Scalar,Dynamic,Dynamic> res(2*L,2*L); res = 0;
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(2*i,2*j) = res_tmp(i,j);
		if (ANCILLA_HOPPING)
		{
			res(2*i+1,2*j+1) = res_tmp(i,j);
		}
	}
	
	for (int i=0; i<2*L-1; ++i)
	{
		res(i,i+1) += bugfix;
		res(i+1,i) += bugfix;
	}
	
//	cout << res.real() << endl;
//	cout << endl;
//	cout << res.imag() << endl;
	return res;
}

ArrayXXd hopping_spinChain (int L, double JA, double JB, double JpA, double JpB, bool ANCILLA_HOPPING=false, bool PBC=false)
{
	ArrayXXd res_tmp(L,L);
	
	if (PBC)
	{
		lout << termcolor::yellow << "Warning: ignoring JB, JpB for PBC!" << termcolor::reset << endl;
		res_tmp = create_1D_PBC(L,JA,JpA);
	}
	else
	{
		res_tmp.setZero();
		res_tmp.matrix().diagonal<1> ()(Eigen::seq(0,Eigen::last,2)).setConstant(JA);
		res_tmp.matrix().diagonal<1> ()(Eigen::seq(1,Eigen::last,2)).setConstant(JB);
		res_tmp.matrix().diagonal<2> ()(Eigen::seq(0,Eigen::last,2)).setConstant(JpA);
		res_tmp.matrix().diagonal<2> ()(Eigen::seq(1,Eigen::last,2)).setConstant(JpB);
		res_tmp += res_tmp.transpose().eval();
	}
	return res_tmp;
}

ArrayXXd hopping_spinChain_T (int L, double JA, double JB, double JpA, double JpB, bool ANCILLA_HOPPING=false, double bugfix=1e-7, bool PBC=false)
{
	ArrayXXd res_tmp;
	if (PBC)
	{
		lout << termcolor::yellow << "Warning: ignoring JB, JpB for PBC!" << termcolor::reset << endl;
		res_tmp = create_1D_PBC(L,JA,JpA);
	}
	else
	{
		res_tmp = hopping_spinChain(L,JA,JB,JpA,JpB,false);
	}
	
	ArrayXXd res(2*L,2*L); res = 0;
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(2*i,2*j) = res_tmp(i,j);
		if (ANCILLA_HOPPING)
		{
			res(2*i+1,2*j+1) = res_tmp(i,j);
		}
	}
	
	for (int i=0; i<2*L-1; ++i)
	{
		res(i,i+1) += bugfix;
		res(i+1,i) += bugfix;
	}
	
	return res;
}

// returns J, R and offset for bilinear-biquadratic Hamiltonian with J1=1 and J2=beta
// H = J*S_i*S_{i+1} - beta*(S_i*S_{i+1})^2
//   = (J+beta/2)*S_i*S_{i+1} - beta/2 Q_i*Q_{i+1}
tuple<double,double,double> params_bilineraBiquadratic (boost::rational<int> beta_input = boost::rational<int>(-1,3))
{
	double beta = boost::rational_cast<double>(beta_input);
	double J = 1.+0.5*beta;
	double R = -0.5*beta;
	double offset = -4./3.*beta;
	return make_tuple(J,R,offset);
}

#endif
