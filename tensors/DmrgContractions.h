#ifndef STRAWBERRY_DMRGCONTRACTIONS_WITH_Q
#define STRAWBERRY_DMRGCONTRACTIONS_WITH_Q

/// \cond
#include <unordered_set>
/// \endcond

//include "Biped.h"
//include "Multipede.h"
#include "tensors/DmrgIndexGymnastics.h"
//include "symmetry/functions.h"

enum CONTRACT_LR_MODE {FULL, TRIANGULAR, FIXED};

/**
 * \ingroup Tensors
 * Contracts a left transfer matrix \p Lold with two MpsQ tensors \p Abra, \p Aket and an MpoQ tensor \p W as follows:
 * \dotfile contractQ_L.dot
 * \param Lold
 * \param Abra
 * \param W
 * \param IS_HAMILTONIAN : If the Mpo is a Hamiltonian, the calculation can be optimized
 * \param Aket
 * \param qloc : local basis
 * \param qOp : operator basis
 * \param Lnew : new transfer matrix to be written to
 * \param RANDOMIZE : if \p true, set right blocks but fill result with random numbersAA
 * \param MODE_input : if \p FULL, simple contraction, 
 *                     if \p TRIANGULAR, contract only the lower triangle \f$a<b\f$, 
 *                     if \p FIXED contract with fixed \f$a\f$
 * \note The quantum number flow for the left environment is \f$i+a=j\f$ where \f$i\f$ is the index L.out (ket layer),
 * \f$a\f$ the index L.mid (MPO layer) and \f$j\f$ the index L.in (bra layer).
 * This corresponds to the CGC \f$C^{i,a\rightarrow j}_{m_i,m_a\rightarrow m_j}\f$.
 */
template<typename Symmetry, typename MatrixType, typename MatrixType2, typename MpoScalar>
void contract_L (const Tripod<Symmetry,MatrixType2> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const bool &IS_HAMILTONIAN,
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOp, 
                 Tripod<Symmetry,MatrixType2> &Lnew,
                 bool RANDOMIZE = false,
                 tuple<CONTRACT_LR_MODE,size_t> MODE_input = make_pair(FULL,0))
{
	MpoScalar factor_cgc;
	Lnew.clear();
	Lnew.setZero();
	auto [MODE,fixed_b] = MODE_input;
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		if (!Symmetry::validate(qarray3<Symmetry::Nq>{qloc[s2], qOp[k], qloc[s1]})) {continue;}
		
		for (size_t qL=0; qL<Lold.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = LAWA(Lold.in(qL), Lold.out(qL), Lold.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix, IS_HAMILTONIAN);
			
			if (FOUND_MATCH)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					swap(quple[0], quple[1]);
					size_t qAbra = get<1>(ix[n]);
					size_t qAket = get<2>(ix[n]);
					
					if (Aket[s2].block[qAket].size() == 0) {continue;}
					if (Abra[s1].block[qAbra].size() == 0) {continue;}
					
					if constexpr ( Symmetry::NON_ABELIAN )
					{
						factor_cgc = Symmetry::coeff_buildL(Aket[s2].in[qAket], qloc[s2], Aket[s2].out[qAket],
						                                    Lold.mid(qL),            qOp[k],   quple[2],
						                                    Abra[s1].in[qAbra], qloc[s1], Abra[s1].out[qAbra]);
					}
					else
					{
						factor_cgc = 1.;
					}
					if (abs(factor_cgc) < ::mynumeric_limits<MpoScalar>::epsilon()) {continue;}
					
					for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
					{
						size_t a = iW.row();
						size_t b = iW.col();
						
						// 0 is the Hamiltonian block. Only singlet couplings are neccessary.
						if (b == 0 and IS_HAMILTONIAN and quple[2] != Symmetry::qvacuum()) {continue;}
						
						if (MODE == FULL or
						   (MODE == TRIANGULAR and a>fixed_b) or
						   (MODE == FIXED and b==fixed_b))
						{
//							if (MODE == FIXED)
//							{
//								cout << "fixed_b=" << fixed_b << ", a=" << a << "/" << W[s1][s2][k].rows() << ", b=" << b << "/" << W[s1][s2][k].cols() << endl;
//								cout << "Lold.block[qL][a][0].size()=" << Lold.block[qL][a][0].size() << endl;
//							}
							
							if (Lold.block[qL][a][0].size() != 0)
							{
								MatrixType2 Mtmp;
								if (RANDOMIZE)
								{
									Mtmp.resize(Abra[s1].block[qAbra].cols(), Aket[s2].block[qAket].cols());
									Mtmp.setRandom();
								}
								else
								{
									optimal_multiply(factor_cgc * iW.value(),
									                 Abra[s1].block[qAbra].adjoint(),
									                 Lold.block[qL][a][0],
									                 Aket[s2].block[qAket],
									                 Mtmp);
								}
								
								auto it = Lnew.dict.find(quple);
								if (it != Lnew.dict.end())
								{
									if (Lnew.block[it->second][b][0].rows() != Mtmp.rows() or 
										Lnew.block[it->second][b][0].cols() != Mtmp.cols())
									{
										Lnew.block[it->second][b][0] = Mtmp;
									}
									else
									{
										Lnew.block[it->second][b][0] += Mtmp;
									}
								}
								else
								{
									boost::multi_array<MatrixType2,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].cols()][1]);
									Mtmpvec[b][0] = Mtmp;
									Lnew.push_back(quple, Mtmpvec);
								}
							}
						}
					}
				}
			}
		}
	}
}

/**
 * \ingroup Tensors
 * Contracts a right transfer matrix \p Rold with two Mps tensors \p Abra, \p Aket and an Mpo tensor \p V as follows:
 * \dotfile contractQ_R.dot
 * \param Rold
 * \param Abra
 * \param W
 * \param IS_HAMILTONIAN : If the Mpo is a Hamiltonian, the calculation can be optimized
 * \param Aket
 * \param qloc : local basis
 * \param qOp : operator basis
 * \param Rnew : new transfer matrix to be written to
 * \param RANDOMIZE : if \p true, set right blocks but fill result with random numbers
 * \param MODE_input : if \p FULL, simple contraction, 
 *                     if \p TRIANGULAR, contract only the lower triangle \f$a<b\f$, 
 *                     if \p FIXED contract with fixed \f$a\f$
 * \note The quantum number flow for the right environment is \f$i+a=j\f$ where \f$i\f$ is the index R.in (ket layer),
 * \f$a\f$ the index R.mid (MPO layer) and \f$j\f$ the index R.out (bra layer).
 * This corresponds to the CGC \f$C^{i,a\rightarrow j}_{m_i,m_a\rightarrow m_j}\f$.
 */
template<typename Symmetry, typename MatrixType, typename MatrixType2, typename MpoScalar>
void contract_R (const Tripod<Symmetry,MatrixType2> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const bool &IS_HAMILTONIAN,
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOp, 
                 Tripod<Symmetry,MatrixType2> &Rnew,
                 bool RANDOMIZE = false,
                 tuple<CONTRACT_LR_MODE,size_t> MODE_input = make_pair(FULL,0))
{
	MpoScalar factor_cgc;
	Rnew.clear();
	Rnew.setZero();
	auto [MODE,fixed_a] = MODE_input;
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		if (!Symmetry::validate(qarray3<Symmetry::Nq>{qloc[s2], qOp[k], qloc[s1]})) {continue;}
		
		for (size_t qR=0; qR<Rold.dim; ++qR)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = AWAR(Rold.in(qR), Rold.out(qR), Rold.mid(qR), s1, s2, qloc, k, qOp, Abra, Aket, ix, IS_HAMILTONIAN);
			
			if (FOUND_MATCH)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					swap(quple[0], quple[1]);
					size_t qAbra = get<1>(ix[n]);
					size_t qAket = get<2>(ix[n]);
					
					if (Aket[s2].block[qAket].size() == 0) {continue;}
					if (Abra[s1].block[qAbra].size() == 0) {continue;}
					
					if constexpr (Symmetry::NON_ABELIAN)
					{
						factor_cgc = Symmetry::coeff_buildR(Aket[s2].in[qAket], qloc[s2], Aket[s2].out[qAket],
						                                    quple[2]          , qOp[k],   Rold.mid(qR),
						                                    Abra[s1].in[qAbra], qloc[s1], Abra[s1].out[qAbra]);
					}
					else
					{
						factor_cgc = 1.;
					}
					if (abs(factor_cgc) < ::mynumeric_limits<MpoScalar>::epsilon()) {continue;}
					
					for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
					{
						size_t a = iW.row();
						size_t b = iW.col();
						
						// Daux-1 is the Hamiltonian block. Only singlet couplings are neccessary.
						if (a == W[s1][s2][k].rows()-1 and IS_HAMILTONIAN and quple[2] != Symmetry::qvacuum()) {continue;}
						
						if (MODE == FULL or
						   (MODE == TRIANGULAR and fixed_a>b) or
						   (MODE == FIXED and a==fixed_a))
						{
//							if (MODE == FIXED)
//							{
//								cout << "fixed_a=" << fixed_a << ", a=" << a << "/" << W[s1][s2][k].rows() << ", b=" << b << "/" << W[s1][s2][k].cols() << endl;
//								cout << "Rold.block[qR][b][0].rows()=" << Rold.block[qR][b][0].rows() << endl;
//							}
							
							if (Rold.block[qR][b][0].rows() != 0)
							{
								MatrixType2 Mtmp;
								if (RANDOMIZE)
								{
									Mtmp.resize(Aket[s2].block[qAket].rows(), Abra[s1].block[qAbra].rows());
									Mtmp.setRandom();
								}
								else
								{
									optimal_multiply(factor_cgc * iW.value(),
									                 Aket[s2].block[qAket],
									                 Rold.block[qR][b][0],
									                 Abra[s1].block[qAbra].adjoint(),
									                 Mtmp);
								}
								
								auto it = Rnew.dict.find(quple);
								if (it != Rnew.dict.end())
								{
									if (Rnew.block[it->second][a][0].rows() != Mtmp.rows() or 
										Rnew.block[it->second][a][0].cols() != Mtmp.cols())
									{
										Rnew.block[it->second][a][0] = Mtmp;
									}
									else
									{
										Rnew.block[it->second][a][0] += Mtmp;
									}
								}
								else
								{
									boost::multi_array<MatrixType2,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].rows()][1]);
									Mtmpvec[a][0] = Mtmp;
									Rnew.push_back(quple, Mtmpvec);
								}
							}
						}
					}
				}
			}
		}
	}
}

/**
 * \ingroup Tensors
 * Contracts a left transfer matrix \p Lold with two Mps tensors \p Abra, \p Aket and a block dependent Mpo tensor \p V as follows:
 * \dotfile contractQ_L.dot
 * \param Lold
 * \param Abra
 * \param V
 * \param Aket
 * \param qloc : local basis
 * \param qOp : operator basis
 * \param Lnew : new transfer matrix to be written to
 * \note This function is used, when the squared Mpo with SU(2) symmetry was precalculated, 
 * since in this case the Mpo matrices depend on the symmetry block. For this reason the special member Vsq of the Mpo is used.
 */
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_L (const Tripod<Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const unordered_map<tuple<size_t,size_t,size_t,qarray<Symmetry::Nq>,qarray<Symmetry::Nq> >,SparseMatrix<MpoScalar> > &V, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOp, 
                 Tripod<Symmetry,MatrixType> &Lnew)
{
	MpoScalar factor_cgc;
	Lnew.clear();
	Lnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		if(!Symmetry::validate(qarray3<Symmetry::Nq>{qloc[s2], qOp[k], qloc[s1]})) {continue;}
		for (size_t qL=0; qL<Lold.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = LAWA(Lold.in(qL), Lold.out(qL), Lold.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix);
		
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					swap(quple[0], quple[1]);
					size_t qAbra = get<1>(ix[n]);
					size_t qAket = get<2>(ix[n]);
					if constexpr ( Symmetry::NON_ABELIAN )
						{
							factor_cgc = Symmetry::coeff_buildL(Aket[s2].in[qAket], qloc[s2], Aket[s2].out[qAket],
																Lold.mid(qL),            qOp[k],   quple[2],
																Abra[s1].in[qAbra], qloc[s1], Abra[s1].out[qAbra]);
						}
					else
					{
						factor_cgc = 1.;
					}
					if (std::abs(factor_cgc) < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
					auto key = make_tuple(s1,s2,k,Lold.mid(qL),quple[2]);
					if(auto it=V.find(key); it == V.end()) { continue; }
					for (int r=0; r<V.at(key).outerSize(); ++r)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iW(V.at(key),r); iW; ++iW)
					{
						size_t a1 = iW.row();
						size_t a2 = iW.col();
						if (Lold.block[qL][a1][0].size() != 0)
						{
							MatrixType Mtmp;
							optimal_multiply(factor_cgc*iW.value(),
											 Abra[s1].block[qAbra].adjoint(),
											 Lold.block[qL][a1][0],
											 Aket[s2].block[qAket],
											 Mtmp);
							// if (Mtmp.norm() < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
							
							auto it = Lnew.dict.find(quple);
							if (it != Lnew.dict.end())
							{
								if (Lnew.block[it->second][a2][0].rows() != Mtmp.rows() or 
									Lnew.block[it->second][a2][0].cols() != Mtmp.cols())
								{
									Lnew.block[it->second][a2][0] = Mtmp;
								}
								else
								{
									Lnew.block[it->second][a2][0] += Mtmp;
									// if (Lnew.block[it->second][a2][0].norm() < ::mynumeric_limits<MpoScalar>::epsilon())
									// {
									// 	Lnew.block[it->second][a2][0].resize(0,0);
									// 	continue;
									// }
								}
							}
							else
							{
								boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[V.at(key).cols()][1]);
								Mtmpvec[a2][0] = Mtmp;
								Lnew.push_back(quple, Mtmpvec);
							}
						}
					}
				}
			}
		}
	}
}

/**
 * \ingroup Tensors
 * Contracts a right transfer matrix \p Rold with two Mps tensors \p Abra, \p Aket and a block dependent Mpo tensor \p V as follows:
 * \dotfile contractQ_R.dot
 * \param Rold
 * \param Abra
 * \param V
 * \param Aket
 * \param qloc : local basis
 * \param qOp : operator basis
 * \param Rnew : new transfer matrix to be written to
 * \note This function is used, when the squared Mpo with SU(2) symmetry was precalculated, 
 * since in this case the Mpo matrices depend on the symmetry block. For this reason the special member Vsq of the Mpo is used.
 */
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_R (const Tripod<Symmetry,MatrixType> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra,
                 const unordered_map<tuple<size_t,size_t,size_t,qarray<Symmetry::Nq>,qarray<Symmetry::Nq> >,SparseMatrix<MpoScalar> > &V,
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOp, 
                 Tripod<Symmetry,MatrixType> &Rnew)
{
	MpoScalar factor_cgc;
	Rnew.clear();
	Rnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		if(!Symmetry::validate(qarray3<Symmetry::Nq>{qloc[s2],qOp[k],qloc[s1]})) {continue;}
		
		for (size_t qR=0; qR<Rold.dim; ++qR)
		{
			auto qRouts = Symmetry::reduceSilent(Rold.out(qR),Symmetry::flip(qloc[s1]));
			auto qRins = Symmetry::reduceSilent(Rold.in(qR),Symmetry::flip(qloc[s2]));
			
			for(const auto& qRout : qRouts)
			for(const auto& qRin : qRins)
			{
				qarray2<Symmetry::Nq> cmp1 = {qRout, Rold.out(qR)};
				qarray2<Symmetry::Nq> cmp2 = {qRin, Rold.in(qR)};
				
				auto q1 = Abra[s1].dict.find(cmp1);
				auto q2 = Aket[s2].dict.find(cmp2);
				
				if (q1!=Abra[s1].dict.end() and 
				    q2!=Aket[s2].dict.end())
				{
					qarray<Symmetry::Nq> new_qin  = Aket[s2].in[q2->second]; // A.in
					qarray<Symmetry::Nq> new_qout = Abra[s1].in[q1->second]; // A†.out = A.in
					auto qRmids = Symmetry::reduceSilent(Rold.mid(qR),Symmetry::flip(qOp[k]));
					
					for(const auto& new_qmid : qRmids)
					{
						qarray3<Symmetry::Nq> quple = {new_qin, new_qout, new_qmid};
						if constexpr (Symmetry::NON_ABELIAN)
						{
							factor_cgc = Symmetry::coeff_buildR(Aket[s2].in[q2->second], qloc[s2], Aket[s2].out[q2->second],
																quple[2]          , qOp[k],   Rold.mid(qR),
																Abra[s1].in[q1->second], qloc[s1], Abra[s1].out[q1->second]);
						}
						else
						{
							factor_cgc = 1.;
						}
						if (std::abs(factor_cgc) < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
						auto key = make_tuple(s1,s2,k,quple[2],Rold.mid(qR));
						if(auto it=V.find(key); it == V.end()) { continue; }
						for (int r=0; r<V.at(key).outerSize(); ++r)
						for (typename SparseMatrix<MpoScalar>::InnerIterator iW(V.at(key),r); iW; ++iW)
						{
							size_t a1 = iW.row();
							size_t a2 = iW.col();
								
							if (Rold.block[qR][a2][0].rows() != 0)
							{
								MatrixType Mtmp;
								optimal_multiply(factor_cgc*iW.value(),
								                 Aket[s2].block[q2->second],
								                 Rold.block[qR][a2][0],
								                 Abra[s1].block[q1->second].adjoint(),
								                 Mtmp);
								// if(Mtmp.norm() < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
								
								auto it = Rnew.dict.find(quple);
								if (it != Rnew.dict.end())
								{
									if (Rnew.block[it->second][a1][0].rows() != Mtmp.rows() or 
										Rnew.block[it->second][a1][0].cols() != Mtmp.cols())
									{
										Rnew.block[it->second][a1][0] = Mtmp;
									}
									else
									{
										Rnew.block[it->second][a1][0] += Mtmp;
										// if(Rnew.block[it->second][a1][0].norm() < ::mynumeric_limits<MpoScalar>::epsilon())
										// {
										// 	Rnew.block[it->second][a1][0].resize(0,0);
										// 	continue;
										// }
									}
								}
								else
								{
									boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[V.at(key).rows()][1]);
									Mtmpvec[a1][0] = Mtmp;
									Rnew.push_back(quple, Mtmpvec);
								}
							}
						}
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename MatrixType, typename MatrixType2>
void contract_L (const Biped<Symmetry,MatrixType2> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 Biped<Symmetry,MatrixType2> &Lnew,
                 bool RANDOMIZE = false,
                 bool CLEAR = false)
{
	if (CLEAR)
	{
		Lnew.clear();
	}
	else
	{
		Lnew.outerResize(Lold);
		Lnew.setZero();
	}
	
	for (size_t s=0; s<qloc.size(); ++s)
	for (size_t qL=0; qL<Lold.dim; ++qL)
	{
		vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > ix;
		bool FOUND_MATCH = LAA(Lold.in[qL], Lold.out[qL], s, qloc, Abra, Aket, ix);
		
		if (FOUND_MATCH)
		{
			for (size_t n=0; n<ix.size(); n++)
			{
				qarray2<Symmetry::Nq> quple = get<0>(ix[n]);
				swap(quple[0], quple[1]);
				if (!Symmetry::validate(quple)) {continue;}
				size_t qAbra = get<1>(ix[n]);
				size_t qAket = get<2>(ix[n]);
				
				if (Lold.block[qL].rows() != 0)
				{
					MatrixType2 Mtmp;
					if (RANDOMIZE)
					{
						Mtmp.resize(Abra[s].block[qAbra].cols(), Aket[s].block[qAket].cols());
						Mtmp.setRandom();
					}
					else
					{
//						print_size(Abra[s].block[qAbra].adjoint(), "Abra[s].block[qAbra].adjoint()");
//						print_size(Lold.block[qL], "Lold.block[qL]");
//						print_size(Aket[s].block[qAket], "Aket[s].block[qAket]");
//						cout << Abra[s].out[qAbra] << ", " << Abra[s].in[qAbra] << endl;
//						cout << Lold.in[qL] << ", " << Lold.out[qL] << endl;
//						cout << Aket[s].in[qAket] << ", " << Aket[s].out[qAket] << endl;
//						cout << endl;
						
						optimal_multiply(1.,
						                 Abra[s].block[qAbra].adjoint(),
						                 Lold.block[qL],
						                 Aket[s].block[qAket],
						                 Mtmp);
					}
					
					auto it = Lnew.dict.find(quple);
					if (it != Lnew.dict.end())
					{
						if (Lnew.block[it->second].rows() != Mtmp.rows() or 
							Lnew.block[it->second].cols() != Mtmp.cols())
						{
							Lnew.block[it->second] = Mtmp;
						}
						else
						{
							Lnew.block[it->second] += Mtmp;
						}
					}
					else
					{
						Lnew.push_back(quple, Mtmp);
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename MatrixType, typename MatrixType2>
void contract_R (const Biped<Symmetry,MatrixType2> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 Biped<Symmetry,MatrixType2> &Rnew,
                 bool RANDOMIZE = false,
                 bool CLEAR = false)
{
	if (CLEAR)
	{
		Rnew.clear();
	}
	else
	{
		Rnew.outerResize(Rold);
		Rnew.setZero();
	}
	
	for (size_t s=0; s<qloc.size(); ++s)
	for (size_t qR=0; qR<Rold.dim; ++qR)
	{
		vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > ix;
		bool FOUND_MATCH = AAR(Rold.in[qR], Rold.out[qR], s, qloc, Abra, Aket, ix);
		
		if (FOUND_MATCH)
		{
			for (size_t n=0; n<ix.size(); n++)
			{
				qarray2<Symmetry::Nq> quple = get<0>(ix[n]);
				swap(quple[0], quple[1]);
				if (!Symmetry::validate(quple)) {continue;}
				size_t qAbra = get<1>(ix[n]);
				size_t qAket = get<2>(ix[n]);
				
				double factor_cgc = Symmetry::coeff_rightOrtho(Abra[s].out[qAbra],
				                                               Abra[s].in [qAbra]);
				
				if (Rold.block[qR].rows() != 0)
				{
					MatrixType2 Mtmp;
					if (RANDOMIZE)
					{
						Mtmp.resize(Aket[s].block[qAket].rows(), Abra[s].block[qAbra].rows());
						Mtmp.setRandom();
					}
					else
					{
						optimal_multiply(factor_cgc,
					                     Aket[s].block[qAket],
					                     Rold.block[qR],
					                     Abra[s].block[qAbra].adjoint(),
					                     Mtmp);
					}
					
					auto it = Rnew.dict.find(quple);
					if (it != Rnew.dict.end())
					{
						if (Rnew.block[it->second].rows() != Mtmp.rows() or 
							Rnew.block[it->second].cols() != Mtmp.cols())
						{
							Rnew.block[it->second] = Mtmp;
						}
						else
						{
							Rnew.block[it->second] += Mtmp;
						}
					}
					else
					{
						Rnew.push_back(quple, Mtmp);
					}
				}
			}
		}
	}
}

// template<typename Symmetry, typename MatrixType>
// void contract_TT (const Tripod<Symmetry,MatrixType>  &T1,
// 				  const Tripod<Symmetry,MatrixType>  &T2,
// 				  Biped<Symmetry,MatrixType> &Res)
// {
// 	Res.clear();
// 	Res.setZero();
	
// 	// for (size_t s=0; s<qloc.size(); ++s)
// 	// {
// 		for (size_t qT1=0; qT1<T1.dim; ++qT1)
// 		{
// 			// auto qRouts = Symmetry::reduceSilent(Rold.out[qR],Symmetry::flip(qloc[s]));
// 			// auto qRins = Symmetry::reduceSilent(Rold.in[qR],Symmetry::flip(qloc[s]));
			
// 			// for(const auto& qRout : qRouts)
// 			// for(const auto& qRin : qRins)
// 			// {
// 			qarray3<Symmetry::Nq> quple = { T1.out(qT1),T1.in(qT1),T1.mid(qT1); }
// 			if(auto qT2=T2.dict.find(quple); qT2 != T2.dict.end())
// 			{
// 				qarray<Symmetry::Nq> new_qin  = T1.in(qT1);
// 				qarray<Symmetry::Nq> new_qout = T1.out(qT2->second);
// 				qarray2<Symmetry::Nq> quple2 = {new_qin, new_qout};
// 				if (!Symmetry::validate(quple2)) {continue;}
					
// 				double factor_cgc = Symmetry::coeff_rightOrtho(Abra[s].out[q1->second],
// 															   Abra[s].in[q1->second]);
					
// 					if (Rold.block[qR].rows() != 0)
// 					{
// 						MatrixType Mtmp;
// 						optimal_multiply(factor_cgc,
// 						                 Aket[s].block[q2->second],
// 						                 Rold.block[qR],
// 						                 Abra[s].block[q1->second].adjoint(),
// 						                 Mtmp);
						
// 						auto it = Rnew.dict.find(quple);
// 						if (it != Rnew.dict.end())
// 						{
// 							if (Rnew.block[it->second].rows() != Mtmp.rows() or 
// 								Rnew.block[it->second].cols() != Mtmp.cols())
// 							{
// 								Rnew.block[it->second] = Mtmp;
// 							}
// 							else
// 							{
// 								Rnew.block[it->second] += Mtmp;
// 							}
// 						}
// 						else
// 						{
// 							Rnew.push_back(quple, Mtmp);
// 						}
// 					}
// 				}
// 			}
// 	// }
// }

template<typename Symmetry, typename Scalar>
void contract_GRALF (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L,
                     const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                     const vector<vector<vector<SparseMatrixXd> > > &W, 
                     const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                     const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                     const vector<qarray<Symmetry::Nq> > &qloc,
                     const vector<qarray<Symmetry::Nq> > &qOp,
                     Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Tout,
                     DMRG::DIRECTION::OPTION DIR)
{
	std::array<typename Symmetry::qType,3> qCheck;
	Scalar factor_cgc, factor_merge;
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		qCheck = {qloc[s2],qOp[k],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		
		for (size_t qL=0; qL<L.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = LAWA(L.in(qL), L.out(qL), L.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix, PROP::HAMILTONIAN);
			
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					auto qR = R.dict.find(quple);
					
					if (qR != R.dict.end())
					{
						swap(quple[0], quple[1]);
						size_t qAbra = get<1>(ix[n]);
						size_t qAket = get<2>(ix[n]);
						if (Aket[s2].block[qAket].size() == 0) {continue;}
						if (Abra[s1].block[qAbra].size() == 0) {continue;}
						if constexpr (Symmetry::NON_ABELIAN)
						{
							if (DIR == DMRG::DIRECTION::RIGHT)
							{
								factor_cgc = Symmetry::coeff_buildL(Aket[s2].in[qAket], qloc[s2], Aket[s2].out[qAket],
																	L.mid(qL)         , qOp[k]  , quple[2],
																	Abra[s1].in[qAbra], qloc[s1], Abra[s1].out[qAbra]);

								factor_merge = Symmetry::coeff_rightOrtho(Abra[s1].out[qAbra],
																		  Aket[s2].out[qAket]);
							}
							else if (DIR == DMRG::DIRECTION::LEFT)
							{
								factor_cgc = Symmetry::coeff_buildR(Aket[s2].in[qAket], qloc[s2], Aket[s2].out[qAket],
																	L.mid(qL)         , qOp[k]  , quple[2],
																	Abra[s1].in[qAbra], qloc[s1], Abra[s1].out[qAbra]);
								factor_merge = Symmetry::coeff_rightOrtho(L.in(qL),
																		  L.out(qL));
							}
						}
						else
						{
							factor_cgc = 1.;
							factor_merge = 1.;
						}
						if (std::abs(factor_cgc*factor_merge) < ::mynumeric_limits<Scalar>::epsilon()) {continue;}
						
						for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
						for (SparseMatrixXd::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
						{
							size_t a1 = iW.row();
							size_t a2 = iW.col();
							
							if (L.block[qL][a1][0].size() != 0 and
								R.block[qR->second][a2][0].size() != 0)
							{
								Matrix<Scalar,Dynamic,Dynamic> Mtmp;
								if (DIR == DMRG::DIRECTION::RIGHT)
								{
									// RALF
									optimal_multiply(iW.value() * factor_cgc * factor_merge,
													 R.block[qR->second][a2][0],
													 Abra[s1].block[qAbra].adjoint(),
													 L.block[qL][a1][0],
													 Aket[s2].block[qAket],
													 Mtmp);
								}
								else if (DIR == DMRG::DIRECTION::LEFT)
								{
									// GRAL
									optimal_multiply(iW.value() * factor_cgc * factor_merge,
													 Aket[s2].block[qAket],
													 R.block[qR->second][a2][0],
													 Abra[s1].block[qAbra].adjoint(),
													 L.block[qL][a1][0],
													 Mtmp);
								}
								
								qarray2<Symmetry::Nq> qTout; 
								if (DIR == DMRG::DIRECTION::RIGHT)
								{
									qTout = {Aket[s2].out[qAket], Aket[s2].out[qAket]};
								}
								 else if (DIR == DMRG::DIRECTION::LEFT)
								{
									qTout = {Aket[s2].in[qAket], Aket[s2].in[qAket]};
								}
								auto it = Tout.dict.find(qTout);
								if (it != Tout.dict.end())
								{
									if (Tout.block[it->second].rows() != Mtmp.rows() or 
										Tout.block[it->second].cols() != Mtmp.cols())
									{
										Tout.block[it->second] = Mtmp;
									}
									else
									{
										Tout.block[it->second] += Mtmp;
									}
								}
								else
								{
									Tout.push_back(qTout,Mtmp);
								}
							}
						}
					}
				}
			}
		}
	}
}

/**
 * \ingroup Tensors
 * Calculates the contraction between a left transfer matrix \p L, 
 * two MpsQ tensors \p Abra, \p Aket, an MpoQ tensor \p W and a right transfer matrix \p R. Not really that much useful.
 * \param L
 * \param Abra
 * \param W
 * \param Aket
 * \param R
 * \param qloc : local basis
 * \param qOp : operator basis
 * \returns : result of contraction
 * \warning Not working for non-abelian symmetries.
 */
template<typename Symmetry, typename Scalar>
Scalar contract_LR (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L,
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                    const vector<vector<vector<SparseMatrixXd> > > &W, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                    const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                    const vector<qarray<Symmetry::Nq> > &qloc,
                    const vector<qarray<Symmetry::Nq> > &qOp)
{
	Scalar res = 0.;
	std::array<typename Symmetry::qType,3> qCheck;
	Scalar factor_cgc;

	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		qCheck = {qloc[s2],qOp[k],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		for (size_t qL=0; qL<L.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = LAA(L.in(qL), L.out(qL), L.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix);
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					auto qR = R.dict.find(quple);
					
					if (qR != R.dict.end())
					{
						swap(quple[0], quple[1]);
						size_t qAbra = get<1>(ix[n]);
						size_t qAket = get<2>(ix[n]);
						if constexpr ( Symmetry::NON_ABELIAN )
						{
							factor_cgc = Symmetry::coeff_buildL(Aket[s2].in[qAket], qloc[s2], Aket[s2].out[qAket],
																L.mid(qL)         , qOp[k]  , quple[2],
																Abra[s1].in[qAbra], qloc[s1], Abra[s1].out[qAbra]);
						}
						else
						{
							factor_cgc = 1.;
						}
						if (std::abs(factor_cgc) < ::mynumeric_limits<Scalar>::epsilon()) { continue; }
						
						for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
						for (SparseMatrixXd::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
						{
							size_t a1 = iW.row();
							size_t a2 = iW.col();
							
							if (L.block[qL][a1][0].rows() != 0 and
								R.block[qR->second][a2][0].rows() != 0)
							{
//						Matrix<Scalar,Dynamic,Dynamic> Mtmp  = iW.value() *
//						                                       (Abra[s1].block[qAbra].adjoint() *
//						                                        L.block[qL][a1][0] * 
//						                                        Aket[s2].block[qAket] * 
//						                                        R.block[qR->second][a2][0]);
//						res += Mtmp.trace();
						
								Matrix<Scalar,Dynamic,Dynamic> Mtmp = L.block[qL][a1][0] * 
									Aket[s2].block[qAket] * 
									R.block[qR->second][a2][0];
								for (size_t i=0; i<Abra[s1].block[qAbra].cols(); ++i)
								{
									res += iW.value() * Abra[s1].block[qAbra].col(i).dot(Mtmp.col(i));
								}
							}
						}
					}
				}
			}
		}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
Scalar contract_LR (size_t fixed_b, 
                    const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L,
                    const Biped <Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R)
{
	Scalar res = 0;
	
	for (size_t qL=0; qL<L.dim; ++qL)
	{
		// Not necessarily vacuum for the structure factor, so this function cannot be used!
		assert(L.out(qL) == L.in(qL) and "contract_LR(Tripod,Biped) error!");
		
		if (L.mid(qL) == Symmetry::qvacuum())
		{
			qarray2<Symmetry::Nq> quple = {L.out(qL), L.in(qL)};
			auto qR = R.dict.find(quple);
			
			if (qR != R.dict.end())
			{
				if (L.block[qL][fixed_b][0].size() != 0 and
				    R.block[qR->second].size() != 0)
				{
					res += (L.block[qL][fixed_b][0] * R.block[qR->second]).trace() * Symmetry::coeff_dot(L.out(qL));
				}
			}
		}
	}
	
	return res;
}

template<typename Symmetry, typename Scalar>
Scalar contract_LR (size_t fixed_a, 
                    const Biped <Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L,
                    const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R)
{
	Scalar res = 0;
	
	for (size_t qR=0; qR<R.dim; ++qR)
	{
		// Not necessarily vacuum for the structure factor, so this function cannot be used!
		assert(R.out(qR) == R.in(qR) and "contract_LR(Biped,Tripod) error!");
		
		if (R.mid(qR) == Symmetry::qvacuum())
		{
			qarray2<Symmetry::Nq> quple = {R.out(qR), R.in(qR)};
			auto qL = L.dict.find(quple);
			
			if (qL != L.dict.end())
			{
				if (R.block[qR][fixed_a][0].size() != 0 and
				    L.block[qL->second].size() != 0)
				{
					res += (L.block[qL->second] * R.block[qR][fixed_a][0]).trace() * Symmetry::coeff_dot(R.out(qR));
				}
			}
		}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
Scalar contract_LR (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L,
                    const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R)
{
	Scalar res = 0;
	
	for (size_t qL=0; qL<L.dim; ++qL)
	{
		qarray3<Symmetry::Nq> quple = {L.out(qL), L.in(qL), L.mid(qL)};
		auto qR = R.dict.find(quple);
		
		if (qR != R.dict.end())
		{
			assert(L.block[qL].shape()[0] == R.block[qR->second].shape()[0]);
			
			for (size_t a=0; a<L.block[qL].shape()[0]; ++a)
			{
				if (L.block[qL][a][0].size() != 0 and
				    R.block[qR->second][a][0].size() != 0)
				{
					res += (L.block[qL][a][0] * R.block[qR->second][a][0]).trace() * Symmetry::coeff_dot(L.in(qL));
				}
			}
		}
	}
	
	return res;
}

//template<typename Symmetry, typename MatrixType>
//void contract_LR (const Tripod<Symmetry,MatrixType> &L,
//                  const Tripod<Symmetry,MatrixType> &R, 
//                  const std::array<qarray<Symmetry::Nq>,D> &qloc, 
//                  Tripod<Symmetry,MatrixType> &Bres)
//{
//	Bres.clear();
//	Bres.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t qL=0; qL<L.dim; ++qL)
//	{
//		qarray3<Symmetry::Nq> quple = {L.out(qL), L.in(qL), L.mid(qL)};
//		auto qR = R.dict.find(quple);
//		
//		if (qR != R.dict.end())
//		{
//			if (L.block[qL][a1][0].rows() != 0 and
//			    R.block[qR->second][a2][0].rows() != 0)
//			{
////						cout << Abra[s1].block[qAbra].adjoint().rows() << "\t" << Abra[s1].block[qAbra].adjoint().cols() << endl;
////						cout << L.block[qL][a1][0].rows() << "\t" << L.block[qL][a1][0].cols() << endl;
////						cout << Aket[s2].block[qAket].rows() << "\t" << Aket[s2].block[qAket].cols() << endl;
////						cout << R.block[qR->second][a2][0].rows() << "\t" << R.block[qR->second][a2][0].cols() << endl;
////						cout << endl;
//				
//				MatrixType Mtmp = L.block[qL][a1][0] * R.block[qR->second][a2][0]);
//				
//				cout << Mtmp.rows() << "\t" << Mtmp.cols() << endl << Mtmp << endl << endl;
//				
////						auto it = Bres.dict.find(quple);
////						if (it != Bres.dict.end())
////						{
////							if (Bres.block[it->second][a2][0].rows() != Mtmp.rows() or 
////								Bres.block[it->second][a2][0].cols() != Mtmp.cols())
////							{
////								Bres.block[it->second][a2][0] = Mtmp;
////							}
////							else
////							{
////								Bres.block[it->second][a2][0] += Mtmp;
////							}
////						}
////						else
////						{
////							boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2].block[qW].cols()][1]);
////							Mtmpvec[a2][0] = Mtmp;
////							Bres.push_back(quple, Mtmpvec);
////							cout << "in:  " << quple[0] << ", out: " << quple[1] << ", mid: " << quple[2] << endl;
////						}
//			}
//		}
//	}
//}

//template<typename Symmetry, typename MatrixType>
//void dryContract_L (const Tripod<Symmetry,MatrixType> &Lold, 
//                    const vector<Biped<Symmetry,MatrixType> > &Abra, 
//                    const std::array<std::array<Biped<Symmetry,SparseMatrixXd>,D>,D> &W, 
//                    const vector<Biped<Symmetry,MatrixType> > &Aket, 
//                    const std::array<qarray<Symmetry::Nq>,D> &qloc, 
//                    Tripod<Symmetry,MatrixType> &Lnew, 
//                    vector<tuple<qarray3<Symmetry::Nq>,std::array<size_t,8> > > &ix)
//{
//	Lnew.setZero();
//	
//	MatrixType Mtmp(1,1); Mtmp << 1.;
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t qL=0; qL<Lold.dim; ++qL)
//	{
//		qarray2<Symmetry::Nq> cmp1 = {Lold.in(qL),  Lold.in(qL)+qloc[s1]};
//		qarray2<Symmetry::Nq> cmp2 = {Lold.out(qL), Lold.out(qL)+qloc[s2]};
//		qarray2<Symmetry::Nq> cmpW = {Lold.mid(qL), Lold.mid(qL)+qloc[s1]-qloc[s2]};
//		
//		auto q1 = Abra[s1].dict.find(cmp1);
//		auto q2 = Aket[s2].dict.find(cmp2);
//		auto qW = W[s1][s2].dict.find(cmpW);
//		
//		if (q1!=Abra[s1].dict.end() and 
//		    q2!=Aket[s2].dict.end() and 
//		    qW!=W[s1][s2].dict.end())
//		{
//			qarray<Symmetry::Nq> new_qin  = Abra[s1].out[q1->second]; // A†.in = A.out
//			qarray<Symmetry::Nq> new_qout = Aket[s2].out[q2->second]; // A.in
//			qarray<Symmetry::Nq> new_qmid = W[s1][s2].out[qW->second];
//			qarray3<Symmetry::Nq> quple = {new_qin, new_qout, new_qmid};
//			
//			size_t Wcols = W[s1][s2].block[qW->second].cols();
//			
//			for (int k=0; k<W[s1][s2].block[qW->second].outerSize(); ++k)
//			for (SparseMatrixXd::InnerIterator iW(W[s1][s2].block[qW->second],k); iW; ++iW)
//			{
//				size_t a1 = iW.row();
//				size_t a2 = iW.col();
//				
//				if (Lold.block[qL][a1][0].rows() != 0)
//				{
//					std::array<size_t,9> juple = {s1, s2, q1->second, qW->second, q2->second, qL, a1, a2};
//					ix.push_back(make_tuple(quple,juple));
//					
//					auto it = Lnew.dict.find(quple);
//					if (it != Lnew.dict.end())
//					{
//						if (Lnew.block[it->second][a2][0].rows() == 0)
//						{
//							Lnew.block[it->second][a2][0] = Mtmp;
//						}
//					}
//					else
//					{
//						boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[Wcols][1]);
//						Mtmpvec[a2][0] = Mtmp;
//						Lnew.push_back(quple, Mtmpvec);
//					}
//				}
//			}
//		}
//	}
//}

//template<typename Symmetry, typename MatrixType>
//void contract_L (const Tripod<Symmetry,MatrixType> &Lold, 
//                 const vector<tuple<qarray3<Symmetry::Nq>,std::array<size_t,8> > > ix, 
//                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
//                 const std::array<std::array<Biped<Symmetry,SparseMatrixXd>,D>,D> &W, 
//                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
//                 const std::array<qarray<Symmetry::Nq>,D> &qloc, 
//                 Tripod<Symmetry,MatrixType> &Lnew)
//{
//	Lnew.setZero();
//	
//	for (size_t i=0; i<ix.size(); ++i)
//	{
//		auto quple = get<0>(ix);
//		size_t s1 = get<1>(ix)[0];
//		size_t s2 = get<1>(ix)[1];
//		size_t q1 = get<1>(ix)[2];
//		size_t qW = get<1>(ix)[3];
//		size_t q2 = get<1>(ix)[4];
//		size_t qL = get<1>(ix)[5];
//		size_t a1 = get<1>(ix)[6];
//		size_t a2 = get<1>(ix)[7];
//		
//		for (int k=0; k<W[s1][s2].block[qW].outerSize(); ++k)
//		for (SparseMatrixXd::InnerIterator iW(W[s1][s2].block[qW],k); iW; ++iW)
//		{
//			size_t a1 = iW.row();
//			size_t a2 = iW.col();
//			
//			if (Lold.block[qL][a1][0].rows() != 0)
//			{
//				MatrixType Mtmp = iW.value() *
//				                  (Abra[s1].block[q1].adjoint() *
//				                   Lold.block[qL][a1][0] * 
//				                   Aket[s2].block[q2]);
//				
//				auto it = Lnew.dict.find(quple);
//				if (it != Lnew.dict.end())
//				{
//					if (Lnew.block[it->second][a2][0].rows() != Mtmp.rows() or 
//						Lnew.block[it->second][a2][0].cols() != Mtmp.cols())
//					{
//						Lnew.block[it->second][a2][0] = Mtmp;
//					}
//					else
//					{
//						Lnew.block[it->second][a2][0] += Mtmp;
//					}
//				}
//				else
//				{
//					boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2].block[qW].cols()][1]);
//					Mtmpvec[a2][0] = Mtmp;
//					Lnew.push_back(quple, Mtmpvec);
//				}
//			}
//		}
//	}
//}

/**
 * \ingroup Tensors
 * Calculates the contraction between a right transfer matrix \p Rold, two MpsQ tensors \p Abra, \p Aket and two MpoQ tensors \p Wbot, \p Wtop.
 * Needed, for example, when calculating \f$\left<H^2\right>\f$ and no MpoQ represenation of \f$H^2\f$ is available.
*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_R (const Tripod<Symmetry,MatrixType> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra,
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wbot,
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wtop,
                 const vector<Biped<Symmetry,MatrixType> > &Aket,
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOpBot,
                 const vector<qarray<Symmetry::Nq> > &qOpTop,
                 const Qbasis<Symmetry> &baseRightBot,
                 const Qbasis<Symmetry> &baseRightTop,
                 const Qbasis<Symmetry> &baseLeftBot,
                 const Qbasis<Symmetry> &baseLeftTop,
                 Tripod<Symmetry,MatrixType> &Rnew)
{
	// cout << baseRightTop << endl << baseLeftTop << endl;
	auto leftTopQs = baseLeftTop.unordered_qs();
	auto leftBotQs = baseLeftBot.unordered_qs();

	auto TensorBaseRight = baseRightBot.combine(baseRightTop);
	auto TensorBaseLeft = baseLeftBot.combine(baseLeftTop);

	std::array<typename Symmetry::qType,3> qCheck;

	MpoScalar factor_cgc, factor_merge, factor_check;
	Rnew.clear();
	Rnew.setZero();

	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t s3=0; s3<qloc.size(); ++s3)
	for (size_t k1=0; k1<qOpTop.size(); ++k1)
	for (size_t k2=0; k2<qOpBot.size(); ++k2)
	{
		qCheck = {qloc[s3],qOpTop[k1],qloc[s2]};
		if(!Symmetry::validate(qCheck)) {continue;}
		qCheck = {qloc[s2],qOpBot[k2],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}

		auto ks = Symmetry::reduceSilent(qOpTop[k2],qOpBot[k1]);
		for(const auto& k : ks)
		{
			qCheck = {qloc[s3],k,qloc[s1]};
			if(!Symmetry::validate(qCheck)) {continue;}
			
			// product in physical space:
			factor_check = Symmetry::coeff_prod(qloc[s1],qOpBot[k2],qloc[s2],
												qOpTop[k1],qloc[s3],k);
			if (std::abs(factor_check) < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
			for (size_t qR=0; qR<Rold.dim; ++qR)
			{
				auto qRouts = Symmetry::reduceSilent(Rold.out(qR),Symmetry::flip(qloc[s1]));
				auto qRins =  Symmetry::reduceSilent(Rold.in(qR),Symmetry::flip(qloc[s3]));
				auto qrightAuxs = Sym::split<Symmetry>(Rold.mid(qR),baseRightTop.qs(),baseRightBot.qs());
				
				for(const auto& qRout : qRouts)
				for(const auto& qRin : qRins)
				{
					auto q1 = Abra[s1].dict.find({qRout, Rold.out(qR)});
					auto q3 = Aket[s3].dict.find({qRin, Rold.in(qR)});
					if (q1!=Abra[s1].dict.end() and q3!=Aket[s3].dict.end())
					{
						auto qRmids = Symmetry::reduceSilent(Rold.mid(qR),Symmetry::flip(k));
						for(const auto& new_qmid : qRmids)
						{
							qarray3<Symmetry::Nq> quple = {Aket[s3].in[q3->second], Abra[s1].in[q1->second], new_qmid};
							factor_cgc = Symmetry::coeff_buildR(Aket[s3].in[q3->second], qloc[s3], Aket[s3].out[q3->second],
																new_qmid               , k       , Rold.mid(qR),
																Abra[s1].in[q1->second], qloc[s1], Abra[s1].out[q1->second]);
							
							if (std::abs(factor_cgc) < std::abs(::mynumeric_limits<MpoScalar>::epsilon())) { continue; }
							for(const auto& [qrightAux,qrightAuxP] : qrightAuxs)
							{
								Eigen::Index left2=TensorBaseRight.leftAmount(Rold.mid(qR),{qrightAuxP, qrightAux});
								auto qleftAuxs = Symmetry::reduceSilent(qrightAux,Symmetry::flip(qOpTop[k1]));
								for(const auto& qleftAux : qleftAuxs)
								{
									if(auto it=leftTopQs.find(qleftAux) != leftTopQs.end())
									{
										auto qleftAuxPs = Symmetry::reduceSilent(qrightAuxP,Symmetry::flip(qOpBot[k2]));
										for(const auto& qleftAuxP : qleftAuxPs)
										{											
											if(auto it=leftBotQs.find(qleftAuxP) != leftBotQs.end())
											{
												factor_merge = Symmetry::coeff_tensorProd(qleftAuxP,qleftAux,new_qmid,
																						  qOpBot[k2],qOpTop[k1],k,
																						  qrightAuxP,qrightAux,Rold.mid(qR));
												if (std::abs(factor_merge) < std::abs(::mynumeric_limits<MpoScalar>::epsilon())) { continue; }
												Eigen::Index left1=TensorBaseLeft.leftAmount(new_qmid,{qleftAuxP, qleftAux});
												for (int ktop=0; ktop<Wtop[s2][s3][k1].outerSize(); ++ktop)
												for (typename SparseMatrix<MpoScalar>::InnerIterator iWtop(Wtop[s2][s3][k1],ktop); iWtop; ++iWtop)
												for (int kbot=0; kbot<Wbot[s1][s2][k2].outerSize(); ++kbot)
												for (typename SparseMatrix<MpoScalar>::InnerIterator iWbot(Wbot[s1][s2][k2],kbot); iWbot; ++iWbot)
												{
													size_t br = iWbot.row();
													size_t bc = iWbot.col();
													size_t tr = iWtop.row();
													size_t tc = iWtop.col();
													MpoScalar Wfactor = iWbot.value() * iWtop.value();
													
													size_t a1 = left1+br*Wtop[s2][s3][k1].rows()+tr;
													size_t a2 = left2+bc*Wtop[s2][s3][k1].cols()+tc;
													
													if (Rold.block[qR][a2][0].rows() != 0)
													{
														MatrixType Mtmp;
														optimal_multiply(factor_check*factor_merge*factor_cgc*Wfactor,
														                 Aket[s3].block[q3->second],
														                 Rold.block[qR][a2][0],
														                 Abra[s1].block[q1->second].adjoint(),
														                 Mtmp);
														auto it = Rnew.dict.find(quple);
														
														if (it != Rnew.dict.end())
														{
															if (Rnew.block[it->second][a1][0].rows() != Mtmp.rows() or 
																Rnew.block[it->second][a1][0].cols() != Mtmp.cols())
															{
																Rnew.block[it->second][a1][0] = Mtmp;
															}
															else
															{
																Rnew.block[it->second][a1][0] += Mtmp;
															}
														}
														else
														{
															boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[TensorBaseLeft.inner_dim(new_qmid)][1]);
															Mtmpvec[a1][0] = Mtmp;
															Rnew.push_back(quple, Mtmpvec);
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

/**
 * \ingroup Tensors
 * Calculates the contraction between a left transfer matrix \p Lold, two MpsQ tensors \p Abra, \p Aket and two MpoQ tensors \p Wbot, \p Wtop.
 * Needed, for example, when calculating \f$\left<H^2\right>\f$ and no MpoQ represenation of \f$H^2\f$ is available.
 */
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_L (const Multipede<4,Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wbot, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wtop, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOpBot,
                 const vector<qarray<Symmetry::Nq> > &qOpTop,
                 Multipede<4,Symmetry,MatrixType> &Lnew)
{
	std::array<typename Symmetry::qType,3> qCheck;

	Lnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t s3=0; s3<qloc.size(); ++s3)
	for (size_t k1=0; k1<qOpBot.size(); ++k1)
	for (size_t k2=0; k2<qOpTop.size(); ++k2)
	{
		qCheck = {qloc[s2],qOpBot[k1],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		qCheck = {qloc[s3],qOpTop[k2],qloc[s2]};
		if(!Symmetry::validate(qCheck)) {continue;}
		for (size_t qL=0; qL<Lold.dim; ++qL)
		{
			tuple<qarray4<Symmetry::Nq>,size_t,size_t> ix;
			bool FOUND_MATCH = AWWA(Lold.in(qL), Lold.out(qL), Lold.bot(qL), Lold.top(qL), 
									s1, s2, s3, qloc, k1, qOpBot, k2, qOpTop, Abra, Aket, ix);
			auto   quple = get<0>(ix);
			swap(quple[0],quple[1]);
			size_t qAbra = get<1>(ix);
			size_t qAket = get<2>(ix);
		
			if (FOUND_MATCH == true)
			{
				for (int kbot=0; kbot<Wbot[s1][s2][k1].outerSize(); ++kbot)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iWbot(Wbot[s1][s2][k1],kbot); iWbot; ++iWbot)
				for (int ktop=0; ktop<Wtop[s2][s3][k2].outerSize(); ++ktop)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iWtop(Wtop[s2][s3][k2],ktop); iWtop; ++iWtop)
				{
					size_t br = iWbot.row();
					size_t bc = iWbot.col();
					size_t tr = iWtop.row();
					size_t tc = iWtop.col();
					MpoScalar Wfactor = iWbot.value() * iWtop.value();
				
					if (Lold.block[qL][br][tr].rows() != 0)
					{
						MatrixType Mtmp;
						optimal_multiply(Wfactor,
							             Abra[s1].block[qAbra].adjoint(),
							             Lold.block[qL][br][tr],
							             Aket[s3].block[qAket],
							             Mtmp);
					
						if (Mtmp.norm() != 0.)
						{
							auto it = Lnew.dict.find(quple);
							if (it != Lnew.dict.end())
							{
								if (Lnew.block[it->second][bc][tc].rows() != Mtmp.rows() or 
									Lnew.block[it->second][bc][tc].cols() != Mtmp.cols())
								{
									Lnew.block[it->second][bc][tc] = Mtmp;
								}
								else
								{
									Lnew.block[it->second][bc][tc] += Mtmp;
								}
							}
							else
							{
								size_t bcols = Wbot[s1][s2][k1].cols();
								size_t tcols = Wtop[s2][s3][k2].cols();
								boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[bcols][tcols]);
								Mtmparray[bc][tc] = Mtmp;
								Lnew.push_back(quple, Mtmparray);
							}
						}
					}
				}
			}
		}
	}
}

/**For details see: Stoudenmire, White (2010)*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_C0 (vector<qarray<Symmetry::Nq> > qloc,
                  vector<qarray<Symmetry::Nq> > qOp,
                  const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                  const vector<Biped<Symmetry,MatrixType> >   &Aket, 
                  vector<Tripod<Symmetry,MatrixType> >        &Cnext)
{
	Cnext.clear();
	Cnext.resize(qloc.size());
	std::array<typename Symmetry::qType,3> qCheck;

	for (size_t s2=0; s2<qloc.size(); ++s2)
	{
		qarray2<Symmetry::Nq> cmpA = {Symmetry::qvacuum(), Symmetry::qvacuum()+qloc[s2]};
		auto qA = Aket[s2].dict.find(cmpA);
		
		if (qA != Aket[s2].dict.end())
		{
			for (size_t s1=0; s1<qloc.size(); ++s1)
			for (size_t k=0; k<qOp.size(); ++k)
			{
				qCheck = {qloc[s2],qOp[k],qloc[s1]};
				if(!Symmetry::validate(qCheck)) {continue;}
				for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
				{
					MatrixType Mtmp = iW.value() * Aket[s2].block[qA->second];
					
					qarray3<Symmetry::Nq> cmpC = {Symmetry::qvacuum(), Aket[s2].out[qA->second], Symmetry::qvacuum()+qloc[s1]-qloc[s2]};
					auto qCnext = Cnext[s1].dict.find(cmpC);
					if (qCnext != Cnext[s1].dict.end())
					{
						if (Cnext[s1].block[qCnext->second][iW.col()][0].rows() != Mtmp.rows() or 
							Cnext[s1].block[qCnext->second][iW.col()][0].cols() != Mtmp.cols())
						{
							Cnext[s1].block[qCnext->second][iW.col()][0] = Mtmp;
						}
						else
						{
							Cnext[s1].block[qCnext->second][iW.col()][0] += Mtmp;
						}
					}
					else
					{
						boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].cols()][1]);
						Mtmpvec[iW.col()][0] = Mtmp;
						Cnext[s1].push_back({Symmetry::qvacuum(), Aket[s2].out[qA->second], Symmetry::qvacuum()+qloc[s1]-qloc[s2]}, Mtmpvec);
					}
				}
			}
		}
	}
}

/**For details see: Stoudenmire, White (2010)
\dotfile contract_C.dot*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_C (vector<qarray<Symmetry::Nq> > qloc,
                 vector<qarray<Symmetry::Nq> > qOp,
                 const vector<Biped<Symmetry,MatrixType> >   &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const vector<Biped<Symmetry,MatrixType> >   &Aket, 
                 const vector<Tripod<Symmetry,MatrixType> >  &C, 
                 vector<Tripod<Symmetry,MatrixType> >        &Cnext)
{
	Cnext.clear();
	Cnext.resize(qloc.size());
	std::array<typename Symmetry::qType,3> qCheck;

	for (size_t s=0; s<qloc.size(); ++s)
	for (size_t qC=0; qC<C[s].dim; ++qC)
	{
		qarray2<Symmetry::Nq> cmpU = {C[s].in(qC), C[s].in(qC)+qloc[s]};
		auto qU = Abra[s].dict.find(cmpU);
		
		if (qU != Abra[s].dict.end())
		{
			for (size_t s1=0; s1<qloc.size(); ++s1)
			for (size_t s2=0; s2<qloc.size(); ++s2)
			for (size_t k=0; k<qOp.size(); ++k)
			{
				qCheck = {qloc[s2],qOp[k],qloc[s1]};
				if(!Symmetry::validate(qCheck)) {continue;}
				
				qarray2<Symmetry::Nq> cmpA = {C[s].out(qC), C[s].out(qC)+qloc[s2]};
				auto qA = Aket[s2].dict.find(cmpA);
				
				if (qA != Aket[s2].dict.end())
				{
					for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
					{
						if (C[s].block[qC][iW.row()][0].rows() != 0)
						{
//							MatrixType Mtmp = iW.value() * (Abra[s].block[qU->second].adjoint() * 
//							                                C[s].block[qC][iW.row()][0] * 
//							                                Aket[s2].block[qA->second]);
							MatrixType Mtmp;
							optimal_multiply(iW.value(),
							                 Abra[s].block[qU->second].adjoint(),
							                 C[s].block[qC][iW.row()][0],
							                 Aket[s2].block[qA->second],
							                 Mtmp);
							
							qarray3<Symmetry::Nq> cmpC = {Abra[s].out[qU->second], Aket[s2].out[qA->second], C[s].mid(qC)+qloc[s1]-qloc[s2]};
							auto qCnext = Cnext[s1].dict.find(cmpC);
							if (qCnext != Cnext[s1].dict.end())
							{
								if (Cnext[s1].block[qCnext->second][iW.col()][0].rows() != Mtmp.rows() or 
									Cnext[s1].block[qCnext->second][iW.col()][0].cols() != Mtmp.cols())
								{
									Cnext[s1].block[qCnext->second][iW.col()][0] = Mtmp;
								}
								else
								{
									Cnext[s1].block[qCnext->second][iW.col()][0] += Mtmp;
								}
							}
							else
							{
								boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][0].cols()][1]);
								Mtmpvec[iW.col()][0] = Mtmp;
								Cnext[s1].push_back({Abra[s].out[qU->second], Aket[s2].out[qA->second], C[s].mid(qC)+qloc[s1]-qloc[s2]}, Mtmpvec);
							}
						}
					}
				}
			}
		}
	}
}

//template<typename Symmetry, typename Scalar>
//void contract_AA (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A1, 
//                  vector<qarray<Symmetry::Nq> > qloc1, 
//                  const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A2, 
//                  vector<qarray<Symmetry::Nq> > qloc2, 
//                  vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair)
//{
//	auto tensor_basis = Symmetry::tensorProd(qloc1,qloc2);
//	Apair.resize(tensor_basis.size());
//	
//	for (size_t s1=0; s1<qloc1.size(); ++s1)
//	for (size_t s2=0; s2<qloc2.size(); ++s2)
//	{
//		auto qmerges = Symmetry::reduceSilent(qloc1[s1], qloc2[s2]);
//		
//		for (const auto &qmerge:qmerges)
//		{
//			auto qtensor = make_tuple(qloc1[s1], s1, qloc2[s2], s2, qmerge);
//			auto s1s2 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor));
//			
//			for (size_t q1=0; q1<A1[s1].dim; ++q1)
//			{
//				auto qmids = Symmetry::reduceSilent(A1[s1].out[q1], qloc2[s2]);
//				
//				for (const auto &qmid:qmids)
//				{
//					qarray2<Symmetry::Nq> quple = {A1[s1].out[q1], qmid};
//					auto q2 = A2[s2].dict.find(quple);
//					
//					if (q2 != A2[s2].dict.end())
//					{
//						Scalar factor_cgc = Symmetry::coeff_Apair(A1[s1].in[q1], qloc1[s1], A1[s1].out[q1], 
//						                                          qloc2[s2], A2[s2].out[q2->second], qmerge);
//						
//						if (abs(factor_cgc) > abs(mynumeric_limits<Scalar>::epsilon()))
//						{
//							Matrix<Scalar,Dynamic,Dynamic> Mtmp = factor_cgc * A1[s1].block[q1] * A2[s2].block[q2->second];
//							
//							qarray2<Symmetry::Nq> qupleApair = {A1[s1].in[q1], A2[s2].out[q2->second]};
//							
//							auto qApair = Apair[s1s2].dict.find(qupleApair);
//							
//							if (qApair != Apair[s1s2].dict.end())
//							{
//								Apair[s1s2].block[qApair->second] += Mtmp;
//							}
//							else
//							{
//								Apair[s1s2].push_back(qupleApair, Mtmp);
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void contract_AW (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Ain, 
                  const vector<qarray<Symmetry::Nq> > &qloc,
                  const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                  const vector<qarray<Symmetry::Nq> > &qOp,
                  const Qbasis<Symmetry> &qauxAl,
                  const Qbasis<Symmetry> &qauxWl,
                  const Qbasis<Symmetry> &qauxAr,
                  const Qbasis<Symmetry> &qauxWr,
                  vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aout)
{
	for (size_t s=0; s<qloc.size(); ++s)
	{
		Aout[s].clear();
	}

	MpoScalar factor_cgc;

	auto tensorBasis_l = qauxAl.combine(qauxWl);
	auto tensorBasis_r = qauxAr.combine(qauxWr);
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0;  k<qOp.size();   ++k)
	{
		qarray3<Symmetry::Nq> qCheck_ = {qloc[s2],qOp[k],qloc[s1]};
		if (!Symmetry::validate(qCheck_)) { continue; }
		// if(W[s1][s2][k].size() == 0) { continue; } //Checks whether QNs s1, s2 and k fit together.
		for (size_t q=0; q<Ain[s2].size(); q++)
		// for (const auto &[qWl,qWl_dim,qWl_plain] : qauxWl) // cpp is to stupid to call cbegin() and cend() here... 
		for (auto it=qauxWl.cbegin(); it != qauxWl.cend(); it++)
		{
			auto [qWl, qWl_dim, qWl_plain] = *it;
			auto qWrs = Symmetry::reduceSilent(qWl,qOp[k]);
			for (const auto &qWr : qWrs)
			{
				if (qauxWr.find(qWr) == false) {continue;}
				auto qmerge_ls = Symmetry::reduceSilent(Ain[s2].in[q] ,qWl);
				auto qmerge_rs = Symmetry::reduceSilent(Ain[s2].out[q],qWr);
				for (const auto qmerge_l : qmerge_ls)
				for (const auto qmerge_r : qmerge_rs)
				{
					qarray3<Symmetry::Nq> qCheck = {qmerge_l,qloc[s1],qmerge_r};
					if (!Symmetry::validate(qCheck)) { continue; }
					if (tensorBasis_l.find(qmerge_l) == false) {continue;}
					if (tensorBasis_r.find(qmerge_r) == false) {continue;}
					if constexpr (Symmetry::NON_ABELIAN)
								 {
									 factor_cgc = Symmetry::coeff_AW(Ain[s2].in[q], qloc[s2], Ain[s2].out[q],
									 								 qWl          , qOp[k]  , qWr,
									 								 qmerge_l     , qloc[s1], qmerge_r);
								 }
					else { factor_cgc = MpoScalar(1); }

					if (abs(factor_cgc) < abs(mynumeric_limits<MpoScalar>::epsilon())) { continue; }

					Matrix<Scalar,Dynamic,Dynamic> Mtmp(tensorBasis_l.inner_dim(qmerge_l), tensorBasis_r.inner_dim(qmerge_r)); Mtmp.setZero();
					int left_l = tensorBasis_l.leftAmount(qmerge_l, { Ain[s2].in[q] , qWl } );
					int left_r = tensorBasis_r.leftAmount(qmerge_r, { Ain[s2].out[q], qWr } );

					for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
					{
						size_t wr = iW.row();
						size_t wc = iW.col();
						Mtmp.block(wr+left_l,wc+left_r,Ain[s2].block[q].rows(),Ain[s2].block[q].cols()) += Ain[s2].block[q] * iW.value() * factor_cgc;
					}
					qarray2<Symmetry::Nq> cmp = qarray2<Symmetry::Nq>{qmerge_l, qmerge_r}; //auxiliary quantum numbers of Aout
					auto it = Aout[s1].dict.find(cmp);
					if (it == Aout[s1].dict.end())
					{
						Aout[s1].push_back(cmp,Mtmp);
					}
					else
					{
						Aout[s1].block[it->second] += Mtmp;
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void contract_AA (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A1, 
                  const vector<qarray<Symmetry::Nq> > &qloc1, 
                  const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A2, 
                  const vector<qarray<Symmetry::Nq> > &qloc2, 
                  const qarray<Symmetry::Nq> &Qtop, 
                  const qarray<Symmetry::Nq> &Qbot, 
                  vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair, 
                  bool DRY = false)
{
	auto tensor_basis = Symmetry::tensorProd(qloc1,qloc2);
	Apair.resize(tensor_basis.size());
	
	vector<qarray<Symmetry::Nq> > qsplit = calc_qsplit (A1, qloc1, A2, qloc2, Qtop, Qbot);
	
	vector<qarray<Symmetry::Nq> > A1in;
	vector<qarray<Symmetry::Nq> > A2out;
	
	// gather all qin at the left:
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t q=0; q<A1[s1].dim; ++q)
	{
		A1in.push_back(A1[s1].in[q]);
	}
	// gather all qout at the right:
	for (size_t s2=0; s2<qloc2.size(); ++s2)
	for (size_t q=0; q<A2[s2].dim; ++q)
	{
		A2out.push_back(A2[s2].out[q]);
	}
	
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t m=0; m<qsplit.size(); ++m)
	{
		auto qins = Symmetry::reduceSilent(qsplit[m], Symmetry::flip(qloc1[s1]));
		
		for (const auto &qin:qins)
		{
			for (size_t s2=0; s2<qloc2.size(); ++s2)
			{
				auto qmerges = Symmetry::reduceSilent(qloc1[s1], qloc2[s2]);
				
				for (const auto &qmerge:qmerges)
				{
					auto qtensor = make_tuple(qloc1[s1], s1, qloc2[s2], s2, qmerge);
					auto s1s2 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor));
					
					auto qouts = Symmetry::reduceSilent(qsplit[m], qloc2[s2]);
					for (const auto &qout:qouts)
					{
						auto qA1 = find(A1in.begin(),  A1in.end(),  qin);
						auto qA2 = find(A2out.begin(), A2out.end(), qout);
						if (qA1 != A1in.end() and qA2 != A2out.end())
						{
							Apair[s1s2].try_create_block({qin,qout});
						}
					}
				}
			}
		}
	}
	
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t s2=0; s2<qloc2.size(); ++s2)
	{
		auto qmerges = Symmetry::reduceSilent(qloc1[s1], qloc2[s2]);
		
		for (const auto &qmerge:qmerges)
		{
			auto qtensor = make_tuple(qloc1[s1], s1, qloc2[s2], s2, qmerge);
			auto s1s2 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor));
			
			for (size_t q1=0; q1<A1[s1].dim; ++q1)
			{
				auto qouts = Symmetry::reduceSilent(A1[s1].out[q1], qloc2[s2]);
				
				for (const auto &qout:qouts)
				{
					qarray2<Symmetry::Nq> quple = {A1[s1].out[q1], qout};
					auto q2 = A2[s2].dict.find(quple);
					
					if (q2 != A2[s2].dict.end())
					{
						Scalar factor_cgc = Symmetry::coeff_Apair(A1[s1].in[q1], qloc1[s1], A1[s1].out[q1], 
						                                          qloc2[s2], A2[s2].out[q2->second], qmerge);						
						if (abs(factor_cgc) > abs(mynumeric_limits<Scalar>::epsilon()))
						{
							Matrix<Scalar,Dynamic,Dynamic> Mtmp;
							if (!DRY)
							{
								Mtmp = factor_cgc * A1[s1].block[q1] * A2[s2].block[q2->second];
							}
							
							if (Mtmp.size() != 0)
							{
								qarray2<Symmetry::Nq> qupleApair = {A1[s1].in[q1], A2[s2].out[q2->second]};
								
								auto qApair = Apair[s1s2].dict.find(qupleApair);
								
								if (qApair != Apair[s1s2].dict.end() and 
								    Apair[s1s2].block[qApair->second].size() == Mtmp.size())
								{
									Apair[s1s2].block[qApair->second] += Mtmp;
								}
								else if (qApair != Apair[s1s2].dict.end() and 
									     Apair[s1s2].block[qApair->second].size() == 0)
								{
									Apair[s1s2].block[qApair->second] = Mtmp;
								}
								else
								{
									Apair[s1s2].push_back(qupleApair, Mtmp);
								}
							}
						}
					}
				}
			}
		}
	}
}

/**for VUMPS 4-site unit cell*/
template<typename Symmetry, typename Scalar>
void contract_AAAA (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A1, 
                    vector<qarray<Symmetry::Nq> > qloc1, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A2, 
                    vector<qarray<Symmetry::Nq> > qloc2, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A3, 
                    vector<qarray<Symmetry::Nq> > qloc3, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A4, 
                    vector<qarray<Symmetry::Nq> > qloc4, 
                    boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquartett)
{
	Aquartett.resize(boost::extents[qloc1.size()][qloc2.size()][qloc3.size()][qloc4.size()]);
	
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t s2=0; s2<qloc2.size(); ++s2)
	for (size_t s3=0; s3<qloc3.size(); ++s3)
	for (size_t s4=0; s4<qloc4.size(); ++s4)
	for (size_t q1=0; q1<A1[s1].dim; ++q1)
	{
		qarray2<Symmetry::Nq> quple2 = {A1[s1].out[q1], A1[s1].out[q1]+qloc2[s2]};
		auto q2 = A2[s2].dict.find(quple2);
		
		if (q2 != A2[s2].dict.end())
		{
			qarray2<Symmetry::Nq> quple3 = {A2[s2].out[q2->second], A2[s2].out[q2->second]+qloc3[s3]};
			auto q3 = A3[s3].dict.find(quple3);
			
			if (q3 != A3[s3].dict.end())
			{
				qarray2<Symmetry::Nq> quple4 = {A3[s3].out[q3->second], A3[s3].out[q3->second]+qloc4[s4]};
				auto q4 = A4[s4].dict.find(quple4);
				
				if (q4 != A4[s4].dict.end())
				{
					Matrix<Scalar,Dynamic,Dynamic> Mtmp = A1[s1].block[q1] * 
					                                      A2[s2].block[q2->second] * 
					                                      A3[s3].block[q3->second] * 
					                                      A4[s4].block[q4->second];
					
					qarray2<Symmetry::Nq> qupleAquartett = {A1[s1].in[q1], A4[s4].out[q4->second]};
					auto qAquartett = Aquartett[s1][s2][s3][s4].dict.find(qupleAquartett);
					
					if (qAquartett != Aquartett[s1][s2][s3][s4].dict.end())
					{
						Aquartett[s1][s2][s3][s4].block[qAquartett->second] += Mtmp;
					}
					else
					{
						Aquartett[s1][s2][s3][s4].push_back(qupleAquartett, Mtmp);
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void split_AA (DMRG::DIRECTION::OPTION DIR, const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair,
			   const vector<qarray<Symmetry::Nq> >& qloc_l, vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Al,
			   const vector<qarray<Symmetry::Nq> >& qloc_r, vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Ar,
			   const qarray<Symmetry::Nq>& qtop, const qarray<Symmetry::Nq>& qbot, double eps_svd, size_t min_Nsv, size_t max_Nsv)
{
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Cdump;
	double truncDump, Sdump;
	split_AA(DIR, Apair, qloc_l, Al, qloc_r, Ar, qtop, qbot,
			 Cdump, false, truncDump, Sdump, eps_svd,min_Nsv,max_Nsv);
}
			   
template<typename Symmetry, typename Scalar>
void split_AA (DMRG::DIRECTION::OPTION DIR, const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair,
			   const vector<qarray<Symmetry::Nq> >& qloc_l, vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Al,
			   const vector<qarray<Symmetry::Nq> >& qloc_r, vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Ar,
			   const qarray<Symmetry::Nq>& qtop, const qarray<Symmetry::Nq>& qbot,
			   Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &C, bool SEPARATE_SV, double &truncWeight, double &S, double eps_svd, size_t min_Nsv, size_t max_Nsv)
{
	vector<qarray<Symmetry::Nq> > midset = calc_qsplit(Al, qloc_l, Ar, qloc_r, qtop, qbot);

	for (size_t s=0; s<qloc_l.size(); ++s)
	{
		Al[s].clear();
	}
	for (size_t s=0; s<qloc_r.size(); ++s)
	{
		Ar[s].clear();
	}
	
	ArrayXd truncWeightSub(midset.size()); truncWeightSub.setZero();
	ArrayXd entropySub(midset.size()); entropySub.setZero();
	
	auto tensor_basis = Symmetry::tensorProd(qloc_l, qloc_r);
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
	for (size_t qmid=0; qmid<midset.size(); ++qmid)
	{
		map<pair<size_t,qarray<Symmetry::Nq> >,vector<pair<size_t,qarray<Symmetry::Nq> > > > s13map;
		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<Scalar> > cgcmap;
		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<size_t> > q13map;
		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<size_t> > s1s3map;
		
		for (size_t s1=0; s1<qloc_l.size(); ++s1)
		for (size_t s3=0; s3<qloc_r.size(); ++s3)
		{
			auto qmerges = Symmetry::reduceSilent(qloc_l[s1], qloc_r[s3]);
			
			for (const auto &qmerge:qmerges)
			{
				auto qtensor = make_tuple(qloc_l[s1], s1, qloc_r[s3], s3, qmerge);
				auto s1s3 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor));
				
				for (size_t q13=0; q13<Apair[s1s3].dim; ++q13)
				{
					auto qlmids = Symmetry::reduceSilent(Apair[s1s3].in[q13], qloc_l[s1]);
					auto qrmids = Symmetry::reduceSilent(Apair[s1s3].out[q13], Symmetry::flip(qloc_r[s3]));
					
					for (const auto &qlmid:qlmids)
					for (const auto &qrmid:qrmids)
					{
						if (qlmid == midset[qmid] and qrmid == midset[qmid])
						{
							s13map[make_pair(s1,Apair[s1s3].in[q13])].push_back(make_pair(s3,Apair[s1s3].out[q13]));
							
							Scalar factor_cgc = Symmetry::coeff_Apair(Apair[s1s3].in[q13], qloc_l[s1], midset[qmid], 
							                                          qloc_r[s3], Apair[s1s3].out[q13], qmerge);
							if (DIR==DMRG::DIRECTION::LEFT)
							{
								factor_cgc *= Symmetry::coeff_leftSweep(Apair[s1s3].out[q13],
																		midset[qmid]);
							}
							
							cgcmap[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(factor_cgc);
							q13map[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(q13);
							s1s3map[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(s1s3);
						}
					}
				}
			}
		}
		
		if (s13map.size() != 0)
		{
			map<pair<size_t,qarray<Symmetry::Nq> >,Matrix<Scalar,Dynamic,Dynamic> > Aclumpvec;
			size_t istitch = 0;
			size_t jstitch = 0;
			vector<size_t> get_s3;
			vector<size_t> get_Ncols;
			vector<qarray<Symmetry::Nq> > get_qr;
			bool COLS_ARE_KNOWN = false;
			
			for (size_t s1=0; s1<qloc_l.size(); ++s1)
			{
				auto qls = Symmetry::reduceSilent(midset[qmid], Symmetry::flip(qloc_l[s1]));
				
				for (const auto &ql:qls)
				{
					for (size_t s3=0; s3<qloc_r.size(); ++s3)
					{
						auto qrs = Symmetry::reduceSilent(midset[qmid], qloc_r[s3]);
						
						for (const auto &qr:qrs)
						{
							auto s3block = find(s13map[make_pair(s1,ql)].begin(), s13map[make_pair(s1,ql)].end(), make_pair(s3,qr));
							
							if (s3block != s13map[make_pair(s1,ql)].end())
							{
								Matrix<Scalar,Dynamic,Dynamic>  Mtmp;
								for (size_t i=0; i<q13map[make_tuple(s1,ql,s3,qr)].size(); ++i)
								{
									size_t q13 = q13map[make_tuple(s1,ql,s3,qr)][i];
									size_t s1s3 = s1s3map[make_tuple(s1,ql,s3,qr)][i];
									
									if (Mtmp.size() == 0)
									{
										Mtmp = cgcmap[make_tuple(s1,ql,s3,qr)][i] * Apair[s1s3].block[q13];
									}
									else if (Mtmp.size() > 0 and Apair[s1s3].block[q13].size() > 0)
									{
										Mtmp += cgcmap[make_tuple(s1,ql,s3,qr)][i] * Apair[s1s3].block[q13];
									}
								}
								if (Mtmp.size() == 0) {continue;}
								
								addRight(Mtmp, Aclumpvec[make_pair(s1,ql)]);
								
								if (COLS_ARE_KNOWN == false)
								{
									get_s3.push_back(s3);
									get_Ncols.push_back(Mtmp.cols());
									get_qr.push_back(qr);
								}
							}
						}
					}
					if (get_s3.size() != 0) {COLS_ARE_KNOWN = true;}
				}
			}
			
			vector<size_t> get_s1;
			vector<size_t> get_Nrows;
			vector<qarray<Symmetry::Nq> > get_ql;
			Matrix<Scalar,Dynamic,Dynamic>  Aclump;
			for (size_t s1=0; s1<qloc_l.size(); ++s1)
			{
				auto qls = Symmetry::reduceSilent(midset[qmid], Symmetry::flip(qloc_l[s1]));
				
				for (const auto &ql:qls)
				{
					size_t Aclump_rows_old = Aclump.rows();
					
					// If cols don't match, it means that zeros were cut, restore them 
					// (happens in MpsCompressor::polyCompress):
					if (Aclumpvec[make_pair(s1,ql)].cols() < Aclump.cols())
					{
						size_t dcols = Aclump.cols() - Aclumpvec[make_pair(s1,ql)].cols();
						Aclumpvec[make_pair(s1,ql)].conservativeResize(Aclumpvec[make_pair(s1,ql)].rows(), Aclump.cols());
						Aclumpvec[make_pair(s1,ql)].rightCols(dcols).setZero();
					}
					else if (Aclumpvec[make_pair(s1,ql)].cols() > Aclump.cols())
					{
						size_t dcols = Aclumpvec[make_pair(s1,ql)].cols() - Aclump.cols();
						Aclump.conservativeResize(Aclump.rows(), Aclump.cols()+dcols);
						Aclump.rightCols(dcols).setZero();
					}
					
					addBottom(Aclumpvec[make_pair(s1,ql)], Aclump);
					
					if (Aclump.rows() > Aclump_rows_old)
					{
						get_s1.push_back(s1);
						get_Nrows.push_back(Aclump.rows()-Aclump_rows_old);
						get_ql.push_back(ql);
					}
				}
			}
			if (Aclump.size() == 0)
			{
//				if (DIR == DMRG::DIRECTION::RIGHT)
//				{
//					this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
//				}
//				else
//				{
//					this->pivot = (loc==0)? 0 : loc;
//				}
				continue;
			}
			
			#ifdef DONT_USE_BDCSVD
			JacobiSVD<Matrix<Scalar,Dynamic,Dynamic> > Jack; // standard SVD
			#else
			BDCSVD<Matrix<Scalar,Dynamic,Dynamic> > Jack; // "Divide and conquer" SVD (only available in Eigen)
			#endif
			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
			VectorXd SV = Jack.singularValues();
			
			// retained states:
			size_t Nret = (SV.array().abs() > eps_svd).count();
			Nret = max(Nret, min_Nsv);
			Nret = min(Nret, max_Nsv);
			truncWeightSub(qmid) = Symmetry::degeneracy(midset[qmid]) * SV.tail(SV.rows()-Nret).cwiseAbs2().sum();
			size_t Nnz = (Jack.singularValues().array() > 0.).count();
			entropySub(qmid) = -Symmetry::degeneracy(midset[qmid]) *
                 			   (SV.head(Nnz).array().square() * SV.head(Nnz).array().square().log()).sum();
			
			Matrix<Scalar,Dynamic,Dynamic>  Aleft, Aright, Cmatrix;
			if (DIR == DMRG::DIRECTION::RIGHT)
			{
				Aleft = Jack.matrixU().leftCols(Nret);
				if (SEPARATE_SV)
				{
					Aright = Jack.matrixV().adjoint().topRows(Nret);
					Cmatrix = Jack.singularValues().head(Nret).asDiagonal();
				}
				else
				{
					Aright = Jack.singularValues().head(Nret).asDiagonal() * Jack.matrixV().adjoint().topRows(Nret);
				}
//				this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
			}
			else
			{
				Aright = Jack.matrixV().adjoint().topRows(Nret);
				if (SEPARATE_SV)
				{
					Aleft = Jack.matrixU().leftCols(Nret);
					Cmatrix = Jack.singularValues().head(Nret).asDiagonal();
				}
				else
				{
					Aleft = Jack.matrixU().leftCols(Nret) * Jack.singularValues().head(Nret).asDiagonal();
				}
//				this->pivot = (loc==0)? 0 : loc;
			}
			
			// update Al
			istitch = 0;
			for (size_t i=0; i<get_s1.size(); ++i)
			{
				size_t s1 = get_s1[i];
				size_t Nrows = get_Nrows[i];
				
				qarray2<Symmetry::Nq> quple = {get_ql[i], midset[qmid]};
				auto q = Al[s1].dict.find(quple);
				if (q != Al[s1].dict.end())
				{
					Al[s1].block[q->second] += Aleft.block(istitch,0, Nrows,Nret);
				}
				else
				{
					Al[s1].push_back(get_ql[i], midset[qmid], Aleft.block(istitch,0, Nrows,Nret));
				}
				istitch += Nrows;
			}
			
			// update Ar
			jstitch = 0;
			for (size_t i=0; i<get_s3.size(); ++i)
			{
				size_t s3 = get_s3[i];
				size_t Ncols = get_Ncols[i];
				
				qarray2<Symmetry::Nq> quple = {midset[qmid], get_qr[i]};
				auto q = Ar[s3].dict.find(quple);
				Scalar factor_cgc3 = (DIR==DMRG::DIRECTION::LEFT)? Symmetry::coeff_leftSweep(midset[qmid],
																							 get_qr[i]):1.;
				if (q != Ar[s3].dict.end())
				{
					Ar[s3].block[q->second] += factor_cgc3 * Aright.block(0,jstitch, Nret,Ncols);
				}
				else
				{
					Ar[s3].push_back(midset[qmid], get_qr[i], factor_cgc3 * Aright.block(0,jstitch, Nret,Ncols));
				}
				jstitch += Ncols;
			}
			
			if (SEPARATE_SV)
			{
				qarray2<Symmetry::Nq> quple = {midset[qmid], midset[qmid]};
				auto q = C.dict.find(quple);
				if (q != C.dict.end())
				{
					C.block[q->second] += Cmatrix;
				}
				else
				{
					C.push_back(midset[qmid], midset[qmid], Cmatrix);
				}
			}
		}
	}
	
	// remove unwanted zero-sized blocks
	for (size_t s=0; s<qloc_l.size(); ++s)
	{
		Al[s] = Al[s].cleaned();
	}
	for (size_t s=0; s<qloc_r.size(); ++s)
	{
		Ar[s] = Ar[s].cleaned();
	}
	
	truncWeight = truncWeightSub.sum();
	S = entropySub.sum();
	// if (DIR == DMRG::DIRECTION::RIGHT)
	// {
	// 	int bond = (loc==this->N_sites-1)? -1 : loc;
	// 	if (bond != -1)
	// 	{
	// 		S(loc) = entropySub.sum();
	// 	}
	// }
	// else
	// {
	// 	int bond = (loc==0)? -1 : loc;
	// 	if (bond != -1)
	// 	{
	// 		S(loc-1) = entropySub.sum();
	// 	}
	// }
}

#endif
