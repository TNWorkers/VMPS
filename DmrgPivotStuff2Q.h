#ifndef STRAWBERRY_DMRGHEFFSTUFF2SITE_WITH_Q
#define STRAWBERRY_DMRGHEFFSTUFF2SITE_WITH_Q

template<size_t Nq, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrixQ<Nq,Scalar,MpoScalar> &H1, 
          const PivotMatrixQ<Nq,Scalar,MpoScalar> &H2, 
          const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Aket1, 
          const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Aket2, 
          const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Abra1, 
          const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Abra2, 
          const vector<qarray<Nq> > &qloc1, const vector<qarray<Nq> > &qloc2, 
          vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > &Apair)
{
//	vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > Apair;
	Apair.resize(qloc1.size());
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	{
		Apair[s1].resize(qloc2.size());
	}
	
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t s2=0; s2<qloc1.size(); ++s2)
	for (size_t qL=0; qL<H1.L.dim; ++qL)
	{
		tuple<qarray3<Nq>,size_t,size_t> ix12;
		bool FOUND_MATCH12 = AWA(H1.L.in(qL), H1.L.out(qL), H1.L.mid(qL), s1, s2, qloc1, Abra1, Aket1, ix12);
		
		if (FOUND_MATCH12)
		{
			qarray3<Nq> quple12 = get<0>(ix12);
			swap(quple12[0], quple12[1]);
			size_t qA12 = get<2>(ix12);
			
			for (size_t s3=0; s3<qloc2.size(); ++s3)
			for (size_t s4=0; s4<qloc2.size(); ++s4)
			{
				tuple<qarray3<Nq>,size_t,size_t> ix34;
				bool FOUND_MATCH34 = AWA(quple12[0], quple12[1], quple12[2], s3, s4, qloc2, Abra2, Aket2, ix34);
				
				if (FOUND_MATCH34)
				{
					qarray3<Nq> quple34 = get<0>(ix34);
					size_t qA34 = get<2>(ix34);
					auto qR = H2.R.dict.find(quple34);
					
					if (qR != H2.R.dict.end())
					{
						if (H1.L.mid(qL) + qloc1[s1] - qloc1[s2] == 
						    H2.R.mid(qR->second) - qloc2[s3] + qloc2[s4])
						{
							for (int k12=0; k12<H1.W[s1][s2].outerSize(); ++k12)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(H1.W[s1][s2],k12); iW12; ++iW12)
							for (int k34=0; k34<H2.W[s3][s4].outerSize(); ++k34)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(H2.W[s3][s4],k34); iW34; ++iW34)
							{
								Matrix<Scalar,Dynamic,Dynamic> Mtmp;
								MpoScalar Wfactor = iW12.value() * iW34.value();
								
								if (H1.L.block[qL][iW12.row()][0].rows() != 0 and
									H2.R.block[qR->second][iW34.col()][0].rows() !=0 and
									iW12.col() == iW34.row())
								{
//									Mtmp = Wfactor * 
//									       (H1.L.block[qL][iW12.row()][0] * 
//									       Aket1[loc1][s2].block[qA12] * 
//									       Aket2[s4].block[qA34] * 
//									       H2.R.block[qR->second][iW34.col()][0]);
									optimal_multiply(Wfactor, 
									                 H1.L.block[qL][iW12.row()][0],
									                 Aket1[s2].block[qA12],
									                 Aket2[s4].block[qA34],
									                 H2.R.block[qR->second][iW34.col()][0],
									                 Mtmp);
								}
								
								if (Mtmp.rows() != 0)
								{
									qarray2<Nq> qupleApair = {H1.L.in(qL), H2.R.out(qR->second)};
									auto qApair = Apair[s1][s3].dict.find(qupleApair);
									
									if (qApair != Apair[s1][s3].dict.end())
									{
										Apair[s1][s3].block[qApair->second] += Mtmp;
									}
									else
									{
										Apair[s1][s3].push_back(qupleApair, Mtmp);
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

#endif
