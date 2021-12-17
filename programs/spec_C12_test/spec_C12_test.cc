using namespace std;

enum DAMPING {GAUSS, LORENTZ, NODAMPING};

size_t L;
double tmax, dt;
int tpoints;

// continuous Fourier transform using Ooura integration, with t=tvals, and a complex column of data
// wmin: minimal frequency
// wmax: maximal frequency
// wpoints: number of frequency points
VectorXcd FT (const VectorXd &tvals, double tmax, const VectorXcd &data, double wmin, double wmax, int wpoints, DAMPING DAMPING_input=LORENTZ, string filename="")
{
	boost::math::quadrature::ooura_fourier_sin<double> OouraSin = boost::math::quadrature::ooura_fourier_sin<double>();
	boost::math::quadrature::ooura_fourier_cos<double> OouraCos = boost::math::quadrature::ooura_fourier_cos<double>();
	Interpol<GSL> InterpRe(tvals);
	Interpol<GSL> InterpIm(tvals);
	for (int it=0; it<tvals.rows(); ++it)
	{
		InterpRe.insert(it,data(it).real());
		InterpIm.insert(it,data(it).imag());
	}
	InterpRe.set_splines();
	InterpIm.set_splines();
	
	auto fRe = [&InterpRe, &tmax, &DAMPING_input](double t)
	{
		if (t>tmax) return 0.;
//		else
//		{
//			if (DAMPING_input == GAUSS)        return InterpRe(t)*exp(-pow(2.*t/tmax,2));
//			else if (DAMPING_input == LORENTZ) return InterpRe(t)*exp(-4.*t/tmax);
//			else                               return InterpRe(t);
//		}
		else
		{
			return InterpRe(t)*exp(-4.*t/tmax);
		}
	};
	auto fIm = [&InterpIm, &tmax, &DAMPING_input](double t)
	{
		if (t>tmax) return 0.;
//		else
//		{
//			if (DAMPING_input == GAUSS)        return InterpIm(t)*exp(-pow(2.*t/tmax,2));
//			else if (DAMPING_input == LORENTZ) return InterpIm(t)*exp(-4.*t/tmax);
//			else                               return InterpIm(t);
//		}
		else
		{
			return InterpIm(t)*exp(-4.*t/tmax);
		}
	};
	
	double resReSin, resReCos, resImSin, resImCos;
	IntervalIterator w(wmin,wmax,wpoints);
	for (w=w.begin(2); w!=w.end(); ++w)
	{
		double wval = *w;
		complex<double> dataw;
		if (wval == 0.)
		{
			dataw = InterpIm.integrate() + 1.i * InterpIm.integrate();
		}
		else
		{
			resReSin = OouraSin.integrate(fRe,wval).first;
			resReCos = OouraCos.integrate(fRe,wval).first;
			resImSin = OouraSin.integrate(fIm,wval).first;
			resImCos = OouraCos.integrate(fIm,wval).first;
			dataw = resReCos-resImSin + 1.i*(resReSin+resImCos);
		}
		w << dataw;
	}
	if (filename!="")
	{
		w.save(filename);
	}
	InterpRe.kill_splines();
	InterpIm.kill_splines();
	
	return w.get_data().col(1)+1.i*w.get_data().col(2);
}

vector<vector<VectorXcd> > Gt_anUPcrUP;
vector<vector<VectorXcd> > Gt_crUPanUP;
vector<vector<VectorXcd> > Gt_anDNanUP;
vector<vector<VectorXcd> > Gt_crDNcrUP;

vector<vector<VectorXcd> > Gt_anDNcrDN;
vector<vector<VectorXcd> > Gt_crDNanDN;
vector<vector<VectorXcd> > Gt_anUPanDN;
vector<vector<VectorXcd> > Gt_crUPcrDN;

void compute_G (const MODEL &Hbra_annihilate, const MODEL &Hbra_create, const MODEL &Hket, const Eigenstate<RealVector> &g, double tmax, int tpoints)
{
	vector<DECONSTRUCTION> D_TYPES = {ANNIHILATE, CREATE};
	vector<double> t_dir = {+1.,-1.}; // +1:forwards, -1:backwards
	
	Gt_anUPcrUP.resize(L);
	Gt_crUPanUP.resize(L);
	Gt_anDNanUP.resize(L);
	Gt_crDNcrUP.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gt_anUPcrUP[i].resize(L);
		Gt_crUPanUP[i].resize(L);
		Gt_anDNanUP[i].resize(L);
		Gt_crDNcrUP[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gt_anUPcrUP[i][j].resize(tpoints); Gt_anUPcrUP[i][j].setZero();
			Gt_crUPanUP[i][j].resize(tpoints); Gt_crUPanUP[i][j].setZero();
			Gt_anDNanUP[i][j].resize(tpoints); Gt_anDNanUP[i][j].setZero();
			Gt_crDNcrUP[i][j].resize(tpoints); Gt_crDNcrUP[i][j].setZero();
		}
	}
	
	Gt_anDNcrDN.resize(L);
	Gt_crDNanDN.resize(L);
	Gt_anUPanDN.resize(L);
	Gt_crUPcrDN.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gt_anDNcrDN[i].resize(L);
		Gt_crDNanDN[i].resize(L);
		Gt_anUPanDN[i].resize(L);
		Gt_crUPcrDN[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gt_anDNcrDN[i][j].resize(tpoints); Gt_anDNcrDN[i][j].setZero();
			Gt_crDNanDN[i][j].resize(tpoints); Gt_crDNanDN[i][j].setZero();
			Gt_anUPanDN[i][j].resize(tpoints); Gt_anUPanDN[i][j].setZero();
			Gt_crUPcrDN[i][j].resize(tpoints); Gt_crUPcrDN[i][j].setZero();
		}
	}
	
	vector<ComplexVector> init_cUP(L);
	vector<ComplexVector> init_cdagUP(L);
	
	vector<ComplexVector> init_cDN(L);
	vector<ComplexVector> init_cdagDN(L);
	
	// apply creator/annihilator to the initial state
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int d=0; d<D_TYPES.size(); ++d)
	{
		DECONSTRUCTION D_TYPE = D_TYPES[d];
		
		if (D_TYPE == ANNIHILATE)
		{
			RealVector Vtmp1, Vtmp2;
			#if defined(USING_ED)
			{
				ED::Photo PhUP(i,UP,ANNIHILATE,Hbra_annihilate,Hket);
				OxV(PhUP,g.state,Vtmp1);
				
				ED::Photo PhDN(i,DN,ANNIHILATE,Hbra_create,Hket); // annihilate DN = create UP
				OxV(PhDN,g.state,Vtmp2);
			}
			#elif defined(USING_U1)
			{
				OxV_exact(Hbra_annihilate.c<UP>(i), g.state, Vtmp1, 2., DMRG::VERBOSITY::SILENT);
				OxV_exact(Hbra_annihilate.c<DN>(i), g.state, Vtmp2, 2., DMRG::VERBOSITY::SILENT);
			}
			#endif
			init_cUP[i] = Vtmp1.cast<complex<double> >();
			init_cDN[i] = Vtmp2.cast<complex<double> >();
		}
		else if (D_TYPE == CREATE)
		{
			RealVector Vtmp1, Vtmp2;
			#if defined(USING_ED)
			{
				ED::Photo PhUP(i,UP,CREATE,Hbra_create,Hket);
				OxV(PhUP,g.state,Vtmp1);
				
				ED::Photo PhDN(i,DN,CREATE,Hbra_annihilate,Hket); // create DN = annihilate UP
				OxV(PhDN,g.state,Vtmp2);
			}
			#elif defined(USING_U1)
			{
				OxV_exact(Hbra_annihilate.cdag<UP>(i), g.state, Vtmp1, 2., DMRG::VERBOSITY::SILENT);
				OxV_exact(Hbra_annihilate.cdag<DN>(i), g.state, Vtmp2, 2., DMRG::VERBOSITY::SILENT);
			}
			#endif
			init_cdagUP[i] = Vtmp1.cast<complex<double> >();
			init_cdagDN[i] = Vtmp2.cast<complex<double> >();
		}
	}
	
	#pragma omp parallel for collapse(3)
	for (int j=0; j<L; ++j)
	for (int d=0; d<D_TYPES.size(); ++d)
	for (int z=0; z<t_dir.size(); ++z)
	{
		DECONSTRUCTION D_TYPE = D_TYPES[d];
		double tsign = t_dir[z];
		
		ComplexVector PsiUPj = (D_TYPE==ANNIHILATE)? init_cUP[j] : init_cdagUP[j];
		#if not defined(USING_ED)
		TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,ComplexVector> Lutz(Hbra_annihilate,PsiUPj);
		#endif
		
		ComplexVector PsiDNj = (D_TYPE==ANNIHILATE)? init_cDN[j] : init_cdagDN[j];
		#if not defined(USING_ED)
		TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,ComplexVector> Lucy(Hbra_annihilate,PsiDNj);
		#endif
		
		IntervalIterator t(0.,tmax,tpoints);
		for (t=t.begin(); t!=t.end(); ++t)
		{
			complex<double> phase = -1.i * exp(+1.i*tsign*g.energy*(*t));
			
			// compute Green's function G↑↑ and F↓↑
			for (int i=0; i<L; ++i)
			{
				if (D_TYPE == CREATE and z==0) // create, forwards in time
				{
					Gt_anUPcrUP[i][j](t.index()) += phase * dot(init_cdagUP[i],PsiUPj); // <ci(t),c†j>
					Gt_crDNcrUP[i][j](t.index()) += phase * dot(init_cDN   [i],PsiUPj); // <c†i(t),c†j>
				}
				else if (D_TYPE == CREATE and z==1) // create, backwards in time
				{
					Gt_crUPanUP[j][i](t.index()) += phase * dot(init_cdagUP[i],PsiUPj); // <ci(-t),c†j> & i<->j
					Gt_crUPcrDN[j][i](t.index()) += phase * dot(init_cDN   [i],PsiUPj); // <c†i(-t),c†j> & i<->j
				}
				else if (D_TYPE == ANNIHILATE and z==0) // annihilate, forwards in time
				{
					Gt_crUPanUP[i][j](t.index()) += phase * dot(init_cUP   [i],PsiUPj); // <c†i(t),cj>
					Gt_anDNanUP[i][j](t.index()) += phase * dot(init_cdagDN[i],PsiUPj); // <ci(t),cj>
				}
				else if (D_TYPE == ANNIHILATE and z==1) // annihilate, backwards in time
				{
					Gt_anUPcrUP[j][i](t.index()) += phase * dot(init_cUP   [i],PsiUPj); // <c†i(-t),cj> & i<->j
					Gt_anUPanDN[j][i](t.index()) += phase * dot(init_cdagDN[i],PsiUPj); // <ci(-t),cj> & i<->j  
				}
			}
			
			// compute Green's function G↓↓ and F↑↓
			for (int i=0; i<L; ++i)
			{
				if (D_TYPE == CREATE and z==0) // create, forwards in time
				{
					Gt_anDNcrDN[i][j](t.index()) += phase * dot(init_cdagDN[i],PsiDNj); // <ci(t),c†j>
					Gt_crUPcrDN[i][j](t.index()) += phase * dot(init_cUP   [i],PsiDNj); // <c†i(t),c†j>
				}
				else if (D_TYPE == CREATE and z==1) // create, backwards in time
				{
					Gt_crDNanDN[j][i](t.index()) += phase * dot(init_cdagDN[i],PsiDNj); // <ci(-t),c†j> & i<->j
					Gt_crDNcrUP[j][i](t.index()) += phase * dot(init_cUP   [i],PsiDNj); // <c†i(-t),c†j> & i<->j
				}
				else if (D_TYPE == ANNIHILATE and z==0) // annihilate, forwards in time
				{
					Gt_crDNanDN[i][j](t.index()) += phase * dot(init_cDN   [i],PsiDNj); // <c†i(t),cj>
					Gt_anUPanDN[i][j](t.index()) += phase * dot(init_cdagUP[i],PsiDNj); // <ci(t),cj>
				}
				else if (D_TYPE == ANNIHILATE and z==1) // annihilate, backwards in time
				{
					Gt_anDNcrDN[j][i](t.index()) += phase * dot(init_cDN   [i],PsiDNj); // <c†i(-t),cj> & i<->j
					Gt_anDNanUP[j][i](t.index()) += phase * dot(init_cdagUP[i],PsiDNj); // <ci(-t),cj> & i<->j  
				}
			}
			
			// propagate
			Stopwatch<> Timer;
			#if defined(USING_ED)
			{
				LanczosPropagator<MODEL,ComplexVector,complex<double> > Lutz(1e-5); // can set higher tolerance, but leads to more computational effort
				(D_TYPE == ANNIHILATE)? Lutz.t_step(Hbra_annihilate, PsiUPj, -1.i*tsign*dt):
				                        Lutz.t_step(Hbra_create,     PsiUPj, -1.i*tsign*dt);
				if (j==0 and d==0 and z==0)
				{
					lout << Timer.info("timestep") << endl;
					lout << *t << "\t" << Lutz.info() << endl;
				}
				
				LanczosPropagator<MODEL,ComplexVector,complex<double> > Lucy(1e-5); // can set higher tolerance, but leads to more computational effort
				(D_TYPE == ANNIHILATE)? Lucy.t_step(Hbra_create,     PsiDNj, -1.i*tsign*dt): // create DN = annihilate UP
				                        Lucy.t_step(Hbra_annihilate, PsiDNj, -1.i*tsign*dt);
			}
			#else
			{
				Lutz.t_step(Hbra_annihilate, PsiUPj, -1.i*tsign*dt);
				if (j==0 and d==0 and z==0)
				{
					lout << Timer.info("timestep") << endl;
					lout << *t << "\t" << Lutz.info() << endl;
					lout << PsiUPj.info() << endl;
				}
				
				Lucy.t_step(Hbra_annihilate, PsiDNj, -1.i*tsign*dt);
			}
			#endif
		}
	}
}

vector<vector<VectorXcd> > Gw_anUPcrUP;
vector<vector<VectorXcd> > Gw_crUPanUP;
vector<vector<VectorXcd> > Gw_anDNanUP;
vector<vector<VectorXcd> > Gw_crDNcrUP;

vector<vector<VectorXcd> > Gw_anDNcrDN;
vector<vector<VectorXcd> > Gw_crDNanDN;
vector<vector<VectorXcd> > Gw_anUPanDN;
vector<vector<VectorXcd> > Gw_crUPcrDN;

void Fourier_ancr (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_anUPcrUP.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_anUPcrUP[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anUPcrUP[i][j].resize(tpoints); Gw_anUPcrUP[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_anUPcrUP[i][j] = FT(tvals, tmax, Gt_anUPcrUP[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_anUPcrUP","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
	
	///////////////
	
	Gw_anDNcrDN.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_anDNcrDN[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anDNcrDN[i][j].resize(tpoints); Gw_anDNcrDN[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_anDNcrDN[i][j] = FT(tvals, tmax, Gt_anDNcrDN[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_anDNcrDN","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

void Fourier_cran (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_crUPanUP.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_crUPanUP[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crUPanUP[i][j].resize(tpoints); Gw_crUPanUP[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_crUPanUP[i][j] = FT(tvals, tmax, Gt_crUPanUP[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_crUPanUP","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
	
	///////////////
	
	Gw_crDNanDN.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_crDNanDN[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crDNanDN[i][j].resize(tpoints); Gw_crDNanDN[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_crDNanDN[i][j] = FT(tvals, tmax, Gt_crDNanDN[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_crDNanDN","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

void Fourier_anan (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_anDNanUP.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_anDNanUP[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anDNanUP[i][j].resize(tpoints); Gw_anDNanUP[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_anDNanUP[i][j] = FT(tvals, tmax, Gt_anDNanUP[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_anDNanUP","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
	
	///////////////
	
	Gw_anUPanDN.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_anUPanDN[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anUPanDN[i][j].resize(tpoints); Gw_anUPanDN[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_anUPanDN[i][j] = FT(tvals, tmax, Gt_anUPanDN[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_anUPanDN","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

void Fourier_crcr (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_crDNcrUP.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_crDNcrUP[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crDNcrUP[i][j].resize(tpoints); Gw_crDNcrUP[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_crDNcrUP[i][j] = FT(tvals, tmax, Gt_crDNcrUP[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_crDNcrUP","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
	
	///////////////
	
	Gw_crUPcrDN.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_crUPcrDN[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crUPcrDN[i][j].resize(tpoints); Gw_crUPcrDN[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_crUPcrDN[i][j] = FT(tvals, tmax, Gt_crUPcrDN[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("Gw_crUPcrDN","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

vector<vector<ComplexInterpol> > Gw_anUPcrUP_interpol;
vector<vector<ComplexInterpol> > Gw_crUPanUP_interpol;
vector<vector<ComplexInterpol> > Gw_anDNanUP_interpol;
vector<vector<ComplexInterpol> > Gw_crDNcrUP_interpol;

vector<vector<ComplexInterpol> > Gw_anDNcrDN_interpol;
vector<vector<ComplexInterpol> > Gw_crDNanDN_interpol;
vector<vector<ComplexInterpol> > Gw_anUPanDN_interpol;
vector<vector<ComplexInterpol> > Gw_crUPcrDN_interpol;

void set_splines_ancr (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_anUPcrUP_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_anUPcrUP_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anUPcrUP_interpol[i][j] = ComplexInterpol(wvals);
			Gw_anUPcrUP_interpol[i][j] = Gw_anUPcrUP[i][j];
			Gw_anUPcrUP_interpol[i][j].set_splines();
		}
	}
	
	///////////////
	
	Gw_anDNcrDN_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_anDNcrDN_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anDNcrDN_interpol[i][j] = ComplexInterpol(wvals);
			Gw_anDNcrDN_interpol[i][j] = Gw_anDNcrDN[i][j];
			Gw_anDNcrDN_interpol[i][j].set_splines();
		}
	}
}

void set_splines_cran (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_crUPanUP_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_crUPanUP_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crUPanUP_interpol[i][j] = ComplexInterpol(wvals);
			Gw_crUPanUP_interpol[i][j] = Gw_crUPanUP[i][j];
			Gw_crUPanUP_interpol[i][j].set_splines();
		}
	}
	
	///////////////
	
	Gw_crDNanDN_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_crDNanDN_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crDNanDN_interpol[i][j] = ComplexInterpol(wvals);
			Gw_crDNanDN_interpol[i][j] = Gw_crDNanDN[i][j];
			Gw_crDNanDN_interpol[i][j].set_splines();
		}
	}
}

void set_splines_anan (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_anDNanUP_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_anDNanUP_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anDNanUP_interpol[i][j] = ComplexInterpol(wvals);
			Gw_anDNanUP_interpol[i][j] = Gw_anDNanUP[i][j];
			Gw_anDNanUP_interpol[i][j].set_splines();
		}
	}
	
	///////////////
	
	Gw_anUPanDN_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_anUPanDN_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anUPanDN_interpol[i][j] = ComplexInterpol(wvals);
			Gw_anUPanDN_interpol[i][j] = Gw_anUPanDN[i][j];
			Gw_anUPanDN_interpol[i][j].set_splines();
		}
	}
}

void set_splines_crcr (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_crDNcrUP_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_crDNcrUP_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crDNcrUP_interpol[i][j] = ComplexInterpol(wvals);
			Gw_crDNcrUP_interpol[i][j] = Gw_crDNcrUP[i][j];
			Gw_crDNcrUP_interpol[i][j].set_splines();
		}
	}
	
	///////////////
	
	Gw_crUPcrDN_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_crUPcrDN_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_crUPcrDN_interpol[i][j] = ComplexInterpol(wvals);
			Gw_crUPcrDN_interpol[i][j] = Gw_crUPcrDN[i][j];
			Gw_crUPcrDN_interpol[i][j].set_splines();
		}
	}
}

//MatrixXcd Gfull (double omega)
//{
//	MatrixXcd res00(L,L); res00.setZero();
//	MatrixXcd res11(L,L); res11.setZero();
//	MatrixXcd res01(L,L); res01.setZero();
//	MatrixXcd res10(L,L); res10.setZero();
//	
//	for (int i=0; i<L; ++i)
//	for (int j=0; j<L; ++j)
//	{
//		res00(i,j) = Gw_anUPcrUP_interpol[i][j](omega);
//		res11(i,j) = Gw_crUPanUP_interpol[i][j](omega);
//		res01(i,j) = Gw_anDNanUP_interpol[i][j](omega);
//		res10(i,j) = Gw_crDNcrUP_interpol[i][j](omega);
//	}
//	
//	MatrixXcd res(2*L,2*L);
//	res.topLeftCorner(L,L) = res00;
//	res.bottomRightCorner(L,L) = res11;
//	res.topRightCorner(L,L) = res01;
//	res.bottomLeftCorner(L,L) = res10;
//	
//	return res;
//}

MatrixXcd G_anUPcrUP (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_anUPcrUP_interpol[i][j](omega);
	}
	
	return res;
}

MatrixXcd G_crUPanUP (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_crUPanUP_interpol[i][j](omega);
	}
	
	return res;
}

MatrixXcd G_anDNcrDN (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_anDNcrDN_interpol[i][j](omega);
	}
	
	return res;
}

MatrixXcd G_crDNanDN (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_crDNanDN_interpol[i][j](omega);
	}
	
	return res;
}

MatrixXcd F_anDNanUP (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_anDNanUP_interpol[i][j](omega);
	}
	
	return res;
}

MatrixXcd F_crDNcrUP (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_crDNcrUP_interpol[i][j](omega);
	}
	
	return res;
}

MatrixXcd F_anUPanDN (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_anUPanDN_interpol[i][j](omega);
	}
	
	return res;
}

MatrixXcd F_crUPcrDN (double omega)
{
	MatrixXcd res(L,L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = Gw_crUPcrDN_interpol[i][j](omega);
	}
	
	return res;
}

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",12);
	double U = args.get<double>("U",4.);
	double mu = args.get<double>("mu",0.5*U);
	double C0 = args.get<double>("C0",0.); // onsite superconducting field
	string MOL = args.get<string>("MOL","CHAIN");
	int i = args.get<int>("i",L/2);
	
	double wmin = args.get<double>("wmin",-20.);
	double wmax = args.get<double>("wmin",+20.);
	int wpoints = args.get<int>("wpoints",501);
	
	tmax = args.get<double>("tmax",8.);
	dt = args.get<double>("dt",0.1);
	tpoints = tmax/dt+1;
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	string base = make_string("MOL=",MOL,"_L=",L,"_U=",U,"_mu=",mu);
	
	ArrayXXd tFull;
	if (MOL == "RING") // for testing
	{
		tFull = create_1D_PBC(L);
	}
	if (MOL == "CHAIN") // for testing
	{
		tFull = create_1D_OBC(L);
	}
	else if (MOL == "C12")
	{
		tFull = hopping_fullerene(L);
	}
	else if (MOL == "ZERO")
	{
		tFull = create_1D_PBC(L);
		tFull.setZero();
	}
	lout << "tFull=" << endl << tFull << endl;
	ArrayXXd Cfull(L,L); // superconducting field
	Cfull.setZero();
	#if defined(USING_ED)
	Cfull.matrix().diagonal().setConstant(C0);
	#endif
	vector<Param> params;
	params.push_back({"tFull",tFull});
	params.push_back({"U",U});
	params.push_back({"mu",mu});
	params.push_back({"Cfull",Cfull});
	
	MODEL Hket, Hbra_annihilate, Hbra_create;
	
	#if defined(USING_ED)
	{
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				Hket = MODEL(L,params,ED::N_EVN_M0);
				#pragma omp critical
				{
					lout << "Hket=" << endl << Hket.info() << endl;
				}
			}
			#pragma omp section
			{
				Hbra_annihilate = MODEL(L,params,ED::N_ODD_MM1); // odd particle number with M=-1 (annihilating UP electron)
				#pragma omp critical
				{
					lout << "Hbra_annihilate=" << endl << Hbra_annihilate.info() << endl;
				}
			}
			#pragma omp section
			{
				Hbra_create = MODEL(L,params,ED::N_ODD_MP1); // odd particle number with M=+1 (creating UP electron)
				#pragma omp critical
				{
					lout << "Hbra_create=" << endl << Hbra_create.info() << endl;
				}
			}
		}
	}
	#elif defined(USING_U1)
	{
		params.push_back({"t",0.});
		params.push_back({"C",C0});
		Hket = MODEL(L,params);
		lout << "Hket=" << endl << Hket.info() << endl;
		Hbra_annihilate = Hket;
		Hbra_create = Hket;
	}
	#endif
	
	//-------------Get groundstate in the ket space-------------
	Stopwatch<> Timer;
	Eigenstate<RealVector> g;
	#if defined(USING_ED)
	{
		LanczosSolver<MODEL,RealVector,double> Lutz(LANCZOS::REORTHO::FULL);
		Lutz.ground(Hket, g, 1e-7, 1e-4);
		lout << Lutz.info() << endl;
		
//		Eigenstate<RealVector> g2;
//		RealVector Vtmp;
//		LanczosSolver<MODEL,RealVector,double> Lutz2(LANCZOS::REORTHO::FULL);
//		Lutz2.ground(Hbra_annihilate, g2, 1e-7, 1e-4);
//		lout << "E=" << setprecision(16) << g2.energy << endl;
//		ED::Photo PhUP(i,UP,ANNIHILATE,Hbra_annihilate,Hket);
//		OxV(PhUP,g.state,Vtmp);
//		cout << "<cUP>=" << g2.state.dot(Vtmp) << endl;
//		cout << "<ni>=" << Vtmp.dot(Vtmp) << endl;
//		cout << "dot=" <<  g2.state.dot(g2.state) << "\t" << g.state.dot(g.state) << endl;
//		cout << endl;
//		cout << g2.state << endl;
////		cout << endl;
////		cout << g.state << endl;
//		
//		RealVector v(Hbra_annihilate.dim());
//		v.setRandom();
//		normalize(v);
//		RealVector w(Hbra_annihilate.dim()); w.setZero();
//		HxV(Hbra_annihilate,v,w);
//		cout << endl << w << endl;
	}
	#else
	{
		DMRG::CONTROL::GLOB GlobParams;
		GlobParams.Minit = args.get<size_t>("Minit",2ul);
		GlobParams.Mlimit = args.get<size_t>("Mlimit",500ul);
		GlobParams.Qinit = args.get<size_t>("Qinit",2ul);
		GlobParams.min_halfsweeps = args.get<size_t>("min_halfsweeps",1ul);
		GlobParams.CALC_S_ON_EXIT = false;
		
		MODEL::Solver Lutz(DMRG::VERBOSITY::ON_EXIT);
		Lutz.userSetGlobParam();
		Lutz.GlobParam = GlobParams;
		Lutz.edgeState(Hket, g, {0});
		lout << Lutz.info() << endl;
//		
//		Eigenstate<RealVector> g2;
//		MODEL::Solver Lutz2(DMRG::VERBOSITY::ON_EXIT);
//		Lutz2.userSetGlobParam();
//		Lutz2.GlobParam = GlobParams;
//		Lutz2.edgeState(Hket, g2, {-1});
//		
//		MODEL::Solver Lutz3(DMRG::VERBOSITY::ON_EXIT);
//		Lutz3.userSetGlobParam();
//		Lutz3.GlobParam = GlobParams;
//		Lutz3.push_back(g2.state);
//		Lutz3.edgeState(Hket, g2, {-1});
//		
//		cout << "<cUP>=" << avg(g2.state, Hket.c<UP>(i), g.state) << endl;
//		cout << "<nUP>=" << avg(g.state, Hket.n<UP>(i), g.state) << endl;
	}
	#endif
	lout << Timer.info("ground state") << endl;
	lout << "E0=" << setprecision(16) << g.energy << setprecision(6) << endl;
	
	compute_G(Hbra_annihilate, Hbra_create, Hket, g, tmax, tpoints);
	lout << Timer.info("compute G") << endl;
	
	Fourier_ancr(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	lout << Timer.info("FT ancr") << endl;
	Fourier_cran(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	lout << Timer.info("FT cran") << endl;
	Fourier_anan(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	lout << Timer.info("FT anan") << endl;
	Fourier_crcr(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	lout << Timer.info("FT crcr") << endl;
	
	set_splines_ancr(wmin, wmax, wpoints);
	set_splines_cran(wmin, wmax, wpoints);
	set_splines_anan(wmin, wmax, wpoints);
	set_splines_crcr(wmin, wmax, wpoints);
	
	MatrixXcd TestA = G_anUPcrUP(1);
	MatrixXcd TestB = -G_crUPanUP(-1).adjoint();
	MatrixXcd TestC = -G_crUPanUP(-1).conjugate();
	MatrixXcd TestD = -G_crUPanUP(-1).transpose();
	lout << endl;
	lout << "testB=" << (TestA-TestB).norm() << endl;
	lout << "testC=" << (TestA-TestC).norm() << endl;
	lout << "testD=" << (TestA-TestD).norm() << endl;
	lout << "testBC=" << (TestB-TestC).norm() << endl;
	lout << "testAtranspose=" << (TestA-TestA.transpose()).norm() << endl;
	
	// a,DN,a,UP
	MatrixXcd Test1 = F_anDNanUP(1);
	
	MatrixXcd Test2(Test1.rows(),Test1.cols()), Test3(Test1.rows(),Test1.cols()), Test4(Test1.rows(),Test1.cols()), Test5(Test1.rows(),Test1.cols()), Test6(Test1.rows(),Test1.cols());
	Test2.setZero();
	Test3.setZero();
	Test4.setZero();
	Test5.setZero();
	Test6.setZero();
	
	// a,DN,a,UP
	Test2 = F_anDNanUP(-1).transpose();
	lout << endl << "test F_an↓an↑ : F_an↓an↑" << endl;
	lout << "diffnorm12=" << (Test1-Test2).norm() << endl;
	
	// c,DN,c,UP
	Test2 = F_crDNcrUP(-1);
	Test3 = -F_crDNcrUP(-1).transpose();
	Test4 = -F_crDNcrUP(-1).adjoint();
	Test5 = -F_crDNcrUP(-1).conjugate(); // -> this // -> should be this
	Test6 = -F_crDNcrUP(-1).conjugate();
	
	lout << endl << "test F_an↓an↑ : F_cr↓cr↑" << endl;
	lout << "diffnorm12=" << (Test1-Test2).norm() << endl;
	lout << "diffnorm13=" << (Test1-Test3).norm() << endl;
	lout << "diffnorm14=" << (Test1-Test4).norm() << endl;
	lout << "diffnorm15=" << (Test1-Test5).norm() << ", should be 0" << endl;
	lout << "diffnorm16=" << (Test1-Test6).norm() << endl;
	
	// c,UP,c,DN
	Test2 = F_crUPcrDN(1);
	Test3 = F_crUPcrDN(1).transpose(); // -> this
	Test4 = F_crUPcrDN(1).adjoint(); // -> should be this
	Test5 = F_crUPcrDN(1).conjugate();
	Test6 = F_crUPcrDN(-1).conjugate();
	
	lout << endl << "test F_an↓an↑ : F_cr↑cr↓" << endl;
	lout << "diffnorm12=" << (Test1-Test2).norm() << endl;
	lout << "diffnorm13=" << (Test1-Test3).norm() << endl;
	lout << "diffnorm14=" << (Test1-Test4).norm() << ", should be 0" << endl;
	lout << "diffnorm15=" << (Test1-Test5).norm() << endl;
	lout << "diffnorm16=" << (Test1-Test6).norm() << endl;
	
	// a,UP,a,DN
	Test2 = -F_anUPanDN(-1);
	Test3 = -F_anUPanDN(-1).transpose(); // -> should be this
	Test4 = -F_anUPanDN(-1).adjoint();  // -> this
	Test5 = -F_anUPanDN(-1).conjugate();
	Test6 = -F_anUPanDN(1);
	
	lout << endl << Test1 << endl;
	lout << endl << Test2 << endl;
	lout << endl << Test3 << endl;
	lout << endl << Test4 << endl;
	lout << endl << Test5 << endl;
	lout << endl << Test6 << endl;
	lout << endl;
	
	lout << endl << "test F_an↓an↑ : F_an↑an↓" << endl;
	lout << "diffnorm12=" << (Test1-Test2).norm() << endl;
	lout << "diffnorm13=" << (Test1-Test3).norm() << ", should be 0" << endl;
	lout << "diffnorm14=" << (Test1-Test4).norm() << endl;
	lout << "diffnorm15=" << (Test1-Test5).norm() << endl;
	lout << "diffnorm16=" << (Test1-Test6).norm() << endl;
	
	lout << Timer.info("Fourier transform") << endl;
	
	ofstream Filer("Gt.dat");
	for (int i=0; i<Gt_anDNanUP[0][0].rows(); ++i)
	{
		Filer << i << "\t" << Gt_anDNanUP[0][3](i).real() << "\t" << Gt_anDNanUP[0][3](i).imag() << "\t" 
		                   << Gt_crDNcrUP[0][3](i).real() << "\t" << Gt_crDNcrUP[0][3](i).imag() << "\t" 
		                   << Gt_anUPanDN[0][3](i).real() << "\t" << Gt_anUPanDN[0][3](i).imag() << "\t" 
		                   << Gt_crUPcrDN[0][3](i).real() << "\t" << Gt_crUPcrDN[0][3](i).imag() << endl;
	}
	Filer.close();
	
	MatrixXcd Test1t(L,L);
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Test1t(i,j) = Gt_anDNanUP[i][j][10];
	}
	
	MatrixXcd Test2t(L,L);
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Test2t(i,j) = Gt_anUPanDN[i][j][10];
	}
	
	MatrixXcd Test3t(L,L);
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Test3t(i,j) = Gt_crDNcrUP[i][j][10];
	}
	
	MatrixXcd Test4t(L,L);
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Test4t(i,j) = Gt_crUPcrDN[i][j][10];
	}
	
	lout << endl << Test1t << endl;
	lout << endl << Test2t << endl;
	lout << endl << Test3t << endl;
	lout << endl << Test4t << endl;
	
	lout << (Test1t-Test4t.conjugate()).norm() << endl;
	lout << (Test1t+Test2t).norm() << endl;
	lout << (Test1t+Test3t.conjugate()).norm() << endl;
}
