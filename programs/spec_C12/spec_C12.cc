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
		else
		{
			if (DAMPING_input == GAUSS)        return InterpRe(t)*exp(-pow(2.*t/tmax,2));
			else if (DAMPING_input == LORENTZ) return InterpRe(t)*exp(-4.*t/tmax);
			else                               return InterpRe(t);
		}
	};
	auto fIm = [&InterpIm, &tmax, &DAMPING_input](double t)
	{
		if (t>tmax) return 0.;
		else
		{
			if (DAMPING_input == GAUSS)        return InterpIm(t)*exp(-pow(2.*t/tmax,2));
			else if (DAMPING_input == LORENTZ) return InterpIm(t)*exp(-4.*t/tmax);
			else                               return InterpIm(t);
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

vector<vector<VectorXcd> > Gt_norm00;
vector<vector<VectorXcd> > Gt_norm11;
vector<vector<VectorXcd> > Gt_anom01;
vector<vector<VectorXcd> > Gt_anom10;

void compute_G (const MODEL &Hbra_annihilate, const MODEL &Hbra_create, const MODEL &Hket, const Eigenstate<RealVector> &g, double tmax, int tpoints)
{
	vector<DECONSTRUCTION> D_TYPES = {ANNIHILATE, CREATE};
	vector<double> t_dir = {+1.,-1.}; // +1:forwards, -1:backwards
	
	Gt_norm00.resize(L);
	Gt_norm11.resize(L);
	Gt_anom01.resize(L);
	Gt_anom10.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gt_norm00[i].resize(L);
		Gt_norm11[i].resize(L);
		Gt_anom01[i].resize(L);
		Gt_anom10[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gt_norm00[i][j].resize(tpoints); Gt_norm00[i][j].setZero();
			Gt_norm11[i][j].resize(tpoints); Gt_norm11[i][j].setZero();
			Gt_anom01[i][j].resize(tpoints); Gt_anom01[i][j].setZero();
			Gt_anom10[i][j].resize(tpoints); Gt_anom10[i][j].setZero();
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
				
				ED::Photo PhDN(i,DN,ANNIHILATE,Hbra_create,Hket);
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
				
				ED::Photo PhDN(i,DN,CREATE,Hbra_annihilate,Hket);
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
		
		IntervalIterator t(0.,tmax,tpoints);
		for (t=t.begin(); t!=t.end(); ++t)
		{
			complex<double> phase = -1.i * exp(+1.i*tsign*g.energy*(*t));
			
			// compute Green's function
			for (int i=0; i<L; ++i)
			{
				if (D_TYPE == CREATE and z==0) // create, forwards in time
				{
					Gt_norm00[i][j](t.index()) += phase * dot(init_cdagUP[i],PsiUPj); // <ci(t),c†j>
					Gt_anom10[i][j](t.index()) += phase * dot(init_cDN   [i],PsiUPj); // <c†i(t),c†j>
				}
				else if (D_TYPE == CREATE and z==1) // create, backwards in time
				{
					Gt_norm11[j][i](t.index()) += phase * dot(init_cdagUP[i],PsiUPj); // <ci(-t),c†j> & i<->j
					Gt_anom10[j][i](t.index()) += phase * dot(init_cDN   [i],PsiUPj); // <c†i(-t),c†j> & i<->j
				}
				else if (D_TYPE == ANNIHILATE and z==0) // annihilate, forwards in time
				{
					Gt_norm11[i][j](t.index()) += phase * dot(init_cUP   [i],PsiUPj); // <c†i(t),cj>
					Gt_anom01[i][j](t.index()) += phase * dot(init_cdagDN[i],PsiUPj); // <ci(t),cj>
				}
				else if (D_TYPE == ANNIHILATE and z==1) // annihilate, backwards in time
				{
					Gt_norm00[j][i](t.index()) += phase * dot(init_cUP   [i],PsiUPj); // <c†i(-t),cj> & i<->j
					Gt_anom01[j][i](t.index()) += phase * dot(init_cdagDN[i],PsiUPj); // <c†i(-t),cj> & i<->j 
				}
			}
			
			// propagate
			Stopwatch<> Timer;
			// measure time of one MVM to test:
//			VectorXcd Vtmp;
//			(D_TYPE == ANNIHILATE)? HxV(Hbra_annihilate,PsiUPj,Vtmp):HxV(Hbra_create,PsiUPj,Vtmp);
//			lout << Timer.info("MVM") << endl;
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
			}
			#endif
		}
	}
}

vector<vector<VectorXcd> > Gw_norm00;
vector<vector<VectorXcd> > Gw_norm11;
vector<vector<VectorXcd> > Gw_anom01;
vector<vector<VectorXcd> > Gw_anom10;

void Fourier_G00 (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_norm00.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_norm00[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_norm00[i][j].resize(tpoints); Gw_norm00[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_norm00[i][j] = FT(tvals, tmax, Gt_norm00[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("norm00","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

void Fourier_G11 (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_norm11.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_norm11[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_norm11[i][j].resize(tpoints); Gw_norm11[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_norm11[i][j] = FT(tvals, tmax, Gt_norm11[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("norm11","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

void Fourier_G01 (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_anom01.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_anom01[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anom01[i][j].resize(tpoints); Gw_anom01[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_anom01[i][j] = FT(tvals, tmax, Gt_anom01[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("anom01","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

void Fourier_G10 (double tmax, int tpoints, double wmin, double wmax, int wpoints, bool SAVE=true, DAMPING DAMPING_input=LORENTZ)
{
	IntervalIterator t(0., tmax, tpoints);
	VectorXd tvals = t.get_abscissa();
	
	Gw_anom10.resize(L);
	for (int i=0; i<L; ++i)
	{
		Gw_anom10[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anom10[i][j].resize(tpoints); Gw_anom10[i][j].setZero();
		}
	}
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		Gw_anom10[i][j] = FT(tvals, tmax, Gt_anom10[i][j], wmin, wmax, wpoints, DAMPING_input, (SAVE)?make_string("anom10","_i=",i,"_j=",j,"_DAMPING=LORENTZ",".dat"):"");
	}
}

vector<vector<ComplexInterpol> > Gw_norm00_interpol;
vector<vector<ComplexInterpol> > Gw_norm11_interpol;
vector<vector<ComplexInterpol> > Gw_anom01_interpol;
vector<vector<ComplexInterpol> > Gw_anom10_interpol;

void set_splines00 (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_norm00_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_norm00_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_norm00_interpol[i][j] = ComplexInterpol(wvals);
			Gw_norm00_interpol[i][j] = Gw_norm00[i][j];
			Gw_norm00_interpol[i][j].set_splines();
		}
	}
}

void set_splines11 (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_norm11_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_norm11_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_norm11_interpol[i][j] = ComplexInterpol(wvals);
			Gw_norm11_interpol[i][j] = Gw_norm11[i][j];
			Gw_norm11_interpol[i][j].set_splines();
		}
	}
}

void set_splines01 (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_anom01_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_anom01_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anom01_interpol[i][j] = ComplexInterpol(wvals);
			Gw_anom01_interpol[i][j] = Gw_anom01[i][j];
			Gw_anom01_interpol[i][j].set_splines();
		}
	}
}

void set_splines10 (double wmin, double wmax, int wpoints)
{
	IntervalIterator w(wmin, wmax, wpoints);
	VectorXd wvals = w.get_abscissa();
	
	Gw_anom10_interpol.resize(L);
	
	for (int i=0; i<L; ++i)
	{
		Gw_anom10_interpol[i].resize(L);
		
		for (int j=0; j<L; ++j)
		{
			Gw_anom10_interpol[i][j] = ComplexInterpol(wvals);
			Gw_anom10_interpol[i][j] = Gw_anom10[i][j];
			Gw_anom10_interpol[i][j].set_splines();
		}
	}
}

MatrixXcd Gfull (double omega)
{
	MatrixXcd res00(L,L); res00.setZero();
	MatrixXcd res11(L,L); res11.setZero();
	MatrixXcd res01(L,L); res01.setZero();
	MatrixXcd res10(L,L); res10.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res00(i,j) = Gw_norm00_interpol[i][j](omega);
		res11(i,j) = Gw_norm11_interpol[i][j](omega);
		res01(i,j) = Gw_anom01_interpol[i][j](omega);
		res10(i,j) = Gw_anom10_interpol[i][j](omega);
	}
	
	MatrixXcd res(2*L,2*L);
	res.topLeftCorner(L,L) = res00;
	res.bottomRightCorner(L,L) = res11;
	res.topRightCorner(L,L) = res01;
	res.bottomLeftCorner(L,L) = res10;
	
	return res;
}

MatrixXcd G (double omega)
{
	MatrixXcd res00(L,L); res00.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res00(i,j) = Gw_norm00_interpol[i][j](omega);
	}
	
	return res00;
}

MatrixXcd F (double omega)
{
	MatrixXcd res01(L,L); res01.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res01(i,j) = Gw_anom01_interpol[i][j](omega);
	}
	
	return res01;
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
	
	Fourier_G00(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	Fourier_G11(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	Fourier_G01(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	Fourier_G10(tmax, tpoints, wmin, wmax, wpoints, true); // SAVE=true
	
	set_splines00(wmin, wmax, wpoints);
	set_splines11(wmin, wmax, wpoints);
	set_splines01(wmin, wmax, wpoints);
	set_splines10(wmin, wmax, wpoints);
	
//	MatrixXcd Test1 = Gfull(0.5).topLeftCorner(L,L);
//	MatrixXcd Test2 = Gfull(0.5).bottomRightCorner(L,L);
	MatrixXcd Test1 = Gfull(0.5).topRightCorner(L,L);
	MatrixXcd Test2 = -Gfull(-0.5).adjoint().bottomLeftCorner(L,L);
	
//	for (int i=0; i<L; ++i)
//	for (int j=0; j<L; ++j)
//	{
//		Test2 *= pow(-1.,i+j);
//	}
	
//	MatrixXcd Test1 = Gfull(0.5).bottomRightCorner(L,L);
//	MatrixXcd Test2 = -G(-0.5).adjoint();
	
	lout << endl << Test1 << endl;
	lout << endl << Test2 << endl;
	lout << "diffnorm=" << (Test1-Test2).norm() << endl;
	
	lout << Timer.info("Fourier transform") << endl;
	
}
