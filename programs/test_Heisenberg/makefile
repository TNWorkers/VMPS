EIGEN=../../../../CPPLIB/EIGEN/
ARMA=../../../../CPPLIB/ARMADILLO/include
#ARMA=../../../../CPPLIB/armadillo-3.920.4/include #g++-4.4 compatible
ALGS=../../../ALGS/
CHEB=../../../CHEBYSHEV/
HIL=../../../HILBERT/
TOOLS=../../../TOOLS/
PLOT=../../../PLOT/
INT=../../../INTEGRATION/
FUNC=../../../FUNCTIONS/
DMRG=../../
BOOST=../../../../CPPLIB/BOOST/
#EINSPLINE=../../../../CPPLIB/EINSPLINE
GSL=../../../../CPPLIB/GSL
LAPACK=../../../../CPPLIB/LAPACK
FFTW=../../../../CPPLIB/FFTW
VIENNACL=../../../../CPPLIB/VIENNACL/
TEXTTABLE=../../../../CPPLIB/cpp-text-table/

BUILD=$(notdir $(CURDIR))
#program name is equal to folder name

COMPILER=
UNAME=$(shell uname)
THISHOST=$(shell hostname)
ifeq ($(UNAME), Linux)
	ifeq ($(THISHOST), asterix01)
		COMPILER=PATH=~/opt/bin LD_LIBRARY_PATH=~/opt/lib64 ~/opt/bin/g++ -fopenmp -lgfortran
	else
		COMPILER=g++-7 -fopenmp 
		#-static -Wl,-upthread_mutex_destroy
	endif
endif
ifeq ($(UNAME), Darwin)
	COMPILER=clang++ -Wno-return-type -Wno-deprecated-register
endif

OBJ=$(BUILD).o # append .o to BUILD
CXX=$(COMPILER) -std=c++1z
MACROS=-fmax-errors=5
INCS=-I $(EIGEN) -I $(ARMA) -I $(BOOST) -I $(GSL)/include -I $(FFTW)
#-I $(ARMA) -I $(FFTW)/include -I $(EINSPLINE) -I $(VIENNACL)
MYINCS=-I $(ALGS) -I $(HIL) -I $(INT) -I $(TOOLS) -I $(CHEB) -I $(FUNC) -I $(PLOT) -I $(DMRG) -I $(VIENNACL) -I $(TEXTTABLE)
LIBS=$(GSL)/lib/libgsl.a $(GSL)/lib/libgslcblas.a $(FFTW)/lib/libfftw3.a $(LAPACK)/liblapack.a $(LAPACK)/librefblas.a /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.a
#$(LAPACK)/liblapack.a $(LAPACK)/librefblas.a -lgfortran -lm
#$(LAPACK)/liblapack.a
#$(FFTW)/lib/libfftw3.a $(FFTW)/lib/libfftw3_omp.a ../../../../CPPLIB/lapack-3.5.0/liblapacke.a
#$(FFTW)/lib/libfftw3.a $(FFTW)/lib/libfftw3_omp.a
#-llapack
#-lfftw3
#$(EINSPLINE)/lib/libeinspline.a 
OPTIMIZATION=-O3
DEBUG=
EXEC=$(BUILD).out # append .out to BUILD
CLEXEC=$(BUILD)_cl.out # append .out to BUILD
DEPFILE=.depend
CPPFLAGS=$(MACROS) $(INCS) $(MYINCS)
CXXFLAGS=$(OPTIMIZATION) $(DEBUG)
LDFLAGS=$(OPTIMIZATION) $(DEBUG)

BACKUP_TAR=backup-$(shell date +%F).tar.gz #backup tar with current date
BACKUPLIST=.backup #list with all files to be backed up

all: dep $(EXEC)
$(EXEC): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)
dep:
	@$(CXX) -MM $(OBJ:%.o=%.cc) $(CPPFLAGS) > $(DEPFILE) #create depfile, replace .o by .cc
clean:
	rm -f $(OBJ) $(EXEC) $(DEPFILE)

cluster: $(CLEXEC)
$(CLEXEC): $(BUILD).cc
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)
cclean:
	rm -f $(CLEXEC)

backup:
	@$(CXX) -MM $(OBJ:%.o=%.cc) $(MYINCS) > $(BACKUPLIST) #create backup list only with MYINCS, replace .o by .cc
	@perl -pi -e 's/ \\//g' $(BACKUPLIST) #remove slashes from backup list
	@perl -i -ne 'print unless 1 .. 1' $(BACKUPLIST) #remove first line of backup list
	tar -c -z --file=$(BACKUP_TAR) *.cc makefile `cat $(BACKUPLIST)` #create backup tar, include .cc file and this makefile

-include $(DEPFILE)
