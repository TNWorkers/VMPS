#ifndef DMRG_HAMILTON_TERMS
#define DMRG_HAMILTON_TERMS

/// \cond
#include <vector>
#include <tuple>
#include <string>
/// \endcond

#include "numeric_limits.h" // from TOOLS
#include "tensors/SiteOperator.h"
#include "symmetry/qarray.h"
#include "SuperMatrix.h"



/*
 *		-> class SiteOperator, ohne namespace
 */
template<typename Symmetry, typename Scalar> bool operator== (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
	if(O1.Q == O2.Q)
	{
		if((O1.data - O2.data).norm() < 1e-10)
		{
			return true;
		}
	}
	return false;
}

/*
 *		-> class SiteOperator, ohne namespace
 */
template<typename Symmetry, typename Scalar> SiteOperator<Symmetry, Scalar> operator*(const SiteOperator<Symmetry,Scalar> &op,const Scalar &lambda)
{
	return lambda*op;
}

/*
 *		-> class SiteOperator, ohne namespace
 */
template<typename Symmetry, typename Scalar> std::vector<SiteOperator<Symmetry,Scalar>> operator*(const std::vector<SiteOperator<Symmetry,Scalar>> &ops, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
	assert(ops.size() == mat.rows() and "Dimensions of vector and matrix do not match!");
	std::vector<SiteOperator<Symmetry, Scalar>> out;
    for(std::size_t j=0; j<mat.cols(); ++j)
	{
		SiteOperator<Symmetry, Scalar> temp;
        std::size_t i=0;
		for(; std::abs(mat(i,j)) < 1e-10 && i<mat.rows(); ++i){}
		if(i == mat.rows()-1)
		{	
 			temp = 0*ops[j];
		}
		else
		{
			temp = ops[i]*mat(i,j);
			++i;
		}
		for(; i<mat.rows(); ++i)
		{
			if(std::abs(mat(i,j)) > 1e-10)
			{
				temp += ops[i]*mat(i,j);
			}
		}
		out.push_back(temp);
	}
	return out;	
}

/*
 *		-> class SiteOperator, ohne namespace
 */
template<typename Symmetry, typename Scalar> std::vector<SiteOperator<Symmetry,Scalar>> operator*(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat, const std::vector<SiteOperator<Symmetry,Scalar>> &ops)
{
	assert(ops.size() == mat.cols() and "Dimensions of matrix and vector do not match!");
	std::vector<SiteOperator<Symmetry, Scalar>> out;
    for(std::size_t i=0; i<mat.rows(); ++i)
	{
		SiteOperator<Symmetry, Scalar> temp;
        std::size_t j=0;
		for(; std::abs(mat(i,j)) < 1e-10 && j<mat.cols(); ++j){}
		if(j == mat.cols()-1)
		{	
 			temp = 0*ops[i];
		}
		else
		{
			temp = mat(i,j)*ops[j];
			++j;
		}

		for(; j<mat.cols(); ++j)
		{
			if(std::abs(mat(i,j)) > 1e-10)
			{
				temp += mat(i,j)*ops[j];
			}
		}
		out.push_back(temp);
	}
	return out;	
}

template<typename Symmetry, typename Scalar> class HamiltonTerms
{
	private:
		typedef SiteOperator<Symmetry,Scalar> OperatorType;
		typename Symmetry::qType qvac = Symmetry::qvacuum();

        /**
         *  Local terms of Hamiltonian, a simple vector with index structure (Lattice Site)
         */
		std::vector<OperatorType> local;

        /**
         *	Stores whether the local operator has been set yet, a simple vector
         *	Index structure (Lattice Site)
         */
		std::vector<bool> localSet;
	
        /**
         *	Incoming nearest-neighbour terms, a twofold vector
         *	Index structure (Lattice Site i, Index incoming operator at i)
         */
		std::vector<std::vector<OperatorType>> tight_in;

        /**
         *	Outgoing nearest-neighbour terms, a twofold vector
         *	Index structure (Lattice Site i, Index outgoing operator at i)
         */
		std::vector<std::vector<OperatorType>> tight_out;

        /**
         *	Nearest-neighbour interactions, a vector of matrices
         *	Index structure (Lattice Site i, Index outgoing operator at i, Index incoming operator at i+1)
         */
		std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> tight_coupl;

        /**
         *	Incoming next-nearest-neighbour terms, a threefold vector
         *	Index structure (Lattice Site i, Index transfer operator at i-1, Index incoming operator at i)
         */
		std::vector<std::vector<std::vector<OperatorType>>> nextn_in;

        /**
         *	Outgoing next-nearest-neighbour terms, a threefold vector
         *	Index structure (Lattice Site i, Index transfer operator at i+1, Index outgoing operator at i)
         */
		std::vector<std::vector<std::vector<OperatorType>>> nextn_out;

        /**
         *	Transfer operators for next-nearest-neighbour interaction, a twofold vector
         *	Index structure (Lattice Site i, Index transfer operator at i)
         */
		std::vector<std::vector<OperatorType>> nextn_TransOps;

        /**
         *	Next-nearest-neighbour interactions, a twofold vector of matrices
         *	Index structure (Lattice Site i, transfer operator at i+1, Index outgoing operator at i, Index incoming operator at i+2)
         */
		std::vector<std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>> nextn_coupl;

        /**
         *	Local hilbert space dimensions, a simple vector
         *	Index structure (Lattice Site). Initialized with 0 and set whenever a operator is added at a given site
         */
		std::vector<int> hilbert_dimension;

        /**
         *	Number of lattice sites
         */
        std::size_t N_sites;

        /**
         *	All informations stored about the HamiltonTerms, twofold vector
         *	Index structure (Lattice Site i, Index of information string)
         */
        std::vector<std::vector<std::string>> info;

        /**
         *	Compressed incoming nearest-neighbour terms, a twofold vector
         *	Index structure (Lattice Site i, Index incoming operator at i)
         */
		std::vector<std::vector<OperatorType>> tight_in_compressed;

        /**
         *	Compressed outgoing nearest-neighbour terms, a twofold vector
         *	Index structure (Lattice Site i, Index outgoing operator at i)
         */
		std::vector<std::vector<OperatorType>> tight_out_compressed;

        /**
         *	Compressed incoming next-nearest-neighbour terms, a threefold vector
         *	Index structure (Lattice Site i, Index transfer operator at i-1, Index incoming operator at i)
         */
		std::vector<std::vector<std::vector<OperatorType>>> nextn_in_compressed;

        /**
         *	Compressed outgoing next-nearest-neighbour terms, a threefold vector
         *	Index structure (Lattice Site i, Index transfer operator at i+1, Index outgoing operator at i)
         */
		std::vector<std::vector<std::vector<OperatorType>>> nextn_out_compressed;

        /**
         *	"Compressed" transfer operators for next-nearest-neighbour interaction, a threefold vector
         *	Index structure (Lattice Site i, Index transfer operator at i, Index repetition of operator)
         */
		std::vector<std::vector<std::vector<OperatorType>>> nextn_trans_compressed;

        /**
         *	Stores, whether the compressed Term vectors are up to date
         */
		bool compressed;
    
        /**
         *  A given name for the HamiltonTerms, such as Heisenberg
         */
        std::string name="";
    
        /**
         *  Stores whether to use periodic boundary conditions or not
         */
        bool open_bc;
    
        /**
         *  Takes the plain interaction vectors and compresses them by singular value decomposition
         */
        void compress();
    
	public:
        /**
         *  Constructor
         *  @param L    Lattice size
         *  @param bc   Open boundary conditions
         */
		HamiltonTerms(std::size_t L, bool bc = true);
    
        /**
         *  Adds a new local interaction to the HamiltonTerms
         *  @param loc      Lattice site
         *  @param Op       SiteOperator acting on the local Hilbert space of site \p loc
         *  @param lambda   Scalar of interaction strength that is multiplied to the operator
         */
        void push_local(std::size_t loc, Scalar lambda, OperatorType Op);
    
        /**
         *  Adds a new nearest-neighbour interaction to the HamiltonTerms
         *  @param loc      Lattice site of first site
         *  @param Op1      SiteOperator acting on the local Hilbert space of site \p loc
         *  @param Op2      SiteOperator acting on the local Hilbert space of site \p loc+1
         *  @param lambda   Scalar of interaction strength that is multiplied to the operator
         */
		void push_tight(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Op2);

        /**
         *  Adds a new next-nearest-neighbour interaction to the HamiltonTerms
         *  @param loc      Lattice site of first site
         *  @param Op1      SiteOperator acting on the local Hilbert space of site \p loc
         *  @param Trans    SiteOperator acting as transfer operator on the local Hilbert space of site \p loc+1
         *  @param Op2      SiteOperator acting on the local Hilbert space of site \p loc+2
         *  @param lambda   Scalar of interaction strength that is multiplied to the operator
         */
        void push_nextn(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Trans, OperatorType Op2);

        /**
         *  @param loc      Lattice site
         *  @param label    Information
         */
        void save_label(std::size_t loc, const std::string& label);
    
        /**
         *  @param label    Name to be given to this instance of HamiltonTerms
         */
        void set_name(const std::string& label);
    
        /**
         *  @return A vector of formatted strings that contain information about the HamiltonTerms. Zeroth entry = name.
         */
        std::vector<std::string> get_info() const;

        /**
         *  @param loc  Lattice site
         *  @return The dimension of the local Hilbert space. If not set yet, this will return 0
         */
		std::size_t Hilbert_dimension(std::size_t loc) const;
    
        /**
         *  Constructs a vector of SuperMatrix from the compressed interactions. Compresses the interaction if necessary.
         */
		std::vector<SuperMatrix<Symmetry, Scalar>> construct_Matrix();
};

template<typename Symmetry, typename Scalar> HamiltonTerms<Symmetry,Scalar>::HamiltonTerms(std::size_t L, bool bc) : N_sites(L)
{
    info.resize(N_sites);
    hilbert_dimension.resize(N_sites);
    local.resize(N_sites);
    localSet.resize(N_sites);
    tight_in.resize(N_sites);
    tight_out.resize(N_sites);
    tight_coupl.resize(N_sites);
    nextn_in.resize(N_sites);
    nextn_out.resize(N_sites);
    nextn_TransOps.resize(N_sites);
    nextn_coupl.resize(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        hilbert_dimension[loc] = 0;
        tight_coupl[loc].resize(0,0);
        localSet[loc] = false;
    }
    open_bc = bc;
}

template<typename Symmetry, typename Scalar> void HamiltonTerms<Symmetry,Scalar>::save_label(std::size_t loc, const std::string& label)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    if(label!="")
    {
        info[loc].push_back(label);
    }
}

template<typename Symmetry, typename Scalar> void HamiltonTerms<Symmetry,Scalar>::set_name(const std::string& label)
{
    name = label;
}


template<typename Symmetry, typename Scalar> std::vector<std::string> HamiltonTerms<Symmetry,Scalar>::get_info() const
{
    std::vector<std::string> res(N_sites+1);
    res[0] = name;
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::stringstream ss;
        std::copy(info[loc].begin(), info[loc].end()-1, std::ostream_iterator<std::string>(ss,","));
        ss << info[loc].back();
        
        res[loc] = ss.str();
        
        while (res[loc].find("perp") != std::string::npos) res[loc].replace(res[loc].find("perp"), 4, "⟂");
        while (res[loc].find("para") != std::string::npos) res[loc].replace(res[loc].find("para"), 4, "∥");
        while (res[loc].find("prime") != std::string::npos) res[loc].replace(res[loc].find("prime"), 5, "'");
        while (res[loc].find("Perp") != std::string::npos) res[loc].replace(res[loc].find("Perp"), 4, "⟂");
        while (res[loc].find("Para") != std::string::npos) res[loc].replace(res[loc].find("Para"), 4, "∥");
        while (res[loc].find("Prime") != std::string::npos) res[loc].replace(res[loc].find("Prime"), 5, "'");
        while (res[loc].find("mu") != std::string::npos) res[loc].replace(res[loc].find("mu"), 2, "µ");
    }
    return res;
}




template<typename Symmetry, typename Scalar> std::size_t HamiltonTerms<Symmetry,Scalar>::Hilbert_dimension(std::size_t loc) const
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    return hilbert_dimension[loc];
}



template<typename Symmetry, typename Scalar> void HamiltonTerms<Symmetry,Scalar>::push_local(std::size_t loc, Scalar lambda, OperatorType Op)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
	assert(Op.Q == qvac and "Local operator is not a singlet");
	if(!localSet[loc])
	{
		localSet[loc] = true;
		local[loc] = lambda*Op;
		hilbert_dimension[loc] = Op.data.rows();
	}
	else
	{
		assert(hilbert_dimension[loc] == Op.data.rows() and "Dimensions of operator and local Hilbert space do not match!");
		local[loc] += lambda*Op;
	}
}
	
template<typename Symmetry, typename Scalar> void HamiltonTerms<Symmetry,Scalar>::push_tight(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Op2)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    assert((!open_bc || loc+1 < N_sites) and "Chosen lattice site out of bounds");
	assert(hilbert_dimension[loc] == 0 || hilbert_dimension[loc] == Op1.data.rows() and "Dimensions of first operator and local Hilbert space do not match");
	assert(hilbert_dimension[(loc+1)%N_sites] == 0 || hilbert_dimension[(loc+1)%N_sites] == Op2.data.rows() and "Dimensions of second operator and local Hilbert space do not match");
	compressed = false;
    std::ptrdiff_t firstit = std::distance(tight_out[loc].begin(), find(tight_out[loc].begin(), tight_out[loc].end(), Op1));
	if(firstit >= tight_out[loc].size())    // If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
	{
		tight_out[loc].push_back(Op1);
		tight_coupl[loc].conservativeResize(tight_out[loc].size(), tight_in[(loc+1)%N_sites].size());
		hilbert_dimension[loc] = Op1.data.rows();
	}
    std::ptrdiff_t secondit = std::distance(tight_in[(loc+1)%N_sites].begin(), find(tight_in[(loc+1)%N_sites].begin(), tight_in[(loc+1)%N_sites].end(), Op2));
	if(secondit >= tight_in[(loc+1)%N_sites].size())    // If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
	{
		tight_in[(loc+1)%N_sites].push_back(Op2);
		tight_coupl[loc].conservativeResize(tight_out[loc].size(), tight_in[(loc+1)%N_sites].size());
		hilbert_dimension[(loc+1)%N_sites] = Op2.data.rows();
	}
	tight_coupl[loc](firstit, secondit) += lambda;
}

template<typename Symmetry, typename Scalar> void HamiltonTerms<Symmetry,Scalar>::push_nextn(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Trans, OperatorType Op2)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    assert((!open_bc || loc+2 < N_sites) and "Chosen lattice site out of bounds");
	assert(hilbert_dimension[loc] == 0 || hilbert_dimension[loc] == Op1.data.rows() and "Dimensions of first operator and local Hilbert space do not match");
	assert(hilbert_dimension[(loc+1)%N_sites] == 0 || hilbert_dimension[(loc+1)%N_sites] == Trans.data.rows() and "Dimensions of transfer operator and local Hilbert space do not match");
	assert(hilbert_dimension[(loc+2)%N_sites] == 0 || hilbert_dimension[(loc+2)%N_sites] == Op2.data.rows() and "Dimensions of second operator and local Hilbert space do not match");
	assert(Trans.Q == qvac and "Transfer operator is not a singlet");
	compressed = false;

    std::ptrdiff_t transit = std::distance(nextn_TransOps[(loc+1)%N_sites].begin(), find(nextn_TransOps[(loc+1)%N_sites].begin(), nextn_TransOps[(loc+1)%N_sites].end(), Trans));
	if(transit >= nextn_TransOps[(loc+1)%N_sites].size())   // If the operator cannot be found, push it to the corresponding terms and create a new container for interactions mediated by this transfer operator
	{
		nextn_TransOps[(loc+1)%N_sites].push_back(Trans);
		hilbert_dimension[(loc+1)%N_sites] = Trans.data.rows();
		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(1,1);
		nextn_coupl[(loc+1)%N_sites].push_back(matrix);
		std::vector<OperatorType> temp;
		nextn_out[loc].push_back(temp);
		nextn_in[(loc+2)%N_sites].push_back(temp);
	}


    std::ptrdiff_t firstit = std::distance(nextn_out[loc][transit].begin(), find(nextn_out[loc][transit].begin(), nextn_out[loc][transit].end(), Op1));
	if(firstit >= nextn_out[loc][transit].size())   // If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
	{
		nextn_out[loc][transit].push_back(Op1);
		nextn_coupl[(loc+1)%N_sites][transit].conservativeResize(nextn_out[loc][transit].size(), nextn_in[(loc+2)%N_sites][transit].size());
        nextn_coupl[(loc+1)%N_sites][transit].bottomRows(1).setZero();
		hilbert_dimension[loc] = Op1.data.rows();
	}
    std::ptrdiff_t secondit = std::distance(nextn_in[(loc+2)%N_sites][transit].begin(), find(nextn_in[(loc+2)%N_sites][transit].begin(), nextn_in[(loc+2)%N_sites][transit].end(), Op2));
	if(secondit >= nextn_in[(loc+2)%N_sites][transit].size())   // If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
	{
		nextn_in[(loc+2)%N_sites][transit].push_back(Op2);
		nextn_coupl[(loc+1)%N_sites][transit].conservativeResize(nextn_out[loc][transit].size(), nextn_in[(loc+2)%N_sites][transit].size());
        nextn_coupl[(loc+1)%N_sites][transit].rightCols(1).setZero();
		hilbert_dimension[(loc+2)%N_sites] = Op2.data.rows();
	}
	nextn_coupl[(loc+1)%N_sites][transit](firstit, secondit) += lambda;

}

template<typename Symmetry, typename Scalar> std::vector<SuperMatrix<Symmetry, Scalar>> HamiltonTerms<Symmetry,Scalar>::construct_Matrix()
{
	if(!compressed) compress();
    std::cout << "Es wurde komprimiert!" << std::endl;
    std::vector<SuperMatrix<Symmetry,Scalar> > G;
    for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		SuperMatrix<Symmetry,Scalar> S;
        std::cout << "Site " << loc << ":" << std::endl;
		if(hilbert_dimension[loc] == 0)     //  Create a trivial SuperMatrix if no operator has been set.
		{
            std::cout << "Triviale Behandlung" << std::endl;
			hilbert_dimension[loc] = 1;
			OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
			S.set(2,2,hilbert_dimension[loc]);
			S(0,0) = Id;
			S(1,1) = Id;
		}
		else
		{
            std::cout << "Nicht-triviale Behandlung" << std::endl;
            std::size_t transfer = 0;   //  Stores the total number of transfer operators at lattice site loc
            std::size_t nextn_rows = 0; //  Stores the total number of next-nearest-neighbour interaction terms with loc-2
            std::size_t nextn_cols = 0; //  Stores the total number of next-nearest-neighbour interaction terms with loc+2
            for(std::size_t t=0; t<nextn_in_compressed[loc].size(); ++t)
			{
				nextn_rows += nextn_in_compressed[loc][t].size();
			}
            for(std::size_t t=0; t<nextn_out_compressed[loc].size(); ++t)
			{
				nextn_cols += nextn_out_compressed[loc][t].size();
			}
            for(std::size_t t=0; t<nextn_trans_compressed[loc].size(); ++t)
			{
				transfer += nextn_trans_compressed[loc][t].size();
			}
            std::size_t rows = 2 + tight_in_compressed[loc].size() + nextn_rows + transfer;     //  Total number of rows
            std::size_t cols = 2 + tight_out_compressed[loc].size() + nextn_cols + transfer;    //  Total number of columns
			S.set(rows,cols,hilbert_dimension[loc]);
			OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
            std::size_t current = 0;
            S(current++,0) = Id;        //  Upper left corner: identity

            for(std::size_t i=0; i<tight_in_compressed[loc].size(); ++i)
			{
                S(current++,0) = tight_in_compressed[loc][i];   //  First column: Incoming nearest-neighbour terms
			}
            for(std::size_t t=0; t<nextn_in_compressed[loc].size(); ++t)
			{
                for(std::size_t i=0; i<nextn_in_compressed[loc][t].size(); ++i)
				{
                    S(current++,0) = nextn_in_compressed[loc][t][i];    //  First column: Incoming next-nearest-neighbour terms, ordered w.r.t. their transfer operators
				}
			}
            
            // First column: A sufficient number of rows is skipped for the transfer operators
			
			current = 0;
			if(!localSet[loc])  //  If no local interaction has been added, the local interaction becomes a dummy SiteOperator with correct dimension
			{
				local[loc] = 0*Id;
			}
            
            S(rows-1,current++) = local[loc];   //  Lower left corner: Local interaction


            for(std::size_t t=0; t<tight_out_compressed[loc].size(); ++t)
			{
                S(rows-1, current++) = tight_out_compressed[loc][t];    //  Last row: Outgoing nearest-neighbour terms
			}

            current += transfer;    // Last row: A sufficient number of columns is skipped for the transfer operators

            for(std::size_t t=0; t<nextn_out_compressed[loc].size(); ++t)
			{
                for(std::size_t i=0; i<nextn_out_compressed[loc][t].size(); ++i)
				{
                    S(rows-1,current++) = nextn_out_compressed[loc][t][i];  // Last row: Outgoing next-nearest-neighbour terms, ordered w.r.t. their transfer operators
				}
			}
            S(rows-1,cols-1) = Id;  // Lower right corner: Identity

            std::size_t row_start = 1 + tight_in_compressed[loc].size() + nextn_rows;   //  Where does the block of transfer operators start?
            std::size_t col_start = 1 + tight_out_compressed[loc].size();

			current = 0;
            for(std::size_t t=0; t<nextn_trans_compressed[loc].size(); ++t)
			{
                for(std::size_t i=0; i<nextn_trans_compressed[loc][t].size(); ++i)
				{
                    S(row_start+current,col_start+current) = nextn_trans_compressed[loc][t][i]; //  Since the interaction for each transfer operator is diagonal: only diagonal elements of the transfer block are set.
					current++;
				}
			}			
		}
		G.push_back(S);
	}
    std::cout << "construct_matrix abgeschlossen" << std::endl;
	return G;
}

template<typename Symmetry, typename Scalar> void HamiltonTerms<Symmetry,Scalar>::compress()
{
    tight_in_compressed.resize(N_sites);
    tight_out_compressed.resize(N_sites);
    nextn_in_compressed.resize(N_sites);
    nextn_out_compressed.resize(N_sites);
    nextn_trans_compressed.resize(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        nextn_in_compressed[loc].resize(nextn_in[loc].size());
        nextn_out_compressed[loc].resize(nextn_out[loc].size());
        nextn_trans_compressed[loc].resize(nextn_coupl[loc].size());
    }
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        if(tight_coupl[loc].rows() > tight_coupl[loc].cols())
        {
            tight_out_compressed[loc] = tight_out[loc] * tight_coupl[loc];
            tight_in_compressed[(loc+1)%N_sites] = tight_in[(loc+1)%N_sites];
        }
        else
        {
            tight_out_compressed[loc] = tight_out[loc];
            tight_in_compressed[(loc+1)%N_sites] = tight_coupl[loc] * tight_in[(loc+1)%N_sites];
        }
        for(std::size_t t=0; t<nextn_coupl[(loc+1)%N_sites].size(); ++t)
        {
            if(loc == 5ul)
            {
                std::cout << "Kopplungsmatrix zwischen Gitterplatz 3 und 5:\n" << nextn_coupl[4][0] << std::endl;
            }
            if(nextn_coupl[(loc+1)%N_sites][t].rows() > nextn_coupl[(loc+1)%N_sites][t].cols())
            {
                nextn_out_compressed[loc][t] = nextn_out[loc][t] * nextn_coupl[(loc+1)%N_sites][t];
                nextn_in_compressed[(loc+2)%N_sites][t] = nextn_in[(loc+2)%N_sites][t];

                for(std::size_t i=0; i<nextn_coupl[(loc+1)%N_sites][t].cols(); ++i)
                {
                    nextn_trans_compressed[(loc+1)%N_sites][t].push_back(nextn_TransOps[(loc+1)%N_sites][t]);
                }
            }
            else
            {
                nextn_out_compressed[loc][t] = nextn_out[loc][t];
                nextn_in_compressed[(loc+2)%N_sites][t] = nextn_coupl[(loc+1)%N_sites][t] * nextn_in[(loc+2)%N_sites][t];
                for(std::size_t i=0; i<nextn_coupl[(loc+1)%N_sites][t].rows(); ++i)
                {
                    nextn_trans_compressed[(loc+1)%N_sites][t].push_back(nextn_TransOps[(loc+1)%N_sites][t]);
                }
            }
        }
        if(loc == 5ul)
        {
            for(size_t t=0; t<nextn_in[loc].size(); ++t)
            {
                for(size_t i=0; i<nextn_in[loc][t].size(); ++i)
                {
                    std::cout << "t=" << t << ", i=" << i << ":\n" << nextn_in[loc][t][i].data << std::endl;
                }
            }
            for(size_t t=0; t<nextn_in_compressed[loc].size(); ++t)
            {
                for(size_t i=0; i<nextn_in[loc][t].size(); ++i)
                {
                    std::cout << "t=" << t << ", i=" << i << ":\n" << nextn_in_compressed[loc][t][i].data << std::endl;
                }
            }
        }
    }
    compressed = true;
}

#endif