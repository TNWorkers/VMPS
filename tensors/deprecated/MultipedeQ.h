#ifndef STRAWBERRY_MULTIPEDEQ
#define STRAWBERRY_MULTIPEDEQ

#ifndef HELPERS_CXX17
  #error This file strictly needs the c++-1z standard
#endif

#include <unordered_map>
#include <unordered_set>

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

#include "macros.h"
#include "PolychromaticConsole.h"
#include "numeric_limits.h"
#include "EigenDevice.h"
#include "Stopwatch.h"
#include "qleg.h"
#include "qarray.h"
#include "DictQ.h"
#include "TensorHelpers.h"
#include "DmrgExternalQ.h"
#include "MemCalc.h"

constexpr Eigen::Index hidden_dim( Eigen::Index Rank )
{
	return Rank < 3? 0 : Rank-3;
};

constexpr Eigen::Index correct_dim( Eigen::Index Rank )
{
	return  Rank == 4 ? -1 : 0;
};

/**
Enumeration for different special contractions.
Needed in case of a non-abelian symmetry, where the Clebsch-Gordon coefficients are not stored explicitly.
*/
namespace con {
	const Eigen::Index APAIR = 1;
	const Eigen::Index WPAIR = 0;
	const Eigen::Index UNITY = 0;

	namespace four {
		enum MODE {UNITY,BUILD_L,BUILD_R,H_PSI,APAIR};
	}
	namespace three {
		enum MODE {UNITY,FUSE_WPAIR,DOT};
	}
	namespace two
	{
		enum MODE {WPAIR,APAIR,FUSE01_APAIR,FUSE13_APAIR,SPLIT_APAIR,NONE,OORR,LLRR,OLRR,SIGN,DOT,UNITY};
	}
}

/**
General tensor with quantum number blocks respecting in prinicipal general non abelian symmetries.
\tparam Nlegs : Amount of tensor legs
\describe_Symmetry
\tparam Scalar : Basic data type for all the tensors.
\tparam Nextra : Amount of additional quantum numbers which can be stored for every block. ("Remembering quantum numbers")
\note special typedefs not caught by Doxygen: \n 
\p TripodQ<Nq,Scalar> (\p Nlegs=3) \n
\p QuadrupedQ<Nq,Scalar> (\p Nlegs=4).*/
template<Eigen::Index Nlegs, typename Symmetry, typename Scalar,
		 Eigen::Index Nextra=0, typename base_type_=Eigen::Tensor<Scalar,Nlegs,Eigen::ColMajor,Eigen::Index> >
class MultipedeQ
{
	typedef MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Self;
	typedef typename Symmetry::qType qType;
	typedef Eigen::Index Index;
	// typedef base_type TensorType;
	template<Index Rank> using TensorType = Eigen::Tensor<Scalar,Rank,Eigen::ColMajor,Index>;

public:
	/**Public typedef to the underlying base_type of this MultipedeQ to get access from other classes.*/
	typedef base_type_ base_type;

	/**Share private functions between different intantiations of MultipedeQ.*/
	template<Index Nlegs_,typename Symmetry_, typename Scalar_, Index Nextra_, typename base_type__> friend class MultipedeQ;

	///@{
	/**Constructing just by set \p dim to 0.*/
	MultipedeQ(){dim=0;}
	/**Constructing by set \p dim to 0 and specify the direction of each leg.*/
	MultipedeQ( const std::array<dir,Nlegs> dirs_in ); //, std::array<string,Nlegs> names_in );
	///@}
	
	/**Const reference to the number of legs \p Nlegs */
	inline constexpr Index rank() const {return Nlegs;}

	/**Array which holds the information over all legs. Most important: the direction of the legs.*/
	std::array<qleg,Nlegs> legs;
	
	///@{
	/**Convenience const access to the amount of blocks.
	Equal to either of the following: \p dim, \p index.size(), \p block.size(), cgc.size()*/
	inline std::size_t size() const {return dim;}
	inline void plusplus() {++dim;}
	/**Vector of all quantum numbers.
	The entries are arrays of size \p Nlegs+Nextra+::hidden_dim(Nlegs).*/
	std::vector<std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> > index;
	/**Vector of quantum number blocks describing the reduced tensor elements for the quantum numbers q.
	The tensor of \p block[q] is characterized by the incoming quantum numbers \p index[q]*/
	std::vector<base_type_> block;
	/**Vector of quantum number blocks describing the Clebsch-Gordon coefficients (cgc) for the quantum numbers q.
	   The tensor of \p cgc[q] is characterized by the incoming quantum numbers \p index[q]*/
	std::vector<base_type_> cgc;
	/**Dictionary allowing one to find the index of \p block for a given array of \p Nlegs quantum numbers in \f$O(1)\f$ operations 
	   without looping over the blocks.
	*/
	::DictQ<Nlegs+Nextra+::hidden_dim(Nlegs),Symmetry> dict;
	///@}
	
	///@{
	/**\describe_memory*/
	double memory (const MEMUNIT memunit=GB) const;
	/**\describe_overhead*/
	double overhead (const MEMUNIT memunit=MB) const;
	///@}

	///@{
	/**Returns an Eigen vector of size \p dim containing all Tensor dimensions of leg \p leg for every block nu.
	 \param leg : specific leg of which the dimensions are calculated.*/
	Eigen::VectorXi dimensions ( const std::size_t leg ) const;
	///@}

	/**Function to print the full MultipedeQ 
	\param SHOW_METADATA : if true, general information over the legs is printed (direction,ordering). 
	\param SHOW_MATRICES : if true, all the block- and cgc-tensors are printed.
	\param precision : precision for the \p Scalar tensor components*/
	std::string print ( const bool SHOW_METADATA=false, const bool SHOW_MATRICES=false , const std::size_t precision=3 ) const;
	
	///@{
	/**Returns the adjoint tensor where the quantum number arrows are flipped.*/
	Self adjoint() const;
	///@}
	
	// inline bool indexDirection(size_t q) const { assert( size() > 0 ); return index[0][q].metaData[0][1];}
	// inline string indexDirectionS(size_t q) const { assert( size() > 0 ); return (index[0][q].metaData[0][1])? "incoming" : "outgoing"; }
	inline dir legDirection( const std::size_t leg_in ) const { return legs[leg_in].direction; };
	// inline dir legDirection( string name_in ) const { for(auto& l: legs) { if (l.name == name_in) { return l.direction; } } };
	
	///@{
	/**Deletes the contents of \p index, \p block, \p cgc, \p dict.*/
	void clear();
	/**Sets all matrices in MultipedeQ<Nlegs,Symmetry,Scalar>::block to zero, preserving the rows and columns and the \p cgc.*/
	void setZero();
	/**Sets all matrices in MultipedeQ<Nlegs,Symmetry,Scalar>::block to identity, preserving the rows and columns and the \p cgc.
	 \warning Only implemented for \p Nlegs=2.*/
	void setIdentity();
	// void setConstant(); //not implemented yet
	/**Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to the vacuum (all of them).*/
	void setVacuum();
	/**Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers according to the input \p Q.*/
	void setTarget ( const std::array<qType,Nlegs> Q );
	///@}

	/**This functions perform a contraction of \p this and \p A
	\param A : other Multipede which is contracted together with \p this. 
	\param indices : array with pairs of indices which will be contracted
	\note The ordering of the resulting indices is: remaining indices of \p this, remaining indices of \p A */
	template<Index Nextra_=0, Index NlIn, std::size_t NCon, Index Nextra1, typename base_type1>
	MultipedeQ<Nlegs+NlIn-2*static_cast<Index>(NCon),Symmetry,Scalar,Nextra_>
		contract(const MultipedeQ<NlIn,Symmetry,Scalar,Nextra1,base_type1> &A, const std::array<std::array<Index,2>,NCon> indices,
				 const con::two::MODE MODE = con::two::MODE::UNITY) const;
	/**This function performs a contraction of \p this and \p A1 and \p A2.
	\param A1, \param A2 : other Multipedes which are contracted together with \p this. 
	\param indices12 : array with pairs of indices which will be contracted between \p this and \p A1
	\param indices13 : array with pairs of indices which will be contracted between \p this and \p A2
	\param indices23 : array with pairs of indices which will be contracted between \p A1 and \p A2
	\note The ordering of the resulting indices is: remaining indices of \p this, remaining indices of \p A1, remaining indices of \p A2*/
	template<Index Nextra_=0, Index NlIn1, Index NlIn2,
			 std::size_t NCon12, std::size_t NCon13, std::size_t NCon23,
			 Index Nextra1, Index Nextra2,
			 typename base_type1, typename base_type2>
	MultipedeQ<Nlegs+NlIn1+NlIn2-2*static_cast<Index>(NCon12+NCon13+NCon23),Symmetry,Scalar,Nextra_>
		contract(const MultipedeQ<NlIn1,Symmetry,Scalar,Nextra1,base_type1> &A1, const MultipedeQ<NlIn2,Symmetry,Scalar,Nextra2,base_type2> &A2,
				 const std::array<std::array<Index,2>,NCon12> indices12,
				 const std::array<std::array<Index,2>,NCon13> indices13,
				 const std::array<std::array<Index,2>,NCon23> indices23,
				 con::three::MODE MODE=con::three::MODE::UNITY) const;
		/**This function performs a contraction of \p this and \p A1 and \p A2 and \p A3.
	\param A1, \param A2, \param A3 : other Multipedes which are contracted together with \p this. 
	\param indicesIJ : array with pairs of indices which will be contracted between \p AI and \p AJ
	\note The ordering of the resulting indices is: 
	remaining indices of \p this, remaining indices of \p A1, remaining indices of \p A2, remaining indices of \p A3*/
	template<Index Nextra_=0, Index NlIn1, Index NlIn2, Index NlIn3,
			 std::size_t NCon12, std::size_t NCon13, std::size_t NCon14, std::size_t NCon23, std::size_t NCon24, std::size_t NCon34,
			 Index Nextra1, Index Nextra2, Index Nextra3,
			 typename base_type1, typename base_type2, typename base_type3>
	MultipedeQ<Nlegs+NlIn1+NlIn2+NlIn3-2*static_cast<Index>(NCon12+NCon13+NCon14+NCon23+NCon24+NCon34),Symmetry,Scalar,Nextra_>
		contract(const MultipedeQ<NlIn1,Symmetry,Scalar,Nextra1,base_type1> &A1,
				 const MultipedeQ<NlIn2,Symmetry,Scalar,Nextra2,base_type2> &A2,
				 const MultipedeQ<NlIn3,Symmetry,Scalar,Nextra3,base_type3> &A3,
				 const std::array<std::array<Index,2>,NCon12> indices12,
				 const std::array<std::array<Index,2>,NCon13> indices13,
				 const std::array<std::array<Index,2>,NCon14> indices14,
				 const std::array<std::array<Index,2>,NCon23> indices23,
				 const std::array<std::array<Index,2>,NCon24> indices24,
				 const std::array<std::array<Index,2>,NCon34> indices34,
				 const con::four::MODE MODE = con::four::MODE::UNITY) const;

	// template<Index... NCons, Index... NlIns>
	// MultipedeQ<(Nlegs + (NlIns + ...) - 2*(NCons + ...)),Nq,Scalar> contractM(::mytuple<NCons...> indices, const MultipedeQ<NlIns,Nq,Scalar>... &As) const;
	// template<Index NlIn, Index NCon>
	// MultipedeQ<Nlegs+NlIn-2*NCon,Nq,Scalar> contract(const MultipedeQ<NlIn,Nq,Scalar> &A, std::array<std::array<string,2>,NCon> indices ) const;

	///@{
	/**This function returns a tensor for fusing two legs of \p this
	 \param leg1 : left leg of fusing process.
	 \param leg2 : right leg of fusing process.
	 \note You can use the fuse tensor for splitting by contracting with fuse.adjoint().*/
	MultipedeQ<3,Symmetry,Scalar,0> fuse( const Index leg1, const Index leg2 ) const;
	// MultipedeQ<Nlegs-1,Symmetry,Scalar,0> directFuse( const Index leg1, const Index leg2 ) const;
	
	/**This function reverses a given index of \p this.
	 \param leg1 : leg which will be reversed.
	 \note This function only changes the \p cgc.*/
	MultipedeQ<2,Symmetry,Scalar,0> reverse( const Index leg1 ) const;
	/**This function shuffles the indices of \p this.
	 \param shuffle_dims : new array, containing the reordering of the indices.*/
	Self shuffle( const std::array<Index,Nlegs> shuffle_dims ) const;
	///@}

	///@{
	/** This function computes exp(x*this).
		\tparam expScalar : Type of the Scalar for the multiplication.
		\param x : constant, which gets multiplied with the tensor.
		\warning This function is only implemented for \p Nlegs=2.
		\note This function uses Eigen/MatrixFunctions.h*/
	template<typename expScalar>
	Self exp( const expScalar x=expScalar(1.) ) const;
	/** This function computes sqrt(this).
	 \warning This function is only implemented for \p Nlegs=2.
	 \note This function uses Eigen/MatrixFunctions.h*/
	Self sqrt() const;
	///@}

	///@{
	/** This function standardize the \p cgc by multiplying a constant factor from \p cgc to \p block.
		\warning Only implemented for \p Nlegs=3*/
	void standardizeCGC( const bool SCHALTER = false );
	/** This function calculates the \p cgc by using the reduce-method from DmrgSUNGymnastics.h.
		\warning Only implemented for \p Nlegs=3*/
	void calcCGC();
	/** This function checks if the CGC are canonical.
		\warning Only implemented for \p Nlegs=3*/
	bool validateCGC() const;
	/** This function checks if the blocks containing values larger than ::numeric_limits<Scalar>::infinity().*/
	bool validateBlock() const;
	///@}

	/** This function sums the blocks over the extra "remembering" quantum number and therefor tidys up the MultipedeQ.*/
	template<std::size_t Nextra_>
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra-static_cast<Index>(Nextra_),base_type_> tidyUp( const std::array<Index,Nextra_> legsIn ) const;
	
	/** This function performs explicitly the tensor product between \p block and \p cgc.
		\note The \p cgc of the resulting tensors are all sized with 1x...x1 and set to 1.
		\note For testing purposes only.*/
	MultipedeQ<2*Nlegs,Symmetry,Scalar,Nextra> makeTensorProduct( ) const;

	///@{
	/**Convenience functions to return a quantum number of the block \p nu to preserve the sanity of the programmer.*/
	inline qType getQ (const std::size_t place_in, const std::size_t nu) const
		{ for(auto& l: legs) { if (l.place == place_in) { return l.index[nu][l.place]; } } }
	inline qType in  (const std::size_t nu) const {return index[nu][0];}
	inline qType out (const std::size_t nu) const {return index[nu][1];}
	inline qType mid (const std::size_t nu) const {return index[nu][2];}
	inline qType bot (const std::size_t nu) const {return index[nu][2];}
	inline qType top (const std::size_t nu) const {return index[nu][3];}
	///@}

	///@{
	/**Adds a new \p block and new \p cgc (second only \p block) to the MultipedeQ specified by the incoming quantum numbers \p quple.
	\warning Does not check whether the block for these quantum numbers already exists.*/
	template<class Dummy=Symmetry>
	typename std::enable_if<Dummy::HAS_CGC>::type push_back(
	const std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> quple, const base_type_ &M, const base_type_ &T);
	
	template<class Dummy=Symmetry>
	typename std::enable_if<!Dummy::HAS_CGC>::type push_back(
	const std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> quple, const base_type_ &M);
	
	/**Adds a new \p block and new \p cgc (second only \p block) to the MultipedeQ specified by the initializer list \p qlist (must be of size \p Nlegs).
	\warning Does not check whether the block for these quantum numbers already exists.*/
	template<class Dummy=Symmetry>
	typename std::enable_if<Dummy::HAS_CGC>::type push_back(
	const std::initializer_list<qType> qlist, const base_type_ &M, const base_type_ &T);
	
	template<class Dummy=Symmetry>
	typename std::enable_if<!Dummy::HAS_CGC>::type push_back(
	const std::initializer_list<qType> qlist, const base_type_ &M);
	///@}

private:
	/**Amount of blocks in the MultipedeQ.*/
	Index dim=0;
		
	template<std::size_t rank, std::size_t cdim, std::size_t offset>
	static std::size_t transfer_leg(std::array<std::size_t,cdim> conLegs, std::size_t transLeg);

	// EigenDevice<Eigen::DefaultDevice> device_;
	
	template<std::size_t Nfilter>
	std::vector<std::size_t> filterRows ( const std::array<Index, Nfilter> filterLegs, const std::array<qType, Nfilter> filterQs ) const;

	template<std::size_t Nl>
	std::unordered_set<std::array<qType, Nl> > findUniqueQs ( const std::array<Index, Nl> LegsIn ) const;

	template<class... Ts>
	static inline bool checkSizes (const Ts&... ts) { return ( (ts.size() > 0) and ... ); }
	// template<class... Ts>
	// static inline bool checkDirections (const Ts&... ts) { return ( ts.checkDirection() and ... ); }

};

template<typename Symmetry, typename Scalar> using BipedQ     = MultipedeQ<2,Symmetry,Scalar,0>;
template<typename Symmetry, typename Scalar> using TripodQ    = MultipedeQ<3,Symmetry,Scalar,0>;
template<typename Symmetry, typename Scalar> using QuadrupedQ = MultipedeQ<4,Symmetry,Scalar,0>;

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
MultipedeQ ( const std::array<dir,Nlegs> dirs_in ) //, std::array<string,Nlegs> names_in )
{
	dim=0;
	std::size_t count=0;
	for(auto& l: legs)
	{
		l.place = count;
		l.direction = dirs_in[count];
		// l.name = names_in[count];
		count++;
	}
}
// template<Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra>
// template<Index... NCons, Index... NlIns>
// MultipedeQ<(Nlegs + (NlIns + ...) - 2*(NCons + ...)),Nq,Scalar>
// MultipedeQ<Nlegs,Symmetry,Scalar>::
// contractM(::mytuple<NCons...> indices, const MultipedeQ<NlIns,Nq,Scalar>... &As) const;
// {
// 	assert(indices.validate());
	
// 	MultipedeQ<(Nlegs + (NlIns + ...) - 2*(NCons + ...)),Nq,Scalar> Mout;
// 	return Mout;
// }

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<Eigen::Index Nextra_, Eigen::Index NlIn1, Eigen::Index NlIn2, Eigen::Index NlIn3,
		 std::size_t NCon12, std::size_t NCon13, std::size_t NCon14, std::size_t NCon23, std::size_t NCon24, std::size_t NCon34,
		 Eigen::Index Nextra1, Eigen::Index Nextra2, Eigen::Index Nextra3,
		 typename base_type1, typename base_type2, typename base_type3>
MultipedeQ<Nlegs+NlIn1+NlIn2+NlIn3-2*static_cast<Eigen::Index>(NCon12+NCon13+NCon14+NCon23+NCon24+NCon34),Symmetry,Scalar,Nextra_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
	contract(const MultipedeQ<NlIn1,Symmetry,Scalar,Nextra1,base_type1> &A1,
			 const MultipedeQ<NlIn2,Symmetry,Scalar,Nextra2,base_type2> &A2,
			 const MultipedeQ<NlIn3,Symmetry,Scalar,Nextra3,base_type3> &A3,
			 const std::array<std::array<Index,2>,NCon12> indices12,
			 const std::array<std::array<Index,2>,NCon13> indices13,
			 const std::array<std::array<Index,2>,NCon14> indices14,
			 const std::array<std::array<Index,2>,NCon23> indices23,
			 const std::array<std::array<Index,2>,NCon24> indices24,
			 const std::array<std::array<Index,2>,NCon34> indices34, const con::four::MODE MODE) const
{
	//Check if one of the MultipedeQ is empty and if the contraction legs match with their directions.
	assert(checkSizes(*this,A1,A2,A3) and "One of the MultipedeQs has zero size.");
	for (std::size_t i=0; i<NCon12; i++)
	{
		assert( (this->legDirection(indices12[i][0]) != A1.legDirection(indices12[i][1])) );
	}
	for (std::size_t i=0; i<NCon13; i++)
	{
		assert( (this->legDirection(indices13[i][0]) != A2.legDirection(indices13[i][1])) );
	}
	for (std::size_t i=0; i<NCon14; i++)
	{
		assert( (this->legDirection(indices14[i][0]) != A3.legDirection(indices14[i][1])) );
	}
	for (std::size_t i=0; i<NCon23; i++)
	{
		assert( (A1.legDirection(indices23[i][0]) != A2.legDirection(indices23[i][1])) );
	}
	for (std::size_t i=0; i<NCon24; i++)
	{
		assert( (A1.legDirection(indices24[i][0]) != A3.legDirection(indices24[i][1])) );
	}
	for (std::size_t i=0; i<NCon34; i++)
	{
		assert( (A2.legDirection(indices34[i][0]) != A3.legDirection(indices34[i][1])) );
	}

	constexpr Index resRank = Nlegs+NlIn1+NlIn2+NlIn3-2*static_cast<Index>((NCon12+NCon13+NCon14+NCon23+NCon24+NCon34));
	typedef MultipedeQ<resRank,Symmetry,Scalar,Nextra_> Self_;
	Self_ Mout;

    //Construct the directions and names of the resulting MultipedeQ. 
	std::size_t hits=0;
	for (std::size_t i=0; i<Nlegs; i++)
	{
		if ((this->legs[i].notPresent(indices12,0)) and (this->legs[i].notPresent(indices13,0)) and (this->legs[i].notPresent(indices14,0)))
		{ Mout.legs[hits]=this->legs[i]; hits++; }
	}
	for (std::size_t i=0; i<NlIn1; i++)
	{
		if ((A1.legs[i].notPresent(indices12,1)) and (A1.legs[i].notPresent(indices23,0)) and (A1.legs[i].notPresent(indices24,0)))
		{ Mout.legs[hits]=A1.legs[i]; hits++; }
	}
	for (std::size_t i=0; i<NlIn2; i++)
	{
		if ((A2.legs[i].notPresent(indices13,1)) and (A2.legs[i].notPresent(indices23,1)) and (A2.legs[i].notPresent(indices34,0)))
		{ Mout.legs[hits]=A2.legs[i]; hits++; }
	}
	for (std::size_t i=0; i<NlIn3; i++)
	{
		if ((A3.legs[i].notPresent(indices14,1)) and (A3.legs[i].notPresent(indices24,1)) and (A3.legs[i].notPresent(indices34,1)))
		{ Mout.legs[hits]=A3.legs[i]; hits++; }
	}

	assert(hits == resRank);
	for (std::size_t q=0; q<Mout.rank(); q++) { Mout.legs[q].place = q; }

	//construct the intermediate product_dims and legs.
	std::array<Eigen::IndexPair<Index>, NCon12> product_dims1_2;
	for (std::size_t i=0; i<NCon12; i++) { product_dims1_2[i] = Eigen::IndexPair<Index>(indices12[i][0], indices12[i][1]); }
	std::array<Eigen::IndexPair<Index>, NCon13+NCon23> product_dims12_3;
	std::array<Eigen::IndexPair<Index>, NCon14+NCon24+NCon34> product_dims123_4;

	//restIJ: vectors which hold the remaining indices of Tensor I after the Jth step.
	std::vector<Index> rest11;
	std::vector<Index> rest12;
	std::vector<Index> rest13;	
	std::vector<Index> rest21;
	std::vector<Index> rest22;
	std::vector<Index> rest23;
	std::vector<Index> rest32;
	std::vector<Index> rest33;
	std::vector<Index> rest43;

	std::array<Index,NCon13+NCon23> merged12;	
	hits = 0;
	Index hitsP = 0;
	for (std::size_t i=0; i<Nlegs; i++)
	{
		if ((this->legs[i].notPresent(indices12,0)))
		{
			rest11.push_back(i);
			if (this->legs[i].notPresent(indices13,0))
			{
				rest12.push_back(i);
				if (this->legs[i].notPresent(indices14,0)) {rest13.push_back(i);}
			}
			if (this->legs[i].present(indices13,0))
			{
				merged12[hits] = hitsP; hits++;
			}
			hitsP++;
		}
	}
	for (std::size_t i=0; i<NlIn1; i++)
	{
		if ((A1.legs[i].notPresent(indices12,1)))
		{
			rest21.push_back(i);
			if (A1.legs[i].notPresent(indices23,0))
			{
				rest22.push_back(i);
				if (A1.legs[i].notPresent(indices24,0)) { rest23.push_back(i); }
			}
			if (A1.legs[i].present(indices23,0))
			{
				merged12[hits] = hitsP; hits++;
			}
			hitsP++;
		}
	}
	for (std::size_t i=0; i<NlIn2; i++)
	{
		if ((A2.legs[i].notPresent(indices13,1)) and A2.legs[i].notPresent(indices23,1))
		{
			rest32.push_back(i);
			if (A2.legs[i].notPresent(indices34,0)) { rest33.push_back(i); }			
		}
	}
	for (std::size_t i=0; i<NlIn2; i++)
	{
		if ((A3.legs[i].notPresent(indices14,1)) and A3.legs[i].notPresent(indices24,1) and A3.legs[i].notPresent(indices34,1)) { rest43.push_back(i); }
	}

	for (std::size_t i=0; i<NCon13; i++)
	{
		product_dims12_3[i] = Eigen::IndexPair<Index>(merged12[i], indices13[i][1]);
	}
	for (std::size_t i=NCon13; i<NCon13+NCon23; i++)
	{
		product_dims12_3[i] = Eigen::IndexPair<Index>(merged12[i], indices23[i-NCon13][1]);
	}

	std::array<Index,NCon14+NCon24+NCon34> merged123;
	hits = 0;
	hitsP = 0;
	for (std::size_t i=0; i<Nlegs; i++)
	{
		if ((this->legs[i].notPresent(indices12,0)) and (this->legs[i].notPresent(indices13,0)))
		{
			if (this->legs[i].present(indices14,0))
			{
				merged123[hits] = hitsP; hits++;
			}
			hitsP++;
		}
	}
	for (std::size_t i=0; i<NlIn1; i++)
	{
		if ((A1.legs[i].notPresent(indices12,1)) and (A1.legs[i].notPresent(indices23,0)))
		{
			if (A1.legs[i].present(indices24,0))
			{
				merged123[hits] = hitsP; hits++;
			}
			hitsP++;
		}
	}
	for (std::size_t i=0; i<NlIn2; i++)
	{
		if ((A2.legs[i].notPresent(indices13,1)) and (A2.legs[i].notPresent(indices23,1)))
		{
			if (A2.legs[i].present(indices34,0))
			{
				merged123[hits] = hitsP; hits++;
			}
			hitsP++;
		}
	}

	for (std::size_t i=0; i<NCon14; i++)
	{
		product_dims123_4[i] = Eigen::IndexPair<Index>(merged123[i], indices14[i][1]);
	}
	for (std::size_t i=NCon14; i<NCon14+NCon24; i++)
	{
		product_dims123_4[i] = Eigen::IndexPair<Index>(merged123[i], indices24[i-NCon14][1]);
	}
	for (std::size_t i=NCon14+NCon24; i<NCon14+NCon24+NCon34; i++)
	{
		product_dims123_4[i] = Eigen::IndexPair<Index>(merged123[i], indices34[i-NCon14-NCon24][1]);
	}

	//construct the legs to be searched at the different contraction steps:
	std::array<Index,NCon12> legs; //legs to be searched for first contraction
	for (std::size_t i=0; i<NCon12; i++) {legs[i] = indices12[i][1];}
	std::array<Index,NCon13+NCon23> legs1; //legs to be searched for second contraction
	for (std::size_t i=0; i<NCon13; i++) {legs1[i] = indices13[i][1];}
	for (std::size_t i=NCon13; i<NCon13+NCon23; i++) {legs1[i] = indices23[i-NCon13][1];}
	std::array<Index,NCon14+NCon24+NCon34> legs2; //legs to be searched for third contraction
	for (std::size_t i=0; i<NCon14; i++) {legs2[i] = indices14[i][1];}
	for (std::size_t i=NCon14; i<NCon14+NCon24; i++) {legs2[i] = indices24[i-NCon14][1];}
	for (std::size_t i=NCon14+NCon24; i<NCon14+NCon24+NCon34; i++) {legs2[i] = indices34[i-NCon14-NCon24][1];}
	std::array<Index,resRank+Nextra_+::hidden_dim(resRank)> dummy_legs; //all legs of the resulting MultipedeQ
	std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));

	//initialize variables, needed during the loop:
	std::array<qType,NCon12> conIndex;
	std::array<qType,NCon13+NCon23> conIndex1; //search index after the first contraction
	std::array<qType,NCon14+NCon24+NCon34> conIndex2; //search index after the second contraction
	std::array<qType,resRank+Nextra_+::hidden_dim(resRank)> totIndex; //resulting index for the output MultipedeQ.

	std::array<Index,resRank> new_dimsC; //New dimensions for the CGC-Tensor
	std::array<Index,resRank> new_dimsA; //New dimensions for the Block-Tensor
	TensorType<resRank> Ctmp; //intermediate variable for storing the CGC-Tensor
	TensorType<resRank> Cnow; //intermediate variable for CGC-Tensor from already inserted blocks for prop_to check.
	Scalar factor_cgc; //factor which is relevant for the Block-Tensors from the CGC-Tensors
	TensorType<resRank> Atmp; //intermediate variable for storing the Block-Tensor
	
	//start the loops
	for (std::size_t nu=0; nu<this->size(); ++nu)
	{
		// cout << "ν=" << nu << endl;
		
		for (std::size_t i=0; i<NCon12; i++) { conIndex[i] = this->index[nu][indices12[i][0]]; }
		auto range1 = A1.dict.equal_range(conIndex,legs);	
		for (auto its1 = range1.first; its1 != range1.second; ++its1)
		{
			std::size_t mu = its1->second;
			// cout << "ν=" << nu << ", μ=" << mu << endl;
			
			for (std::size_t i=0; i<NCon13; i++) { conIndex1[i] = this->index[nu][indices13[i][0]]; }
			for (std::size_t i=NCon13; i<NCon13+NCon23; i++) { conIndex1[i] = A1.index[mu][indices23[i-NCon13][0]]; }
			auto range2 = A2.dict.equal_range(conIndex1,legs1);
			for (auto its2 = range2.first; its2 != range2.second; ++its2)
			{
				std::size_t kappa = its2->second;
				// cout << "ν=" << nu << ", μ=" << mu << ", κ=" << kappa << endl;

				for (std::size_t i=0; i<NCon14; i++) { conIndex2[i] = this->index[nu][indices14[i][0]]; }
				for (std::size_t i=NCon14; i<NCon14+NCon24; i++) { conIndex2[i] = A1.index[mu][indices24[i-NCon14][0]]; }
				for (std::size_t i=NCon14+NCon24; i<NCon14+NCon24+NCon34; i++) { conIndex2[i] = A2.index[kappa][indices34[i-NCon14-NCon24][0]]; }
				auto range3 = A3.dict.equal_range(conIndex2,legs2);
				for (auto its3 = range3.first; its3 != range3.second; ++its3)
				{
					std::size_t lambda = its3->second;
					// cout << "ν=" << nu << ", μ=" << mu << ", κ=" << kappa << ", λ=" << lambda << endl;
					
					//contructing the resulting index.
					//Calculate the dimensions for the contracted tensors (necessary for parallelization).
					for(std::size_t i=0; i<rest13.size(); i++)
					{
						totIndex[i] = this->index[nu][rest13[i]];
						if constexpr ( Symmetry::HAS_CGC ) { new_dimsC[i]=this->cgc[nu].dimension(rest13[i]); }
						new_dimsA[i]=this->block[nu].dimension(rest13[i]);
					}
					for(std::size_t i=0; i<rest23.size(); i++)
					{
						totIndex[i+rest13.size()] = A1.index[mu][rest23[i]];
						if constexpr ( Symmetry::HAS_CGC ) { new_dimsC[i+rest13.size()] = A1.cgc[mu].dimension(rest23[i]);	}
						new_dimsA[i+rest13.size()] = A1.block[mu].dimension(rest23[i]);
					}
					for(std::size_t i=0; i<rest33.size(); i++)
					{
						totIndex[i+rest13.size()+rest23.size()] = A2.index[kappa][rest33[i]];
						if constexpr ( Symmetry::HAS_CGC ) { new_dimsC[i+rest13.size()+rest23.size()] = A2.cgc[kappa].dimension(rest33[i]); }
						new_dimsA[i+rest13.size()+rest23.size()] = A2.block[kappa].dimension(rest33[i]);
					}
					for(std::size_t i=0; i<rest43.size(); i++)
					{
						totIndex[i+rest13.size()+rest23.size()+rest33.size()] = A3.index[lambda][rest43[i]];
						if constexpr ( Symmetry::HAS_CGC ) { new_dimsC[i+rest13.size()+rest23.size()+rest33.size()] = A3.cgc[lambda].dimension(rest43[i]); }
						new_dimsA[i+rest13.size()+rest23.size()+rest33.size()] = A3.block[lambda].dimension(rest43[i]);
					}
					if constexpr ( Nextra_ > 0 ) { totIndex[resRank] = A1.index[mu][3]; }
					
					//if Symmetry has CGC calculate the contracted CGC tensor and check the flow equations.
					if constexpr ( Symmetry::HAS_CGC )
						{
							Ctmp.resize(new_dimsC);
							Ctmp.device(device.get()) = ((this->cgc[nu].contract(A1.cgc[mu],product_dims1_2))
												 .contract(A2.cgc[kappa],product_dims12_3)).contract(A3.cgc[lambda],product_dims123_4);
							Scalar norm = ::sumAbs(Ctmp);
							if (norm < ::numeric_limits<Scalar>::epsilon()) { continue; }
						}
					else if constexpr ( Symmetry::SPECIAL )
					{
						if( MODE == con::four::MODE::BUILD_R )
						{
							factor_cgc = Symmetry::coeff_buildR(A1.index[mu][1],A1.index[mu][2],A1.index[mu][0], 
															A3.index[lambda][1],A3.index[lambda][4],A3.index[lambda][0],
															A2.index[kappa][1],A2.index[kappa][2],A2.index[kappa][0]);
							// factor_cgc = Symmetry::coeff_9j(A1.index[mu][1],A1.index[mu][2],A1.index[mu][0], 
							// 								A3.index[lambda][1],A3.index[lambda][4],A3.index[lambda][0],
							// 								A2.index[kappa][1],A2.index[kappa][2],A2.index[kappa][0])*
							// 	std::pow(static_cast<Scalar>(A1.index[mu][0]*A3.index[lambda][0]*A2.index[kappa][1]*A2.index[kappa][2]),Scalar(0.5));
						}
						else if ( MODE == con::four::MODE::BUILD_L )
						{
							factor_cgc = Symmetry::coeff_buildL(A1.index[mu][1],A1.index[mu][2],A1.index[mu][0],
															A3.index[lambda][1],A3.index[lambda][4],A3.index[lambda][0],
															A2.index[kappa][1],A2.index[kappa][2],A2.index[kappa][0]);
							// factor_cgc = Symmetry::coeff_9j(A1.index[mu][1],A1.index[mu][2],A1.index[mu][0],
							// 								A3.index[lambda][1],A3.index[lambda][4],A3.index[lambda][0],
							// 								A2.index[kappa][1],A2.index[kappa][2],A2.index[kappa][0])*
							// 	std::pow(static_cast<Scalar>(A1.index[mu][0]*A3.index[lambda][0]*A2.index[kappa][1]*A2.index[kappa][2]),Scalar(0.5))*
							// 	A2.index[kappa][0]*std::pow(static_cast<Scalar>(A2.index[kappa][1]),-1.);
						}
						else if ( MODE == con::four::MODE::H_PSI )
						{
							factor_cgc = Symmetry::coeff_HPsi(A1.index[mu][1],A1.index[mu][2],A1.index[mu][0], 
															A3.index[lambda][1],A3.index[lambda][4],A3.index[lambda][0],
															A2.index[kappa][1],A3.index[lambda][2],this->index[nu][1]);
							// factor_cgc = Symmetry::coeff_9j(A1.index[mu][1],A1.index[mu][2],A1.index[mu][0], 
							// 								A3.index[lambda][1],A3.index[lambda][4],A3.index[lambda][0],
							// 								A2.index[kappa][1],A3.index[lambda][2],this->index[nu][1])*
							// 	std::pow(static_cast<Scalar>(A1.index[mu][0]*A3.index[lambda][0]*A2.index[kappa][1]*A3.index[lambda][2]),Scalar(0.5))*
							// 	this->index[nu][1]*std::pow(static_cast<Scalar>(A2.index[kappa][1]),-1.);
						}
						else if ( MODE == con::four::MODE::UNITY or MODE == con::four::MODE::APAIR ) { factor_cgc = Symmetry::coeff_unity(); }
						if (std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon()) { continue; }
					}
					else
					{
						factor_cgc = 1.;
					}
					//calculate the contracted A tensor.
					Atmp.resize(new_dimsA);
					Atmp.device(device.get()) = ((this->block[nu].contract(A1.block[mu],product_dims1_2))
													 .contract(A2.block[kappa],product_dims12_3)).contract(A3.block[lambda],product_dims123_4);

					auto inner_range = Mout.dict.equal_range(totIndex,dummy_legs);
					if ( inner_range.first != Mout.dict.end(dummy_legs) )
					{
						// TODO: Get rid of check with prop_to method but check it by "remembering" quantum numbers.
						// check if the new tensor can be added to an existing block.
						for (auto inner_its = inner_range.first; inner_its != inner_range.second; ++inner_its)
						{
							if constexpr ( Symmetry::HAS_CGC )
								{
									Cnow = Mout.cgc[inner_its->second];
									factor_cgc = prop_to(Ctmp, Cnow);
									if ( factor_cgc != std::numeric_limits<Scalar>::infinity() ) //add the block because cgc are proportional
									{
										assert(std::abs(factor_cgc) < ::numeric_limits<Scalar>::infinity() and "factor from cgcs too large");
										Atmp.device(device.get()) = Atmp * factor_cgc;
										Mout.block[inner_its->second].device(device.get()) += Atmp;
										break;
									}
								}
							else
							{
								Atmp.device(device.get()) = Atmp * factor_cgc;
								Mout.block[inner_its->second].device(device.get()) += Atmp;
								break;
							}
						}
						//Remark: the following commented code is only needed if you have a SU(N) symmetry with N>2.
						// if ( factor_cgc == std::numeric_limits<Scalar>::infinity() )
						// {
						// 	// assert(resRank > 3 and "More then one block is only possible for Tensors with rank > 3");
						// 	if constexpr ( Symmetry::HAS_CGC )
						// 		{
						// 			Mout.push_back(totIndex,Atmp,Ctmp);
						// 			break;
						// 		}
						// 	else
						// 	{
						// 		Atmp.device(device.get()) = Atmp * factor_cgc;
						// 		Mout.push_back(totIndex,Atmp);
						// 		break;
						// 	}
						// }
					}
					else //the block does not exist yet -> insert this block.
					{
						if constexpr ( Symmetry::HAS_CGC )
							{
								Mout.push_back(totIndex,Atmp,Ctmp);
							}
						else
						{
							Atmp.device(device.get()) = Atmp * factor_cgc;
							Mout.push_back(totIndex,Atmp);
						}
					}
				}// end of lambda loop
			}//end of kappa loop
		}//end of mu loop
	}//end of nu loop
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<Eigen::Index Nextra_, Eigen::Index NlIn1, Eigen::Index NlIn2,
		 std::size_t NCon12, std::size_t NCon13, std::size_t NCon23,
		 Eigen::Index Nextra1, Eigen::Index Nextra2,
		 typename base_type1, typename base_type2>
MultipedeQ<Nlegs+NlIn1+NlIn2-2*static_cast<Eigen::Index>(NCon12+NCon13+NCon23),Symmetry,Scalar,Nextra_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
	contract(const MultipedeQ<NlIn1,Symmetry,Scalar,Nextra1,base_type1> &A1,
			 const MultipedeQ<NlIn2,Symmetry,Scalar,Nextra2,base_type2> &A2,
			 const std::array<std::array<Index,2>,NCon12> indices12,
			 const std::array<std::array<Index,2>,NCon13> indices13,
			 const std::array<std::array<Index,2>,NCon23> indices23,
			 const con::three::MODE MODE) const
{
	//Check if one of the MultipedeQ is empty and if the contraction legs match with their directions.
	assert(this->size()>0 and A1.size()>0 and A2.size()>0 and "One of the MultipedeQs has zero size.");
	for (std::size_t i=0; i<NCon12; i++)
	{
		assert( (this->legDirection(indices12[i][0]) != A1.legDirection(indices12[i][1])) );
	}
	for (std::size_t i=0; i<NCon13; i++)
	{
		assert( (this->legDirection(indices13[i][0]) != A2.legDirection(indices13[i][1])) );
	}
	for (std::size_t i=0; i<NCon23; i++)
	{
		assert( (A1.legDirection(indices23[i][0]) != A2.legDirection(indices23[i][1])) );
	}

	constexpr Index resRank = Nlegs+NlIn1+NlIn2-2*static_cast<Index>(NCon12+NCon13+NCon23);
	typedef MultipedeQ<resRank,Symmetry,Scalar,Nextra_> Self_;
	
	Self_ Mout;

    //Construct the directions and names of the resulting MultipedeQ. 
	std::size_t hits=0;
	for (std::size_t i=0; i<Nlegs; i++)
	{
		if ((this->legs[i].notPresent(indices12,0)) and (this->legs[i].notPresent(indices13,0))) { Mout.legs[hits]=this->legs[i]; hits++; }
	}
	for (std::size_t i=0; i<NlIn1; i++)
	{
		if ((A1.legs[i].notPresent(indices12,1)) and (A1.legs[i].notPresent(indices23,0))) { Mout.legs[hits]=A1.legs[i]; hits++; }
	}
	for (std::size_t i=0; i<NlIn2; i++)
	{
		if ((A2.legs[i].notPresent(indices13,1)) and (A2.legs[i].notPresent(indices23,1))) { Mout.legs[hits]=A2.legs[i]; hits++; }
	}
	assert(hits == resRank);
	for (std::size_t q=0; q<Mout.rank(); q++) { Mout.legs[q].place = q; }

	std::array<Eigen::IndexPair<Index>, NCon12> product_dims12;
	for (std::size_t i=0; i<NCon12; i++) { product_dims12[i] = Eigen::IndexPair<Index>(indices12[i][0], indices12[i][1]); }
	std::array<Eigen::IndexPair<Index>, NCon13+NCon23> product_dims_merged;

	std::array<Index,NCon13+NCon23> merged;
	hits = 0;
	Index hitsP = 0;
	for (std::size_t i=0; i<Nlegs; i++)
	{
		if ((this->legs[i].notPresent(indices12,0)))
		{
			if (this->legs[i].present(indices13,0))
			{
				merged[hits] = hitsP; hits++;
			}
			hitsP++;
		}
	}
	for (std::size_t i=0; i<NlIn1; i++)
	{
		if ((A1.legs[i].notPresent(indices12,1)))
		{
			if (A1.legs[i].present(indices23,0))
			{
				merged[hits] = hitsP; hits++;
			}
			hitsP++;
		}
	}
	for (std::size_t i=0; i<NCon13; i++)
	{
		product_dims_merged[i] = Eigen::IndexPair<Index>(merged[i], indices13[i][1]);
	}
	for (std::size_t i=NCon13; i<NCon13+NCon23; i++)
	{
		product_dims_merged[i] = Eigen::IndexPair<Index>(merged[i], indices23[i-NCon13][1]);
	}

	//initalize variables needed during the loop.
	std::array<qType,NCon12> conIndex;
	std::array<qType,NCon13+NCon23> conIndex1;
	std::array<qType,resRank+Nextra_+::hidden_dim(resRank)> totIndex;
	std::array<qType,resRank> totIndex_;
	
	std::array<Index,NCon12> legs;
	std::array<Index,NCon13+NCon23> legs1;
	std::array<Index,resRank+::hidden_dim(resRank)> dummy_legs;

	std::array<Index,resRank> new_dimsC;
	std::array<Index,resRank> new_dimsA;
	TensorType<resRank> Atmp;
	TensorType<resRank> Ctmp;
	TensorType<resRank> Cnow;
	Scalar factor_cgc;

	for (std::size_t nu=0; nu<this->size(); ++nu)
	{
		for (std::size_t i=0; i<NCon12; i++) { conIndex[i] = this->index[nu][indices12[i][0]]; legs[i] = indices12[i][1]; }
		auto range1 = A1.dict.equal_range(conIndex,legs);
	
		for (auto its1 = range1.first; its1 != range1.second; ++its1)
		{
			std::size_t mu = its1->second;
			for (std::size_t i=0; i<NCon13; i++) { conIndex1[i] = this->index[nu][indices13[i][0]]; legs1[i] = indices13[i][1]; }
			for (std::size_t i=NCon13; i<NCon13+NCon23; i++) { conIndex1[i] = A1.index[mu][indices23[i-NCon13][0]]; legs1[i] = indices23[i-NCon13][1]; }
			auto range2 = A2.dict.equal_range(conIndex1,legs1);
			for (auto its2 = range2.first; its2 != range2.second; ++its2)
			{
				std::size_t kappa = its2->second;
				// cout << "ν=" << nu << ", μ=" << mu << ", κ=" << kappa << endl;
								
				//contructing the resulting index.
				//Calculate the dimensions for the contracted tensors (necessary for parallelization).
				hits=0;
				for (std::size_t i=0; i<Nlegs; i++)
				{
					if ((this->legs[i].notPresent(indices12,0)) and (this->legs[i].notPresent(indices13,0)))
					{
						totIndex[hits] = this->index[nu][i];
						if constexpr ( Symmetry::HAS_CGC )
							{
								new_dimsC[hits] = this->cgc[nu].dimension(i);
							}
						new_dimsA[hits] = this->block[nu].dimension(i);
						hits++;
					}
				}
				for (std::size_t i=0; i<NlIn1; i++)
				{
					if ((A1.legs[i].notPresent(indices12,1)) and (A1.legs[i].notPresent(indices23,0)))
					{
						totIndex[hits] = A1.index[mu][i];
						if constexpr ( Symmetry::HAS_CGC )
							{
								new_dimsC[hits] = A1.cgc[mu].dimension(i);
							}
						new_dimsA[hits] = A1.block[mu].dimension(i);
						hits++;
					}
				}
				for (std::size_t i=0; i<NlIn2; i++)
				{
					if ((A2.legs[i].notPresent(indices13,1)) and (A2.legs[i].notPresent(indices23,1)))
					{
						totIndex[hits] = A2.index[kappa][i];
						if constexpr ( Symmetry::HAS_CGC )
							{
								new_dimsC[hits] = A2.cgc[kappa].dimension(i);
							}
						new_dimsA[hits] = A2.block[kappa].dimension(i);
						hits++;
					}
				}
				auto hits_rem = hits;
				std::size_t extra_range;
				std::vector<qType> extra;
				if ( MODE == con::three::MODE::FUSE_WPAIR )
				{
					extra = Symmetry::reduceSilent(this->index[nu][7],this->index[nu][8]);
					extra_range = extra.size();
				}
				else { extra_range=1; }
				for ( std::size_t lambda=0; lambda<extra_range; lambda++)
				{
					hits = hits_rem;
					//if we get a MultipedeQ with rank > 3, we need to store the contracted index, as a hidden index
					if constexpr ( ::hidden_dim(resRank) > 0 )
								 {
									 if ( MODE == con::three::MODE::FUSE_WPAIR )
									 {
										 totIndex[hits] = extra[lambda];
										 hits++;
									 }
									 else
									 {
										 for (std::size_t i=0; i<Nlegs; i++)
										 {
											 if ((this->legs[i].present(indices12,0)))
											 {
												 totIndex[hits] = this->index[nu][i];
												 hits++;
											 }
										 }
										 for (std::size_t i=0; i<NlIn1; i++)
										 {
											 if ((A2.legs[i].present(indices23,1)))
											 {
												 totIndex[hits] = A2.index[kappa][i];
												 hits++;
											 }
										 }
									 }
								 }
					//check if you did things correct.
					assert(hits == resRank+Nextra_+::hidden_dim(resRank));
					// std::cout << "tot= "; for(const auto& i : totIndex) {std::cout << i << " ";} std::cout << std::endl;
					//check if the new index is allowed from Symmetry perspektives. Otherwise got to front of the loop.
					std::copy(totIndex.begin(),totIndex.begin()+resRank,totIndex_.begin());
					if( !Symmetry::validate(totIndex_) ) { continue; }

					//calculate the contracted CGC tensor.
					if constexpr ( Symmetry::HAS_CGC )
						{
							Ctmp.resize(new_dimsC);
							Ctmp.device(device.get()) = (this->cgc[nu].contract(A1.cgc[mu],product_dims12)).contract(A2.cgc[kappa],product_dims_merged);
							Scalar norm = ::sumAbs(Ctmp);
							if (norm < ::numeric_limits<Scalar>::epsilon()) { continue; }
						}

					if constexpr ( Symmetry::SPECIAL )
						{
							factor_cgc = Scalar(1);
							if ( MODE == con::three::MODE::FUSE_WPAIR )
							{
								factor_cgc = Symmetry::coeff_Wpair(this->index[nu][0],this->index[nu][7],this->index[nu][6],
																   this->index[nu][8],this->index[nu][3],extra[lambda],
																   this->index[nu][7],this->index[nu][8],extra[lambda],
																   this->index[nu][2],this->index[nu][5],A2.index[kappa][2]);
								// factor_cgc = Symmetry::coeff_6j(this->index[nu][0],this->index[nu][7],this->index[nu][6],
								// 								this->index[nu][8],this->index[nu][3],extra[lambda])*
								// 	std::pow(static_cast<Scalar>(this->index[nu][6]*extra[lambda]),Scalar(0.5))*
								// 	Symmetry::coeff_9j(this->index[nu][1],this->index[nu][4],A1.index[mu][2],
								// 					   this->index[nu][7],this->index[nu][8],extra[lambda],
								// 					   this->index[nu][2],this->index[nu][5],A2.index[kappa][2])*
								// 	std::pow(static_cast<Scalar>(this->index[nu][2]*this->index[nu][5]*A1.index[mu][2]*extra[lambda]),Scalar(0.5))*
								// 	std::pow(Scalar(-1),
								// 			 0.5*(static_cast<Scalar>(this->index[nu][0]+this->index[nu][3]+this->index[nu][7]+this->index[nu][8]-4)));
							}
							else if ( MODE == con::three::MODE::DOT )
							{
								factor_cgc = Symmetry::coeff_dot(A2.index[kappa][1]);
							}
							if ( std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon() ) { continue; }
						}
					else
					{
						factor_cgc = Scalar(1);
					}
					//calculate the contracted A tensor.
					Atmp.resize(new_dimsA);
					Atmp.device(device.get()) = (this->block[nu].contract(A1.block[mu],product_dims12)).contract(A2.block[kappa],product_dims_merged);

					std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));
					auto inner_range = Mout.dict.equal_range(totIndex,dummy_legs);
					std::size_t numbers_before = Mout.dict.count(totIndex,dummy_legs);

					if (numbers_before != 0) //entry already exists.
					{
						// TODO: Get rid of check with prop_to method but check it by "remembering" quantum numbers.
						// check if the new tensor can be added to an existing block.
						if constexpr ( Symmetry::HAS_CGC )
							{
								for (auto inner_its = inner_range.first; inner_its != inner_range.second; ++inner_its)
								{
									Cnow = Mout.cgc[inner_its->second];
									factor_cgc = prop_to(Ctmp, Cnow);
									if ( factor_cgc != std::numeric_limits<Scalar>::infinity() ) //add the block because cgc are proportional
									{
										assert(std::abs(factor_cgc) < ::numeric_limits<Scalar>::infinity() and "factor from cgcs too large");
										Atmp.device(device.get()) = Atmp * factor_cgc;
										Mout.block[inner_its->second].device(device.get()) += Atmp;
										break;
									}
								}
							}
						else
						{
							for (auto inner_its = inner_range.first; inner_its != inner_range.second; ++inner_its)
							{
								Atmp.device(device.get()) = Atmp * factor_cgc;
								// auto dims1 = Mout.block[inner_its->second].dimensions();
								// for ( auto i: dims1 ) {std::cout << i << " ";} std::cout << std::endl;
								// for ( auto i: Mout.index[inner_its->second] ) {std::cout << i << " ";} std::cout << std::endl;
								// auto dims2 = Atmp.dimensions();
								// for ( auto i: dims2 ) {std::cout << i << " ";} std::cout << std::endl;
								// for ( auto i: totIndex ) {std::cout << i << " ";} std::cout << std::endl;
								// std::cout << std::endl;
								Mout.block[inner_its->second].device(device.get()) += Atmp;
							}
						}
						//if not: insert a new block with the same index. Remark: We use std::unorderd_multimap, so this is possible.
						// if ( factor_cgc == std::numeric_limits<Scalar>::infinity() )
						// {
						// 	if constexpr ( Symmetry::HAS_CGC )
						// 		{
						// 			Mout.push_back(totIndex,Atmp,Ctmp);
						// 		}
						// 	else
						// 	{
						// 		Atmp.device(device.get()) = Atmp * factor_cgc;
						// 		Mout.push_back(totIndex,Atmp);								
						// 	}
						// }
					}
					else //the block does not exist yet -> insert this block.
					{
						if constexpr ( Symmetry::HAS_CGC )
							{
								Mout.push_back(totIndex,Atmp,Ctmp);
							}
						else
						{
							Atmp.device(device.get()) = Atmp * factor_cgc;
							Mout.push_back(totIndex,Atmp);								
						}
					}
				} //end of lambda loop
			} //end of kappa loop
		} //end of mu loop
	} //end of nu loop
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<Eigen::Index Nextra_, Eigen::Index NlIn, std::size_t NCon, Eigen::Index Nextra1, typename base_type1>
MultipedeQ<Nlegs+NlIn-2*static_cast<Eigen::Index>(NCon),Symmetry,Scalar,Nextra_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
	contract(const MultipedeQ<NlIn,Symmetry,Scalar,Nextra1,base_type1> &A, const std::array<std::array<Index,2>,NCon> indices, const con::two::MODE MODE) const
{
	//Check if one of the MultipedeQ is empty and if the contraction legs match with their directions.
	assert(this->size()>0 and A.size()>0 and "One of the MultipedeQs has zero size.");
	for (std::size_t i=0; i<NCon; i++)
	{
		assert( (this->legDirection(indices[i][0]) != A.legDirection(indices[i][1])) );
	}

	constexpr Index resRank = Nlegs+NlIn-2*static_cast<Index>(NCon);
	MultipedeQ<resRank,Symmetry,Scalar,Nextra_> Mout;

	//Construct the directions and names of the resulting MultipedeQ. 
	std:size_t hits=0;
	for (std::size_t i=0; i<Nlegs; i++)
	{
		if ((this->legs[i].notPresent(indices,0))) { Mout.legs[hits]=this->legs[i]; hits++; }
	}
	for (std::size_t i=0; i<NlIn; i++)
	{
		if ((A.legs[i].notPresent(indices,1))) { Mout.legs[hits]=A.legs[i]; hits++; }
	}
	assert(hits == resRank);
	// std::copy_if(this->legs.begin(), this->legs.end(), Mout.legs.begin(),
	// 			 [&](qleg leg){return (leg.isPresent(indices,0)); } );
	// std::copy_if(A.legs.begin(), A.legs.end(), Mout.legs.begin()+NCon,
	// 			 [&](qleg leg){return (leg.isPresent(indices,1)); } );
	for (std::size_t q=0; q<Mout.rank(); q++) { Mout.legs[q].place = q; }

	//initalize contraction dims for Eigen::Tensor contract-method.
	std::array<Eigen::IndexPair<Index>, NCon> product_dims;
	for (std::size_t i=0; i<NCon; i++) { product_dims[i] = Eigen::IndexPair<Index>(indices[i][0], indices[i][1]); }

	//initalize variables needed during the loop.
	std::array<Index,NCon> legs;
	std::array<Index,resRank> new_dimsC; //New dimensions for the CGC-Tensor
	std::array<Index,resRank> new_dimsA; //New dimensions for the Block-Tensor					 
	TensorType<resRank> Ctmp; //intermediate variable for storing the CGC-Tensor
	TensorType<resRank> Cnow; //intermediate variable for CGC-Tensor from already inserted blocks for prop_to check.
	Scalar factor_cgc; //factor which is relevant for the Block-Tensors from the CGC-Tensors
	TensorType<resRank> Atmp; //intermediate variable for storing the Block-Tensor
	std::array<Index,resRank+Nextra_+::hidden_dim(resRank)> dummy_legs;
	std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));

	std::array<qType,NCon> conIndex; //search index after the first contraction
	std::array<qType,resRank+Nextra_+::hidden_dim(resRank)> totIndex; //resulting index for the output MultipedeQ.
	std::array<qType,resRank> totIndex_;

	for (std::size_t nu=0; nu<this->size(); ++nu)
	{
		// std::cout << "ν=" << nu << std::endl;
		for (std::size_t i=0; i<NCon; i++) { conIndex[i] = this->index[nu][indices[i][0]]; legs[i] = indices[i][1]; }
		auto range = A.dict.equal_range(conIndex,legs);
	
		for (auto its = range.first; its != range.second; ++its)
		{
			std::size_t mu = its->second;
			// std::cout << "ν=" << nu << ", μ=" << mu << std::endl;
			
			//contructing the resulting index.
			std::size_t hits=0;
			for (std::size_t i=0; i<Nlegs; i++)
			{
				bool dummy=false;
				std::size_t j=0;
				while (!dummy and (j < NCon) ) { dummy = (i == indices[j][0]); j++; }
				if (!dummy)
				{
					totIndex[hits] = this->index[nu][i];
					hits++;
				}
			}
			for (std::size_t i=0; i<NlIn; i++)
			{
				bool dummy=false;
				std::size_t j=0;
				while (!dummy and (j < NCon) ) { dummy = (i == indices[j][1]); j++; }
				if (!dummy)
				{
					totIndex[hits] = A.index[mu][i];
					hits++;
				}
			}
			//if we get a MultipedeQ with rank > 3, we need to store the contracted indices, as a hidden indices
			if constexpr ( ::hidden_dim(resRank) > 0 )
						 {
							 for (std::size_t i=0; i<Nlegs; i++)
							 {
								 if ((this->legs[i].present(indices,0)))
								 {
									 totIndex[hits] = this->index[nu][i];
									 hits++;
								 }
							 }
							 if ( MODE == con::two::MODE::WPAIR )
							 {
								 totIndex[hits] = this->index[nu][4];
								 hits++;
								 totIndex[hits] = A.index[mu][4];
								 hits++;
							 }
						 }
			if constexpr ( Nextra_ > 0 ) //if we want to store quantum numbers for later use, store them also as hidden indices
						 {
							 if ( MODE == con::two::MODE::FUSE13_APAIR )
							 {
								 totIndex[hits] = this->index[nu][4];
								 hits++;
							 }
							 else if ( MODE == con::two::MODE::SPLIT_APAIR )
							 {
								 totIndex[hits] = this->index[nu][3];
								 hits++;
							 }
						 }
			//check if you did things correct.
			assert(hits == resRank+Nextra_+::hidden_dim(resRank));
			
			//check if the new index is allowed from Symmetry perspektives. Otherwise go to front of the loop.
			std::copy(totIndex.begin(),totIndex.begin()+resRank,totIndex_.begin());
			if constexpr( resRank == 2 )
						{
							if (!Symmetry::validate(totIndex_)) { continue; }
						}
			if constexpr ( Symmetry::SPECIAL )
				{
					if ( MODE == con::two::MODE::FUSE01_APAIR )
					{
						if( A.index[mu][4] != this->index[nu][2] ) { continue; }
					}
				}
			
			//Calculate the dimensions for the contracted tensors (necessary for parallelization).
			hits=0;
			for (std::size_t i=0; i<Nlegs; i++)
			{
				if ((this->legs[i].notPresent(indices,0)))
				{
					if constexpr ( Symmetry::HAS_CGC )
						{
							new_dimsC[hits] = this->cgc[nu].dimension(i);
						}
					new_dimsA[hits] = this->block[nu].dimension(i);
					hits++;
				}
			}
			for (std::size_t i=0; i<NlIn; i++)
			{
				if ((A.legs[i].notPresent(indices,1)))
				{
					if constexpr ( Symmetry::HAS_CGC )
						{
							new_dimsC[hits] = A.cgc[mu].dimension(i);
						}
					new_dimsA[hits] = A.block[mu].dimension(i);
					hits++;
				}
			}
			assert(hits == resRank);

			//calculate the contracted CGC tensor.
			if constexpr ( Symmetry::HAS_CGC )
				{
					Ctmp.resize(new_dimsC);
					Ctmp.device(device.get()) = this->cgc[nu].contract(A.cgc[mu],product_dims);
					Scalar norm = ::sumAbs(Ctmp);
					if (norm < ::numeric_limits<Scalar>::epsilon()) { continue; }
				}
			else if constexpr ( Symmetry::SPECIAL )
				{
					factor_cgc = Scalar(1);
					if ( MODE == con::two::MODE::OORR )
					{
						factor_cgc = Symmetry::coeff_rightOrtho(this->out(nu),this->in(nu));
						// factor_cgc = static_cast<Scalar>(this->out(nu))*std::pow(static_cast<Scalar>(this->in(nu)),-1.);
					}
					else if ( MODE == con::two::MODE::OLRR )
					{
						factor_cgc = Symmetry::coeff_leftSweep(this->out(nu),this->in(nu),this->mid(nu));
						// factor_cgc = std::pow(static_cast<Scalar>(this->out(nu)),0.5)*std::pow(static_cast<Scalar>(this->in(nu)),-0.5)*
						// 	(-1.)*std::pow(-1,0.5*(this->mid(nu)+this->out(nu)-this->in(nu)-1));
					}
					else if ( MODE == con::two::MODE::SIGN )
					{
						factor_cgc = Symmetry::coeff_sign(this->out(nu),this->in(nu),this->mid(nu));
						// factor_cgc = std::pow(static_cast<Scalar>(this->in(nu)),0.5)*std::pow(static_cast<Scalar>(this->out(nu)),-0.5)*
						// 	(-1.)*std::pow(-1,0.5*(this->mid(nu)+this->out(nu)-this->in(nu)-1));
					}
					else if ( MODE == con::two::MODE::DOT )
					{
						factor_cgc = Symmetry::coeff_dot(this->out(nu));
						// factor_cgc = static_cast<Scalar>(this->out(nu)[0]);//*static_cast<Scalar>(this->mid(nu));
					}
					else if
						( MODE == con::two::MODE::LLRR or MODE == con::two::MODE::NONE or MODE == con::two::MODE::UNITY or MODE == con::two::MODE::APAIR )
					{
						factor_cgc = Symmetry::coeff_unity();
					}
					else if ( MODE == con::two::MODE::FUSE13_APAIR )
					{
						factor_cgc = Symmetry::coeff_Apair(this->index[nu][0],this->index[nu][1],this->index[nu][4],
														   this->index[nu][3],this->index[nu][2],A.index[mu][2]);
						// factor_cgc = Symmetry::coeff_6j(this->index[nu][0],this->index[nu][1],this->index[nu][4],
						// 								this->index[nu][3],this->index[nu][2],A.index[mu][2])*
						// 	std::pow(static_cast<Scalar>(this->index[nu][4]*A.index[mu][2]),Scalar(0.5))*
						// 	// std::pow(Scalar(-1),0.5*(static_cast<Scalar>(this->index[nu][0]+this->index[nu][1]+this->index[nu][2]+this->index[nu][3]-4)));
						// 	std::pow(Scalar(-1),0.5*(static_cast<Scalar>(this->index[nu][0]+this->index[nu][2]+A.index[mu][2]-3)));
					}
					else if ( MODE == con::two::MODE::SPLIT_APAIR )
					{
						factor_cgc = Symmetry::coeff_Apair(this->index[nu][0],A.index[mu][0],this->index[nu][3],
														   A.index[mu][1],this->index[nu][1],A.index[mu][2]);
						// factor_cgc = Symmetry::coeff_6j(this->index[nu][0],A.index[mu][0],this->index[nu][3],
						// 								A.index[mu][1],this->index[nu][1],A.index[mu][2])*
						// 	std::pow(static_cast<Scalar>(this->index[nu][3]*A.index[mu][2]),Scalar(0.5))*
						// 	std::pow(Scalar(-1),0.5*(static_cast<Scalar>(this->index[nu][0]+this->index[nu][1]+A.index[mu][2]-3)));
					}

					if (std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon()) { continue; }
				}
			else
			{
				factor_cgc = Scalar(1);
			}
			//calculate the contracted A tensor.
			Atmp.resize(new_dimsA);
			Atmp.device(device.get()) = this->block[nu].contract(A.block[mu],product_dims);

			//Get pointers to all blocks added with the same index totIndex.
			auto inner_range = Mout.dict.equal_range(totIndex,dummy_legs);
			std::size_t numbers_before = Mout.dict.count(totIndex,dummy_legs);
			if (numbers_before != 0) //entry already exists.
			{
				// check if the new tensor can be added to an existing block.
				// TODO: Get rid of check with prop_to method but check it by "remembering" quantum numbers.
				if constexpr ( Symmetry::HAS_CGC )
					{
						for (auto inner_its = inner_range.first; inner_its != inner_range.second; ++inner_its)
						{
							Cnow = Mout.cgc[inner_its->second];
							factor_cgc = prop_to(Ctmp, Cnow);
							if ( factor_cgc != std::numeric_limits<Scalar>::infinity() ) //add the block because cgc are proportional
							{
								assert(std::abs(factor_cgc) < ::numeric_limits<Scalar>::infinity() and "factor from cgcs too large");
								Atmp.device(device.get()) = Atmp * factor_cgc;
								Mout.block[inner_its->second].device(device.get()) += Atmp;
								break;
							}
						}
					}
				else
				{
					for (auto inner_its = inner_range.first; inner_its != inner_range.second; ++inner_its)
					{
						Atmp.device(device.get()) = Atmp * factor_cgc;
						Mout.block[inner_its->second].device(device.get()) += Atmp;
					}
				}
				//if not: insert a new block with the same index. Remark: We use std::unorderd_multimap, so this is possible.
				// if ( factor_cgc == std::numeric_limits<Scalar>::infinity() )
				// {
				// 	if constexpr ( Symmetry::HAS_CGC )
				// 		{
				// 			Mout.push_back(totIndex,Atmp,Ctmp);
				// 		}
				// 	else
				// 	{
				//      Atmp.device(device.get()) = Atmp * factor_cgc;
				// 		Mout.push_back(totIndex,Atmp);								
				// 	}
				// }
			}
			else //the block does not exist yet -> insert this block.
			{
				if constexpr ( Symmetry::HAS_CGC )
					{
						Mout.push_back(totIndex,Atmp,Ctmp);
					}
				else
				{
					Atmp.device(device.get()) = Atmp * factor_cgc;
					Mout.push_back(totIndex,Atmp);								
				}
			}
		}//end of mu loop
	}//end of nu loop
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<std::size_t rank, std::size_t cdim, std::size_t offset>
std::size_t MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
transfer_leg(std::array<std::size_t,cdim> conLegs, std::size_t transLeg)
{
	auto it = std::find(conLegs.begin(),conLegs.end(),transLeg);
	if (it != conLegs.end()) { return std::numeric_limits<std::size_t>::infinity(); }
	else
	{
		std::size_t count = 0;
		for (std::size_t i=0; i<transLeg; i++)
		{
			auto it2 = std::find(conLegs.begin(),conLegs.end(),i);
			if (it2 == conLegs.end()) { count++; }
			return offset+count;
		}
	}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<3,Symmetry,Scalar,0> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
fuse( const Index leg1, const Index leg2 ) const
{
	assert( this->legs[leg1].getDir() == this->legs[leg2].getDir() and "leg directions of the fused legs have to be equal.");
	std::array<dir,3> new_dirs;
	// std::array<string,3> new_names;
	new_dirs[0] = this->legs[leg1].getFlipDir();
	new_dirs[1] = this->legs[leg2].getFlipDir();
	new_dirs[2] = this->legs[leg1].getDir();

	// new_names[0] = "left";
	// new_names[1] = "right";
	// new_names[2] = "fused";
	
	MultipedeQ<3,Symmetry,Scalar,0> fuseT(new_dirs); //,new_names);
	for (std::size_t q=0; q<3; q++) {fuseT.legs[q].place = q;}
	std::unordered_set<std::array<qType,2> > doubleCounting;

	//preparing the block structure of the Fuse tensor using the reduce-method and set Clebsch-Gordon-coefficients.
	for (std::size_t nu=0; nu<this->size(); nu++)
	{
		std::array<qType,2> redIndex = { index[nu][leg1], index[nu][leg2] };
		auto it = doubleCounting.find(redIndex);
		if (it != doubleCounting.end()) {continue;}
		else
		{
			doubleCounting.insert(redIndex);
			// Index dim1 = block[nu].dimension(leg1);
			// Index dim2 = block[nu].dimension(leg2);
			auto qs = Symmetry::reduceSilent(index[nu][leg1],index[nu][leg2]);
			// auto Tcgc = ::reduce<Scalar>(static_cast<int>(index[nu][leg1][0]),static_cast<int>(index[nu][leg2][0]));
			for (std::size_t i=0; i<qs.size(); i++)
			{
				std::array<qType,3+0+::hidden_dim(3)> totIndex;
				totIndex[0] = index[nu][leg1];
				totIndex[1] = index[nu][leg2];
				totIndex[2] = qs[i];
				TensorType<3> A;
				if constexpr ( Symmetry::HAS_CGC )
					{
						std::array<Index,3> shuffle_dims{0,2,1};
						TensorType<3> T = Symmetry::reduce(index[nu][leg1],index[nu][leg2],qs[i]).shuffle(shuffle_dims);
						fuseT.push_back(totIndex,A,T);
					}
				else
				{
					fuseT.push_back(totIndex,A);					
				}
			}
		}
	}

	//Setting the sizes and the content of the A-Tensors for the Fuse tensor.
	std::unordered_set<std::size_t> doubleNus;
	std::array<Index,3-2> legFuse = {2};
	auto outsetFuse = fuseT.findUniqueQs(legFuse); //set of unique quantum numbers at output of the fuse Tensor.
	
	std::array<Index,Nlegs+Nextra+::hidden_dim(Nlegs)+::correct_dim(Nlegs)-2> legT;
	// std::array<Index,Nlegs-2> legT;
	std::size_t count=0;
	for (std::size_t i=0; i<this->rank()+Nextra+::hidden_dim(this->rank())+::correct_dim(Nlegs); i++)
	// for (std::size_t i=0; i<this->rank(); i++)
	{
		if ( i != leg1 and i != leg2 ) { legT[count] = static_cast<Index>(i); count++; }
	}
	auto outsetT = this->findUniqueQs(legT); //set of unique quantum numbers at remaining positions of the original tensor.
	for (auto itFuse=outsetFuse.begin(); itFuse!=outsetFuse.end(); itFuse++)
		for (auto itT=outsetT.begin(); itT!=outsetT.end(); itT++)
		{
			std::vector<std::size_t> nuVec, muVec;
			std::vector<Index> dims;
			auto nuList = fuseT.filterRows(legFuse,*itFuse);
			auto muList = this->filterRows(legT,*itT);
			std::unordered_set<std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)+::correct_dim(Nlegs)> > uniqueController;
			for (std::size_t i=0; i<nuList.size(); i++)
				for (std::size_t j=0; j<muList.size(); j++)
				{
					//find nu and mu which merge together. //Does this operator == can be overloaded from Symmetry??? as long as qType=int, no problem
					if ( fuseT.index[nuList[i]][0] == this->index[muList[j]][leg1] and fuseT.index[nuList[i]][1] == this->index[muList[j]][leg2] )
					{
						std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)+::correct_dim(Nlegs)> checkIndex;
						if constexpr( ::correct_dim(Nlegs) == -1 )
									{
										std::copy(this->index[muList[j]].begin(),
												  this->index[muList[j]].begin()+Nlegs+Nextra+::hidden_dim(Nlegs)+::correct_dim(Nlegs),
												  checkIndex.begin());
									}
						else { checkIndex = this->index[muList[j]]; }
						auto it = uniqueController.find(checkIndex);
						if ( it == uniqueController.end() )
						{
							uniqueController.insert(checkIndex);
							dims.push_back(this->block[muList[j]].dimension(leg1) * this->block[muList[j]].dimension(leg2));
							nuVec.push_back(nuList[i]); muVec.push_back(muList[j]);
						}
					}
				}
			
			//Need to sort the different quantum number sectors which merge together!
			std::vector<std::size_t> dummy(dims.size());
			std::iota(dummy.begin(),dummy.end(),0);
			std::stable_sort (dummy.begin(), dummy.end(),
					   [&] (std::size_t n1, std::size_t n2)
					   {
						   std::array<qType,2> q1,q2;
						   std::copy(fuseT.index[nuVec[n1]].begin(),fuseT.index[nuVec[n1]].begin()+2,q1.begin());
						   std::copy(fuseT.index[nuVec[n2]].begin(),fuseT.index[nuVec[n2]].begin()+2,q2.begin());
						   return Symmetry::compare(q1,q2);
					   }
				);
			std::vector<Index> dims2 = dims;
			std::vector<std::size_t> nuVec2 = nuVec;
			std::vector<std::size_t> muVec2 = muVec;
			for (std::size_t i=0; i<dims2.size(); i++)
			{
				dims[i] = dims2[dummy[i]];
				nuVec[i] = nuVec2[dummy[i]];
				muVec[i] = muVec2[dummy[i]];
			}

			//Now set the content of the fuse tensor
			Index dim3 = accumulate(dims.begin(),dims.end(),0);
			Index offset=0;
			for (std::size_t i=0; i<dims.size(); i++)
			{
				auto it=doubleNus.find(nuVec[i]);
				if (it != doubleNus.end())
				{
					if ( fuseT.block[nuVec[i]].dimension(2) < dim3 )
					{
						Index dim1 = this->block[muVec[i]].dimension(leg1);
						Index dim2 = this->block[muVec[i]].dimension(leg2);
						
						fuseT.block[nuVec[i]] = TensorType<3>(dim1,dim2,dim3); fuseT.block[nuVec[i]].setZero();
						for (Index j=0; j<dim1; j++)
							for (Index k=0; k<dim2; k++)
								for (Index l=0; l<dim3; l++)
								{
									if ( j+dim1*k == l ) { fuseT.block[nuVec[i]](j,k,offset+l) = Scalar(1.); }
								}
						offset += dims[i];
					}
				}
				else
				{
					Index dim1 = this->block[muVec[i]].dimension(leg1);
					Index dim2 = this->block[muVec[i]].dimension(leg2);
					fuseT.block[nuVec[i]] = TensorType<3>(dim1,dim2,dim3); fuseT.block[nuVec[i]].setZero();
					for (Index j=0; j<dim1; j++)
						for (Index k=0; k<dim2; k++)
							for (Index l=0; l<dim3; l++)
							{
								if ( j+dim1*k == l ) { fuseT.block[nuVec[i]](j,k,offset+l) = Scalar(1.); }
							}
					offset += dims[i];
					doubleNus.insert(nuVec[i]);
				}
			}
		}
	return fuseT;
}

// template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
// MultipedeQ<Nlegs-1,Symmetry,Scalar,0> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
// directFuse( const Index leg1, const Index leg2 ) const
// {
// 	std::array<dir,2> new_dirs = { dir::in, dir::out };
// 	MultipedeQ<Nlegs-1,Symmetry,Scalar,0> Mout(new_dirs);
// 	for (std::size_t q=0; q<Nlegs-1; q++) {Mout.legs[q].place = q;}

// 	std::unordered_map<qType,std::vector<fuseData> > store;
// 	//dry loop: Collect data
// 	for (std::size_t nu=0; nu<this->size(); nu++)
// 	{
// 		auto qvec = Symmetry::reduceSilent(this->index[nu][leg1],this->index[nu][leg2]);
// 		for ( auto q: qvec )
// 		{
// 			if ( q != index[nu][0] ) { continue; }
// 			auto it = store.find(q);
// 			if ( it == store.end() )
// 			{
// 				fuseData info;
// 				info.nu = nu;
// 				info.dim1 = block[nu].dimension(leg1)*block[nu].dimension(leg2);
// 				std::vector<fuseData> entry;
// 				entry.push_back(info);
// 				store.insert(std::make_pair(q,entry));
// 				TensorType<2> A;
// 				Mout.push_back({index[nu][0],q},A);
// 			}
// 			else
// 			{
// 				fuseData info;
// 				info.nu = nu;
// 				info.dim1 = block[nu].dimension(leg1)*block[nu].dimension(leg2);
// 				(it->second).push_back(info);
// 			}
// 		}
// 	}
// 	//Need to sort the data!
// 	for ( auto it=store.begin(); it!=store.end(); it++ )
// 	{
// 		std::vector<std::size_t> index_sort((it->second).size());
// 		std::iota(index_sort.begin(),index_sort.end(),0);
// 		std::stable_sort (index_sort.begin(), index_sort.end(),
// 						  [&] (std::size_t n1, std::size_t n2)
// 						  {
// 							  std::array<qType,2> q1,q2;
// 							  std::copy(index[(it->second)[n1].nu].begin()+1,index[(it->second)[n1].nu].begin()+3,q1.begin());
// 							  std::copy(index[(it->second)[n2].nu].begin()+1,index[(it->second)[n2].nu].begin()+3,q2.begin());
// 							  return Symmetry::compare(q1,q2);
// 						  }
// 			);
// 		std::vector<fuseData> entry2 = it->second;
// 		for (std::size_t i=0; i<entry2.size(); i++)
// 		{
// 			(it->second)[i] = entry2[index_sort[i]];
// 		}
// 	}

// 	std::array<Index,2> dummy_legs;
// 	std::iota(dummy_legs.begin(),dummy_legs.end(),0);
// 	//working loop.
// 	for (std::size_t nu=0; nu<this->size(); nu++)
// 	{
// 		auto qvec = Symmetry::reduceSilent(this->index[nu][leg1],this->index[nu][leg2]);
// 		for ( auto q : qvec )
// 		{
// 			if ( q != index[nu][0] ) { continue; }
// 			auto it1 = store.find(q);
// 			std::array<qType,2> qs = {index[nu][0],q};
// 			auto it2 = Mout.dict.find(qs,dummy_legs);
// 			if ( Mout.block[it2->second].size() == 0 )
// 			{
// 				Index cols = this->block[nu].dimension(0);
// 				Index rows = 0;
// 				for ( std::size_t i=0; i<(it1->second).size(); i++) { rows += (it1->second)[i].dim1; }
// 				Mout.block[it2->second].resize(cols,rows); Mout.block[it2->second].setZero();
// 			}
// 			std::array<Index,2> new_dimsA = {this->block[nu].dimension(0),this->block[nu].dimension(1)*this->block[nu].dimension(2)};
// 			TensorType<2> A = this->block[nu].reshape(new_dimsA);
// 			std::array<std::pair<Index,Index>,2> padding_dims;
// 			Index left=0, right=0;
// 			bool SCHALTER = false;
// 			for ( std::size_t i=0; i<(it1->second).size(); i++ )
// 			{
// 				if ( (it1->second)[i].nu != nu and SCHALTER == false ) { left += (it1->second)[i].dim1; }
// 				if ( (it1->second)[i].nu != nu and SCHALTER == true ) { right += (it1->second)[i].dim1; }
// 				if ( (it1->second)[i].nu == nu ) { SCHALTER = true; }
// 			}
// 			padding_dims = {std::make_pair(0,0),std::make_pair(left,right)};
// 			TensorType<2> Afinal = A.pad(padding_dims);
// 			Mout.block[it2->second] += Afinal;
// 		}
// 	}
// 	return Mout;
// }

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<2,Symmetry,Scalar,0> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
reverse( const Index leg1 ) const
{
	std::array<dir,2> new_dirs;
	// std::array<string,2> new_names;
	new_dirs[0] = this->legs[leg1].getFlipDir();
	new_dirs[1] = this->legs[leg1].getFlipDir();

	// if (this->legs[leg1].getDir() == dir::in)
	// {
	// 	new_names[0] = "cap";
	// 	new_names[1] = "cap";
	// }
	// else
	// {
	// 	new_names[0] = "cup";
	// 	new_names[1] = "cup";
	// }
	MultipedeQ<2,Symmetry,Scalar,0> Mout(new_dirs); //,new_names);
	for (std::size_t q=0; q<2; q++) {Mout.legs[q].place = q;}
	std::unordered_set<std::array<qType,1> > doubleCounting;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		std::array<qType,1> redIndex;
		redIndex[0] = index[nu][leg1];
		auto it = doubleCounting.find(redIndex);
		if (it != doubleCounting.end()) { continue; }
		else
		{
			doubleCounting.insert(redIndex);
			TensorType<2> reverser;
			if constexpr ( Symmetry::HAS_CGC )
				{
					if ( this->legs[leg1].getDir() == dir::in ) //incoming to outgoing
					{
						reverser = Symmetry::calcCapTensor(index[nu][leg1]);
					}
					else //outgoing to incoming
					{
						reverser = Symmetry::calcCupTensor(index[nu][leg1]);
					}
				}
			TensorType<2> AId(block[nu].dimension(leg1),block[nu].dimension(leg1)); AId.setZero();
			for (std::size_t i=0; i<block[nu].dimension(leg1); i++) { AId(i,i) = Scalar(1.); }
			std::array<qType, 2+0+::hidden_dim(2)> totIndex;
			totIndex[0] = index[nu][leg1];
			totIndex[1] = Symmetry::flip(index[nu][leg1]);
			
			if constexpr ( Symmetry::HAS_CGC )
				{
					Mout.push_back(totIndex,AId,reverser);
				}
			else
			{
				Mout.push_back(totIndex,AId);
			}
		}
	}
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
shuffle( const std::array<Index,Nlegs> shuffle_dims ) const
{
	std::array<qleg,Nlegs> shuffLegs;
	for (std::size_t i=0; i<rank(); i++)
	{
		shuffLegs[i] = this->legs[static_cast<std::size_t>(shuffle_dims[i])];
	}
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout;
	Mout.legs = shuffLegs;
	for (std::size_t q=0; q<Nlegs; q++) {Mout.legs[q].place = q;}
	for (std::size_t nu=0; nu<size(); nu++)
	{
		std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> shuffIndex;
		std::array<Index,Nlegs> new_dimsC;
		std::array<Index,Nlegs> new_dimsA;
		for (std::size_t i=0; i<rank(); i++)
		{
			shuffIndex[i] = this->index[nu][static_cast<size_t>(shuffle_dims[i])];
			if constexpr ( Symmetry::HAS_CGC )
			{
				new_dimsC[i] = this->cgc[nu].dimension(shuffle_dims[i]);
			}
			new_dimsA[i] = this->block[nu].dimension(shuffle_dims[i]);
		}
		for (std::size_t i=rank(); i<rank()+Nextra+::hidden_dim(rank()); i++)
		{
			shuffIndex[i] = index[nu][i];
		}
		base_type_ A; A.resize(new_dimsA);
		A.device(device.get()) = this->block[nu].shuffle(shuffle_dims);
		if constexpr ( Symmetry::HAS_CGC )
			{
				base_type_ T; T.resize(new_dimsC);
				T.device(device.get()) = this->cgc[nu].shuffle(shuffle_dims);
				Mout.push_back(shuffIndex,A,T);
			}
		else
		{
			Mout.push_back(shuffIndex,A);			
		}
	}
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<typename expScalar>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
exp( const expScalar x ) const
{
	static_assert( Nlegs == 2, "It's only possible to calculate the exponential of a matrix" );
	// assert( this->legs[0].getDir() == dir::in and this->legs[1].getDir() == dir::out and "We need a regular matrix for exponentials.");
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout;
	Mout.legs = this->legs;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		base_type_ T;
		base_type_ A;
		if constexpr ( Symmetry::HAS_CGC )
			{
				Scalar norm = cgc[nu](0,0);
				T = cgc[nu]*pow(norm,-1);
				A = block[nu] * norm * x;
			}
		else
		{
				A = block[nu] * x;			
		}
		Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Mtmp;
		Mtmp = Eigen::Map<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> >(A.data(), A.dimension(0), A.dimension(1));
		Mtmp = Mtmp.exp();
		base_type_ Aexp = Eigen::TensorMap<base_type_ >(Mtmp.data(), Mtmp.rows(), Mtmp.cols());
		if constexpr ( Symmetry::HAS_CGC )
			{
				Mout.push_back(this->index[nu],Aexp,T);
			}
		else
		{
			Mout.push_back(this->index[nu],Aexp);			
		}
	}
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
sqrt() const
{
	static_assert( Nlegs == 2, "It's only possible to calculate the square root of a matrix" );
	assert( this->legs[0].getDir() == dir::in and this->legs[1].getDir() == dir::out and "We need a regular matrix for square roots.");
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout;
	Mout.legs = this->legs;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		base_type_ T;
		base_type_ A;
		if constexpr ( Symmetry::HAS_CGC )
			{
				Scalar norm = cgc[nu](0,0);
				T = cgc[nu]*pow(norm,-1);
				A = block[nu] * norm;
			}
		else
		{
				A = block[nu];			
		}

		Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Mtmp;
		Mtmp = Eigen::Map<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> >(A.data(), A.dimension(0), A.dimension(1));
		Mtmp = Mtmp.sqrt();
		base_type_ Asqrt = Eigen::TensorMap<base_type_ >(Mtmp.data(), Mtmp.rows(), Mtmp.cols());
		if constexpr ( Symmetry::HAS_CGC )
			{
				Mout.push_back(this->index[nu],Asqrt,T);
			}
		else
		{
			Mout.push_back(this->index[nu],Asqrt);
		}
	}
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
void MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
setIdentity()
{
	static_assert(Nlegs == 2,"The setIdentity method is only implemented for rank2 tensors.");
	assert( this->legs[0].getDir() == dir::in and this->legs[1].getDir() == dir::out and "We need a regular matrix for the identity.");
	for (std::size_t nu=0; nu<size(); nu++)
	{
		if constexpr ( Symmetry::HAS_CGC )
			{
				assert(cgc[nu].dimension(0) == cgc[nu].dimension(1) and "Matrices have to be quadratic for identities.");
			}
		assert(block[nu].dimension(0) == block[nu].dimension(1) and "Matrices have to be quadratic for identities.");
		if constexpr ( Symmetry::HAS_CGC )
			{
				cgc[nu].setZero();		
				for (std::size_t i=0; i<cgc[nu].dimension(0); i++)
				{
					cgc[nu](i,i) = Scalar(1.);
				}
			}
		block[nu].setZero();
		for (std::size_t i=0; i<block[nu].dimension(0); i++)
		{
			block[nu](i,i) = Scalar(1.);
		}
	}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<std::size_t Nfilter>
std::vector<std::size_t>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
filterRows( const std::array<Index, Nfilter> filterLegs, const std::array<qType, Nfilter> filterQs) const
{
	std::vector<std::size_t> Vout;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		std::size_t i=0;
		bool dummy = true;
		while (dummy and (i < Nfilter) ) { dummy = ( filterQs[i] == index[nu][filterLegs[i]] ); i++; }
		if ( dummy ) { Vout.push_back(nu); }
	}
	return Vout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<std::size_t Nl>
std::unordered_set<std::array<typename MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::qType, Nl> >
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
findUniqueQs( const std::array<Index, Nl> LegsIn ) const
{
	std::unordered_set<std::array<qType, Nl> > Sout;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		std::array<qType, Nl> redIndex;
		for (std::size_t i=0; i<rank()+Nextra+::hidden_dim(rank())+::correct_dim(rank()); i++)
		// for (std::size_t i=0; i<rank(); i++)
			for (std::size_t j=0; j<Nl; j++)
			{
				if ( i == LegsIn[j] ){ redIndex[j] = index[nu][i]; }
			}
		auto it = Sout.find(redIndex);
		if (it != Sout.end()) {continue;}
		else
		{
			Sout.insert(redIndex);
		}
	}
	return Sout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
void MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
calcCGC ( )
{
	if constexpr ( Symmetry::HAS_CGC )
		{
			static_assert(Nlegs==3, "Calculation of CGC is only implemented for rank 3 tensors.");
			for (std::size_t nu=0; nu<size(); ++nu)
			{
				cgc[nu] = Symmetry::reduce(in(nu),mid(nu),out(nu));
			}
		}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
void MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
standardizeCGC ( const bool SCHALTER )
{
	if constexpr ( Symmetry::HAS_CGC )
		{
			static_assert(Nlegs==3 or Nlegs==4, "Standardization of CGC is only implemented for rank 3 and rank 4 tensors.");
			if constexpr ( Nlegs == 3 )
						 {
							 for (std::size_t nu=0; nu<size(); ++nu)
							 {
								 TensorType<3> standCGC;
								 if(SCHALTER)
								 {
									 std::array<int,3> shuffle_dims{1,0,2};
									 standCGC = Symmetry::reduce(out(nu),mid(nu),in(nu)).shuffle(shuffle_dims);
								 }
								 else
								 {
									 standCGC = Symmetry::reduce(in(nu),mid(nu),out(nu));
								 }
								 Scalar prop = prop_to(cgc[nu],standCGC);
								 assert (prop != std::numeric_limits<Scalar>::infinity() and "The CGC structure is damaged and can't be standardized.");
								 block[nu] = block[nu] * prop;
								 cgc[nu] = standCGC;
							 }
						 }
			else if constexpr ( Nlegs == 4 )
							  {
								  for (std::size_t nu=0; nu<size(); ++nu)
								  {
									  TensorType<3> standCGC1, standCGC2;
									  standCGC1 = Symmetry::reduce(index[nu][0],index[nu][4],index[nu][1]);
									  standCGC2 = Symmetry::reduce(index[nu][2],index[nu][4],index[nu][3]);
									  std::array<Eigen::IndexPair<Index>, 1> product_dims = {Eigen::IndexPair<Index>(2,2)};
									  TensorType<4> standCGC = standCGC1.contract(standCGC2,product_dims);
									  Scalar prop = prop_to(cgc[nu],standCGC);
									  assert (prop != std::numeric_limits<Scalar>::infinity() and "The CGC structure is damaged and can't be standardized.");
									  block[nu] = block[nu] * prop;
									  cgc[nu] = standCGC;
								  }
							  }
		}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
bool MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
validateCGC ( ) const
{
	if constexpr ( Symmetry::HAS_CGC )
		{
			bool res = true;
			static_assert(Nlegs==3 or Nlegs==4, "Validation of CGC is only implemented for rank 3 and rank 4 tensors.");
			if constexpr ( Nlegs == 3 )
						 {
							 for (std::size_t nu=0; nu<size(); nu++)
							 {
								 TensorType<3> standCGC = Symmetry::reduce(in(nu),mid(nu),out(nu));
								 Scalar prop = prop_to(cgc[nu],standCGC);
								 if ( std::abs(prop - Scalar(1.)) < ::numeric_limits<Scalar>::epsilon() ) { res = true; }
								 else { return false; }
							 }
							 return res;
						 }
			else if constexpr ( Nlegs == 4 )
							  {
								  for (std::size_t nu=0; nu<size(); ++nu)
								  {
									  TensorType<3> standCGC1, standCGC2;
									  standCGC1 = Symmetry::reduce(index[nu][0],index[nu][4],index[nu][1]);
									  standCGC2 = Symmetry::reduce(index[nu][2],index[nu][4],index[nu][3]);
									  std::array<Eigen::IndexPair<Index>, 1> product_dims = {Eigen::IndexPair<Index>(2,2)};
									  TensorType<4> standCGC = standCGC1.contract(standCGC2,product_dims);
									  Scalar prop = prop_to(cgc[nu],standCGC);
									  if ( std::abs(prop - Scalar(1)) < ::numeric_limits<Scalar>::epsilon() ) { res = true; }
									  block[nu] = block[nu] * prop;
									  cgc[nu] = standCGC;
								  }
								  return res;
							  }
		}
	else
	{
		return true;
	}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
bool MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
validateBlock ( ) const
{
	bool out = true;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		if( ! ::validate(block[nu]) ) { out = false; return out; }
	}
	return out;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<std::size_t Nextra_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra-static_cast<Eigen::Index>(Nextra_),base_type_> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
tidyUp( const std::array<Index,Nextra_> legsIn ) const
{
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra-static_cast<Index>(Nextra_)> Mout;
	
	std::unordered_set<std::array<qType,Nlegs+Nextra-static_cast<Index>(Nextra_)+::hidden_dim(Nlegs)> > doubleCounting;
	std::array<qType,Nlegs+Nextra-static_cast<Index>(Nextra_)+::hidden_dim(Nlegs)> totIndex, tempIndex;
	base_type_ A;

	for (Index q=0; q<Nlegs; q++)
	{
		Mout.legs[q] = this->legs[q];
	}

	for (std::size_t nu=0; nu<size(); nu++)
	{
		std::size_t shift=0;
		for (Index q=0; q<Nlegs+Nextra+::hidden_dim(Nlegs); q++)
		{
			if (std::find(legsIn.begin(),legsIn.end(),q) != legsIn.end()) { shift++; continue; }
			else { totIndex[q-shift] = index[nu][q]; }
		}

		auto it=doubleCounting.find(totIndex);
		if (it != doubleCounting.end()) { continue; }
		else
		{
			doubleCounting.insert(totIndex);
			A = block[nu];
			
			for (std::size_t mu=nu+1; mu<size(); mu++)
			{
				shift=0;
				for (Index q=0; q<Nlegs+Nextra+::hidden_dim(Nlegs); q++)
				{
					if (std::find(legsIn.begin(),legsIn.end(),q) != legsIn.end()) { shift++; continue; }
					else { tempIndex[q-shift] = index[mu][q]; }
				}
				if (totIndex == tempIndex)
				{
					A += block[mu];
				}
			}
			Mout.push_back(totIndex,A);
		}
	}
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
adjoint() const
{	
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout = *this;
	for (std::size_t q=0; q<rank(); q++)
	{
		Mout.legs[q].direction = Mout.legs[q].getFlipDir();
	}
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<2*Nlegs,Symmetry,Scalar,Nextra> MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
makeTensorProduct () const
{
	MultipedeQ<2*Nlegs,Symmetry,Scalar,2*Nextra> Mout;
	if constexpr ( Symmetry::HAS_CGC )
		{
			std::array<qleg,2*Nlegs> new_legs;
			std::copy(legs.begin(),legs.end(),new_legs.begin());
			std::copy(legs.begin(),legs.end(),new_legs.begin()+Nlegs);

			Mout.legs = new_legs;
			for (std::size_t nu=0; nu<size(); nu++)
			{
				std::array<qType,2*(Nlegs+Nextra+::hidden_dim(Nlegs))> totIndex;
				std::copy(index[nu].begin(),index[nu].end(),totIndex.begin());
				std::copy(index[nu].begin(),index[nu].end(),totIndex.begin()+Nlegs);
				std::array<Eigen::IndexPair<Index>, 0> product_dims;
				TensorType<2*Nlegs> A = block[nu].contract(cgc[nu],product_dims);
				std::array<Index,2*Nlegs> id_dims;
				for (size_t i=0; i<2*Nlegs; i++) {id_dims[i] = Index(1);}
				TensorType<2*Nlegs> C; C.resize(id_dims); C.setConstant(Scalar(1.));
				std::array<Index,2*Nlegs> dummy_legs;
				std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));
				auto it = Mout.dict.find(totIndex,dummy_legs);
				if ( it != Mout.dict.end(dummy_legs) ) { Mout.block[it->second] += A; }
				else { Mout.push_back(totIndex,A,C); }
			}
		}
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
std::string MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
print ( const bool SHOW_METADATA, const bool SHOW_MATRICES, const std::size_t precision ) const
{
#ifndef HELPERS_IO_TABLE
	std::stringstream ss;
	std::size_t columnWidth = Nlegs+::hidden_dim(Nlegs) + 6;
	if ( SHOW_METADATA )
	{
		ss << "Number of legs: " << this->rank() << std::endl;
		ss << "with directions:" << std::endl;
		for (std::size_t i=0; i<rank(); i++) {ss << "leg=" << i << "\t" << this->legs[i].getDir() << std::endl;}
		ss << std::endl;
	}
	ss << std::setw(3) << "ν" << std::setw(columnWidth) << "{Q_ν}" << std::setw(columnWidth) << "A_ν";
	if constexpr ( Symmetry::HAS_CGC )
		{
			ss << std::setw(columnWidth) << "C_ν";
		}
	ss  << std::endl;
	ss << std::setfill('-') << std::setw(3) << "" << std::setw(columnWidth) << "" << std::setw(columnWidth) << "" << std::setw(columnWidth) << "" <<
		std::setfill(' ') << std::endl;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		ss << std::setw(2) << nu;
		std::stringstream tt,uu,vv;
		tt << "(";
		for (std::size_t q=0; q<index[nu].size(); q++)
		{
			tt << index[nu][q];
			if(q==index[nu].size()-1) {tt << ")";} else {tt << ",";}
		}
		for (std::size_t q=0; q<this->rank(); q++)
		{
			uu << block[nu].dimension(q);
			if(q==this->rank()-1) {uu << "";} else {uu << "x";}
			if constexpr ( Symmetry::HAS_CGC )
				{
					vv << cgc[nu].dimension(q);
					if(q==this->rank()-1) {vv << "";} else {vv << "x";}
				}
		}
		ss << std::setw(columnWidth+1) << tt.str() << std::setw(columnWidth) << uu.str() << std::setw(columnWidth) << vv.str() << std::endl;
	}
	ss << std::endl;
	if (SHOW_MATRICES)
	{
		ss << TCOLOR(GREEN) << "\e[4mA-matrices:\e[0m" << std::endl;
		for (std::size_t nu=0; nu<dim; nu++)
		{
			ss << TCOLOR(GREEN) << "ν=" << nu << std::endl << std::setprecision(precision) << std::fixed << block[nu] << std::endl;
		}
		ss << TCOLOR(BLACK) << std::endl;
		if constexpr ( Symmetry::HAS_CGC )
			{
				ss << TCOLOR(BLUE) << "\e[4mC-matrices:\e[0m" << std::endl;
				for (std::size_t nu=0; nu<dim; nu++)
				{
					ss << TCOLOR(BLUE) << "ν=" << nu << std::endl << std::setprecision(precision) << std::fixed << TCOLOR(BLUE) << cgc[nu] << std::endl;
				}
				ss << TCOLOR(BLACK) << std::endl;
			}
	}
	return ss.str();
#else //Use TextTable library for nicer output.
	std::stringstream out;
	if ( SHOW_METADATA )
	{
		out << "Amount of legs: " << this->rank() << std::endl;
		for (std::size_t i=0; i<rank(); i++) {out << "leg=" << i << "\t" << this->legs[i].getDir() << std::endl;}
		out << std::endl;
	}

	TextTable t( '-', '|', '+' );
	t.add("ν");
	t.add("Q_ν");
	t.add("A_ν");
	if constexpr ( Symmetry::HAS_CGC ) { t.add("C_ν"); }
	t.endOfRow();
	for (std::size_t nu=0; nu<size(); nu++)
	{
		std::stringstream ss,tt,uu,vv;
		ss << nu;
		tt << "(";
		for (std::size_t q=0; q<index[nu].size(); q++)
		{
			tt << index[nu][q];
			if(q==index[nu].size()-1) {tt << ")";} else {tt << ",";}
		}
		for (std::size_t q=0; q<this->rank(); q++)
		{
			uu << block[nu].dimension(q);
			if(q==this->rank()-1) {uu << "";} else {uu << "x";}
			if constexpr ( Symmetry::HAS_CGC )
				{
					vv << cgc[nu].dimension(q);
					if(q==this->rank()-1) {vv << "";} else {vv << "x";}
				}
		}
		t.add(ss.str());
		t.add(tt.str());
		t.add(uu.str());
		if constexpr ( Symmetry::HAS_CGC ) { t.add(vv.str()); }
		t.endOfRow();
	}
	t.setAlignment( 0, TextTable::Alignment::RIGHT );
	out << t;

	if (SHOW_MATRICES)
	{
		out << TCOLOR(GREEN) << "\e[4mA-tensors:\e[0m" << std::endl;
		for (std::size_t nu=0; nu<dim; nu++)
		{
			out << TCOLOR(GREEN) << "ν=" << nu << std::endl << std::setprecision(precision) << std::fixed << block[nu] << std::endl;
		}
		out << TCOLOR(BLACK) << std::endl;
		if constexpr ( Symmetry::HAS_CGC )
			{
				out << TCOLOR(BLUE) << "\e[4mC-tensors:\e[0m" << std::endl;
				for (std::size_t nu=0; nu<dim; nu++)
				{
					out << TCOLOR(BLUE) << "ν=" << nu << std::endl << std::setprecision(precision) << std::fixed << TCOLOR(BLUE) << cgc[nu] << std::endl;
				}
				out << TCOLOR(BLACK) << std::endl;
			}
	}

	return out.str();
#endif
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
double MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
memory (const MEMUNIT memunit) const
{
	double res = 0.;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		res += calc_memoryT(this->block[nu]);
		if constexpr ( Symmetry::HAS_CGC )
			{
				res += calc_memoryT(this->cgc[nu]);
			}
	}
	return res;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
double MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
overhead (const MEMUNIT memunit) const
{
	double res = 0.;
	res += (Nlegs+Nextra+::hidden_dim(Nlegs)) * Symmetry::Nq * calc_memory<Index>(dim, memunit); // in,out,mid; dict.keys
	res += Symmetry::Nq * calc_memory<std::size_t>(size(), memunit); // dict.vals
	return res;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
Eigen::VectorXi MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
dimensions (const std::size_t leg) const
{
	Eigen::VectorXi Vout(this->size());
	for (std::size_t nu=0; nu<size(); nu++) { Vout[nu] = block[nu].dimension(leg); }
	return Vout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
void MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
setZero()
{
	for (std::size_t nu=0; nu<size(); ++nu)
	{
		block[nu].setZero();
	}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
void MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
clear()
{
	index.clear();
	block.clear();
	if constexpr ( Symmetry::HAS_CGC )
		{
			cgc.clear();
		}
	dict.clear();
	dim = 0;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<class Dummy>
typename std::enable_if<Dummy::HAS_CGC>::type MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
push_back (const std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> quple, const base_type_ &M, const base_type_ &T)
{
	index.push_back(quple);
	block.push_back(M);
	cgc.push_back(T);
	dict.insert({quple, dim});
	++dim;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<class Dummy>
typename std::enable_if<!Dummy::HAS_CGC>::type MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
push_back (const std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> quple, const base_type_ &M)
{
	index.push_back(quple);
	block.push_back(M);
	dict.insert({quple, dim});
	++dim;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<class Dummy>
typename std::enable_if<Dummy::HAS_CGC>::type MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
push_back (const std::initializer_list<qType> qlist, const base_type_ &M, const base_type_ &T)
{
	assert(qlist.size() == Nlegs+Nextra+::hidden_dim(Nlegs));
	std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> quple;
	std::copy(qlist.begin(), qlist.end(), quple.data());
	push_back(quple,M,T);
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
template<class Dummy>
typename std::enable_if<!Dummy::HAS_CGC>::type MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
push_back (const std::initializer_list<qType> qlist, const base_type_ &M)
{
	assert(qlist.size() == Nlegs+Nextra+::hidden_dim(Nlegs));
	std::array<qType,Nlegs+Nextra+::hidden_dim(Nlegs)> quple;
	std::copy(qlist.begin(), qlist.end(), quple.data());
	push_back(quple,M);
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
void MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
setVacuum()
{
	std::array<Index,Nlegs> id_dims;
	for (std::size_t i=0; i<Nlegs; i++) {id_dims[i] = 1;}
	base_type_ Mtmp; Mtmp.resize(id_dims);
	Mtmp.setConstant(1.);
	std::array<qType,Nlegs> quple;
	for (std::size_t leg=0; leg<Nlegs; ++leg)
	{
		quple[leg] = Symmetry::qvacuum();
	}
	if constexpr ( Symmetry::HAS_CGC )
		{
			base_type_ Ttmp; Ttmp.resize(id_dims);
			Ttmp.setConstant(1.);
			push_back(quple, Mtmp, Ttmp);
		}
	else
	{
		push_back(quple, Mtmp);		
	}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
void MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_>::
setTarget (const std::array<qType,Nlegs> Q)
{
	std::array<Index,Nlegs>  id_dimsCGC;
	std::array<Index,Nlegs> id_dimsBLOCK;
	for (std::size_t i=0; i<Nlegs; i++) { id_dimsBLOCK[i] = 1; }
	base_type_ Mtmp; Mtmp.resize(id_dimsBLOCK);
	base_type_ Ttmp;
	if constexpr ( Symmetry::HAS_CGC )
		{
			Mtmp.setConstant(1.);
			for (std::size_t i=0; i<Nlegs; i++) {id_dimsCGC[i] = Symmetry::degeneracy((Q[i]));}
			Ttmp.resize(id_dimsCGC);
			Ttmp.setZero();
			for (std::size_t i=0; i<Q[0][0]; i++)
			{
				Ttmp(i,i,0) = Scalar(1.);
			}
			push_back(Q, Mtmp, Ttmp);
		}
	else
	{			
		Mtmp.setConstant(std::pow(Symmetry::coeff_dot(Q[0]),1.));
		push_back(Q, Mtmp);
	}
//	Ttmp.setConstant(1.);
	// Ttmp = std::pow(static_cast<Scalar>(1,-1) * Ttmp;
}

/**Adds two Multipeds block- and coefficient-wise.*/
template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> operator+ (const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M1,
															  const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M2)
{
	if ( M1.size() < M2.size() ) { return M2+M1; }
	std::vector<std::size_t> blocks_in_2nd_tensor;

	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout;
	Mout.legs = M1.legs;
	std::array<Eigen::Index,Nlegs+Nextra+::hidden_dim(Nlegs)> dummy_legs;
	std::iota(dummy_legs.begin(), dummy_legs.end(), Eigen::Index(0));
	Eigen::Tensor<Scalar,Nlegs,Eigen::ColMajor,Eigen::Index> A;
	for (std::size_t nu=0; nu<M1.size(); nu++)
	{
		auto it1 = M2.dict.find(M1.index[nu],dummy_legs);
		if ( it1 != M2.dict.end(dummy_legs) )
		{
			blocks_in_2nd_tensor.push_back(it1->second);
			A = M1.block[nu] + M2.block[it1->second];
		}
		else
		{
			A = M1.block[nu];
		}
		if constexpr ( Symmetry::HAS_CGC )
			{	
				Mout.push_back(M1.index[nu],A,M1.cgc[nu]);
			}
		else
		{
			Mout.push_back(M1.index[nu],A);
		}
	}
	if(blocks_in_2nd_tensor.size() != M2.size())
	{
		for(std::size_t nu=0; nu<M2.size(); nu++)
		{
			auto it = std::find(blocks_in_2nd_tensor.begin(),blocks_in_2nd_tensor.end(),nu);
			if(it == blocks_in_2nd_tensor.end())
			{
				A=M2.block[nu];
				if constexpr ( Symmetry::HAS_CGC )
					{	
						Mout.push_back(M2.index[nu],A,M2.cgc[nu]);
					}
				else
				{
					Mout.push_back(M2.index[nu],A);
				}
			}
		}
	}
	return Mout;
}

/**Subtracts two Multipeds block- and coefficient-wise.*/
template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> operator- (const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M1,
															  const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M2)
{
	static_assert( Nlegs == 3 or Nlegs == 2, "Method is only implemented for rank 2 and 3 Multipedes.");
	// if ( M1.size() < M2.size() ) { return M2-M1; }
	std::vector<std::size_t> blocks_in_2nd_tensor;
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout;
	Mout.legs = M1.legs;
	std::array<Eigen::Index,Nlegs+Nextra+::hidden_dim(Nlegs)> dummy_legs;
	std::iota(dummy_legs.begin(), dummy_legs.end(), Eigen::Index(0));
	Eigen::Tensor<Scalar,Nlegs,Eigen::ColMajor,Eigen::Index> A;
	for (std::size_t nu=0; nu<M1.size(); nu++)
	{
		auto it1 = M2.dict.find(M1.index[nu],dummy_legs);
		if ( it1 != M2.dict.end(dummy_legs) )
		{
			blocks_in_2nd_tensor.push_back(it1->second);
			A = M1.block[nu] - M2.block[it1->second];
		}
		else
		{
			A = M1.block[nu];
		}
		if constexpr ( Symmetry::HAS_CGC )
			{	
				Mout.push_back(M1.index[nu],A,M1.cgc[nu]);
			}
		else
		{
			Mout.push_back(M1.index[nu],A);
		}	
	}
	if(blocks_in_2nd_tensor.size() != M2.size())
	{
		for(std::size_t nu=0; nu<M2.size(); nu++)
		{
			auto it = std::find(blocks_in_2nd_tensor.begin(),blocks_in_2nd_tensor.end(),nu);
			if(it == blocks_in_2nd_tensor.end())
			{
				A=-M2.block[nu];
				if constexpr ( Symmetry::HAS_CGC )
					{	
						Mout.push_back(M2.index[nu],A,M2.cgc[nu]);
					}
				else
				{
					Mout.push_back(M2.index[nu],A);
				}
			}
		}
	}
	return Mout;
}

/**Divides two Multipeds block- and coefficient-wise.*/
template<std::size_t Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> operator/ (const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M1,
															  const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M2)
{
	static_assert( Nlegs == 3, "Method is only implemented for rank 3 Multipedes.");
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout;
	Mout.legs = M1.legs;
	assert( M1.size() == M2.size() and "Can't divide Multipedes with different sizes." );
	for (std::size_t nu=0; nu<M1.size(); nu++)
		for (std::size_t mu=0; mu<M2.size(); mu++)
		{
			if (M1.index[nu] == M2.index[mu])
			{
				Eigen::Tensor<Scalar,Nlegs,Eigen::ColMajor,Eigen::Index> A = M1.block[nu] / M2.block[mu];
				if constexpr ( Symmetry::HAS_CGC )
					{	
						Mout.push_back(M1.index[nu],A,M1.cgc[nu]);
					}
				else
				{
					Mout.push_back(M1.index[nu],A);
				}
			}
		}
	assert(Mout.size() == M1.size() and "Can't divide Multipedes with different sizes.");
	return Mout;
}

/**Multiplies a Multiped block and coefficient-wise with a scalar.*/
template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> operator* (const Scalar S, const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M)
{
	MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> Mout;
	Mout.legs = M.legs;
	for (std::size_t nu=0; nu<M.size(); nu++)
	{
		Eigen::Tensor<Scalar,Nlegs,Eigen::ColMajor,Eigen::Index> A = M.block[nu] * S;
		if constexpr ( Symmetry::HAS_CGC )
			{	
				Mout.push_back(M.index[nu],A,M.cgc[nu]);
			}
		else
		{
			Mout.push_back(M.index[nu],A);
		}
	}
	assert(Mout.size() == M.size() and "Can't divide Multipedes with different sizes.");
	return Mout;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
bool operator== (const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M1, const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M2)
{
	if constexpr ( Symmetry::HAS_CGC )
		{
			auto fullM1 = M1.makeTensorProduct();
			auto fullM2 = M2.makeTensorProduct();
			bool out;
			MultipedeQ<2*Nlegs,Symmetry,Scalar,Nextra> MCheck;
			MCheck.legs = fullM1.legs;
			for (std::size_t nu=0; nu<fullM1.size(); nu++)
				for (std::size_t mu=0; mu<fullM2.size(); mu++)
				{
					if (fullM1.index[nu] == fullM2.index[mu])
					{
						for (std::size_t i=0; i<2*Nlegs; i++)
						{
							if (fullM1.block[nu].dimension(i) != fullM2.block[mu].dimension(i)) {
								std::cout << "unequal dims" << std::endl; out=false; return out; }
						}
						Eigen::Tensor<Scalar,2*Nlegs,Eigen::ColMajor,Eigen::Index> A = fullM1.block[nu] - fullM2.block[mu];
						std::array<Eigen::Index,2*Nlegs> id_dims;
						for (size_t i=0; i<2*Nlegs; i++) {id_dims[i] = Eigen::Index(1);}
						Eigen::Tensor<Scalar,2*Nlegs,Eigen::ColMajor,Eigen::Index> C; C.resize(id_dims); C.setConstant(1.);
						MCheck.push_back(fullM1.index[nu],A,C);
					}
				}

			(MCheck.size() == 0) ? out=false : out=true; 
			for (std::size_t nu=0; nu<MCheck.size(); nu++)
			{
				if ( sumAbs(MCheck.block[nu]) > ::numeric_limits<Scalar>::epsilon() ) { out = false; std::cout << "test" << std::endl; return out; }
			}
			return out;
		}
	else
	{
		return true;
	}
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
bool compare_cgcStructure (const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M1, const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M2 )
{
	bool out=true;
	if constexpr ( Symmetry::HAS_CGC )
		{
			for (std::size_t nu=0; nu<M1.size(); nu++)
				for (std::size_t mu=0; mu<M2.size(); mu++)
				{
					if (M1.index[nu] == M2.index[mu])
					{
						Scalar prop = prop_to(M1.cgc[nu],M2.cgc[mu]);
						std::cout << "ν=" << nu << ", μ=" << mu << std::endl;
						std::cout << std::setprecision(15) << prop << std::setprecision(6) << std::endl;
						if (std::abs(prop-1.) > ::numeric_limits<Scalar>::epsilon()) { out=false; return out; }
					}
				}
		}
	return out;
}

template<Eigen::Index Nlegs, typename Symmetry, typename Scalar, Eigen::Index Nextra, typename base_type_>
std::ostream& operator<< (std::ostream& os, const MultipedeQ<Nlegs,Symmetry,Scalar,Nextra,base_type_> &M)
{
	os << M.print(false,false);
	return os;
}

#endif
