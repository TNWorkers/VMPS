#include "MpoQ.h"

namespace VMPS
{
/**MPO representation of \f$H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
                                + J \sum_i \mathbf{s}_i\cdot\mathbf{S}_i\f$.*/
	class KondoModell : public MpoQ<8,2,double>
	{
	public:
		/**
		   @param L_input : chain length
		   @param J_input : \f$J\f$
		*/
		KondoModel (size t L_input, double J_input);
		
		/**
		   \f$ckondo_{\uparrow} = \left(
		   \begin{array}{cccccccc}
		   0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   \end{array}
		   \right)\f$
		*/
		static const Eigen::Matrix<double,8,8,RowMajor> ckondo_UP;
				
		/**
		   \f$ckondo_{\downarrow} = \left(
		   \begin{array}{cccccccc}
		   0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   \end{array}
		   \right)\f$
		*/
		static const Eigen::Matrix<double,8,8,RowMajor> ckondo_DOWN;

		/**
		   \f$Skondo_{x} = \left(
		   \begin{array}{cccccccc}
		   0.5 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0.5 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0.5 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0.5 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & -0.5 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & -0.5 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & -0.5 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & -0.5\\
		   \end{array}
		   \right)\f$
		*/
		static const Eigen::Matrix<double,8,8,RowMajor> Skondo_X;

		/**
		   \f$Skondo_{p} = \left(
		   \begin{array}{cccccccc}
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
		   \end{array}
		   \right)\f$
		*/
		static const Eigen::Matrix<double,8,8,RowMajor> Skondo_p;

		/**
		   \f$Skondo_{x} = \left(
		   \begin{array}{cccccccc}
		   0 & 0 & 0 & 0 & 0.5 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0.5 & 0 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0.5 & 0\\
		   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0.5\\
		   0.5 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0.5 & 0 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0.5 & 0 & 0 & 0 & 0 & 0\\
		   0 & 0 & 0 & 0.5 & 0 & 0 & 0 & 0\\
		   \end{array}
		   \right)\f$
		*/
		static const Eigen::Matrix<double,8,8,RowMajor> Skondo_x;

	}
}
