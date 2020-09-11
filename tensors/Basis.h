#ifndef BASIS_H_
#define BASIS_H_

/** \class Basis
  *
  * \ingroup Tensors
  *
  * Class for a plain basis without symmetries. A basis with symmetries is a collection of plain bases for each Symmetry entry
  *
  */
class Basis
{
public:
	/**Does nothing.*/
	Basis() {};
	
	Basis(std::vector<std::string> idents, Eigen::Index dim) {
		Eigen::Index count = 0;
		for(const auto& ident : idents) {
			auto entry = std::make_tuple(ident,count);
			data_.push_back(entry);
			count++;
		}
		curr_dim += dim;
	};

	Basis(Eigen::Index dim) {
		std::vector<std::string> idents(dim,"");
		Basis(idents,dim);
	};

	Eigen::Index size() const {return data_.size();}
	
	void push_back( const std::string& ident) {
		data_.push_back(std::make_tuple(ident,curr_dim));
		curr_dim++;
	};

	void push_back( const std::vector<std::string> &idents) {
		for (const auto &ident:idents)
		{
			data_.push_back(std::make_tuple(ident,curr_dim));
			curr_dim++;
		}
	};

	bool find( std::string ident_in ) {
		for(const auto& elem : data_) {auto [ident,num] = elem; if (ident == ident_in) {return true;}}
		return false;
	}
	Eigen::Index location( std::string ident_in ) {
		for(const auto& elem : data_) {auto [ident,num] = elem; if (ident == ident_in) {return num;}}
		return std::numeric_limits<Eigen::Index>::max();
	}

	/**Adds to bases together.*/
	Basis add (const Basis& other) const;

	/**
	 * Returns the tensor product basis.
	 * This function also saves the history of the combination process for later use. See leftAmount() and rightAmount().
	 */
	Basis combine (const Basis& other) const;

	/**Prints the basis.*/
	std::string print() const;
	
	std::vector<std::tuple<std::string,Eigen::Index> >::iterator begin() { return data_.begin(); }
	std::vector<std::tuple<std::string,Eigen::Index> >::iterator end() { return data_.end(); }

	std::vector<std::tuple<std::string,Eigen::Index> >::const_iterator cbegin() const { return data_.cbegin(); }
	std::vector<std::tuple<std::string,Eigen::Index> >::const_iterator cend() const { return data_.cend(); }

	bool operator==( const Basis& other ) const {return (this->data_ == other.data_);}

	friend ostream& operator<<(ostream& os, const Basis &basis);

//private:
	struct fuseData
	{
		Eigen::Index dim1,dim2;
		std::array<Eigen::Index,2> source(Eigen::Index combined_num) const
			{
				Eigen::Index tmp = combined_num / dim1;
				return {{combined_num - dim1*tmp, tmp}};
			};		
	};
	
	std::vector<std::tuple<std::string,Eigen::Index> > data_;
	Eigen::Index curr_dim = 0;
	Basis::fuseData history;
};

Basis Basis::
add (const Basis& other) const
{
	Basis out;
	
	for (const auto &[ident,num] : data_)
	{
		out.push_back(ident);
	}
	for (const auto &[ident,num] : other.data_)
	{
		out.push_back(ident);
	}
	return out;
}

Basis Basis::
combine (const Basis& other) const
{
	Basis out;
	out.history.dim1 = this->size();
	out.history.dim2 = other.size();
	
	for(const auto& [ident2,num2] : other.data_)
	for(const auto& [ident1,num1] : this->data_)
	{
		// out.data_.push_back(std::make_tuple(ident1+"⊗"+ident2,num1 + this->size()*num2));
		out.data_.push_back(std::make_tuple("",num1 + this->size()*num2));
	}
	return out;
}

std::string Basis::
print() const
{
	std::stringstream out;
	TextTable t( '-', '|', '+' );
	t.add("num");
	t.add("ident");
	t.endOfRow();
	for (const auto& [ident,num] : data_)
	{
		std::stringstream ss, tt;
		ss << ident;
		tt << num;
		t.add(ss.str());
		t.add(tt.str());
		t.endOfRow();
	}
	out << t;
	return out.str();
}

std::ostream& operator<<(std::ostream& os, const Basis &basis)
{
	os << basis.print();
	return os;
}
#endif
