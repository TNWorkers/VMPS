#ifndef BASIS_H_
#define BASIS_H_

/** \class Basis
  *
  * Class for a plain basis without symmetries. A basis with symmetries is a collection of plain bases for each Symmetry entry
  *
  */
class Basis
{
public:
	Basis(std::vector<std::string> idents, Eigen::Index dim) {
		Eigen::Index count = 0;
		for(const auto& ident : idents) {
			auto entry = std::make_tuple(ident,count);
			data_.push_back(entry);
			count++;
		}
	};

	Basis(Eigen::Index dim) {
		std::vector<std::string> idents(dim,"");
		Basis(idents,dim);
	};

	Eigen::Index size() const {return data_.size();}
	void push_back( const std::string& ident) {
		curr_dim++;
		data_.push_back(std::make_tuple(ident,curr_dim));
	};

	bool find( std::string ident_in ) {
		for(const auto& elem : data_) {auto [ident,num] = elem; if (ident == ident_in) {return true;}}
		return false;
	}
	Eigen::Index location( std::string ident_in ) {
		for(const auto& elem : data_) {auto [ident,num] = elem; if (ident == ident_in) {return num;}}
		return std::numeric_limits<Eigen::Index>::max();
	}
	
	std::vector<std::tuple<std::string,Eigen::Index> >::iterator begin() { return data_.begin(); }
	std::vector<std::tuple<std::string,Eigen::Index> >::iterator end() { return data_.end(); }

	std::vector<std::tuple<std::string,Eigen::Index> >::const_iterator cbegin() const { return data_.cbegin(); }
	std::vector<std::tuple<std::string,Eigen::Index> >::const_iterator cend() const { return data_.cend(); }

	bool operator==( const Basis& other ) const {return (this->data_ == other.data_);}
private:
	std::vector<std::tuple<std::string,Eigen::Index> > data_;
	Eigen::Index curr_dim = 0;
};

#endif
