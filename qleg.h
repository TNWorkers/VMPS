#ifndef QLEG
#define QLEG

#include <array>
#include <set>

enum dir {in, out};
// enum Species {normal,qnum,silent}
/**This class contains info and small functions about the legs of the tensors in MultipedeQ.h*/
struct qleg
{
	typedef Eigen::Index Index;
	qleg() {};
	dir direction;
	// Species species;
	// string name;
	std::size_t place;
	// enum SU2_3_Tensor meaning;
	inline dir getFlipDir() const {if (direction == dir::in) {return dir::out;} else {return dir::in;}}
	inline dir getDir() const { return direction; }
	// inline dir getSpecies() const { return species; }
	// inline string getName() const { return name; }
	template<std::size_t N> bool notPresent( std::array<std::array<Index,2>,N > a_in, std::size_t order ) const;
	template<std::size_t N> bool present( std::array<std::array<Index,2>,N > a_in, std::size_t order ) const;
};

template <std::size_t N>
bool qleg::
notPresent( std::array<std::array<Index,2>,N > a_in, std::size_t order ) const
{
	bool out=true;
	for (std::size_t i=0; i<N; i++)
	{
		if (this->place == a_in[i][order]) {out = false; return out;}
	}
	return out;
}

template <std::size_t N>
bool qleg::
present( std::array<std::array<Index,2>,N > a_in, std::size_t order ) const
{
	return !notPresent(a_in,order);
}

inline bool operator == ( dir &dir1, dir &dir2)
{
	if (dir1 == dir::in and dir2 == dir::in) {return true;}
	else if (dir1 == dir::out and dir2 == dir::out) {return true;}
	else if (dir1 == dir::in and dir2 == dir::out) {return false;}
	else if (dir1 == dir::out and dir2 == dir::in) {return false;}
}

inline bool operator != ( dir &dir1, dir &dir2)
{
	if (dir1 == dir::in and dir2 == dir::in) {return false;}
	else if (dir1 == dir::out and dir2 == dir::out) {return false;}
	else if (dir1 == dir::in and dir2 == dir::out) {return true;}
	else if (dir1 == dir::out and dir2 == dir::in) {return true;}
}

inline std::ostream& operator<< (std::ostream& os, dir dir_in)
{
	if (dir_in == dir::in) {os << "in";}
	else {os << "out";}
	return os;
}

#endif
