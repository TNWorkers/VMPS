#ifndef ITERATORFLOOR
#define ITERATORFLOOR

class IteratorFloor
{
public:

	IteratorFloor() {curr_index = 0;}

	size_t curr_index;

	inline size_t index()      {return curr_index;}
	virtual size_t operator*() {return curr_index;}

//	virtual size_t begin (size_t start_index) {return start_index;}
//	virtual size_t end   (size_t end_index)   {return end_index;}

	virtual void operator++() {++curr_index;}
	virtual void operator--() {--curr_index;}
	
	bool operator< (const size_t &other_index) {return curr_index< other_index;}
	bool operator> (const size_t &other_index) {return curr_index> other_index;}
	bool operator==(const size_t &other_index) {return curr_index==other_index;}
	bool operator!=(const size_t &other_index) {return curr_index!=other_index;}
	bool operator<=(const size_t &other_index) {return curr_index<=other_index;}
	bool operator>=(const size_t &other_index) {return curr_index>=other_index;}
	
//	IteratorFloor& operator= (const size_t &other_index) {curr_index = static_cast<size_t>(other_index);}
	
//	friend ostream &operator<< (ostream &out, IteratorFloor i) {out << i.curr_index; return out;}
};

#endif
