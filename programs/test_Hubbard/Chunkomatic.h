#ifndef CHUNKOMATIC
#define CHUNKOMATIC

template<typename IntType>
class Chunkomatic
{
public:

	Chunkomatic (IntType N_input, int chunks_input);

	int index();
	IntType value1();
	IntType value2();
	
	Chunkomatic& operator = (const int &comp) {curr_index=comp; return *this;}
	
	bool operator <  (const int &comp) {return(curr_index< comp)?true:false;}
	bool operator >  (const int &comp) {return(curr_index> comp)?true:false;}
	bool operator <= (const int &comp) {return(curr_index<=comp)?true:false;}
	bool operator >= (const int &comp) {return(curr_index>=comp)?true:false;}
	
	bool operator == (const int &comp) {return(curr_index==comp)?true:false;}
	bool operator != (const int &comp) {return(curr_index!=comp)?true:false;}
	
	IntType begin();
	IntType end();
	void operator++() {++curr_index;};
	
	Eigen::Matrix<IntType,Dynamic,2> get_limits() {return chunk_limits;};
	
private:
	
	int curr_index;
	IntType N;
	int chunks;
	
	Eigen::Matrix<IntType,Dynamic,2> chunk_limits;
};

template<typename IntType>
Chunkomatic<IntType>::
Chunkomatic (IntType N_input, int chunks_input)
:N(N_input), chunks(chunks_input)
{
	chunk_limits.resize(chunks,2);
	
	chunk_limits(0,0) = 0;
	chunk_limits(0,1) = N/chunks;
	
	for (IntType chunk=1; chunk<chunks; ++chunk)
	{
		chunk_limits(chunk,1) = N*(chunk+1)/chunks;
		chunk_limits(chunk,0) = chunk_limits(chunk-1,1);
	}
}

template<typename IntType>
int Chunkomatic<IntType>::
index()
{
	return curr_index;
}

template<typename IntType>
IntType Chunkomatic<IntType>::
begin()
{
	return 0;
}

template<typename IntType>
IntType Chunkomatic<IntType>::
end()
{
	return chunks;
}

template<typename IntType>
IntType Chunkomatic<IntType>::
value1()
{
	return chunk_limits(curr_index,0);
}

template<typename IntType>
IntType Chunkomatic<IntType>::
value2()
{
	return chunk_limits(curr_index,1);
}

#endif
