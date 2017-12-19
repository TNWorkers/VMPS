#ifndef PARAMHANDLER
#define PARAMHANDLER

#include <any>
#include <tuple>
#include <map>
#include <initializer_list>
#include <typeinfo>
#include <typeindex>

template<typename Scalar>
struct param0d
{
	Scalar x;
	string label;
};

template<typename Scalar>
struct param1d
{
	Scalar x;
	Array<Scalar,Dynamic,1> a;
	string label;
};

template<typename Scalar>
struct param2d
{
	Scalar x;
	Array<Scalar,Dynamic,Dynamic> a;
	string label;
};

struct Param
{
	Param (string label_input, std::any value_input, size_t index_input=0)
	:label(label_input), value(value_input), index(index_input)
	{}
	
	Param (tuple<string,std::any> input, size_t index_input=0)
	:label(get<0>(input)), value(get<1>(input)), index(index_input)
	{}
	
	string   label;
	std::any value;
	size_t   index=0;
};

class ParamHandler
{
public:
	
	ParamHandler (const vector<Param> &p_list);
	ParamHandler (const vector<Param> &p_list, const map<string,std::any> &defaults_input);
	
	template<typename Scalar> Scalar get (const string label, const size_t index=0) const;
	template<typename Scalar> Scalar get_default (const string label) const;
	bool HAS (const string label, const size_t index=0) const;
	bool HAS_ANY_OF (const initializer_list<string> &labels, const size_t &index=0) const;
	bool HAS_NONE_OF (const initializer_list<string> &labels, const size_t &index=0) const;
	inline size_t size() const {return params.size();}
	
//	string info() const;
	
	template<typename Scalar> param0d<Scalar> fill_array0d (string label_def, string label_x, size_t loc=0) const;
	template<typename Scalar> param1d<Scalar> fill_array1d (string label_x, string label_a, size_t size_a, size_t loc=0) const;
	template<typename Scalar> param2d<Scalar> fill_array2d (string label_x, string label_a, size_t size_a, size_t loc=0) const;
	
private:
	
	IOFormat arrayFormat;
	
	size_t calc_cellsize (const vector<Param> &p_list);
	
	vector<map<string,std::any> > params;
	map<string,std::any> defaults;
};

ParamHandler::
ParamHandler (const vector<Param> &p_list)
{
	params.resize(calc_cellsize(p_list));
	arrayFormat = IOFormat(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");
	
	for (auto p:p_list)
	{
		params[p.index].insert(make_pair(p.label,p.value));
	}
}

ParamHandler::
ParamHandler (const vector<Param> &p_list, const map<string,std::any> &defaults_input)
:defaults(defaults_input)
{
	params.resize(calc_cellsize(p_list));
	arrayFormat = IOFormat(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");
	
	for (auto p:p_list)
	{
		params[p.index].insert(make_pair(p.label,p.value));
	}
}

template<typename Scalar> 
Scalar ParamHandler::
get (const string label, size_t index) const
{
	assert(index < params.size());
	
	auto it = params[index].find(label);
	
	if (it != params[index].end())
	{
		return any_cast<Scalar>(it->second);
	}
	else
	{
		auto it0 = params[0].find(label);
		if (it0 != params[0].end())
		{
			return any_cast<Scalar>(it0->second);
		}
		else
		{
			return get_default<Scalar>(label);
		}
	}
}

template<typename Scalar> 
Scalar ParamHandler::
get_default (const string label) const
{
	auto it = defaults.find(label);
	if (it == defaults.end())
	{
		cout << "Cannot get default parameter " << label << "!" << endl;
		assert(it != defaults.end());
	}
	else
	{
		return any_cast<Scalar>(it->second);
	}
}

bool ParamHandler::
HAS (const string label, const size_t index) const
{
	auto it = params[index].find(label);
	if (it != params[index].end())
	{
		return true;
	}
	else
	{
		auto it0 = params[0].find(label);
		if (it0 != params[0].end())
		{
			return true;
		}
	}
	return false;
}

bool ParamHandler::
HAS_ANY_OF (const initializer_list<string> &labels, const size_t &index) const
{
	vector<bool> res;
	for (const auto &label:labels) res.push_back(HAS(label,index));
	return (find(res.begin(), res.end(), true) != res.end())? true:false;
}

bool ParamHandler::
HAS_NONE_OF (const initializer_list<string> &labels, const size_t &index) const
{
	return !HAS_ANY_OF(labels,index);
}

size_t ParamHandler::
calc_cellsize (const vector<Param> &p_list)
{
	set<size_t> indices;
	
	for (auto p:p_list)
	{
		indices.insert(p.index);
	}
	
	for (auto i:indices)
	{
		assert(i<indices.size() and "Strange choice of cell indices!");
	}
	
	return indices.size();
}

template<typename Scalar>
param0d<Scalar> ParamHandler::
fill_array0d (string label_def, string label_x, size_t loc) const
{
	param0d<Scalar> res;
	res.x = get_default<Scalar>(label_def);
	
	if (HAS(label_x,loc))
	{
		res.x = get<Scalar>(label_x,loc);
		
		stringstream ss;
		ss << label_x << "=" << res.x;
		res.label = ss.str();
	}
	else if (HAS(label_def,loc))
	{
		res.x = get<Scalar>(label_def,loc);
	}
	
	return res;
}

template<typename Scalar>
param1d<Scalar> ParamHandler::
fill_array1d (string label_x, string label_a, size_t size_a, size_t loc) const
{
	assert(!(HAS(label_x) and HAS(label_a)));
	
	param1d<Scalar> res;
	
	res.a.resize(size_a);
	res.a.setZero();
	res.x = get_default<Scalar>(label_x);
	stringstream ss;
	
	if (HAS(label_x,loc))
	{
		res.x = get<Scalar>(label_x,loc);
		res.a = res.x;
		ss << label_x << "=" << res.x;
		res.label = ss.str();
	}
	else if (HAS(label_a,loc))
	{
		res.a = get<Array<Scalar,Dynamic,1> >(label_a,loc);
		ss << label_a << "=" << res.a.format(arrayFormat);
		res.label = ss.str();
	}
	
	if (HAS(label_x,loc) or HAS(label_a,loc))
	{
		res.label = ss.str();
	}
	
	return res;
}

template<typename Scalar>
param2d<Scalar> ParamHandler::
fill_array2d (string label_x, string label_a, size_t size_a, size_t loc) const
{
	assert(!(HAS(label_x) and HAS(label_a)));
	
	param2d<Scalar> res;
	
	res.x = get_default<double>(label_x);
	res.a.resize(size_a,size_a);
	res.a.setZero();
	res.a.matrix().diagonal().setConstant(res.x);
	stringstream ss;
	
	if (HAS(label_x,loc))
	{
		res.x = get<Scalar>(label_x,loc);
		res.a.matrix().diagonal().setConstant(res.x);
		ss << label_x << "=" << res.x;
		res.label = ss.str();
	}
	else if (HAS(label_a,loc))
	{
		res.a = get<Array<Scalar,Dynamic,Dynamic> >(label_a,loc);
		ss << label_a << "=" << res.a.format(arrayFormat);
		res.label = ss.str();
	}
	
	if (HAS(label_x,loc) or HAS(label_a,loc))
	{
		res.label = ss.str();
	}
	
	return res;
}

// It makes little sense, to print out std::any, but here is a possibility if you know all possible types:

//string ParamHandler::
//info() const
//{
//	stringstream ss;
//	
//	unordered_map<type_index,string> type_names;
//	type_names[type_index(typeid(int))]             = "int";
//	type_names[type_index(typeid(size_t))]          = "size_t";
//	type_names[type_index(typeid(double))]          = "double";
//	type_names[type_index(typeid(complex<double>))] = "complex<double>";
//	type_names[type_index(typeid(vector<double>))]  = "vector<double>";
//	type_names[type_index(typeid(bool))]            = "bool";
//	
//	for (auto p:params)
//	{
//		ss << p.first << "\t";
//		
//		if (type_names[std::type_index(p.second.type())] == "int")
//		{
//			ss << any_cast<int>(p.second);
//		}
//		else if (type_names[std::type_index(p.second.type())] == "size_t")
//		{
//			ss << any_cast<size_t>(p.second);
//		}
//		else if (type_names[std::type_index(p.second.type())] == "double")
//		{
//			ss << any_cast<double>(p.second);
//		}
//		else if (type_names[std::type_index(p.second.type())] == "complex<double>")
//		{
//			ss << any_cast<complex<double> >(p.second);
//		}
//		else if (type_names[std::type_index(p.second.type())] == "vector<double>")
//		{
//			vector<double> pval = any_cast<vector<double> >(p.second);
//			for (int i=0; i<pval.size(); ++i)
//			{
//				ss << any_cast<double>(pval[i]);
//				if (i!=pval.size()-1)
//				{
//					ss << ",";
//				}
//			}
//		}
//		else if (type_names[std::type_index(p.second.type())] == "bool")
//		{
//			ss << any_cast<bool>(p.second);
//		}
//		else
//		{
//			throw ("unknown parameter type!");
//		}
//		ss << endl;
//	}
//	
//	return ss.str();
//}

#endif
