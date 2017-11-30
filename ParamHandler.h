#ifndef PARAMHANDLER
#define PARAMHANDLER

#include <any>
#include <tuple>
#include <map>
#include <initializer_list>
#include <typeinfo>
#include <typeindex>

struct Param
{
	Param (string label_input, std::any value_input)
	:label(label_input), value(value_input)
	{}
	
	Param (tuple<string,std::any> input)
	:label(get<0>(input)), value(get<1>(input))
	{}
	
	string   label;
	std::any value;
};

class ParamHandler
{
public:
	
	ParamHandler (const initializer_list<Param> &p_list);
	ParamHandler (const initializer_list<Param> &p_list, const map<string,std::any> &defaults_input);
	
	template<typename Scalar> Scalar get (const string label) const;
	template<typename Scalar> Scalar get_default (const string label) const;
	bool HAS (const string label) const;
	
	string info() const;
	
private:
	
	map<string,std::any> params;
	map<string,std::any> defaults;
};

ParamHandler::
ParamHandler (const initializer_list<Param> &p_list)
{
	for (auto p:p_list)
	{
		params.insert(make_pair(p.label,p.value));
	}
}

ParamHandler::
ParamHandler (const initializer_list<Param> &p_list, const map<string,std::any> &defaults_input)
:defaults(defaults_input)
{
	for (auto p:p_list)
	{
		params.insert(make_pair(p.label,p.value));
	}
}

template<typename Scalar> 
Scalar ParamHandler::
get (const string label) const
{
	auto it = params.find(label);
	if (it != params.end())
	{
		return any_cast<Scalar>(it->second);
	}
	else
	{
		return get_default<Scalar>(label);
	}
}

template<typename Scalar> 
Scalar ParamHandler::
get_default (const string label) const
{
	auto it = defaults.find(label);
	assert(it != defaults.end());
	return any_cast<Scalar>(it->second);
}

bool ParamHandler::
HAS (const string label) const
{
	return static_cast<bool>(params.find(label) != params.end());
}

string ParamHandler::
info() const
{
	stringstream ss;
	
	unordered_map<type_index,string> type_names;
	type_names[type_index(typeid(int))]             = "int";
	type_names[type_index(typeid(size_t))]          = "size_t";
	type_names[type_index(typeid(double))]          = "double";
	type_names[type_index(typeid(complex<double>))] = "complex<double>";
	type_names[type_index(typeid(vector<double>))]  = "vector<double>";
	type_names[type_index(typeid(bool))]            = "bool";
	
	for (auto p:params)
	{
		ss << p.first << "\t";
		
		if (type_names[std::type_index(p.second.type())] == "int")
		{
			ss << any_cast<int>(p.second);
		}
		else if (type_names[std::type_index(p.second.type())] == "size_t")
		{
			ss << any_cast<size_t>(p.second);
		}
		else if (type_names[std::type_index(p.second.type())] == "double")
		{
			ss << any_cast<double>(p.second);
		}
		else if (type_names[std::type_index(p.second.type())] == "complex<double>")
		{
			ss << any_cast<complex<double> >(p.second);
		}
		else if (type_names[std::type_index(p.second.type())] == "vector<double>")
		{
			vector<double> pval = any_cast<vector<double> >(p.second);
			for (int i=0; i<pval.size(); ++i)
			{
				ss << any_cast<double>(pval[i]);
				if (i!=pval.size()-1)
				{
					ss << ",";
				}
			}
		}
		else if (type_names[std::type_index(p.second.type())] == "bool")
		{
			ss << any_cast<bool>(p.second);
		}
		else
		{
			throw ("unknown parameter type!");
		}
		ss << endl;
	}
	
	return ss.str();
}

#endif
