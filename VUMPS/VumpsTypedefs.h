#ifndef VUMPSTYPEDEFS
#define VUMPSTYPEDEFS

struct GAUGE
{
	enum OPTION {L=0, R=1, C=2};
};

std::ostream& operator<< (std::ostream& s, GAUGE::OPTION g)
{
	if      (g==GAUGE::OPTION::L) {s << "L";}
	else if (g==GAUGE::OPTION::R) {s << "R";}
	else if (g==GAUGE::OPTION::C) {s << "C";}
	return s;
}

#endif
