namespace Project{
/**
\todo 
      - effizienter Lanczos-Fehler, damit ein Ausgang bei jedem Schritt möglich ist

\todo 
      - fehlende Blöcke in Mps::calc_N genau wie in Umps::calc_N (ist das überhaupt nötig?)

\todo 
      - calc_N sowie Erzeugung von Aclump als externe oder static-Funktionen umschreiben, sodass weniger Code-Kopien.
            - Klasse Blocker kann viele Codedopplungen entfernen ist bisher aber noch nicht effizient genug.
		      Das komplette Prozedere sollte in einer Schleife stattfinden. (Wie bisher auch)

\todo 
      - orbitalabhängige t', J'
*/
#define MPS_PROJECT

/**
\todo 
      - Rücktransformation der Basis und bessere print-Anzeige der transformierten Basis mit boost::rational

\todo 
      - Die Parallelisierung in VUMPS muss neu überdacht werden. Momentan laufen Programme auf mehreren Kernen (insbesondere mit SU(2)) äußerst instabil.
*/
	
#define VUMPS_PROJECT

/**
\todo 
      - beliebigreichweitiges Hopping und Mpo-Kompression

\todo 
      - Z(N)-Symmetrieklasse

\todo 
      - Ausnutzung der Impulserhaltung für Zylinder in y-Richtung mit der Z(N)-Klasse
*/
#define LONG_TERM_PROJECT
}
