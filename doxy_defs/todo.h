namespace Project{
/**
\todo - effizienter Lanczos-Fehler, damit ein Ausgang bei jedem Schritt möglich ist

\todo - fehlende Blöcke in Mps::calc_N genau wie in Umps::calc_N (ist das überhaupt nötig?)

\todo - sweepStep2, calc_N als externe oder static-Funktionen umschreiben, sodass weder Code-Kopien noch Dummy-Mps nötig sind? Alternativ Basisklasse für Mps und Umps?

\todo - T- und S- Operatoren in HubbardSO4

\todo - Erzeugung von Aclump in eine separate Funktion auslagern, damit der Code nicht ständig kopiert werden muss

\todo - orbitalabhängige t', J'
*/
#define MPS_PROJECT

/**
\todo - Rücktransformation der Basis und bessere print-Anzeige der transformierten Basis mit boost::rational

\todo - Die Parallelisierung in VUMPS muss neu überdacht werden. Momentan laufen Programme auf mehreren Kernen (insbesondere mit SU(2)) äußerst instabil.

\todo - VmpsSolver::expand_basis für Einheitszellen ist implementiert. Folgende Frage sollte aber nochmal analysiert werden:
        Wie berechnet man den Zweiplatz-A-Tensor: 
		-# AL[i] * AC[i+1] <- Problem: AL wird während des Prozesses verändert, AC nicht.
		-# AL[i] * C[i]AR[i+1] <- Hier sind alle drei objekte beteiligt die verändert werden. <- Problem: Umgebungen währenddessen updaten?

*/
#define VUMPS_PROJECT

/**
\todo - beliebigreichweitiges Hopping und Mpo-Kompression

\todo - Z(N)-Symmetrieklasse

\todo - Ausnutzung der Impulserhaltung für Zylinder in y-Richtung mit der Z(N)-Klasse
*/
#define LONG_TERM_PROJECT
}
