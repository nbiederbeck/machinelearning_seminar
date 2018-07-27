# Machine Learning Seminar
Eine Idee für den Abschlussbericht.


## 01_einleitung_motivation_fragestellung.tex
Warum Wolken klassifizieren?
- Teil der Wettervorhersage mit einfachen Mitteln
    - Raspberry Pi etc.
    - Wolken starkes Feature (verglichen mit Temperatur, Feuchtigkeit)
    - Verlauf der Wolken wichtig
    - Wolken in Abhängigkeit der Temperatur --> Feature Generation (?)


Warum Machine Learning?
- Live Analyse
- Schnelle Auswertung mit kleiner Leistung
- nicht Foto sondern Klasse abspeichern als Feature
    - geringer Datentransfer

Warum Neuronales Netz?
- Lerne Formen und Farben
    - Convolution (insb. Formen)
- langes Training, schnelles Auswerten
- Modell distributierbar (keine Daten nötig wie bei kNN, SVM (wenig))
    - viele Wetterstationen, ein Modell

Warum Random Forest (Alternativmethode)?
- Nur auf Farben
    - Erwartung: schlechter da weniger Features
    - weniger Rechenaufwand (insb Training)
- Generalisiertes Modell out of the box

## 02_auswahl_charakterisierung_datensatz.tex
Datennahme
- Fotos von Raspberry Pis
- regelmäßig (auch Nachts)
    - Zeitreihenanalyse Wetter
- Fotos zeitabhängig
    - Klassen zeitabhängig (Sommer / Winter)
- aber: im Modell unabhängig von Zeit
- 11 Klassen (inkl. bad picture ?)
    - 2 verwerfen, da zu wenig samples
- selbst gelabelt

Erstes Preprocessing
- Nachtfotos entfernen
    - Mittlere Helligkeit statt Zeitangaben
        - kann ein Pi auch direkt rechnen, unabhängig von Standort, Systemzeit etc.

## 03_motivation_dokumentation_loesungsansatz.tex
## 04_dokumentation_interpretation_ergebnisse.tex
## 05_zusammenfassung.tex
## 06_code_anhang.tex
