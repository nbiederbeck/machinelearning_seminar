# Wolkenklassifikation
*Ein Bericht über den Prozess des Machine Learnings anhand eines selbstgestellten Problems für das Seminar "Machine Learning for Physicists" an der Technischen Universität Dortmund*


## Einleitung, Motivation, Fragestellung
Ein langfristiges Projekt ist das Bauen einer autarken Wetterstation basierend auf dem Raspberry Pi^[raspberrypi.org]. Dies ist ein linuxbasiertes System-On-A-Chip (SoC). 
Es werden die Temperatur, Luftfeuchtigkeit und Luftdruck von Sensoren gemessen.
Langfristig soll die Wetterstation nicht nur aktuelle Wetterdaten aufzeichnen und aufbereiten, sondern das Wetter auch vorhersagen.
Da sich verschiedene Wolkentypen bei unterschiedlichen Witterungsverhältnissen bilden, kam die Idee auf, die aktuelle Wolkenlage als Parameter der Wettervorhersage zu nutzen.
Insbesondere die Veränderungen der Wolkenlage lässt eindeutige Rückschlüsse über den Verlauf des Wetters zu.

Die Rechenleistung und der verfügbare Speicher des Raspberry Pi ist gering, sodass es sinnvoll ist, die notwendigen Rechnungen und den Umfang aufgenommener Daten zu beschränken.
Aus diesem Grund soll die autarke Wetterstation nur den aktuellen Wolkentyp für die Vorhersage speichern.

Die Wetterstation ist mit einer Kamera bestückt, die regelmäßig Wetterdaten und Himmelfotos aufnimmt. Diese Fotos sollen klassifiziert werden, sodass die Datenspeicherung aus einer Klassenrepräsentation, statt aus einem Foto besteht.

Ein maschinelles Lernverfahren soll genutzt werden, um die Himmelfotos direkt auf dem Raspberry Pi zu klassifizieren.
Die Wahl fällt auf ein Neuronales Netz. 
Neuronal Netze bestehen aus mehreren sogenannten Schichten, die verschiedene Matrixmultiplikationen verwenden, um zum einen Dimensionen zu reduzieren und Eingangsparameter im Bezug auf die zugehörige Klasse zu gewichten. 
Ein großer Vorteil gegenüber anderen Maschinellen Lernverfahren ist, dass das Modell unabhängig von den Trainingsdaten verwendet werden kann (vergleiche kNN, hier muss der gesamte Trainingsdatensatz gespeichert werden), was für die Wettervorhersage sehr sinnvoll ist, da der Speicherplatz begrenzt ist (s. o.). 
Außerdem besteht so die Möglichkeit, mehrere Wetterstationen zu kombinieren und für die Klassifikation muss nur das Modell weitergegeben werden.
Die Matrixmultiplikationen sind der Grund dafür, dass die Auswertungszeit auch mit geringer Rechenleistung kurz bleibt.
Ein Neuronales Netz wird mit sogenannten `Convolutional` Schichten die Formen der Wolkenklassen lernen und zusammen mit den Information der drei Farbkanäle (rot, grün, blau) in `Dense` Schichten gewichten, um eine Klassifikation zu ermöglichen. 

Eine Alternativmethode zum Neuronalen Netz ist der Random Forest. 
Auch hier ist kann das Modell zum Auswerten ohne die Trainingsdaten weitergegeben werden und ist somit gut geeignet für kleine Systeme wie den Raspberry Pi.
Der Random Forest hat weniger Kapazitäten als das Neuronale Netz, da er nur die Information der Farbkanäle zum Trainieren bekommt. 
Der Random Forest ist ein beliebter Machine-Learning Algorithmus, da er mit wenig Optimierung brauchbare Ergebnisse liefert und seine Trainingszeit sehr gering ist. 
Ein Random Forest besteht aus einem Ensemble von mehreren binären Entscheidungsbäumen. 


## Auswahl und Beschreibung des Datensatzes
Der Raspberry Pi nimmt regelmäßig Fotos auf.
Diese werden von uns in **11** Klassen eingeteilt und gelabelt.
![**img:** Klassenverteilung (something else)]()
Besonders hervorzuheben sind "Bad Pictures", da diese auf diesen zum einen gar kein Himmel zu sehen ist, da die Kamera in die falsche Richtung gezeigt hat ![**img:** Beispiel]() oder unbekannte Fehler während der Aufnahme auftraten [rotes Bild, grünes Bild]. 
Auf weiteren Bad Pictures sind zu viele Regentropfen auf der Kameralinse, sodass eine vernünftige Klassifikation von unserer Seite nicht zuverlässig wäre ![**img:** Beispiel](), was einen unreinen Datensatz zur Folge hätte. 
Weiterhin nimmt die Wetterstation auch nachts Fotos auf, welche aufgrund des schlechten Lichtsensors in der Kamera nicht gut aufgelöst werden können uns somit großflächig zu dunkel ist, um Wolken zu erkennen. 

Fotos bei Nacht und Bad Pictures werden nicht mit abgegeben, um den benötigten Speicherplatz für den Datensatz nicht unnötig zu vergrößern. 
Die beiden Klassen werden auch nicht in das Training der Algorithmen aufgenommen, da die selbstgestellte Aufgabe ist, fotografierte Wolken in Wolkenklassen einzuteilen, nicht zu entscheiden, ob es ein gutes (persönliche/menschliche Wertung) Foto ist. 
Auf langfristige Sicht sollte der Anteil von Bad Pictures statistisch gering sein. 

Der Datensatz besteht ohne Bad Pictures und Fotos bei Nacht aus **N** Fotos mit 1024x768 Pixeln mit je 3 Farbkanälen und belegt unkomprimiert **6**GB Speicherplatz. 
Er ist aufgeteilt in **9** Wolkenklassen, von denen **n** aus weniger als **k** Fotos bestehen.
![**img:** Verteilung 7 Klassen]()

## Darstellung des Lösungsansatzes
Im Folgenden wird erklärt, wie die aufgenommenen Fotos vorverarbeitet werden und wie sich die Architekturen und das Training der Maschinellen Lernverfahren auf die gewählten Leistungsmerkmale (Genauigkeit?) auswirken. 

### Vorverarbeitung
Zuerst werden Fotos bei Nacht aus dem Datensatz entfernt.
Die erste Idee war, die Fotos anhand der Aufnahmezeit zu filtern.
Dies wurde schnell verworfen, da bewusst war, dass bei einer langfristigen Nutzung der
Wetterstation die Zeiten der Sonnenauf- und -untergänge variieren.
Die Fotos werden stattdessen bei einer mittleren Pixelhelligkeit unter **kk** verworfen.
Dies hat den Vorteil, dass keine Rücksicht auf den Standort und die Systemzeiteinstellung
verschiedener Wetterstationen Rücksicht genommen werden muss.

Die Aufnahme ist so programmiert, dass die Fotos noch vor dem Abspeichern den Helligkeitsfilter durchlaufen, sodass dunkle Fotos nicht gespeichert werden.

Anschließend werden Schnitte auf den Farbkanälen der einzelnen Fotos angewendet.
Dies geschieht aus zwei Gründen, die in der dargestellten Reihenfolge während der Arbeit aufkamen:
1. Entfernung von fotografierten Objekten, die nicht zum Himmel gehören.
    Hierzu zählen insbesondere Bäume und Hauswände, die sich im Sichtfeld der Kamera befinden.
    Diese Objekte sollen mit einem *allgemein anwendbaren* Algorithmus, anstelle eines harten Schnittes auf entsprechenden Bildregionen, entfernt werden, damit verschiedene Wetterstationen denselben Code verwenden können.
    Die Objekte sollen *entfernt* werden, damit das Nauronale Netz von Anfang an
    ausschließlich die Formen von Wolken lernt.
2. Entfernung von Farbwerten wie grün und rot, die im Farbraum von den dominanten Himmelfarben blau, weiß und
   grau weit entfernt sind, damit die Parameter des Random Forest die Himmelfarben und nicht
   Umgebungsfarben sind.

Es wird hierzu ein Cut der Form
$$ b > [g/r] ^ 2,$$
mit $r, g, b \in [0, 255]$ den Farbwerten der Pixel,
auf den Farbkanälen angewendet.
Die Pixel, die geschnitten werden sollen, werden auf den RGB Wert $(0, 0, 0)$, also schwarz gesetzt.
![**img:** Beispiele auf Farbwürfel und echten Bildern (offensichtlich/nicht-offensichtlich)]()

### Neuronales Netz
Das Neuronale Netz wird mit dem Python Paket *keras*^[keras.io] implementiert.
Es wird die Klasse `keras.models.Sequential` verwendet.

##### *Merkmale*
**Hier noch herausfinden, wie das Netz aktuell gebaut ist.**
Die Eingangsparameter des Neuronalen Netzes sind die vollständigen Fotos, also 1024x768x3 Parameter.
Es werden **k** Abfolgen von `Convolutional` Schichten und `MaxPooling` Schichten verwendet.
Die Schichten werden zur Dimensionsreduktion verwendet.
Zusätzlich dazu ist die `Convolutional` Schicht geeignet, um Kanten zu finden und somit fähig, die Form der Wolken zu lernen.
Im Anschluss folgt eine `Flatten` Schicht, die die $(n \times N)$-Dimensionalität auf eine $(n \cdot N \times 1)$-Dimension reduziert.
Dies ist notwendig, um im Anschluss klassische `Dense` Schichten zu verwenden.

Es folgen `Dense` Schichten der Dimensionen $x_0, x_1, \ldots, x_l$, jeweils gepaart mit `Noise` Schichten und `Dropout` Schichten.
Die `Dropout` Schichten werden genutzt, um die Gewichte in einer Größenordnung zu halten,
die `Noise` Schichten, um die Gewichte klein zu halten.
Beides ist in der Matrixmultiplikation von Vorteil,
da so **herausfinden, warum**.

##### *Auswirkungen*

### Random Forest
Der Random Forest wird mit dem Python Paket *scikit-learn*^[scikit-learn.org] implementiert.
Es wird die Klasse `sklearn.ensemble.RandomForestClassifier` verwendet.

##### *Merkmale*
Die Eingangsparameter des Random Forest sind die histogrammierten Farbkanäle der Fotos, auf denen
die Farbschnitte angewendet wurden.
Es werden die einzelnen Kanäle im Bereich $[1, 255]$ in 10 Bins histogrammiert und normiert.
![**img:** Histogramm und Foto]()
Der Farbwert $0$ wird nicht in das Histogramm aufgenommen, da sonst die durch die Farbschnitte entfernten Objekte in die Trainingsdaten aufgenommen werden würden.



##### *Auswirkungen*



## Darstellung der Ergebnisse
## (( Code, Daten im Anhang ))
