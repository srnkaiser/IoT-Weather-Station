# AI Weather Station — KI-Modul

**Team 7** · Bruno, Luka, Sören, Michael
Projekt 7: Weather Station · ESP32 + DHT22 + BMP180

Externes, lokal trainiertes KI-Modell für Wettervorhersage und Anomalie-Erkennung.
Erfüllt die Vorgabe von Dr. Saha: eigenes trainiertes Modell, Python, lokal, eigener Datensatz.

---

## 1. Hardwarebezug

| ThingSpeak | Größe | Sensor | Rolle |
|---|---|---|---|
| field1 | Temperatur °C | DHT22 | Zielgröße + Feature |
| field2 | Luftfeuchte % | DHT22 | Feature |
| field3 | Luftdruck hPa | BMP180 | Feature (als Tendenz) |
| field4 | Temperatur °C | BMP180 | Kreuzvergleich Sensorfehler |

> **Hinweis zur Bezeichnung:** Der Sensor heißt **BMP180**, nicht BME180.
> BMP = *Pressure*, BME = *Environment* (mit Feuchte). Im Paper korrekt schreiben.

**Zentrale Design-Regel:** Das Modell darf ausschließlich Größen verwenden, die
DHT22 und BMP180 später auch liefern. Ein auf Windgeschwindigkeit trainiertes
Modell ist wertlos, wenn der ESP32 sie nicht messen kann.

---

## 2. Warum nicht auf ThingSpeak-Daten trainiert wird

In Wokwi werden Sensorwerte per Regler oder Szenario-Datei gesetzt. Was in eurem
ThingSpeak-Kanal liegt, sind also **eingestellte Werte, keine echten Wetterverläufe**.
Als Trainingsdaten unbrauchbar.

Deshalb die saubere Trennung:

```
Echter historischer Wetterdatensatz  ──►  Training (offline, einmalig)
                                              │
                                              ▼
Wokwi / ESP32 ──► ThingSpeak ────────────►  fertiges Modell ──► Vorhersage + Alarm
```

ThingSpeak wird **nur gelesen**, nie zum Trainieren benutzt.

---

## 3. Installation

```bash
git clone <euer-repo>
cd weather_ai
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Sofort-Test mit synthetischen Daten (die komplette Pipeline in einem Befehl):

```bash
./run_all.sh
```

---

## 4. Den echten Datensatz besorgen

Ihr sollt laut Dozent selbst einen geeigneten Datensatz finden. Zwei Kandidaten,
die exakt zu eurer Sensorik passen:

### Empfehlung: Jena Climate (Max-Planck-Institut für Biogeochemie)

Zeitreihe 2009–2016, 10-Minuten-Auflösung, enthält `T (degC)`, `rh (%)`, `p (mbar)` —
deckt sich exakt mit DHT22 + BMP180. Über 400.000 Messpunkte.

```bash
# CSV herunterladen und ablegen als:
data/jena_climate_2009_2016.csv
```

Dann in `config.py`:

```python
DATASET = "jena"
```

### Alternative: DWD Open Data

Stündliche Stationsdaten des Deutschen Wetterdienstes, z. B. Stuttgart-Echterdingen.
Gibt eurem Paper lokalen Bezug. Temperatur und Feuchte stecken im Produkt `TU`,
der Luftdruck in `P0` — beide Dateien müsst ihr über `MESS_DATUM` zusammenführen.
Format ist Semikolon-separiert, Fehlwerte sind `-999`.

Danach `DATASET = "dwd"` und ggf. `RESAMPLE = "1h"`, `STEP_MINUTES = 60` setzen.

> Nach dem Wechsel auf echte Daten **alle Skripte neu laufen lassen**. Die Zahlen
> aus dem synthetischen Datensatz dürfen nicht ins Paper.

---

## 5. Projektstruktur

```
weather_ai/
├── config.py                  Alle Parameter zentral
├── run_all.sh                 Komplette Pipeline
├── requirements.txt
├── data/                      Datensätze
├── models/                    Trainierte Modelle (.joblib)
├── results/                   Metriken, Tabellen, Plots → fürs Paper
└── src/
    ├── data_loader.py         Laden, Bereinigen, einheitliches Schema
    ├── make_testdata.py       Synthetische Testdaten
    ├── features.py            Feature Engineering
    ├── train_forecast.py      Vorhersagemodell + Modellvergleich
    ├── train_anomaly.py       Isolation Forest + Fehlerinjektion
    ├── sensor_check.py        Layer 2: Kreuzvergleich der Sensoren
    ├── evaluate_combined.py   Gesamtauswertung beider Layer
    ├── predict_live.py        ThingSpeak → Vorhersage + Alarm
    └── train_lstm.py          Optional: LSTM für die Vergleichstabelle
```

---

## 6. Die drei Design-Entscheidungen, die ihr im Paper begründen müsst

### 6.1 Persistenz-Baseline

Wetter ist stark autokorreliert. Eine 1-Stunden-Temperaturvorhersage sieht
in absoluten Zahlen immer gut aus — R² über 0,97 bekommt ihr geschenkt.
Aussagekräftig ist nur der Abstand zur naiven Vorhersage „die Temperatur bleibt
gleich". Deshalb läuft die Persistenz-Baseline durch die gesamte Auswertung mit,
und der **Skill Score** (Anteil des Baseline-Fehlers, den das Modell entfernt) ist
die eigentliche Kennzahl.

### 6.2 Chronologischer Split, niemals zufällig

Bei einem Zufalls-Split sieht das Modell im Training die Zukunft und wird auf der
Vergangenheit bewertet. Jede Metrik wird dadurch geschönt. Wir schneiden strikt
zeitlich: erste 80 % Training, letzte 20 % Test.

### 6.3 Höhenunabhängiger Luftdruck

Absoluter Luftdruck hängt von der Stationshöhe ab. Jena liegt auf ca. 155 m,
euer Standort nicht. Ein auf absolute hPa trainiertes Modell schleppt diesen
Offset als gelernten Bias mit. Deshalb: absolute Druckwerte raus, stattdessen
Abweichung vom eigenen 24-h-Mittel plus die **Drucktendenz** über 3 und 6 Stunden.
Die ist höhenunabhängig — und fallender Druck ist ohnehin der klassische
synoptische Prädiktor für eine heranziehende Front.

Abschaltbar über `ALTITUDE_INVARIANT_PRESSURE = False` in `config.py`.

---

## 7. Die zweischichtige Anomalie-Architektur

Der wichtigste konzeptionelle Teil des Projekts — und das, was euch von anderen
Gruppen unterscheidet.

```
                    Messwert
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ Layer 1           │      │ Layer 2              │
│ Isolation Forest  │      │ Sensor-Kreuzvergleich│
│ statistisch       │      │ deterministisch      │
└─────────┬─────────┘      └──────────┬───────────┘
          │                           │
          └───────────┬───────────────┘
                      ▼
       Layer 2 stumm + Layer 1 feuert → echtes Wetterereignis
       Layer 2 feuert                 → Hardwaredefekt, Daten unbrauchbar
```

**Warum zwei Schichten?** Der Isolation Forest ist bei zwei Fehlerarten
strukturell blind:

- **stuck** (eingefrorener Wert): ein konstanter Wert ist statistisch nicht
  extrem, er ist nur konstant.
- **drift** (langsam wachsender Offset): jeder Einzelwert bleibt im plausiblen
  Bereich.

Beide fängt Layer 2 über Hardware-Redundanz ab: DHT22 und BMP180 messen dieselbe
Temperatur am selben Ort. **Wetter trifft beide Sensoren. Ein Defekt trifft einen.**

Die Schwelle wird über *Median + k · MAD* gebildet, nicht über Mittelwert + k · σ.
MAD ist robust — ein driftender Sensor bläht die Standardabweichung auf und würde
eine σ-basierte Schwelle über genau den Fehler heben, den sie erkennen soll.

**Bewertung ohne Labels:** Anomalie-Erkennung ist unüberwacht, also gibt es
nichts zu messen. Wir injizieren deshalb Fehler bekannten Typs an bekannten
Positionen in den Testabschnitt und messen Precision/Recall dagegen. Das ergibt
eine echte quantitative Tabelle statt „das Modell hat ein paar Anomalien gefunden".

---

## 8. Live-Betrieb

`config.py` ausfüllen:

```python
THINGSPEAK = dict(
    channel_id="1234567",
    read_api_key="XXXXXXXX",   # leer lassen, wenn Kanal öffentlich
    ...
)
```

Dann:

```bash
python3 src/predict_live.py              # einmalig
python3 src/predict_live.py --watch 300  # alle 5 Minuten
python3 src/predict_live.py --json       # maschinenlesbar
```

Optional Rückschreiben in einen zweiten ThingSpeak-Kanal (Vorhersage,
Anomalie-Flag, Sensorfehler-Flag) über `write_api_key`.

**Achtung Vorlaufzeit:** Die Features brauchen ca. 24 h lückenlose Historie
(längstes Rolling-Fenster). Vorher liefert `predict_live.py` einen Fehler.
Für die Demo: Wokwi mit beschleunigtem Szenario laufen lassen oder
`ROLLING_MIN = [60, 360]` setzen.

---

## 9. Zuordnung zur geforderten Paper-Struktur

| Abschnitt im Paper | Woher |
|---|---|
| Proposed AI-enabled Architecture | Abschnitt 2 + 7 dieser README |
| Hardware Design | Tabelle Abschnitt 1 |
| Cloud Integration (ThingSpeak) | `predict_live.py` |
| AI Model | `features.py`, `train_forecast.py`, Abschnitt 6 |
| Experimental Results | `results/forecast_metrics.md`, `results/combined_results.md` |
| | Plots: `results/*.png` |
| Conclusion / Future Scope | Abschnitt 10 |

---

## 10. Limitationen — ehrlich benennen

Ein Gutachter sucht genau danach. Selbst nennen ist stärker als gefunden werden.

1. **Feuchte hängt an einem einzigen Sensor.** Der BMP180 kann keine Feuchte.
   Fällt der DHT22 aus, ist die Größe komplett weg — für sie gibt es keine
   Redundanz und damit keinen Kreuzvergleich.
2. **Ein Standort, ein Zeitraum.** Trainiert auf Daten einer Station. Übertrag
   auf ein anderes Klima ist nicht validiert.
3. **Layer 1 erkennt drift und stuck kaum** (Recall unter 10 %). Das ist keine
   Schwäche der Umsetzung, sondern eine prinzipielle Grenze punktweiser
   Ausreißererkennung — und genau die Begründung für Layer 2.
4. **Contamination ist ein manuell gesetzter Parameter.** Er bestimmt direkt
   die Falschalarmrate und lässt sich ohne Labels nicht sauber optimieren.
5. **Nur ein Vorhersagehorizont.** 60 Minuten. Längere Horizonte sind über
   `FORECAST_HORIZON_MIN` erreichbar, aber nicht evaluiert.
6. **Kein Niederschlag, kein Wind.** Ohne entsprechende Sensorik ist echte
   Wettervorhersage im meteorologischen Sinn nicht möglich — es geht um
   Kurzfrist-Extrapolation lokaler Messgrößen.

---

## 11. Nächste Schritte

- [ ] Echten Datensatz herunterladen, `DATASET` umstellen, alles neu trainieren
- [ ] ThingSpeak-Zugangsdaten in `config.py` eintragen
- [ ] Feldbelegung field1–field4 gegen euren Kanal prüfen
- [ ] Optional `train_lstm.py` für die Vergleichstabelle
- [ ] Plots aus `results/` ins Paper übernehmen
