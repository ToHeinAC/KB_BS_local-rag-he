# Reranked Summaries Debug Output - 20250925_100014

# Full State Dump

## user_query

Was ist ein Kd Wert und wie ist er für Ra-226?

## additional_context

```
1. Der Nutzer benötigt eine präzise Definition des **Kd‑Werts** (Sorption‑/Partition‑Koeffizienten) inklusive der zugrunde liegenden physikalisch‑chemischen Konzepte und seiner Relevanz für die Umweltradioökologie von Ra‑226.  
2. Er verlangt eine **tiefgehende technische Analyse** des Kd‑Werts bei Ra‑226, die experimentelle Messmethoden, Modellannahmen und die Einflussfaktoren (z. B. Boden‑pH, organischer Gehalt, Adsorptions­isomere) detailliert darstellt.  
3. Der Fokus liegt auf einer **umfassenden Literaturrecherche**: der Nutzer möchte einen ausführlichen Text mit zusammenfassenden Tabellen, die aktuelle Messwerte, Referenzbereiche und Literaturquellen zu Ra‑226‑Kd‑Werten systematisch auflisten.  
4. Ergänzend wird die **Umgebungskontextualisierung** gefordert – insbesondere die Bedeutung des Kd‑Werts für die Mobilität und Risikoabschätzung von Ra‑226 in Grundwasser‑ und Bodenschichten, inklusive praktischer Implikationen für die Umweltüberwachung.
```

## report_llm

gpt-oss:20b

## detected_language

German

## search_summaries

### Was ist ein Kd Wert und wie ist er für Ra-226?

#### Item 1 (Rank 1)

**Rerank Score:** 2.0
**Original Index:** 0
**Position:** 0

**Content:**
```
Content: **Definition und Bedeutung von Kd (Ra)**  
Kd‑Werte geben an, wie stark Radium adsorbiert wird; höhere Werte deuten auf stärkere Adsorption hin, wobei Radium bei hohen Kd‑Werten nahezu vollständig reversibel adsorbiert ist. [EPA_Kd-c.pdf--250804]  

**Kd‑Messungen und Einflussfaktoren**  
- In der Studie von Nathwani und Phillips (1979b) zeigen sich Kd‑Werte, die mit steigender Calcium‑Konzentration abnehmen, was die Sensitivität von Radium‑Adsorption gegenüber Calcium verdeutlicht. [EPA_Kd-c.pdf--250804]  
- Serne (1979) stellte einen Zusammenhang zwischen Kd und der spezifischen Adsorptionskapazität (CEC) her, insbesondere bei sandigen Böden. [EPA_Kd-c.pdf--250804]  
- Für sandige Böden liegen die Kd‑Werte zwischen 214 und 467 ml g⁻¹ (Table 4.1, trs476). [trs476_TheEnvironBehaviorRadium.pdf--250804]  
- Die EPA‑Studie ergab ungewöhnlich hohe Kd‑Werte, die auf eine mögliche Kalzium‑Präzipitation hinweisen. [EPA_Kd-c.pdf--250804]  

**Tabellarische Übersicht (TABLE 4.1, trs476)**  
Soil Group | Kd (Ra) (L kg⁻¹) | Samples | Min | Max  
Sand | 3 100 | 1 | 3 100 | 3 100  
Loam | 1 100 | 1 | 1 100 | 1 100  
Clay | 38 000 | 1 | 38 000 | 38 000  
All soils | 2 500 | 1 000 | 950 | 950 000  
Geometric Mean | 2.5 × 10³ | – | – | –  
Geometric SD | 13 | – | – | –  
Samples | 1 000 | – | – | –  
Variability | 2 – 5 Ordnung der Größenordnung | – | – | –  
(4.2.1.6)  

**Vertikale Mobilität (4.1)**  
\[
\frac{dC}{dt} = \frac{K_d}{\theta}\frac{d\theta}{dt} + \frac{K_d}{\theta}\frac{d\theta}{dt}
\]  
Beispielrechnung (4.2.1.6):  
- Niedriges Ionen‑Stoff‑Wasser, 20 cm a⁻¹ Niederschlagsüberschuss, ρ = 1,4 g mL⁻¹, θ = 0,2 mL mL⁻¹, Kd = 2 500 L kg⁻¹ → interstitielle Geschwindigkeit 100 cm a⁻¹, Radium‑Geschwindigkeit VRa ≈ 0,005 cm a⁻¹. [trs476_TheEnvironBehaviorRadium.pdf--250804]  

**Relevanz für Umwelt‑ und Gesundheitsbewertung**  
- Hohe Kd‑Werte für Radium (z. B. 38 000 L kg⁻¹ in Tonnenböden, 2 500 L kg⁻¹ im IAEA‑Katalog) deuten auf geringe Mobilität und damit auf ein geringeres Transport‑Risiko in Grundwasser hin. [trs476_TheEnvironBehaviorRadium.pdf--250804]  
- Die starke Sensitivität gegenüber Calcium‑Konzentration und die Abnahme der Kd‑Werte bei erhöhtem Calcium sind entscheidend für die Modellierung von Radium‑Transport in Böden. [EPA_Kd-c.pdf--250804]  
- Die hohe Variabilität (2–5 Ordnung der Größenordnung) unterstreicht die Notwendigkeit bodenspezifischer Kd‑Messungen für genaue Risikobewertungen. [trs476_TheEnvironBehaviorRadium.pdf--250804]  
- Kd‑Werte werden in vertikalen Mobilitätsmodellen verwendet, um die Geschwindigkeit des Radium‑Transportes in Grundwasser zu schätzen, was für die Bewertung von Strahlenschutzmaßnahmen von zentraler Bedeutung ist. [trs476_TheEnvironBehaviorRadium.pdf--250804]
Importance_score: 7.82
Source_filename: EPA_Kd-c.pdf--250804, trs476_TheEnvironBehaviorRadium.pdf--250804, BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804
Source_path: kb/insert_data/EPA_Kd-c.pdf, kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf, kb/insert_data/BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf
```

**Metadata:**
```json
{
  "position": 0,
  "query": "Was ist ein Kd Wert und wie ist er f\u00fcr Ra-226?",
  "name": [
    "EPA_Kd-c.pdf--250804",
    "trs476_TheEnvironBehaviorRadium.pdf--250804",
    "BfS_2012_SW_14_12-Nat\u00fcrliche Radioaktivit\u00e4t in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804"
  ],
  "path": [
    "kb/insert_data/EPA_Kd-c.pdf",
    "kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf",
    "kb/insert_data/BfS_2012_SW_14_12-Nat\u00fcrliche Radioaktivit\u00e4t in Baumaterialien und die daraus resultierende Strahlenexposition.pdf"
  ],
  "rerank_score": 2.0,
  "original_index": 0
}
```

### Definition und physikalisch‑chemische Grundlagen des Kd‑Werts bei Radium‑226

#### Item 1 (Rank 1)

**Rerank Score:** 5.0
**Original Index:** 0
**Position:** 1

**Content:**
```
Content: Kd‑Wert (Sorption‑/Partition‑Koeffizient) beschreibt das Verhältnis der Konzentration eines Stoffes in der festen Phase zu seiner Konzentration in der flüssigen Phase und wird in ml g⁻¹ angegeben. Für Radium‑226 (Ra‑226) liegen in der Literatur sehr unterschiedliche Werte vor, die stark von Boden­eigenschaften wie pH, organischem Gehalt, Sand‑/Silt‑/Clay‑anteil und CEC abhängen.  

In der EPA‑Publikation „Kd‑c“ werden mehrere Messungen und Zusammenstellungen von Kd‑Werten für Ra‑226 präsentiert.  
- Looney et al. (1987) empfehlen einen Kd‑Wert von 100 ml g⁻¹ für Ra‑226 und geben einen möglichen Wertebereich von 10 bis 1 000 000 ml g⁻¹ an. Diese Empfehlung ist spezifisch für den Savannah River Plant‑Standort und muss bei anderen Standorten überprüft werden. [EPA_Kd-c.pdf--250804]  
- Serne (1974) hat Kd‑Werte für vier sandige, aride Bodenproben aus Utah gemessen:  
  - Boden I (pH 7.9–8.0): 354 ± 15 ml g⁻¹  
  - Boden II (pH 7.6–7.7): 289 ± 7 ml g⁻¹  
  - Boden III (pH 7.8–7.9): 467 ± 15 ml g⁻¹  
  - Boden IV (pH 7.6–7.8): 214 ± 15 ml g⁻¹  
  Diese Werte verdeutlichen die starke Abhängigkeit von Boden­pH und Textur. [EPA_Kd-c.pdf--250804]  
- Thibault et al. (1990) haben eine umfangreiche Kompilation von Kd‑Werten für Ra‑226 aus verschiedenen Quellen erstellt, um die Migration von Radium in geologischen Lagerstätten zu bewerten. Die genauen Zahlen aus dieser Kompilation sind im vorliegenden Auszug nicht aufgeführt, werden jedoch als Referenz für weitere Analysen genannt. [EPA_Kd-c.pdf--250804]  
- Table 5.26 aus der EPA‑Publikation listet die physikalisch‑chemischen Eigenschaften von Böden, für die Kd‑Werte ermittelt wurden:  
  - Wendover Silty Clay: pH 5.4, organischer Gehalt 16.2 %, Sand 6.7 %, Silt 47.9 %, Clay 45.4 %, CEC 34.7 meq / 100 g  
  - Grimsby Silt Loam: pH 4.3, organischer Gehalt 1.0 %, Sand 43.7 %, Silt 48.9 %, Clay 7.4 %, CEC 10.4 meq / 100 g  
  - St. Thomas Sand: pH 5.2, organischer Gehalt 3.1 %, Sand 91.1 %, Silt 6.8 %, Clay 1.3 %, CEC 10.9 meq / 100 g  
  Diese Parameter sind entscheidend für die Interpretation der Kd‑Werte. [EPA_Kd-c.pdf--250804]  

Das Dokument „strlsch_messungen_gamma_natrad_bf.pdf“ beschreibt Messmethoden zur Bestimmung von Ra‑226 (z. B. Auswertung der 186,2‑keV‑Linie oder der Gammalinien der Nachbarkerne), liefert jedoch keine Kd‑Werte. [strlsch_messungen_gamma_natrad_bf.pdf--250804]  

Die Kd‑Werte sind für die Umweltüberwachung von entscheidender Bedeutung, da sie die Mobilität von Ra‑226 in Grundwasser‑ und Bodenschichten bestimmen. Hohe Kd‑Werte deuten auf starke Adsorption und damit auf geringe Mobilität hin, während niedrige Werte auf eine höhere Freisetzung und potenziell höhere Strahlenbelastung des Grundwassers hinweisen. Diese Informationen werden bei Risikobewertungen, Standortwahl für Lagerstätten und bei der Entwicklung von Sanierungsstrategien verwendet.
Importance_score: 6.78
Source_filename: strlsch_messungen_gamma_natrad_bf.pdf--250804, EPA_Kd-c.pdf--250804, EPA_Kd-c.pdf--250804
Source_path: kb/insert_data/strlsch_messungen_gamma_natrad_bf.pdf, kb/insert_data/EPA_Kd-c.pdf, kb/insert_data/EPA_Kd-c.pdf
```

**Metadata:**
```json
{
  "position": 1,
  "query": "Definition und physikalisch\u2011chemische Grundlagen des Kd\u2011Werts bei Radium\u2011226",
  "name": [
    "strlsch_messungen_gamma_natrad_bf.pdf--250804",
    "EPA_Kd-c.pdf--250804",
    "EPA_Kd-c.pdf--250804"
  ],
  "path": [
    "kb/insert_data/strlsch_messungen_gamma_natrad_bf.pdf",
    "kb/insert_data/EPA_Kd-c.pdf",
    "kb/insert_data/EPA_Kd-c.pdf"
  ],
  "rerank_score": 5.0,
  "original_index": 0
}
```

### Experimentelle Messmethoden und Standardprotokolle zur Bestimmung des Kd‑Werts von Ra‑226 in Böden

#### Item 1 (Rank 1)

**Rerank Score:** 2.0
**Original Index:** 0
**Position:** 2

**Content:**
```
Content: **Definition des Kd‑Werts**  
Der Kd‑Wert (Sorption‑/Partition‑Koeffizient) ist das Verhältnis der Konzentration eines Soluts, das an einer festen Phase adsorbiert ist, zur Konzentration des Soluts in der flüssigen Phase bei Gleichgewicht. Dieser Koeffizient beschreibt, wie stark ein Stoff an Boden‑ oder Sedimentpartikeln bindet und damit seine Mobilität im Grundwasser bestimmt. [EPA_Kd-a.pdf--250804]  

**Relevanz für Ra‑226**  
Ra‑226 ist ein radioaktives Radionuklid, das in Trinkwasserproben nach den deutschen Leitlinien (Kapitel 7.2.5) in vier Quartalen gemessen wird, wenn die Gesamt‑Alpha‑Aktivität den Grenzwert überschreitet. Der Kd‑Wert von Ra‑226 ist entscheidend für die Abschätzung seiner Mobilität in Böden und Grundwasser sowie für die Risikobewertung von Kontaminationen. [EPA_Kd-a.pdf--250804]  

**Experimentelle Messmethoden**  
1. **Laborbatch‑Methode** – Boden‑Proben werden mit einer definierten Konzentration des Radionuklids in einer Lösung inkubiert; nach Erreichen des Gleichgewichts wird die adsorbierte Menge gemessen. [EPA_Kd-a.pdf--250804]  
2. **In‑situ‑Batch‑Methode** – ähnliche Prinzipien wie die Laborbatch‑Methode, jedoch direkt im Feld, um natürliche Bedingungen zu berücksichtigen. [EPA_Kd-a.pdf--250804]  
3. **Laborfluss‑Durch‑Methode** – Bodenproben werden in einem kontinuierlichen Fluss von Lösung durchlaufen; die Adsorption wird über die Zeit verfolgt. [EPA_Kd-a.pdf--250804]  
4. **Feldmodell‑Methode** – Modellierung der Transportprozesse im Feld unter Verwendung von Messdaten aus der Umgebung. [EPA_Kd-a.pdf--250804]  
5. **Koc‑Methode** – Berechnung des Kd aus dem organischen Kohlenstoff‑Sorption‑Koeffizienten (Koc) der Bodenprobe. [EPA_Kd-a.pdf--250804]  

**Modellannahmen und Einflussfaktoren**  
- **Gleichgewicht**: Annahme, dass Adsorption und Desorption im Gleichgewicht sind.  
- **Homogenität**: Bodenpartikel und Lösung gelten als homogen.  
- **Einflussfaktoren**: Boden‑pH, organischer Gehalt, Partikelgröße, Lösung‑zu‑Feststoff‑Verhältnis, Temperatur, Elektrolytkonzentration. Diese Faktoren können die gemessenen Kd‑Werte um bis zu drei Größenordnungen variieren. [EPA_Kd-a.pdf--250804]  

**Variabilität der Kd‑Werte**  
Eine interlaboratorische Übung mit neun Labors zeigte, dass die Kd‑Werte für Cesium von 1,3 ± 0,4 bis 880 ± 160 ml/g und für Plutonium von 70 ± 36 bis 63 000 ± 19 000 ml/g schwankten – eine Variation von bis zu drei Größenordnungen. Für Strontium lagen die Werte innerhalb einer Größenordnung (1,4 ± 0,2 bis 14,9 ± 4,6 ml/g). Die Hauptursachen waren: Tracer‑Zugabe, Lösung‑zu‑Feststoff‑Verhältnis, Anfangskonzentration, Partikelgröße, Trennmethoden, Behältnisse und Temperatur. [EPA_Kd-a.pdf--250804]  

**Literaturwerte (Beispiel für andere Radionuklide)**  

| Radionuklid | Kd‑Bereich (ml/g) | Quelle |
|-------------|-------------------|--------|
| Cs | 1,3 ± 0,4 – 880 ± 160 | [EPA_Kd-a.pdf--250804] |
| Pu | 70 ± 36 – 63 000 ± 19 000 | [EPA_Kd-a.pdf--250804] |
| Sr | 1,4 ± 0,2 – 14,9 ± 4,6 | [EPA_Kd-a.pdf--250804] |

**Keine spezifischen Kd‑Werte für Ra‑226 in den vorliegenden Quellen**  
Die bereitgestellten Dokumente enthalten keine direkten Messwerte oder Literaturangaben für den Kd‑Wert von Ra‑226. Die Relevanz des Kd‑Werts für Ra‑226 wird jedoch aus den allgemeinen Prinzipien der Radionuklid‑Transportmodellierung und den Anforderungen der Trinkwasser‑Leitlinien abgeleitet. [EPA_Kd-a.pdf--250804]  

**Umweltkontextualisierung**  
Der Kd‑Wert bestimmt, ob Ra‑226 in Böden adsorbiert und damit aus dem Grundwasser zurückgehalten wird oder mobil bleibt und potenziell in die Trinkwasserversorgung gelangt. Ein hoher Kd‑Wert bedeutet starke Bindung an Bodenpartikel, geringere Mobilität und geringeres Risiko für die Trinkwasserversorgung. Umgekehrt führt ein niedriger Kd‑Wert zu erhöhter Mobilität und höherem Risiko. Die Bewertung der Mobilität von Ra‑226 erfolgt daher häufig in Kombination mit Messungen der Gesamt‑Alpha‑Aktivität und der spezifischen Radiumkonzentration in Wasserproben. [EPA_Kd-a.pdf--250804]  

**Praktische Implikationen für die Umweltüberwachung**  
- Regelmäßige Messungen der Ra‑226‑Konzentration in Wasserproben (Kapitel 7.2.5).  
- Anwendung geeigneter Batch‑ oder Flow‑Through‑Methoden zur Bestimmung des Kd‑Werts in lokalen Böden.  
- Berücksichtigung von Boden‑pH, organischem Gehalt und Partikelgröße bei der Interpretation der Kd‑Werte.  
- Nutzung der Kd‑Werte in Transport‑ und Risikomodellen, um die potenzielle Ausbreitung von Ra‑226 im Grundwasser zu prognostizieren. [EPA_Kd-a.pdf--250804]
Importance_score: 6.20
Source_filename: 20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf--250804, EPA_Kd-a.pdf--250804, EPA_Kd-a.pdf--250804
Source_path: kb/insert_data/20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf, kb/insert_data/EPA_Kd-a.pdf, kb/insert_data/EPA_Kd-a.pdf
```

**Metadata:**
```json
{
  "position": 2,
  "query": "Experimentelle Messmethoden und Standardprotokolle zur Bestimmung des Kd\u2011Werts von Ra\u2011226 in B\u00f6den",
  "name": [
    "20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf--250804",
    "EPA_Kd-a.pdf--250804",
    "EPA_Kd-a.pdf--250804"
  ],
  "path": [
    "kb/insert_data/20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf",
    "kb/insert_data/EPA_Kd-a.pdf",
    "kb/insert_data/EPA_Kd-a.pdf"
  ],
  "rerank_score": 2.0,
  "original_index": 0
}
```

### Tabellarische Übersicht aktueller Literaturwerte des Kd‑Werts für Ra‑226 in verschiedenen Boden‑ und Grundwasserproben

#### Item 1 (Rank 1)

**Rerank Score:** 2.0
**Original Index:** 0
**Position:** 3

**Content:**
```
Content: Die vorliegenden Quellen enthalten ausschließlich Angaben zu Aktivitätskonzentrationen von Radium‑226 (Ra‑226) in Bau­materialien und in Oberflächengewässern, jedoch keine Daten oder Berechnungen zum Sorptions‑/Partition‑Koeffizienten (Kd) von Ra‑226.  

**1. Aktivitätskonzentrationen in Bau­materialien**  
Die Tabellen aus den Dokumenten *BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf* und *BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf* listen die spezifische Aktivität von Ra‑226 in verschiedenen Baustoffen (z. B. Ziegel, Mauerwerk, Beton) in Bq kg⁻¹. Beispiele:  
- Ziegel: 570 ± 60 Bq kg⁻¹ (Kd‑Wert nicht angegeben) [BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf]  
- Mauerwerk: 1120 ± 79 Bq kg⁻¹ (Kd‑Wert nicht angegeben) [BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf]  

**2. Aktivitätskonzentrationen in Oberflächengewässern**  
Die Quelle *fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf* enthält Messwerte von Ra‑226 in unfiltrierten und filtrierten Gewässern, die aus NORM‑Industrien stammen. Beispielwerte:  
- Fossa Eugeniana: 29 mBq l⁻¹ (unfiltriert), 20 mBq l⁻¹ (filtriert) [fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf]  
- Rheinberger Altrhein: 19 mBq l⁻¹ (unfiltriert), 14 mBq l⁻¹ (filtriert) [fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf]  

**3. Fehlende Kd‑Informationen**  
Kein Abschnitt, keine Tabelle und keine Formel in den bereitgestellten Dokumenten beschreibt den Kd‑Wert, seine Berechnung, Einflussfaktoren (z. B. Boden‑pH, organischer Gehalt) oder Literaturwerte für Ra‑226. Daher kann keine technische Analyse oder Literaturübersicht zu Kd‑Werten für Ra‑226 aus diesen Quellen erstellt werden.  

**Fazit**  
Die vorhandenen Dokumente liefern ausschließlich Aktivitätsdaten von Ra‑226 in Bau­materialien und Gewässern, enthalten jedoch keine Informationen zum Sorptions‑/Partition‑Koeffizienten (Kd) von Ra‑226. Für eine detaillierte Kd‑Analyse wären weitere, spezifisch zu Sorptionsstudien gehörende Quellen erforderlich.
Importance_score: 5.65
Source_filename: BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf--250804, BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804, fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf--250804
Source_path: kb/insert_data/BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf, kb/insert_data/BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf, kb/insert_data/fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf
```

**Metadata:**
```json
{
  "position": 3,
  "query": "Tabellarische \u00dcbersicht aktueller Literaturwerte des Kd\u2011Werts f\u00fcr Ra\u2011226 in verschiedenen Boden\u2011 und Grundwasserproben",
  "name": [
    "BfS_2012_Nat\u00fcrliche Radioaktivit\u00e4t in Baumaterialien.pdf--250804",
    "BfS_2012_SW_14_12-Nat\u00fcrliche Radioaktivit\u00e4t in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804",
    "fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf--250804"
  ],
  "path": [
    "kb/insert_data/BfS_2012_Nat\u00fcrliche Radioaktivit\u00e4t in Baumaterialien.pdf",
    "kb/insert_data/BfS_2012_SW_14_12-Nat\u00fcrliche Radioaktivit\u00e4t in Baumaterialien und die daraus resultierende Strahlenexposition.pdf",
    "kb/insert_data/fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf"
  ],
  "rerank_score": 2.0,
  "original_index": 0
}
```

### Einflussfaktoren auf den Kd‑Wert von Ra‑226: Boden‑pH, organischer Gehalt, Adsorptionsisomere und Minerallösungen

#### Item 1 (Rank 1)

**Rerank Score:** 4.0
**Original Index:** 0
**Position:** 4

**Content:**
```
Content: Ein Kd‑Wert (Sorption‑/Partition‑Koeffizient) ist definiert als das Verhältnis der Konzentration des Radionuklids in der festen Phase zur Konzentration in der Boden­lösung: „concentration in the solid phase divided by the concentration of the radionuclide in the soil solution obtained from the contaminated soil“ [trs476_TheEnvironBehaviorRadium.pdf--250804].  

**Physikalisch‑chemische Grundlagen für Ra‑226**  
- Ra‑226 existiert im Boden als Ra²⁺ über den pH‑Bereich 4–8 und bildet kaum komplexe Spezies. Es hat eine hohe Affinität zu den regulären Austausch‑stellen des Bodens [trs476_TheEnvironBehaviorRadium.pdf--250804].  
- Die Adsorption von Ra‑226 wird stark von der Cation‑Exchange‑Capacity (CEC) und dem organischen Gehalt (OM) bestimmt. Vandenhove und Van Hees fanden die Beziehungen  
  \[
  K_d(\text{Ra}) = 0.71 \times \text{CEC} - 0.64 \quad (R^2=0.91)
  \]  
  \[
  K_d(\text{Ra}) = 27 \times \text{OM} - 27 \quad (R^2=0.83)
  \]  
  [trs476_TheEnvironBehaviorRadium.pdf--250804].  
- Der Boden‑pH hat einen positiven Einfluss auf die Adsorption: mit steigendem pH steigt die Kd‑Werte [trs476_TheEnvironBehaviorRadium.pdf--250804].  
- Calcium‑Ionen reduzieren die Adsorption von Ra‑226; höhere Ca²⁺‑Konzentrationen führen zu niedrigeren Kd‑Werten, was in den Messungen von Nathwani und Phillips (1979b) beobachtet wurde [EPA_Kd-c.pdf--250804].  

**Literaturwerte und Messmethoden**  
| Quelle | Bodenart | Kd‑Wert | Einheit | Bemerkung |
|--------|----------|---------|---------|-----------|
| Serne (1974) | Sandige, aride Böden aus Utah | 214–467 | ml g⁻¹ | pH 7,6–8,0, Kd korreliert mit CEC | [EPA_Kd-c.pdf--250804] |
| Nathwani & Phillips (1979b) | Verschiedene Böden | sehr groß, >10⁴ ml g⁻¹ | – | Ungewöhnlich hohe Werte, möglicher Radium‑Precipitation | [EPA_Kd-c.pdf--250804] |
| Sheppard et al. (TRS 476) | Verschiedene Bodenklassen | 47 L kg⁻¹ (n=37) | L kg⁻¹ | Unabhängig von Bodenart, Empfehlung für allgemeine Anwendung | [trs476_TheEnvironBehaviorRadium.pdf--250804] |
| TRS 364 (IAEA) | Sand | 490 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |
| TRS 364 (IAEA) | Loam | 36 000 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |
| TRS 364 (IAEA) | Clay | 9 000 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |
| TRS 364 (IAEA) | Organic | 2 400 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |
| IAEA (aktuelle Kompilation) | Verschiedene | geometrisches Mittel variiert um 2–5 Ordnung | – | Kd‑Werte für Clay > Loam, erklärt durch höhere CEC | [trs476_TheEnvironBehaviorRadium.pdf--250804] |

**Umwelt‑ und Risikokontext**  
- Ra‑226 ist im Vergleich zu Uran weniger mobil, wie die niedrigen Mobilitätsanteile (12 % für 226Ra vs. 81 % für 238U) zeigen [trs476_TheEnvironBehaviorRadium.pdf--250804].  
- Hohe Kd‑Werte bedeuten starke Boden‑Adsorption, was die Mobilität in Grundwasser reduziert und das Risiko für die Umwelt senkt.  
- Boden­parameter wie CEC, OM und pH sind entscheidend für die Vorhersage der Ra‑226‑Mobilität; daher werden Kd‑Werte in Umwelt‑Monitoring‑Modellen verwendet, um die Ausbreitung von Ra‑226 in Böden und Grundwasser zu bewerten.  
- Die große Streuung der Kd‑Werte (bis zu 5 Ordnung) unterstreicht die Notwendigkeit bodenspezifischer Messungen und die Vorsicht bei der Anwendung allgemeiner Schätzwerte.  

Diese Zusammenfassung liefert die präzise Definition des Kd‑Werts, die physikalisch‑chemischen Grundlagen für Ra‑226, eine systematische Auflistung aktueller Messwerte aus der Literatur, sowie die Bedeutung des Kd‑Werts für die Mobilität und Risikoabschätzung von Ra‑226 in Böden und Grundwasser.
Importance_score: 6.80
Source_filename: trs476_TheEnvironBehaviorRadium.pdf--250804, EPA_Kd-c.pdf--250804, trs476_TheEnvironBehaviorRadium.pdf--250804
Source_path: kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf, kb/insert_data/EPA_Kd-c.pdf, kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf
```

**Metadata:**
```json
{
  "position": 4,
  "query": "Einflussfaktoren auf den Kd\u2011Wert von Ra\u2011226: Boden\u2011pH, organischer Gehalt, Adsorptionsisomere und Minerall\u00f6sungen",
  "name": [
    "trs476_TheEnvironBehaviorRadium.pdf--250804",
    "EPA_Kd-c.pdf--250804",
    "trs476_TheEnvironBehaviorRadium.pdf--250804"
  ],
  "path": [
    "kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf",
    "kb/insert_data/EPA_Kd-c.pdf",
    "kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf"
  ],
  "rerank_score": 4.0,
  "original_index": 0
}
```

## scoring_statistics

### total_summaries_scored

```
5
```

### average_score

```
3.0
```

### highest_score

```
5.0
```

### lowest_score

```
2.0
```

### number_of_queries_processed

```
5
```

---

# FULL ResearcherStateV2 WORKFLOW STATE AT RERANK_SUMMARIES STEP

```json
{
  "user_query": "Was ist ein Kd Wert und wie ist er für Ra-226?",
  "current_position": 3,
  "detected_language": {
    "detected_language": "German",
    "current_position": "detect_language"
  },
  "research_queries": [
    "Was ist ein Kd Wert und wie ist er für Ra-226?",
    "Definition und physikalisch‑chemische Grundlagen des Kd‑Werts bei Radium‑226",
    "Experimentelle Messmethoden und Standardprotokolle zur Bestimmung des Kd‑Werts von Ra‑226 in Böden",
    "Tabellarische Übersicht aktueller Literaturwerte des Kd‑Werts für Ra‑226 in verschiedenen Boden‑ und Grundwasserproben",
    "Einflussfaktoren auf den Kd‑Wert von Ra‑226: Boden‑pH, organischer Gehalt, Adsorptionsisomere und Minerallösungen"
  ],
  "retrieved_documents": {
    "Was ist ein Kd Wert und wie ist er für Ra-226?": [
      "Document objects with metadata containing Unknown references"
    ],
    "Definition und physikalisch‑chemische Grundlagen des Kd‑Werts bei Radium‑226": [
      "Document objects with metadata containing Unknown references"
    ],
    "Experimentelle Messmethoden und Standardprotokolle zur Bestimmung des Kd‑Werts von Ra‑226 in Böden": [
      "Document objects with metadata containing Unknown references"
    ],
    "Tabellarische Übersicht aktueller Literaturwerte des Kd‑Werts für Ra‑226 in verschiedenen Boden‑ und Grundwasserproben": [
      "Document objects with metadata containing Unknown references"
    ],
    "Einflussfaktoren auf den Kd‑Wert von Ra‑226: Boden‑pH, organischer Gehalt, Adsorptionsisomere und Minerallösungen": [
      "Document objects with metadata containing Unknown references"
    ]
  },
  "search_summaries": {
    "Was ist ein Kd Wert und wie ist er für Ra-226?": [
      {
        "content": "Content: **Definition und Bedeutung von Kd (Ra)**  \nKd‑Werte geben an, wie stark Radium adsorbiert wird; höhere Werte deuten auf stärkere Adsorption hin, wobei Radium bei hohen Kd‑Werten nahezu vollständig reversibel adsorbiert ist. [EPA_Kd-c.pdf--250804]  \n\n**Kd‑Messungen und Einflussfaktoren**  \n- In der Studie von Nathwani und Phillips (1979b) zeigen sich Kd‑Werte, die mit steigender Calcium‑Konzentration abnehmen, was die Sensitivität von Radium‑Adsorption gegenüber Calcium verdeutlicht. [EPA_Kd-c.pdf--250804]  \n- Serne (1979) stellte einen Zusammenhang zwischen Kd und der spezifischen Adsorptionskapazität (CEC) her, insbesondere bei sandigen Böden. [EPA_Kd-c.pdf--250804]  \n- Für sandige Böden liegen die Kd‑Werte zwischen 214 und 467 ml g⁻¹ (Table 4.1, trs476). [trs476_TheEnvironBehaviorRadium.pdf--250804]  \n- Die EPA‑Studie ergab ungewöhnlich hohe Kd‑Werte, die auf eine mögliche Kalzium‑Präzipitation hinweisen. [EPA_Kd-c.pdf--250804]  \n\n**Tabellarische Übersicht (TABLE 4.1, trs476)**  \nSoil Group | Kd (Ra) (L kg⁻¹) | Samples | Min | Max  \nSand | 3 100 | 1 | 3 100 | 3 100  \nLoam | 1 100 | 1 | 1 100 | 1 100  \nClay | 38 000 | 1 | 38 000 | 38 000  \nAll soils | 2 500 | 1 000 | 950 | 950 000  \nGeometric Mean | 2.5 × 10³ | – | – | –  \nGeometric SD | 13 | – | – | –  \nSamples | 1 000 | – | – | –  \nVariability | 2 – 5 Ordnung der Größenordnung | – | – | –  \n(4.2.1.6)  \n\n**Vertikale Mobilität (4.1)**  \n\\[\n\\frac{dC}{dt} = \\frac{K_d}{\\theta}\\frac{d\\theta}{dt} + \\frac{K_d}{\\theta}\\frac{d\\theta}{dt}\n\\]  \nBeispielrechnung (4.2.1.6):  \n- Niedriges Ionen‑Stoff‑Wasser, 20 cm a⁻¹ Niederschlagsüberschuss, ρ = 1,4 g mL⁻¹, θ = 0,2 mL mL⁻¹, Kd = 2 500 L kg⁻¹ → interstitielle Geschwindigkeit 100 cm a⁻¹, Radium‑Geschwindigkeit VRa ≈ 0,005 cm a⁻¹. [trs476_TheEnvironBehaviorRadium.pdf--250804]  \n\n**Relevanz für Umwelt‑ und Gesundheitsbewertung**  \n- Hohe Kd‑Werte für Radium (z. B. 38 000 L kg⁻¹ in Tonnenböden, 2 500 L kg⁻¹ im IAEA‑Katalog) deuten auf geringe Mobilität und damit auf ein geringeres Transport‑Risiko in Grundwasser hin. [trs476_TheEnvironBehaviorRadium.pdf--250804]  \n- Die starke Sensitivität gegenüber Calcium‑Konzentration und die Abnahme der Kd‑Werte bei erhöhtem Calcium sind entscheidend für die Modellierung von Radium‑Transport in Böden. [EPA_Kd-c.pdf--250804]  \n- Die hohe Variabilität (2–5 Ordnung der Größenordnung) unterstreicht die Notwendigkeit bodenspezifischer Kd‑Messungen für genaue Risikobewertungen. [trs476_TheEnvironBehaviorRadium.pdf--250804]  \n- Kd‑Werte werden in vertikalen Mobilitätsmodellen verwendet, um die Geschwindigkeit des Radium‑Transportes in Grundwasser zu schätzen, was für die Bewertung von Strahlenschutzmaßnahmen von zentraler Bedeutung ist. [trs476_TheEnvironBehaviorRadium.pdf--250804]\nImportance_score: 7.82\nSource_filename: EPA_Kd-c.pdf--250804, trs476_TheEnvironBehaviorRadium.pdf--250804, BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804\nSource_path: kb/insert_data/EPA_Kd-c.pdf, kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf, kb/insert_data/BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf",
        "metadata": {
          "position": 0,
          "query": "Was ist ein Kd Wert und wie ist er für Ra-226?",
          "name": [
            "EPA_Kd-c.pdf--250804",
            "trs476_TheEnvironBehaviorRadium.pdf--250804",
            "BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804"
          ],
          "path": [
            "kb/insert_data/EPA_Kd-c.pdf",
            "kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf",
            "kb/insert_data/BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf"
          ],
          "rerank_score": 2.0,
          "original_index": 0
        }
      }
    ],
    "Definition und physikalisch‑chemische Grundlagen des Kd‑Werts bei Radium‑226": [
      {
        "content": "Content: Kd‑Wert (Sorption‑/Partition‑Koeffizient) beschreibt das Verhältnis der Konzentration eines Stoffes in der festen Phase zu seiner Konzentration in der flüssigen Phase und wird in ml g⁻¹ angegeben. Für Radium‑226 (Ra‑226) liegen in der Literatur sehr unterschiedliche Werte vor, die stark von Boden­eigenschaften wie pH, organischem Gehalt, Sand‑/Silt‑/Clay‑anteil und CEC abhängen.  \n\nIn der EPA‑Publikation „Kd‑c“ werden mehrere Messungen und Zusammenstellungen von Kd‑Werten für Ra‑226 präsentiert.  \n- Looney et al. (1987) empfehlen einen Kd‑Wert von 100 ml g⁻¹ für Ra‑226 und geben einen möglichen Wertebereich von 10 bis 1 000 000 ml g⁻¹ an. Diese Empfehlung ist spezifisch für den Savannah River Plant‑Standort und muss bei anderen Standorten überprüft werden. [EPA_Kd-c.pdf--250804]  \n- Serne (1974) hat Kd‑Werte für vier sandige, aride Bodenproben aus Utah gemessen:  \n  - Boden I (pH 7.9–8.0): 354 ± 15 ml g⁻¹  \n  - Boden II (pH 7.6–7.7): 289 ± 7 ml g⁻¹  \n  - Boden III (pH 7.8–7.9): 467 ± 15 ml g⁻¹  \n  - Boden IV (pH 7.6–7.8): 214 ± 15 ml g⁻¹  \n  Diese Werte verdeutlichen die starke Abhängigkeit von Boden­pH und Textur. [EPA_Kd-c.pdf--250804]  \n- Thibault et al. (1990) haben eine umfangreiche Kompilation von Kd‑Werten für Ra‑226 aus verschiedenen Quellen erstellt, um die Migration von Radium in geologischen Lagerstätten zu bewerten. Die genauen Zahlen aus dieser Kompilation sind im vorliegenden Auszug nicht aufgeführt, werden jedoch als Referenz für weitere Analysen genannt. [EPA_Kd-c.pdf--250804]  \n- Table 5.26 aus der EPA‑Publikation listet die physikalisch‑chemischen Eigenschaften von Böden, für die Kd‑Werte ermittelt wurden:  \n  - Wendover Silty Clay: pH 5.4, organischer Gehalt 16.2 %, Sand 6.7 %, Silt 47.9 %, Clay 45.4 %, CEC 34.7 meq / 100 g  \n  - Grimsby Silt Loam: pH 4.3, organischer Gehalt 1.0 %, Sand 43.7 %, Silt 48.9 %, Clay 7.4 %, CEC 10.4 meq / 100 g  \n  - St. Thomas Sand: pH 5.2, organischer Gehalt 3.1 %, Sand 91.1 %, Silt 6.8 %, Clay 1.3 %, CEC 10.9 meq / 100 g  \n  Diese Parameter sind entscheidend für die Interpretation der Kd‑Werte. [EPA_Kd-c.pdf--250804]  \n\nDas Dokument „strlsch_messungen_gamma_natrad_bf.pdf“ beschreibt Messmethoden zur Bestimmung von Ra‑226 (z. B. Auswertung der 186,2‑keV‑Linie oder der Gammalinien der Nachbarkerne), liefert jedoch keine Kd‑Werte. [strlsch_messungen_gamma_natrad_bf.pdf--250804]  \n\nDie Kd‑Werte sind für die Umweltüberwachung von entscheidender Bedeutung, da sie die Mobilität von Ra‑226 in Grundwasser‑ und Bodenschichten bestimmen. Hohe Kd‑Werte deuten auf starke Adsorption und damit auf geringe Mobilität hin, während niedrige Werte auf eine höhere Freisetzung und potenziell höhere Strahlenbelastung des Grundwassers hinweisen. Diese Informationen werden bei Risikobewertungen, Standortwahl für Lagerstätten und bei der Entwicklung von Sanierungsstrategien verwendet.\nImportance_score: 6.78\nSource_filename: strlsch_messungen_gamma_natrad_bf.pdf--250804, EPA_Kd-c.pdf--250804, EPA_Kd-c.pdf--250804\nSource_path: kb/insert_data/strlsch_messungen_gamma_natrad_bf.pdf, kb/insert_data/EPA_Kd-c.pdf, kb/insert_data/EPA_Kd-c.pdf",
        "metadata": {
          "position": 1,
          "query": "Definition und physikalisch‑chemische Grundlagen des Kd‑Werts bei Radium‑226",
          "name": [
            "strlsch_messungen_gamma_natrad_bf.pdf--250804",
            "EPA_Kd-c.pdf--250804",
            "EPA_Kd-c.pdf--250804"
          ],
          "path": [
            "kb/insert_data/strlsch_messungen_gamma_natrad_bf.pdf",
            "kb/insert_data/EPA_Kd-c.pdf",
            "kb/insert_data/EPA_Kd-c.pdf"
          ],
          "rerank_score": 5.0,
          "original_index": 0
        }
      }
    ],
    "Experimentelle Messmethoden und Standardprotokolle zur Bestimmung des Kd‑Werts von Ra‑226 in Böden": [
      {
        "content": "Content: **Definition des Kd‑Werts**  \nDer Kd‑Wert (Sorption‑/Partition‑Koeffizient) ist das Verhältnis der Konzentration eines Soluts, das an einer festen Phase adsorbiert ist, zur Konzentration des Soluts in der flüssigen Phase bei Gleichgewicht. Dieser Koeffizient beschreibt, wie stark ein Stoff an Boden‑ oder Sedimentpartikeln bindet und damit seine Mobilität im Grundwasser bestimmt. [EPA_Kd-a.pdf--250804]  \n\n**Relevanz für Ra‑226**  \nRa‑226 ist ein radioaktives Radionuklid, das in Trinkwasserproben nach den deutschen Leitlinien (Kapitel 7.2.5) in vier Quartalen gemessen wird, wenn die Gesamt‑Alpha‑Aktivität den Grenzwert überschreitet. Der Kd‑Wert von Ra‑226 ist entscheidend für die Abschätzung seiner Mobilität in Böden und Grundwasser sowie für die Risikobewertung von Kontaminationen. [EPA_Kd-a.pdf--250804]  \n\n**Experimentelle Messmethoden**  \n1. **Laborbatch‑Methode** – Boden‑Proben werden mit einer definierten Konzentration des Radionuklids in einer Lösung inkubiert; nach Erreichen des Gleichgewichts wird die adsorbierte Menge gemessen. [EPA_Kd-a.pdf--250804]  \n2. **In‑situ‑Batch‑Methode** – ähnliche Prinzipien wie die Laborbatch‑Methode, jedoch direkt im Feld, um natürliche Bedingungen zu berücksichtigen. [EPA_Kd-a.pdf--250804]  \n3. **Laborfluss‑Durch‑Methode** – Bodenproben werden in einem kontinuierlichen Fluss von Lösung durchlaufen; die Adsorption wird über die Zeit verfolgt. [EPA_Kd-a.pdf--250804]  \n4. **Feldmodell‑Methode** – Modellierung der Transportprozesse im Feld unter Verwendung von Messdaten aus der Umgebung. [EPA_Kd-a.pdf--250804]  \n5. **Koc‑Methode** – Berechnung des Kd aus dem organischen Kohlenstoff‑Sorption‑Koeffizienten (Koc) der Bodenprobe. [EPA_Kd-a.pdf--250804]  \n\n**Modellannahmen und Einflussfaktoren**  \n- **Gleichgewicht**: Annahme, dass Adsorption und Desorption im Gleichgewicht sind.  \n- **Homogenität**: Bodenpartikel und Lösung gelten als homogen.  \n- **Einflussfaktoren**: Boden‑pH, organischer Gehalt, Partikelgröße, Lösung‑zu‑Feststoff‑Verhältnis, Temperatur, Elektrolytkonzentration. Diese Faktoren können die gemessenen Kd‑Werte um bis zu drei Größenordnungen variieren. [EPA_Kd-a.pdf--250804]  \n\n**Variabilität der Kd‑Werte**  \nEine interlaboratorische Übung mit neun Labors zeigte, dass die Kd‑Werte für Cesium von 1,3 ± 0,4 bis 880 ± 160 ml/g und für Plutonium von 70 ± 36 bis 63 000 ± 19 000 ml/g schwankten – eine Variation von bis zu drei Größenordnungen. Für Strontium lagen die Werte innerhalb einer Größenordnung (1,4 ± 0,2 bis 14,9 ± 4,6 ml/g). Die Hauptursachen waren: Tracer‑Zugabe, Lösung‑zu‑Feststoff‑Verhältnis, Anfangskonzentration, Partikelgröße, Trennmethoden, Behältnisse und Temperatur. [EPA_Kd-a.pdf--250804]  \n\n**Literaturwerte (Beispiel für andere Radionuklide)**  \n\n| Radionuklid | Kd‑Bereich (ml/g) | Quelle |\n|-------------|-------------------|--------|\n| Cs | 1,3 ± 0,4 – 880 ± 160 | [EPA_Kd-a.pdf--250804] |\n| Pu | 70 ± 36 – 63 000 ± 19 000 | [EPA_Kd-a.pdf--250804] |\n| Sr | 1,4 ± 0,2 – 14,9 ± 4,6 | [EPA_Kd-a.pdf--250804] |\n\n**Keine spezifischen Kd‑Werte für Ra‑226 in den vorliegenden Quellen**  \nDie bereitgestellten Dokumente enthalten keine direkten Messwerte oder Literaturangaben für den Kd‑Wert von Ra‑226. Die Relevanz des Kd‑Werts für Ra‑226 wird jedoch aus den allgemeinen Prinzipien der Radionuklid‑Transportmodellierung und den Anforderungen der Trinkwasser‑Leitlinien abgeleitet. [EPA_Kd-a.pdf--250804]  \n\n**Umweltkontextualisierung**  \nDer Kd‑Wert bestimmt, ob Ra‑226 in Böden adsorbiert und damit aus dem Grundwasser zurückgehalten wird oder mobil bleibt und potenziell in die Trinkwasserversorgung gelangt. Ein hoher Kd‑Wert bedeutet starke Bindung an Bodenpartikel, geringere Mobilität und geringeres Risiko für die Trinkwasserversorgung. Umgekehrt führt ein niedriger Kd‑Wert zu erhöhter Mobilität und höherem Risiko. Die Bewertung der Mobilität von Ra‑226 erfolgt daher häufig in Kombination mit Messungen der Gesamt‑Alpha‑Aktivität und der spezifischen Radiumkonzentration in Wasserproben. [EPA_Kd-a.pdf--250804]  \n\n**Praktische Implikationen für die Umweltüberwachung**  \n- Regelmäßige Messungen der Ra‑226‑Konzentration in Wasserproben (Kapitel 7.2.5).  \n- Anwendung geeigneter Batch‑ oder Flow‑Through‑Methoden zur Bestimmung des Kd‑Werts in lokalen Böden.  \n- Berücksichtigung von Boden‑pH, organischem Gehalt und Partikelgröße bei der Interpretation der Kd‑Werte.  \n- Nutzung der Kd‑Werte in Transport‑ und Risikomodellen, um die potenzielle Ausbreitung von Ra‑226 im Grundwasser zu prognostizieren. [EPA_Kd-a.pdf--250804]\nImportance_score: 6.20\nSource_filename: 20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf--250804, EPA_Kd-a.pdf--250804, EPA_Kd-a.pdf--250804\nSource_path: kb/insert_data/20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf, kb/insert_data/EPA_Kd-a.pdf, kb/insert_data/EPA_Kd-a.pdf",
        "metadata": {
          "position": 2,
          "query": "Experimentelle Messmethoden und Standardprotokolle zur Bestimmung des Kd‑Werts von Ra‑226 in Böden",
          "name": [
            "20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf--250804",
            "EPA_Kd-a.pdf--250804",
            "EPA_Kd-a.pdf--250804"
          ],
          "path": [
            "kb/insert_data/20180530_Leitfaden Trinkwasser_mit_Formblaettern.pdf",
            "kb/insert_data/EPA_Kd-a.pdf",
            "kb/insert_data/EPA_Kd-a.pdf"
          ],
          "rerank_score": 2.0,
          "original_index": 0
        }
      }
    ],
    "Tabellarische Übersicht aktueller Literaturwerte des Kd‑Werts für Ra‑226 in verschiedenen Boden‑ und Grundwasserproben": [
      {
        "content": "Content: Die vorliegenden Quellen enthalten ausschließlich Angaben zu Aktivitätskonzentrationen von Radium‑226 (Ra‑226) in Bau­materialien und in Oberflächengewässern, jedoch keine Daten oder Berechnungen zum Sorptions‑/Partition‑Koeffizienten (Kd) von Ra‑226.  \n\n**1. Aktivitätskonzentrationen in Bau­materialien**  \nDie Tabellen aus den Dokumenten *BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf* und *BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf* listen die spezifische Aktivität von Ra‑226 in verschiedenen Baustoffen (z. B. Ziegel, Mauerwerk, Beton) in Bq kg⁻¹. Beispiele:  \n- Ziegel: 570 ± 60 Bq kg⁻¹ (Kd‑Wert nicht angegeben) [BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf]  \n- Mauerwerk: 1120 ± 79 Bq kg⁻¹ (Kd‑Wert nicht angegeben) [BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf]  \n\n**2. Aktivitätskonzentrationen in Oberflächengewässern**  \nDie Quelle *fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf* enthält Messwerte von Ra‑226 in unfiltrierten und filtrierten Gewässern, die aus NORM‑Industrien stammen. Beispielwerte:  \n- Fossa Eugeniana: 29 mBq l⁻¹ (unfiltriert), 20 mBq l⁻¹ (filtriert) [fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf]  \n- Rheinberger Altrhein: 19 mBq l⁻¹ (unfiltriert), 14 mBq l⁻¹ (filtriert) [fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf]  \n\n**3. Fehlende Kd‑Informationen**  \nKein Abschnitt, keine Tabelle und keine Formel in den bereitgestellten Dokumenten beschreibt den Kd‑Wert, seine Berechnung, Einflussfaktoren (z. B. Boden‑pH, organischer Gehalt) oder Literaturwerte für Ra‑226. Daher kann keine technische Analyse oder Literaturübersicht zu Kd‑Werten für Ra‑226 aus diesen Quellen erstellt werden.  \n\n**Fazit**  \nDie vorhandenen Dokumente liefern ausschließlich Aktivitätsdaten von Ra‑226 in Bau­materialien und Gewässern, enthalten jedoch keine Informationen zum Sorptions‑/Partition‑Koeffizienten (Kd) von Ra‑226. Für eine detaillierte Kd‑Analyse wären weitere, spezifisch zu Sorptionsstudien gehörende Quellen erforderlich.\nImportance_score: 5.65\nSource_filename: BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf--250804, BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804, fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf--250804\nSource_path: kb/insert_data/BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf, kb/insert_data/BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf, kb/insert_data/fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf",
        "metadata": {
          "position": 3,
          "query": "Tabellarische Übersicht aktueller Literaturwerte des Kd‑Werts für Ra‑226 in verschiedenen Boden‑ und Grundwasserproben",
          "name": [
            "BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf--250804",
            "BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf--250804",
            "fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf--250804"
          ],
          "path": [
            "kb/insert_data/BfS_2012_Natürliche Radioaktivität in Baumaterialien.pdf",
            "kb/insert_data/BfS_2012_SW_14_12-Natürliche Radioaktivität in Baumaterialien und die daraus resultierende Strahlenexposition.pdf",
            "kb/insert_data/fkz_3615_s_12232_strahlenexpositionen_norm_bf.pdf"
          ],
          "rerank_score": 2.0,
          "original_index": 0
        }
      }
    ],
    "Einflussfaktoren auf den Kd‑Wert von Ra‑226: Boden‑pH, organischer Gehalt, Adsorptionsisomere und Minerallösungen": [
      {
        "content": "Content: Ein Kd‑Wert (Sorption‑/Partition‑Koeffizient) ist definiert als das Verhältnis der Konzentration des Radionuklids in der festen Phase zur Konzentration in der Boden­lösung: „concentration in the solid phase divided by the concentration of the radionuclide in the soil solution obtained from the contaminated soil“ [trs476_TheEnvironBehaviorRadium.pdf--250804].  \n\n**Physikalisch‑chemische Grundlagen für Ra‑226**  \n- Ra‑226 existiert im Boden als Ra²⁺ über den pH‑Bereich 4–8 und bildet kaum komplexe Spezies. Es hat eine hohe Affinität zu den regulären Austausch‑stellen des Bodens [trs476_TheEnvironBehaviorRadium.pdf--250804].  \n- Die Adsorption von Ra‑226 wird stark von der Cation‑Exchange‑Capacity (CEC) und dem organischen Gehalt (OM) bestimmt. Vandenhove und Van Hees fanden die Beziehungen  \n  \\[\n  K_d(\\text{Ra}) = 0.71 \\times \\text{CEC} - 0.64 \\quad (R^2=0.91)\n  \\]  \n  \\[\n  K_d(\\text{Ra}) = 27 \\times \\text{OM} - 27 \\quad (R^2=0.83)\n  \\]  \n  [trs476_TheEnvironBehaviorRadium.pdf--250804].  \n- Der Boden‑pH hat einen positiven Einfluss auf die Adsorption: mit steigendem pH steigt die Kd‑Werte [trs476_TheEnvironBehaviorRadium.pdf--250804].  \n- Calcium‑Ionen reduzieren die Adsorption von Ra‑226; höhere Ca²⁺‑Konzentrationen führen zu niedrigeren Kd‑Werten, was in den Messungen von Nathwani und Phillips (1979b) beobachtet wurde [EPA_Kd-c.pdf--250804].  \n\n**Literaturwerte und Messmethoden**  \n| Quelle | Bodenart | Kd‑Wert | Einheit | Bemerkung |\n|--------|----------|---------|---------|-----------|\n| Serne (1974) | Sandige, aride Böden aus Utah | 214–467 | ml g⁻¹ | pH 7,6–8,0, Kd korreliert mit CEC | [EPA_Kd-c.pdf--250804] |\n| Nathwani & Phillips (1979b) | Verschiedene Böden | sehr groß, >10⁴ ml g⁻¹ | – | Ungewöhnlich hohe Werte, möglicher Radium‑Precipitation | [EPA_Kd-c.pdf--250804] |\n| Sheppard et al. (TRS 476) | Verschiedene Bodenklassen | 47 L kg⁻¹ (n=37) | L kg⁻¹ | Unabhängig von Bodenart, Empfehlung für allgemeine Anwendung | [trs476_TheEnvironBehaviorRadium.pdf--250804] |\n| TRS 364 (IAEA) | Sand | 490 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |\n| TRS 364 (IAEA) | Loam | 36 000 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |\n| TRS 364 (IAEA) | Clay | 9 000 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |\n| TRS 364 (IAEA) | Organic | 2 400 L kg⁻¹ | L kg⁻¹ | – | [trs476_TheEnvironBehaviorRadium.pdf--250804] |\n| IAEA (aktuelle Kompilation) | Verschiedene | geometrisches Mittel variiert um 2–5 Ordnung | – | Kd‑Werte für Clay > Loam, erklärt durch höhere CEC | [trs476_TheEnvironBehaviorRadium.pdf--250804] |\n\n**Umwelt‑ und Risikokontext**  \n- Ra‑226 ist im Vergleich zu Uran weniger mobil, wie die niedrigen Mobilitätsanteile (12 % für 226Ra vs. 81 % für 238U) zeigen [trs476_TheEnvironBehaviorRadium.pdf--250804].  \n- Hohe Kd‑Werte bedeuten starke Boden‑Adsorption, was die Mobilität in Grundwasser reduziert und das Risiko für die Umwelt senkt.  \n- Boden­parameter wie CEC, OM und pH sind entscheidend für die Vorhersage der Ra‑226‑Mobilität; daher werden Kd‑Werte in Umwelt‑Monitoring‑Modellen verwendet, um die Ausbreitung von Ra‑226 in Böden und Grundwasser zu bewerten.  \n- Die große Streuung der Kd‑Werte (bis zu 5 Ordnung) unterstreicht die Notwendigkeit bodenspezifischer Messungen und die Vorsicht bei der Anwendung allgemeiner Schätzwerte.  \n\nDiese Zusammenfassung liefert die präzise Definition des Kd‑Werts, die physikalisch‑chemischen Grundlagen für Ra‑226, eine systematische Auflistung aktueller Messwerte aus der Literatur, sowie die Bedeutung des Kd‑Werts für die Mobilität und Risikoabschätzung von Ra‑226 in Böden und Grundwasser.\nImportance_score: 6.80\nSource_filename: trs476_TheEnvironBehaviorRadium.pdf--250804, EPA_Kd-c.pdf--250804, trs476_TheEnvironBehaviorRadium.pdf--250804\nSource_path: kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf, kb/insert_data/EPA_Kd-c.pdf, kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf",
        "metadata": {
          "position": 4,
          "query": "Einflussfaktoren auf den Kd‑Wert von Ra‑226: Boden‑pH, organischer Gehalt, Adsorptionsisomere und Minerallösungen",
          "name": [
            "trs476_TheEnvironBehaviorRadium.pdf--250804",
            "EPA_Kd-c.pdf--250804",
            "trs476_TheEnvironBehaviorRadium.pdf--250804"
          ],
          "path": [
            "kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf",
            "kb/insert_data/EPA_Kd-c.pdf",
            "kb/insert_data/trs476_TheEnvironBehaviorRadium.pdf"
          ],
          "rerank_score": 4.0,
          "original_index": 0
        }
      }
    ]
  },
  "final_answer": "",
  "linked_final_answer": null,
  "selected_database": "NORM__Qwen--Qwen3-Embedding-0.6B--3000--600",
  "quality_check": null,
  "report_llm": "gpt-oss:20b",
  "summarization_llm": "gpt-oss:20b",
  "enable_quality_checker": false,
  "all_reranked_summaries": null,
  "reflection_count": 0,
  "internet_result": null,
  "internet_search_term": null,
  "web_search_enabled": false,
  "human_feedback": "1. Radioökologie. 2. Literaturrecherche benötigt. 3. ausführlicher Text mit zusammenfassenden Tabellen und Quellenangaben",
  "analysis": "```json\n{\n  \"detected_language\": \"German\",\n  \"current_position\": \"detect_language\",\n  \"Key Insights\": [\n    \"Der Fokus liegt auf der Radioökologie, insbesondere der Umweltverträglichkeit von Ra‑226.\",\n    \"Der Nutzer verlangt eine umfangreiche Literaturrecherche.\",\n    \"Er erwartet einen ausführlichen Text, der wichtige Werte zusammenfasst und mit Tabellen sowie Quellenangaben präsentiert.\"\n  ],\n  \"Research Focus\": [\n    \"Verhalten von Ra‑226 in natürlichen Ökosystemen (Verteilung zwischen Wasser, Boden, Sedimenten).\",\n    \"Bestimmung des Partitionierungskoeffizienten (Kd) unter variierenden Umweltbedingungen (pH, Temperatur, Bindegliedern).\",\n    \"Bewertung von Transport- und Retentionsmechanismen sowie Risikoeinschätzung für Ökosysteme.\"\n  ],\n  \"Constraints\": [\n    \"Die Ergebnisse müssen durch Literatur belegt werden (Peer‑Reviewed Journale, Fachbücher, offizielle Datenbanken).\",\n    \"Die Darstellung soll Tabellen enthalten, die Werte und Studienergebnisse übersichtlich zusammenfassen.\",\n    \"Quellenangaben sind exakt (APA, Vancouver oder einheitliches Format).\"\n  ],\n  \"Additional Context\": [\n    \"Keine spezifischen zusätzlichen Informationen wurden bereitgestellt; die Anfrage ist allgemein gehalten.\"\n  ],\n  \"Recommendations\": [\n    \"Starte mit einer systematischen Literaturrecherche (z. B. PubMed, Web of Science, Scopus, Google Scholar).\",\n    \"Nutze Suchbegriffe wie \\\"Ra‑226 Kd\\\", \\\"Radium 226 partition coefficient\\\", \\\"radioecology Ra‑226\\\".\",\n    \"Klassifiziere gefundene Studien nach Umweltmatrix (Wasser, Sediment, Boden) und Parameter (pH, Temperatur, organische Bindung).\",\n    \"Erstelle für jede Kategorie eine Tabelle, die Kd‑Werte, Referenz, Methodik und Bedingungen angibt.\",\n    \"Füge einen ausführlichen Text hinzu, der die Zusammenhänge erklärt, Trends diskutiert und Implikationen für die Umweltbewertung hervorhebt.\",\n    \"Vergiss nicht, am Ende eine Bibliographie in einheitlichem Format aufzuführen.\"\n  ]\n}\n```",
  "follow_up_questions": "1. Welche spezifischen Umweltmatrices (z. B. Grundwasser, Oberflächenwasser, Böden, Sedimente) sollen in der Untersuchung der Kd‑Werte für Ra‑226 besonders berücksichtigt werden, und gibt es bestimmte Regionen oder Gebiete, die für die Analyse Priorität haben?  \n2. Welche vorhandenen Literaturquellen oder Datenbanken haben Sie bereits geprüft, und welche Arten von Studien (z. B. experimentelle Messungen, Modellierungen, Langzeitüberwachungsdaten) sind für Sie am relevantesten?  \n3. In welchem Zitierstil und mit welchen Formatierungsrichtlinien möchten Sie die Ergebnisse präsentieren, und bevorzugen Sie dabei eine bestimmte Datenvisualisierung (z. B. Tabellen, Grafiken, GIS‑Karten)?",
  "additional_context": "1. Der Nutzer benötigt eine präzise Definition des **Kd‑Werts** (Sorption‑/Partition‑Koeffizienten) inklusive der zugrunde liegenden physikalisch‑chemischen Konzepte und seiner Relevanz für die Umweltradioökologie von Ra‑226.  \n2. Er verlangt eine **tiefgehende technische Analyse** des Kd‑Werts bei Ra‑226, die experimentelle Messmethoden, Modellannahmen und die Einflussfaktoren (z. B. Boden‑pH, organischer Gehalt, Adsorptions­isomere) detailliert darstellt.  \n3. Der Fokus liegt auf einer **umfassenden Literaturrecherche**: der Nutzer möchte einen ausführlichen Text mit zusammenfassenden Tabellen, die aktuelle Messwerte, Referenzbereiche und Literaturquellen zu Ra‑226‑Kd‑Werten systematisch auflisten.  \n4. Ergänzend wird die **Umgebungskontextualisierung** gefordert – insbesondere die Bedeutung des Kd‑Werts für die Mobilität und Risikoabschätzung von Ra‑226 in Grundwasser‑ und Bodenschichten, inklusive praktischer Implikationen für die Umweltüberwachung."
}
```

