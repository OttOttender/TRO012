Lõpuprojekt - Oma objekti tuvastus

Tuvastatakse reaalajas 5 eritüüpi patareid: RainbowAA, DuracellAA, DuracellAAA, PanasonicAA ja PanasonicAAA.
Lisaks tuvastakse mitu antud klassi patareid asub hetkel ekraanil ning palju on patareid ekraanil kokku.
Patareide filmimiseks on kasutatud IPhone 12 tagakaamerat, mis filmib ligikaudu 50 cm kõrguselt.
Projekti dataseti loomiseks tegin 215 pilti: 30 igast patareist üksida, 10 iga patarei gruppidest (2 või enam sama tüüpi) ja 15 pilti erinevates patarei tüüpidest koos.
Dataseti lõin robotflows ning mudeli selle põhjal harjustasin custom scriptiga.

Projekti käigus esile tulnud probleemid:

Kuidas saada live-feed telefonis arvutisse?
Alguses oli ambitsioonikalt ka soov tuvastada patarei + ja - poolt aga selle klasifitseerimine RoboFLows oleks olnud liiga aeganõudev.
Dataset on siiski liiga väike, et saaks suurema confidence variableiga piiritleda false positive.
AA ja AAA tuvastus tundub olema ka väga juhuslik.
Datasetis oleks pidanud klassifitseerima ka patareid, mis ei kuulunud 5 klassi juurde.

Mida saaks tulevikus arendada:

Suurendada dataseti.
Treenida mudelid veel rohkem.
Lisada juurde "Unknown" klass, mis tunneb ära kui tundmatu patarei esineb ekraanil.
Tõsta mingil viisil kaamera FPSi, et tuvastus oleks kasutuskõlblikum reaalses lahenduses.
Õppida selgemaks, kuidas Yolo mudel töötab, et seda effektiivsemalt oma dataseti korrigeerida.
