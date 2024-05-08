# Skriptit
- LUMIlle ja MAHTIlle sama mpi-skripti kuin PUHTIlla jo on
- omat skriptit, joilla voi ajaa muutkin testit
- yksi skripti, joka vetää koodin reposta, kääntää ja ajaa kaikki testit

# TDD demoamista?
- toteuta karkea versio ensin (käyttäen TDD:tä), joka viimeistellään demotessa
- godboltissa?
- omalla läppärillä?
- demotaan jonkun yksinkertaisen hommelin testaamista:
    - vaikka liittyen johonkin stencil-operaatioon
    - selitystä miten otetaan huomioon GPU/MPI-versiointi?
        - jaetaan toiminnallisuus mahdollisimman yksinkertaisiin osioihin
- Demotaan tdd:tä luomalla `struct Field`, jota voi käyttää GPU:lla tai CPU:lla
    - fieldillä on koko ja pointteri muistiin, eli field on ns. "view" johonkin dataan
    - joku muu huolehtii muistin olemassaolosta

