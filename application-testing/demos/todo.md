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

Väsäillään tällainen aluksi, sitten halutaan alkaa rakentaa sum-funktiota, joka laskee datan summan.
Luodaan testejä:
- annetaan nollavektori, assert 0
- annetaan ykkösvektori, assert n
- annetaan iotavektori, assert n(n + 1)/2
- annetaan iotavektori väärinpäin, assert n(n + 1) / 2
- annetaan kakkosvektori, assert 2n

Tehdään sum-toteutus vaiheittaan:
- palauta 0
- palauta 0, jos eka 0, muuten palauta n
- palauta 0, jos eka 0, palauta n jos ekat kaksi 1, muuten palauta n (n + 1) / 2
- toteuta kunnolla/oikein

```cpp
template <typename T>
struct Field<T> {
    const size_t num_rows = 0;
    const size_t num_cols = 0;
    T* data = nullptr;

    Field(T* data, size_t num_rows, size_t num_cols): num_rows(num_rows), num_cols(num_cols), data(data) {}

    // Return the number of values needed to store the data
    static size_t size_requirement(size_t num_rows, size_t num_cols);
};
```
