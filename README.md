# Simulacija upravljanja dvostrukim obrnutim njihalom na kolicima

Završni rad — Sveučilište Josipa Jurja Strossmayera u Osijeku, Fakultet primijenjene matematike i informatike.

Rad razmatra sustav dvostrukog obrnutog njihala na pokretnim kolicima: izvodi se nelinearni fizikalni model, model se linearizira oko uspravne ravnoteže, provjerava se upravljivost sustava, analiziraju polovi otvorene petlje, te se dizajnira regulator stanja metodom linearno-kvadratne regulacije (LQR). Dobiveni regulator naposljetku se testira na punom nelinearnom modelu za nekoliko različitih početnih otklona.

## Sadržaj repozitorija

| Datoteka | Opis |
|---|---|
| `Upravljivost.py` | Provjera upravljivosti sustava — konstruira upravljivu matricu $\mathcal C=[B\ AB\ \cdots\ A^5B]$, računa njen rang i singularne vrijednosti preko `numpy.linalg.svd` / `matrix_rank`. |
| `Racunanje_polova.py` | Rješava algebarsku Riccatijevu jednadžbu (`scipy.linalg.solve_continuous_are`), računa LQR pojačanje $K$ i matricu zatvorene petlje $A_{cl}=A-BK$, te uspoređuje polove otvorene i zatvorene petlje. |
| `Q_R_test.py` | Usporedba tri različita izbora težinskih matrica $Q,R$ (standardna, stroža na kuteve, štedljiva na silu) — za svaku ispisuje $K$, maksimalne otklone, potrebnu silu i vrijeme smirivanja na punom nelinearnom modelu. |
| `Simulacija_Animacija.py` | Glavna simulacija: linearizacija, LQR dizajn, primjena regulatora na puni nelinearni model, grafovi odziva ($q,\theta_1,\theta_2,u$ kroz vrijeme) i animacija njihala s kamerom koja prati kolica. |
| `Simulacija_bez_regulatora.py` | Ista fizika, ali bez regulatora ($u=0$) — pokazuje da sustav bez upravljanja ne ostaje u uspravnom položaju, korišteno za usporedbu u radu. |

## Preduvjeti

```bash
pip install numpy scipy matplotlib
```

Za spremanje animacija (`.gif`) potreban je i `pillow`:

```bash
pip install pillow
```

## Pokretanje

Svaka skripta je samostalna i pokreće se izravno, npr.:

```bash
python Simulacija_Animacija.py
```

Skripte koje generiraju slike/animacije (`Simulacija_Animacija.py`, `Simulacija_bez_regulatora.py`) spremaju izlazne `.png` i `.gif` datoteke u isti direktorij iz kojeg se pokreću.

## Fizikalni parametri korišteni u svim skriptama

| Parametar | Vrijednost | Značenje |
|---|---|---|
| $m$ | 1.0 | masa kolica |
| $m_1$ | 0.3 | masa na vrhu prvog štapa |
| $m_2$ | 0.2 | masa na vrhu drugog štapa |
| $l_1$ | 0.3 | duljina prvog štapa |
| $l_2$ | 0.25 | duljina drugog štapa |
| $d_1,d_2,d_3$ | 0.1, 0.05, 0.05 | koeficijenti trenja |
| $g$ | 9.81 | gravitacijsko ubrzanje |

## Napomena o oznakama u nazivima datoteka/grafova

Parovi brojeva poput `(20,-20)` označavaju **početne otklone** $\theta_1(0),\theta_2(0)$ (u stupnjevima) od uspravnog ravnotežnog položaja s kojima simulacija/animacija kreće — npr. `(20,-20)` znači da prvi štap kreće otklonjen $20^\circ$, a drugi $-20^\circ$ od uspravnog položaja.
