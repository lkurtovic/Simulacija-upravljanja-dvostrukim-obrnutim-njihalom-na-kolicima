# Simulacija upravljanja dvostrukim obrnutim njihalom na kolicima

Završni praktični projekt — Sveučilište Josipa Jurja Strossmayera u Osijeku, Fakultet primijenjene matematike i informatike.

Završni praktični projekt razmatra sustav dvostrukog obrnutog njihala na pokretnim kolicima: izvodi se nelinearni fizikalni model, model se linearizira oko uspravne ravnoteže, provjerava se upravljivost sustava, analiziraju polovi otvorene petlje, te se dizajnira regulator stanja metodom linearno-kvadratne regulacije (LQR). Dobiveni regulator naposljetku se testira na punom nelinearnom modelu za nekoliko različitih početnih otklona, na vanjsku smetnju, te se istražuje njegovo područje atrakcije.

## Sadržaj repozitorija

| Datoteka | Opis |
|---|---|
| `Upravljivost.py` | Provjera upravljivosti sustava — konstruira upravljivu matricu $\mathcal C=[B\ AB\ \cdots\ A^5B]$, računa njen rang i singularne vrijednosti preko `numpy.linalg.svd` / `matrix_rank`. |
| `Racunanje_polova.py` | Rješava algebarsku Riccatijevu jednadžbu (`scipy.linalg.solve_continuous_are`), računa LQR pojačanje $K$ i matricu zatvorene petlje $A_{cl}=A-BK$, te uspoređuje polove otvorene i zatvorene petlje. |
| `Q_R_test.py` | Usporedba tri različita izbora težinskih matrica $Q,R$ (standardna, stroža na kuteve, štedljiva na silu) — za svaku ispisuje $K$, maksimalne otklone, potrebnu silu i vrijeme smirivanja na punom nelinearnom modelu. |
| `QR_utjecaj_na_K.py` | Analiza kako pojedinačna promjena $q_2$, $R$, ili skaliranje cijelog $Q$/$R$ utječe na dobiveno LQR pojačanje $K$. |
| `Simulacija&Animacija.py` | Glavna simulacija: linearizacija, LQR dizajn, primjena regulatora na puni nelinearni model, grafovi odziva ($q,\theta_1,\theta_2,u$ kroz vrijeme) i animacija njihala s kamerom koja prati kolica. |
| `Simulacija_bez_regulatora.py` | Ista fizika, ali bez regulatora ($u=0$) — pokazuje da sustav bez upravljanja ne ostaje u uspravnom položaju, korišteno za usporedbu u radu. |
| `Simulacija_smetnja.py` | Sustav polazi iz uspravne ravnoteže; tijekom simulacije djeluje kratkotrajna vanjska smetnja $w_2$ (moment na prvi zglob, npr. udar vjetra) — pokazuje da regulator uspješno odbacuje i vanjske smetnje, ne samo početne otklone. Uključuje i animaciju s vizualnom oznakom trenutka udara. |
| `Podrucje_atrakcije.py` | Testira gustu mrežu početnih otklona $(\theta_1(0),\theta_2(0))$ i za svaku bilježi stabilizira li se sustav ili divergira — rezultat je vizualizacija područja atrakcije regulatora. Rezultati se spremaju u `rezultati_cache.json`, pa se pri ponovnom/proširenom pokretanju već izračunate točke ne računaju iznova. |

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

Skripte koje generiraju slike/animacije (`Simulacija_Animacija.py`, `Simulacija_bez_regulatora.py`, `Simulacija_smetnja.py`, `Podrucje_atrakcije.py`) spremaju izlazne `.png` i `.gif` datoteke u isti direktorij iz kojeg se pokreću.

`Podrucje_atrakcije.py` može, ovisno o odabranoj rezoluciji i rasponu mreže, trajati i po nekoliko minuta pri prvom pokretanju — napredak se sprema nakon svakog retka mreže u `rezultati_cache.json`, pa se izvođenje sigurno može prekinuti i kasnije nastaviti bez gubitka već izračunatog.

## Fizikalni parametri korišteni u svim skriptama

| Parametar | Vrijednost | Jedinica | Značenje |
|---|---|---|---|
| $m$ | 1.0 | kg | masa kolica |
| $m_1$ | 0.3 | kg | masa na vrhu prvog štapa |
| $m_2$ | 0.2 | kg | masa na vrhu drugog štapa |
| $l_1$ | 0.3 | m | duljina prvog štapa |
| $l_2$ | 0.25 | m | duljina drugog štapa |
| $d_1$ | 0.1 | kg/s | koeficijent trenja kolica |
| $d_2,d_3$ | 0.05, 0.05 | kg·m²/s | koeficijenti trenja zglobova |
| $g$ | 9.81 | m/s² | gravitacijsko ubrzanje |

## Napomena o oznakama u nazivima datoteka/grafova

Parovi brojeva poput `(20,-20)` označavaju **početne otklone** $\theta_1(0),\theta_2(0)$ (u stupnjevima) od uspravnog ravnotežnog položaja s kojima simulacija/animacija kreće — npr. `(20,-20)` znači da prvi štap kreće otklonjen $20^\circ$, a drugi $-20^\circ$ od uspravnog položaja.

## Napomena o `rezultati_cache.json`

Datoteku generira `Podrucje_atrakcije.py`. Ključevi su oblika `"th1,th2"` (cjelobrojni stupnjevi), a vrijednosti `true`/`false` označavaju stabilizira li se sustav iz tog početnog otklona. Datoteku nije potrebno ručno uređivati; ako je slučajno obrišete, sljedeće pokretanje skripte jednostavno počinje računati ispočetka.