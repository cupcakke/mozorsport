# JAIDE — CPU tesztek és benchmarkok

Ez a dokumentum a JAIDE / RSF (Visszafordítható Szórt Folyam) verem tényleges,
reprodukálható futási bizonyítékait rögzíti CPU-n. A cél nem a nyers sebesség
demonstrálása, hanem annak igazolása, hogy a rendszer **ténylegesen létezik,
végigfut és matematikailag konzisztens** — a definiáló primitíve nem a
`σ(W·x + b)` alak, hanem egy bijektív, invertálható kereszt-affin csatolás.

## Környezet

| Tétel | Érték |
|---|---|
| CPU | Intel(R) Xeon(R) @ 2.60 GHz |
| Magok (nproc) | 2 |
| OS | Linux 6.1 |
| Fordító | Zig 0.13.0 |
| Build mód | ReleaseFast (benchmarkok), Debug (tesztek) |
| GPU gyorsítás | kikapcsolva (`gpu_acceleration = false`) — tiszta CPU út |
| RSF forward/backward | egyszálú, kézi SIMD nélküli skalár referencia-implementáció |

Megjegyzés: az RSF forward/backward út szándékosan naiv skalár Zig
(0 szál, 0 explicit `@Vector`). A számok ehhez a referencia-szinthez
értendők, nem GPU-hoz vagy vektorizált kernelhez.

## Tesztek — teljes CPU-készlet zöld (314 teszt)

Minden tesztgyökér Zig 0.13.0-val, CPU-n, `gpu_acceleration = false` mellett fut.

| Modul | Eredmény |
|---|---|
| `src/core/tensor.zig` | All 29 tests passed |
| `src/core/memory.zig` | All 19 tests passed |
| `src/core/learned_embedding.zig` | All 30 tests passed |
| `src/core_relational/nsir_core.zig` | All 3 tests passed |
| `src/core_relational/reasoning_orchestrator.zig` | All 59 tests passed |
| `src/core_relational/crev_pipeline.zig` | All 48 tests passed |
| `src/core_relational/surprise_memory.zig` | All 31 tests passed |
| `src/core_relational/temporal_graph.zig` | All 31 tests passed |
| `src/processor/oftb.zig` | All 31 tests passed |
| `src/processor/rsf.zig` | All 33 tests passed |
| **Összesen** | **314 teszt, mind PASS** |

Az `rsf.zig` tesztjei között ott a két invertálhatósági kulcsteszt:

- „RSF forward then inverse returns input within 1e-4 tolerance”
- „RSF with OFTB forward then inverse returns input within 1e-4 tolerance”

Ezek közvetlenül azt bizonyítják, hogy a réteg alapművelete bijektív:
`inverse(forward(x)) == x` a numerikus tűrésen belül.

## Benchmarkok — mind PASS

### bench-rsf (RSF forward/backward)

Teljes forward + backward, `dim=128, layers=4, batch=16, iters=30`:

| Fázis | Idő / iteráció | Áteresztés |
|---|---|---|
| forward | 1.94 ms | ~2.11 M elem/s |
| backward | 3.80 ms | ~1.08 M elem/s |
| backward/forward arány | 1.96× | — |

Nagy konfiguráció (`dim=512, layers=12, batch=64`), forward:
~464 ms/iter ≈ **1.74 GFLOP/s** egyszálú, nem vektorizált skalár úton.

Kontextus: a `backward ≈ 2× forward` arány azért adódik, mert a visszaút
nemcsak a súly- és bemenet-gradienst számolja, hanem az aktivációkat is
**menet közben rekonstruálja** (`inverse`) ahelyett, hogy tárolná őket — ez az
RSF `O(dim)` (mélységtől független) aktivációs memóriájának mérhető lenyomata.
A gradiens a `[dim × (dim+1)]` homogén súlymátrixon folyik át, amelynek utolsó
oszlopa a beolvasztott eltolás (bias) — külön bias-tenzor nélkül.

### bench-matmul (referencia mátrixszorzás, cache-barát i-p-j, 1 mag, skalár)

| Méret | ms / iter | GFLOP/s |
|---|---|---|
| 128×128 | 1.46 | 2.87 |
| 256×256 | 10.99 | 3.05 |
| 512×512 | 87.51 | 3.07 |
| 1024×1024 | 710.74 | 3.02 |

Ez a nyers referencia-sebesség ugyanezen a magon. Az RSF forward
(~1.74 GFLOP/s) ennek a ~57%-a, ami elvárható: az RSF rétegenként két
mátrix-vektor műveletet végez, plusz `exp`/clip és determinisztikus szórt
permutáció overheadet.

### bench-tensor-ops (elemenkénti műveletek, 16 MB tenzor, 500 iter)

| Művelet | ns / elem | Sávszélesség |
|---|---|---|
| fill | 0.35 | 11.32 GB/s |
| add | 0.58 | 6.91 GB/s |
| mul | 0.58 | 6.89 GB/s |

### bench-sfd (SFD optimalizáló: FP4 kvantálás + SpectralNorm)

| Mérés | Eredmény |
|---|---|
| FP4 kvantálás (1 048 576 elem) | ~1.76 G elem/s |
| SpectralNorm, 20 power-iteráció | 10.08 ms/iter |
| SpectralNorm, 5 power-iteráció (ritka) | 3.00 ms/iter |
| gyorsítás (teljes / ritka) | 3.36× |

## Mit bizonyítanak ezek

1. **Létezik és végigfut.** A teljes forward + backward + optimalizáló lánc
   CPU-n lefut, PASS-t ad, memóriaszivárgás nélkül (a tesztek a
   `GeneralPurposeAllocator` szivárgás-ellenőrzésével futnak).
2. **A primitív tényleg invertálható.** Ha a réteg alapművelete nem lenne
   bijektív, a backward (amely `inverse`-szel rekonstruál) numerikusan
   szétesne; a roundtrip tesztek 1e-4 tűréssel átmennek.
3. **Nincs benne `σ(W·x + b)` és nincs perceptron.** A kódbázisban sehol nincs
   softmax, attention, konvolúció, RNN/LSTM/GRU, layer/batch norm, sem külön
   `weights + biases + activation` réteg. Az eltolás homogén koordinátákkal be
   van olvasztva a súlymátrixba (`[dim × (dim+1)]`), így a fő primitív a szórt,
   invertálható kereszt-affin csatolás — és semmi más a négy kanonikus
   paradigmából (perceptron/CNN/RNN/Transformer).

## Reprodukció

A hivatalos build a Zig eszközláncot és a Futhark-generált C kerneleket
használja (`zig build test-all`, `zig build bench`). GPU nélküli, tiszta
CPU-ellenőrzéshez a fenti számok `gpu_acceleration = false` mellett készültek;
a Futhark C entry-k CPU-úton nem hívódnak (`gpu_enabled = false`), így a
neurális RSF út a `src/processor/rsf.zig` skalár implementációján fut.
