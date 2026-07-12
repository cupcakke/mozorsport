JAIDE ÁTTEKINTÉS

A JAIDE egy foundation nagy nyelvi modell, amely az 5. gyök architektúra paradigmán alapul. Ez a modell eltér a hagyományos Perceptron, CNN, RNN és Transformer architektúráktól azáltal, hogy Visszafordítható Szórt Folyam (RSF) vermet alkalmaz. Ez a tervezés biztosítja, hogy minden neurális réteg bijektív és invertálható legyen, lehetővé téve az O(dim) memória komplexitást a visszaterjesztés során, mivel az aktivációk menet közben rekonstruálhatók ahelyett, hogy gyorsítótárban tárolnák őket.

A JAIDE az első ténylegesen létező, működő architektúra, amelynek definiáló primitívje nem a σ(W·x + b) alak, és nem is annak valamely variánsa. A RSF réteg alapművelete egy bijektív, invertálható kereszt-affin csatolás (skála- és fordításkomponensekkel, determinisztikus szórt permutációkkal), nem pedig egy nemlineáris aktivációval lezárt affin transzformáció.

A rendszer ezt a visszafordítható neurális gerincet egy magas szintű kognitív réteggel integrálja, amelyet Mag Relációs Rétegnek neveznek, és amely kvantum-inspirált relációs gráfokat és fraktál dinamikát alkalmaz az érveléshez. A teljes verem — beleértve a VPU vektorprocesszort, az FNDSManager fraktál adatstruktúrákat, a CREVPipeline oksági kivonót, a FormalVerificationEngine invariáns ellenőrzőt, a SecurityProofEngine információáramlás igazolót és a QuantumTaskAdapter kvantum szubgráf végrehajtót — közvetlenül részt vesz minden tanítási lépésben és minden következtetési kérésben; egyetlen komponens sincs null-ként inicializálva utólagos aktiválásra várva.

Az 5. gyök paradigma: Visszafordítható Szórt Folyam (RSF)

A JAIDE neurális feldolgozásának magja az RSFLayer, amely kereszt-affin csatoló rétegekből és determinisztikus szórt permutációkból áll. A Transformerekkel ellentétben, amelyek O(L · S · d) memória skálázástól szenvednek a figyelemmechanizmusok miatt, a JAIDE fix memória lábnyomot tart fenn az L mélységtől függetlenül.

Mérési bizonyíték: a teljes RSF verem CPU-n végigfut (forward + backward + optimalizáló), a teljes tesztkészlet zöld, és a benchmarkok is lefutnak. A backward út nem tárolja az aktivációkat, hanem az invertálható primitívvel menet közben rekonstruálja őket; a forward→inverse roundtrip 1e-4 tűréssel átmegy, ami közvetlenül igazolja, hogy a réteg alapművelete bijektív. A részletes teszt- és benchmark-eredményeket, valamint a környezeti adatokat lásd: BENCHMARKS.md.

Főbb neurális komponensek:

- RSF réteg: Megvalósítja a forwardInPlace és inverseInPlace műveleteket skála (S) és fordítás (T) komponensek segítségével.
- OFTB (Ortogonális Fraktál Transzformációs Blokk): Paraméter nélküli, determinisztikus Haar-wavelet szóró/gyűjtő réteg.
- SFD (Spektrális Fisher Diagonalizáló): Másodrendű optimalizáló, amely közelíti az átlós Fisher információs mátrixot spektrális vágással.
- GradientFlowController: initWithConfig segítségével inicializált stabilizáló, amely a spectral_power_iterations, gradient_clip_norm és use_normalized_gradient_flow mezőket használja a beágyazás visszaterjesztéskor a gradiens vágáshoz és normalizáláshoz.

Mag Relációs Réteg

A JAIDE túlmutat az egyszerű token előrejelzésen azáltal, hogy fenntart egy Önhasonló Relációs Gráfot (SSRG / NSIR). Ez a réteg explicit módon tárolja a tokenek közötti kapcsolatokat ritka gráfként, lehetővé téve a szelektív figyelmet O(d) komplexitással.

Főbb alrendszerek:

- NSIR: Csomópontokat és éleket kezel EdgeQuality állapotokkal (szuperpozíció, összefonódott, koherens, összeomlott, fraktál).
- ReasoningOrchestrator: Három szintű megismerést kezel: helyi, globális és meta.
- ZRuntime: Egy relációs végrehajtó motor, amely olyan műveleteket dolgoz fel, mint az entangle_variables és a quantum_circuit.
- VPU: Vektorprocesszor, amely computeGraphEmbeddings, quantumVectorOps és computeSimilarityMatrix útján gráfba ágyazott vektorokat állít elő, kvantum operátorokat alkalmaz rájuk, és hasonlósági mátrixot számol — a mátrix első eleme közvetlenül modulálja a tanítási learning_rate-et.
- FNDSManager: Fraktál Neurális Dinamikus Rendszer kezelő, amely createTree, insertIntoTree, createIndex és addPatternToIndex útján minden tanítási lépéshez és minden inferencia kéréshez létrehoz egy fraktál fát, tokeneket illeszt be, mintaindexet készít, és PatternLocation entitásokat regisztrál.
- CREVPipeline: processTextStream útján a bemeneti szöveg minden inferencia kérésben és minden buildKnowledgeGraph hívásban áthalad az oksági kivonón.
- FormalVerificationEngine: verifyGraph útján a következtetési kérésekben ellenőrzi az NSIR gráf strukturális invariánsait.
- SecurityProofEngine: proveInformationFlowSecurity útján a következtetési kérésekben Bell-LaPadula/Biba/nem-interferencia bizonyítékokat konstruál a gráf éleire.
- QuantumTaskAdapter: identifyQuantumSubgraphs → executeQuantumTask → applyResultsToGraph ciklust futtat a következtetés részeként, a lokális állapotvektor szimulátorral vagy IBM hardverrel.

Navigáció és aloldalak

A dokumentáció speciális szakaszokba van szervezve, amelyek a teljes vermet lefedik a hardver RTL-től a magas szintű érvelésig.

Kezdeti lépések és Build rendszer

Lefedi a Zig 0.14.0 eszközlánc és a Futhark fordító követelményeit. Részletezi, hogyan kell felépíteni a jaide-inference-server, jaide-distributed-futhark, jaide-rtl-sim és jaide-c-api-test futtatható fájlokat a -Dgpu, -Dzk, -Dverify és -Drtl jelzők segítségével.

Rendszerarchitektúra áttekintés

Mélyreható betekintést nyújt a kétrétegű interakciós modellbe: hogyan kezeli a processor/ verem a nagy dimenziós numerikus folyamokat, miközben a core_relational/ réteg szimbolikus és kvantum-relációs struktúrákat kezel — beleértve a VPU vektor műveleteket, az FNDS fraktál indexelést, a formális invariáns ellenőrzést, a biztonsági bizonyításokat és a kvantum szubgráf orchestrálást.

Összefoglaló táblázat: Főbb alrendszerek

| Alrendszer | Elsődleges felelősség | Főbb kódfájlok |
| :--- | :--- | :--- |
| Numerikus mag | Tenzorok, memória, SIMD | src/core/tensor.zig, src/core/memory.zig |
| Neurális verem | RSF rétegek, OFTB keverés | src/processor/rsf.zig, src/processor/oftb.zig |
| Optimalizáló | SFD, GradientFlowController | src/optimizer/sfd.zig |
| Relációs réteg | NSIR gráf, érvelés | src/core_relational/nsir_core.zig, src/core_relational/reasoning_orchestrator.zig |
| Vektorprocesszor | Gráf-beágyazás, kvantum vektor ops, hasonlósági mátrix | src/core_relational/vpu.zig |
| Fraktál indexelés | Fraktál fák, önhasonló index, mintakeresés | src/core_relational/fnds.zig |
| Oksági kivonás | Hármas kivonás, tudásgráf integráció | src/core_relational/crev_pipeline.zig |
| Formális ellenőrzés | Invariánsok, Hoare-logika, tétel bizonyítás | src/core_relational/formal_verification.zig, src/verification/oftb.lean |
| Biztonsági bizonyítás | Információáramlás, hozzáférés vezérlés, integritás | src/core_relational/security_proofs.zig |
| Kvantum feladat | Szubgráf azonosítás, IBM/szimulátor végrehajtás | src/core_relational/quantum_task_adapter.zig |
| Elosztott tanítás | GPUCoordinator, DistributedTrainerFuthark, Modal | src/distributed/gpu_coordinator.zig, src/distributed/distributed_trainer_futhark.zig, src/distributed/modal_gpu.zig |
| Hardver gyorsítás | Futhark kernelek, CUDA | src/hw/accel/, src/main_distributed_futhark.zig |
| Hardver RTL | Haskell MemoryArbiter, RankerCore, SSISearch, Zig szimulátor | src/hw/rtl/, src/hw/rtl/rtl_sim_main.zig |
| Kiszolgálás/Index | SSI, Ranker, HTTP API | src/index/ssi.zig, src/api/inference_server.zig, src/inference_server_main.zig |
| C API | C linkage kötések tesztje | src/core_relational/c_api.zig, src/tests/c_api_test.c |
| ZK áramkör | Circom + snarkjs Groth16 pipeline | src/zk/inference_trace.circom |

---

1.1 KEZDETI LÉPÉSEK ÉS BUILD RENDSZER

Ez az oldal részletezi a JAIDE rendszer build infrastruktúráját, eszközlánc-követelményeit és elsődleges végrehajtási belépési pontjait. A JAIDE egy hibrid build rendszert alkalmaz, amely a Zig eszközlánc köré épül, és C-alapú Futhark kerneleket, Haskell-alapú RTL modulokat, Circom-alapú ZK áramköröket és Lean4-alapú formális bizonyításokat integrál a hardver-gyorsított neurális, relációs, hardveres és kriptográfiai feldolgozáshoz.

Eszközlánc-követelmények

A JAIDE felépítéséhez és futtatásához a következő környezet szükséges:

- Zig fordító: A 0.14.0 verzió szükséges, ahogy azt a build.zig.zon build manifest meghatározza.
- Futhark: Szükséges a src/hw/accel/futhark_kernels.fut és src/hw/accel/main.fut C kerneleinek generálásához. A build rendszer futhark_cpu_step és futhark_gpu_step lépések formájában automatikusan meghívja a futhark c és futhark opencl parancsokat.
- C eszközlánc: Rendszer C fordító (pl. GCC vagy Clang) és libc a generált Futhark kód linkeléséhez és a jaide-c-api-test futtatható fájl fordításához.
- CUDA Toolkit (opcionális, -Dgpu=true esetén): Szükséges a GPU-gyorsított elosztott tanításhoz, kifejezetten a Futhark CUDA/OpenCL backendekkel kompatibilis verzió, cuda, cudart, nvrtc, nccl könyvtárakkal a /usr/local/cuda/lib64 és /usr/local/cuda/lib64/stubs útvonalakon.
- Circom + snarkjs (opcionális, -Dzk=true esetén): Szükséges az inference_trace.circom áramkör lefordításához R1CS/WASM/SYM formátumra és a Groth16 megbízható beállítás végrehajtásához pot12_final.ptau alapján.
- Lake + Lean4 (opcionális, -Dverify=true esetén): Szükséges az src/verification/ könyvtárban található oftb.lean formális bizonyítások fordításához lake build útján.
- GHC (opcionális, -Drtl=true esetén): Szükséges az src/hw/rtl/ könyvtárban található Haskell modulok (MemoryArbiter, RankerCore, SSISearch) megosztott könyvtárba (librtl_sim.so) fordításához.

Build konfiguráció és opciók

A JAIDE build rendszerét a build.zig kezeli. Négy elsődleges konfigurációs kapcsolót biztosít a hardver gyorsítás, a nulla-tudás áramkör fordítás, a formális ellenőrzés és a hardver RTL szimuláció vezérléséhez.

Build opciók

| Opció | Típus | Leírás | Alapértelmezett |
| :--- | :--- | :--- | :--- |
| gpu | bool | Engedélyezi a GPU/CUDA gyorsítást a Futhark CUDA backenden keresztül és összefordítja a jaide-distributed-futhark futtatható fájlt | false |
| zk | bool | Lefordítja a Circom ZK áramköröket (R1CS, WASM, SYM), végrehajtja a Groth16 megbízható beállítást snarkjs-szel és exportálja a verifikációs kulcsot | false |
| verify | bool | Futtatja a Lean4 formális ellenőrzést a src/verification/ könyvtárban lake build segítségével; a test-all is függ ettől | false |
| rtl | bool | GHC-vel megosztott könyvtárba fordítja a MemoryArbiter, RankerCore és SSISearch Haskell modulokat, majd összefordítja a jaide-rtl-sim Zig futtatható fájlt | false |

Ezek az opciók rögzítésre kerülnek a build szkriptben, és build_options modulként propagálódnak a Zig forráskódba gpu_acceleration, zk_enabled, verify_enabled és rtl_enabled mezőnevek alatt.

Függőségek GPU buildekhez

Ha a -Dgpu=true kerül átadásra, a build rendszer megkísérli a következő rendszerkönyvtárakhoz való linkelést a jaide-distributed-futhark számára:

- cuda, cudart, nvrtc (NVIDIA futtatókörnyezet és fordító).
- nccl (NVIDIA Kollektív Kommunikációs Könyvtár) a több GPU-s szinkronizáláshoz.

A build szkript feltételezi a szabványos CUDA útvonalakat a /usr/local/cuda/include és /usr/local/cuda/lib64 helyeken, valamint a /usr/local/cuda/lib64/stubs stub könyvtárat.

Futhark kernel generálás

A build.zig két futhark rendszerparancs lépést tartalmaz:

- futhark_cpu_step: futhark c --library src/hw/accel/futhark_kernels.fut -o src/hw/accel/futhark_kernels — minden Zig futtatható fájl, amely linkeli a futhark_kernels.c-t (jaide-inference-server, jaide-distributed-futhark, összes bench-*), a build.zig-ben step.dependOn(&futhark_cpu_step.step) függést hordozza.
- futhark_gpu_step: futhark opencl --library src/hw/accel/main.fut -o src/hw/accel/main_gpu — a jaide-distributed-futhark futtatható fájl -Dgpu=true esetén ehhez a lépéshez is függést hordoz.

ZK áramkör pipeline (-Dzk=true)

A zk build step három rendszerparancsot futtat láncolt függőséggel:

1. circom src/zk/inference_trace.circom --r1cs --wasm --sym -o src/zk/ — előállítja az R1CS constraint rendszert, a WASM witness generátort és a szimbólum térképet.
2. snarkjs groth16 setup src/zk/inference_trace.r1cs pot12_final.ptau src/zk/inference_trace.zkey — Groth16 megbízható beállítás.
3. snarkjs zkey export verificationkey src/zk/inference_trace.zkey src/zk/verification_key.json — verifikációs kulcs export.

Lean4 formális ellenőrzés (-Dverify=true)

A verify build step a lake build parancsot futtatja a src/verification/ munkakönyvtárban. Ez a lépés a test-all cél függősége is, így a formális bizonyítások futtatása bekerül a teljes teszt csomagba.

Haskell RTL fordítás (-Drtl=true)

Az rtl build step két részből áll:

1. ghc -O2 -dynamic -shared -fPIC src/hw/rtl/MemoryArbiter.hs src/hw/rtl/RankerCore.hs src/hw/rtl/SSISearch.hs -o src/hw/rtl/librtl_sim.so — a Haskell modulok dinamikus megosztott könyvtárrá fordítása.
2. Ezután összefordítódik a jaide-rtl-sim Zig futtatható fájl az src/hw/rtl/rtl_sim_main.zig-ből, amely önmagában futtatható RTL szimulációt biztosít MemoryArbiter, RankerCore és SSISearch statisztikáival.

Elsődleges futtatható fájlok

A build rendszer az alábbi artifaktumokat állítja elő a konfigurációtól függően.

1. jaide-inference-server

A szabványos következtetési motor. HTTP interfészt biztosít a modell interakcióhoz.

- Forrás: src/inference_server_main.zig.
- Függőségek: Linkel a futhark_kernels.c fájlhoz (futhark_cpu_step után), importálja a core_relational modult.
- Cél: Kezeli a teljes kérési folyamatot a tokenizálástól, a beágyazáson, RSF előre menetén, NSIR gráf kódoláson, VPU gráf beágyazáson, FNDS mintaindexelésen, CREV oksági kivonáson, formális invariáns ellenőrzésen, biztonsági bizonyításon, kvantum szubgráf feldolgozáson és a token generálásig.

2. jaide-distributed-futhark

A nagy teljesítményű elosztott tanítási és feldolgozási motor, csak akkor érhető el, ha a gpu engedélyezve van.

- Forrás: src/main_distributed_futhark.zig.
- Függőségek: Linkel a main_gpu.c és futhark_kernels.c fájlokhoz (mindkét futhark step után), teljes CUDA/NCCL linkelést igényel.
- Cél: Kezeli a több rangú GPU tanítást, a gradiens all-reduce-t és a nagy léptékű RSF modell frissítéseket. Támogatja a jaide-distributed-futhark --deploy <model_path> <dataset_path> parancssori módot, amely a ModalGPUClient segítségével közvetlenül a Modal API-hoz posztolja a tanítási feladatot MODAL_API_TOKEN környezeti változó alapján és 30 másodpercenként lekérdezi az állapotot.

3. jaide-rtl-sim (-Drtl=true esetén)

Önmagában futtatható RTL szimulátor. Argumentumai: cycles banks requests_per_cycle. Kiadja a memória arbiter kihasználtságot, a banki nyomást, a ranker pontszámokat és az SSI keresés hit arányt.

4. jaide-c-api-test

C nyelvű smoketest a c_api ABI validálásához. A build install célként állítja elő.

Benchmarking és tesztelési csomag

A JAIDE átfogó benchmark és egységteszt csomagot tartalmaz a neurális-relációs verem teljesítményének és helyességének biztosítására.

Benchmarking csomag

A benchmarking infrastruktúra a src/_bench_deps.zig fájlban van összesítve, amely belső modulokat tesz elérhetővé a benchmark futtatók számára. Minden bench futtatható a futhark_cpu_step függéssel fordul.

| Benchmark névtér | Célmodul | Teljesítmény mérőszámok |
| :--- | :--- | :--- |
| rsf | processor/rsf.zig | Előre/visszafelé áteresztőképesség és visszafordítható réteg késleltetés. |
| core_tensor | core/tensor.zig | SIMD elemenként végzett műveletek és csempézett matmul GFLOPS. |
| sfd | optimizer/sfd.zig | Sztochasztikus Fisher átlós frissítési sebesség és K-FAC előkondicionálás. |

Egységtesztek

A build rendszer specifikus lépéseket definiál az egyes alrendszerek tesztjeinek futtatásához. Ezek a zig build <lépés_neve> paranccsal hajthatók végre.

- test-tensor: Validálja a src/core/tensor.zig fájlt (alak/lépés, szórás).
- test-memory: Validálja a src/core/memory.zig fájlt (arena, slab, pool, buddy allokátorok).
- test-rsf: Validálja a src/processor/rsf.zig fájlt (affin csatolás, visszafordíthatóság).
- test-oftb: Validálja a src/processor/oftb.zig fájlt (pillangó keverés).
- test-embedding: Validálja a src/core/learned_embedding.zig fájlt (beágyazás előre/visszafelé menet).
- test-nsir: Validálja a src/core_relational/nsir_core.zig fájlt (gráf topológia, qubit primitívek).
- test-reasoning: Validálja a src/core_relational/reasoning_orchestrator.zig fájlt (energia számítás, hierarchia).
- test-crev: Validálja a src/core_relational/crev_pipeline.zig fájlt (oksági érvelés és hármas kivonás).
- test-surprise: Validálja a src/core_relational/surprise_memory.zig fájlt (Jaccard-disszimilaritás, CAS küszöbök).
- test-temporal: Validálja a src/core_relational/temporal_graph.zig fájlt (állapot pillanatképek).
- test-vpu: Validálja a src/core_relational/vpu.zig fájlt (SIMD vektor típusok, gráf beágyazás, kvantum vektor ops, hasonlósági mátrix).
- test-fnds: Validálja a src/core_relational/fnds.zig fájlt (fraktál fa, önhasonló index, mintakeresés, PatternLocation életciklus).
- test-formal: Validálja a src/core_relational/formal_verification.zig fájlt (invariáns, Hoare-triplet, tétel bizonyítás).
- test-security: Validálja a src/core_relational/security_proofs.zig fájlt (Bell-LaPadula, Biba, nem-interferencia, hozzáférés vezérlés).
- test-quantum-adapter: Validálja a src/core_relational/quantum_task_adapter.zig fájlt (szubgráf azonosítás, feladat végrehajtás, eredmény visszaírás a gráfba).
- test-signal: Validálja a src/core_relational/signal_propagation.zig fájlt (jelterjedés, aktivációs nyomkövetés, inferencia hookok).
- stress-refcount: Futtatja a src/tests/stress_tensor_refcount.zig fájlt a Tensor referenciaszámláló szálbiztonságának validálásához.
- test-c-api: Lefordítja és futtatja a src/tests/c_api_test.c C fájlt a c_api ABI szintű smoketestjéhez.
- test-all: Futtatja a teljes tesztcsomagot, beleértve az összes felsorolt teszt lépést, a stress-refcount-ot és a test-c-api-t is; -Dverify=true esetén a Lean4 lake build-et is meghívja.

---

1.2 RENDSZERARCHITEKTÚRA ÁTTEKINTÉS

A JAIDE architektúra az 5. gyök architektúra paradigmára való átmenetet képviseli, túllépve a hagyományos Perceptron, CNN, RNN és Transformer modelleken. Két elsődleges tartományból álló, szorosan összekapcsolt rendszerként van felépítve: egy Neurális Feldolgozó Réteg (RSF) a nagy dimenziós numerikus transzformációhoz és egy Mag Relációs Réteg a szimbolikus, oksági és kvantum-relációs megismeréshez.

Magas szintű architektúrális rétegek

A rendszer két elsődleges tartományra oszlik, amelyek egy neurális-relációs hídon keresztül kommunikálnak:

1. Neurális Feldolgozó Réteg (RSF): Bijektív, aktiváció-gyorsítótár-mentes Visszafordítható Szórt Folyam rétegek verme. Kezeli a nyers token beágyazásokat és a numerikus jellemzőkivonást O(dim) memória komplexitással.
2. Mag Relációs Réteg: Egy kognitív alrendszer, amely magas szintű érvelést kezel az Önhasonló Relációs Gráfon (NSIR), az oksági ellenőrzésen (CREV), a fraktál dinamikus rendszereken (FNDS), a vektorprocesszoron (VPU), a formális ellenőrzésen (FormalVerificationEngine), a biztonsági bizonyításokon (SecurityProofEngine) és a kvantum feladat orchestráláson (QuantumTaskAdapter) keresztül.

Az RSF neurális verem (processor/)

A neurális réteg magja a Visszafordítható Szórt Folyam (RSF). A Transformerekkel ellentétben, amelyek O(L · S · d) memóriát igényelnek az aktivációkhoz, az RSF rétegek bijektívek. Ez lehetővé teszi, hogy a visszafelé irányuló menet rekonstruálja az aktivációkat a kimenetekből, csökkentve a memória terhelést O(dim)-re, az L rétegek számától függetlenül.

- RSFLayer: Kereszt-affin csatolást valósít meg. Skála (S) és fordítás (T) komponenseket használ az adatok transzformálásához.
- OFTB (Ortogonális Fraktál Transzformációs Blokk): Paraméter nélküli, determinisztikus Haar-wavelet szóró/gyűjtő réteg.
- SFD (Spektrális Fisher Diagonalizáló): Másodrendű optimalizáló FP4→FP32 vegyes pontossággal.
- GradientFlowController: A propagateEmbeddingGradients útvonalon initWithConfig(.{ .gradient_clip_norm, .use_normalized_gradient_flow, .spectral_power_iterations }) hívással inicializálódik, majd manuális normavágást alkalmaz a beágyazás visszaterjesztési vektorára a gradiens robbanás megelőzéséhez.

Mag Relációs Réteg (core_relational/)

A relációs réteg tíz aktív komponensből áll, amelyek a következtetés és a tanítás minden ciklusában részt vesznek:

- SelfSimilarRelationalGraph (NSIR): Csomópontok és élek szálbiztos tárolója EdgeQuality állapotokkal.
- ReasoningOrchestrator: Háromszintű érvelési hierarchia (helyi, globális, meta).
- ChaosCoreKernel: Tartalom-címezhető tárolás és dinamikus feladatütemezés.
- CREVPipeline: Oksági hármas kivonás processTextStream útján.
- ZRuntime: Kvantum-relációs változó végrehajtó motor.
- SignalPropagationEngine: Nem opcionális mező a trainerben; kötelező jelterjedés minden runCoreRelationalPass ciklusban.
- VPU: computeGraphEmbeddings, quantumVectorOps és computeSimilarityMatrix a tanítási és következtetési útvonalakon.
- FNDSManager: Fraktál fák és önhasonló indexek dinamikus létrehozása minden kérés/lépés során.
- FormalVerificationEngine: verifyGraph a következtetéskor az NSIR gráf strukturális invariánsainak ellenőrzésére.
- SecurityProofEngine: proveInformationFlowSecurity a következtetéskor az információáramlás biztonságának validálására.
- QuantumTaskAdapter: identifyQuantumSubgraphs → executeQuantumTask → applyResultsToGraph teljes ciklus minden inferencia kérésben.
- SurpriseMemoryManager és TemporalGraph: online tanulás és időbélyeg alapú állapot pillanatképek.
- RelationalGraphProcessingUnit (R-GPU): Gráf elosztás fizikai magokra.

Ez a kompozíció biztosítja, hogy egyetlen komponens sincs null-ként inicializálva utólagos aktiválásra várva. A DistributedTrainerFuthark initWithComponents útvonalán minden mező a struktúra létrehozásakor beköttetik, és a signal_engine nem opcionális.

Neurális-relációs híd

Az RSF verem és a Mag Relációs Réteg között a következő adatáramlás történik:

1. Az RSF előrelépés kimenete tenzor-bájtsorozat.
2. Ez a bájtsorozat átfut a nsir_graph.encodeInformation-en, ami frissíti a topology_hash-t.
3. A VPU computeGraphEmbeddings hívása F64x4 vektorokat produkál a gráf csomópontokból; a topology_hash első két bájtja meghatározza a θ és φ szögeket a quantumVectorOps számára; a computeSimilarityMatrix első cellája (koherencia) modulálja a tanítási learning_rate-et (felső határ 0.1).
4. Az FNDSManager létrehoz egy fraktál fát, beszúrja a tokeneket, létrehoz egy mintaindexet és regisztrálja a PatternLocation-öket.
5. A ReasoningOrchestrator hierarchikus érvelést futtat 50 belső ciklussal.
6. A SurpriseMemoryManager rögzíti a magas meglepetés-értékű mintákat.
7. A TemporalGraph nanoszekundum pontosságú állapot pillanatképet készít.
8. A SignalPropagationEngine egy propagateStep-et hajt végre a rebindált gráfon és flow analyzer-en.
9. A ZRuntime létrehoz egy változót a globális lépéshez.

Az inferencia útvonalon ezekhez még hozzáadódik a CREVPipeline processTextStream, a FormalVerificationEngine verifyGraph, a SecurityProofEngine proveInformationFlowSecurity és a QuantumTaskAdapter teljes ciklusa.

---

2 NUMERIKUS MAG

A JAIDE numerikus magja a Tensor, memória allokátorok és I/O primitívek gyűjteménye, amelyek a magasabb szintű alrendszerek alapját képezik.

2.1 TENZOR ARITMETIKA

A JAIDE Tensor alrendszere biztosítja a mag numerikus infrastruktúrát az összes RSF művelethez, magas szintű optimalizálási sémákat alkalmazva, mint a SIMD vektorizáció, csempézett memória hozzáférés és a másolás-íráskor (CoW) referenciaszámlálás.

1. Tenzor magstruktúra és Alak

A Tensor struktúra allokátort, adatpuffert és Shape struktúrát tartalmaz. A Shape metaadatokat, dimenziókat és lépéseket tárolja, lehetővé téve a rendszer számára, hogy sokdimenziós tenzorokat kezeljen egyetlen, folytonos allokációs struktúraként. A lépések meghatározzák a memória ugrást, amely szükséges egy lépés megtételéhez egy adott tengely mentén.

- Folytonosság: Egy tenzor folytonosnak tekinthető, ha lépései megfelelnek a szabványos sor-főbb elrendezésnek.
- Szórás: A rendszer támogatja a szórást, lehetővé téve a különböző alakú tenzorok közötti műveleteket, ha dimenzióik kompatibilisek.

2. Memóriakezelés és Másolás-íráskor (CoW)

A neurális menetek során a drága allokációk minimalizálása érdekében a rendszer atomi referenciaszámlálási mechanizmust alkalmaz Másolás-íráskor logikával kombinálva.

- retain(): Atomikusan növeli a referenciaszámlálót és beállítja a cow jelzőt true értékre, jelölve az adatokat megosztottként.
- release(): Csökkenti a számlálót és felszabadítja a memóriát, ha nullára csökken.
- ensureWritable(): Bármely helyben végzett mutáció előtt ez az ellenőrzés biztosítja, hogy ha a tenzor megosztott (cow == true), friss másolat készüljön a mellékhatások megelőzésére a rendszer más részein.

3. Matematikai műveletek

SIMD-vektorizált elemenként végzett műveletek

A rendszer a Zig @Vector típusát alkalmazza hardver-gyorsított műveletekhez. A tenzorok 32 bájtos határokhoz vannak igazítva az AVX/SIMD utasítások hatékony támogatásához.

- Vektor szélesség: A rendszer alapértelmezés szerint 8-as szélességet használ (f32 elemek).
- Műveletek: Elemenként végzett összeadás, kivonás, szorzás és osztás vektorizált ciklusokkal valósul meg folytonos tenzorokhoz, TensorIterator tartalékkal a nem folytonos nézetekhez.

Többszálú csempézett Matmul

Nagy mátrixszorzásokhoz a rendszer csempézett megközelítést alkalmaz a gyorsítótár lokalitás maximalizálásához és a munkaterhelést több szálon osztja el.

- matmul: Orchestrálja két tenzor szorzatát. Validálja a dimenziókat és kiválasztja az optimális végrehajtási útvonalat.
- MatmulComptime: Speciális struktúra kis, rögzített dimenziójú szorzásokhoz (M, K, N), amely inline ciklusokat használ a maximális teljesítményért.

Dekompozíciók és lineáris algebra

A rendszer fejlett algebrai műveleteket biztosít az RSF (Visszafordítható Szórt Folyam) rétegekhez:

- Determináns és inverz: Négyzetes mátrixokhoz számítva, elengedhetetlen a visszafordítható rétegek Jacobi számításaihoz.
- Transzponálás: Nulla másolású művelet, amely felcseréli a dimenziókat és lépéseket.

4. Bináris szerializációs formátum

A tenzorok speciális bináris formátumban kerülnek tárolásra, amely gyors I/O-ra van tervezve memória-leképezésen keresztül.

Szerializációs leképezés:

A Tensor struktúra (shape.dims, shape.strides, data (f32)) a bináris fájlba (JAIDE40) kerül: Mágikus fejléc (4 bájt), Rang (u32), Dimenziók (N * u64), Lépések (N * u64), Adatpuffer (f32 blokkok).

- Formátum: A save függvény írja a tenzor rangját, majd a dimenziókat, lépéseket és a nyers f32 adatpuffert.
- Kompatibilitás: A tenzorok exportálhatók/importálhatók az NSIR-be (Önhasonló Relációs Gráf) kvantum-relációs feldolgozáshoz.

5. Főbb függvények összefoglalója

| Függvény | Fájl elérési út | Leírás |
| :--- | :--- | :--- |
| init | src/core/tensor.zig | Új tenzort allokál a megadott dimenziókkal. |
| retain | src/core/tensor.zig | Atomikusan növeli a referenciaszámlálót a megosztott tulajdonhoz. |
| ensureWritable | src/core/tensor.zig | Másolás-íráskor végrehajtása, ha a tenzor megosztott. |
| add | src/core/tensor.zig | SIMD-gyorsított elemenként végzett összeadás. |
| matmul | src/core/tensor.zig | Csempézett, többszálú mátrixszorzás. |
| transpose | src/core/tensor.zig | A tenzor transzponált nézetét adja vissza. |

---

2.2 MEMÓRIAKEZELÉS

A JAIDE memóriakezelési rendszer speciális allokátorok és szinkronizációs primitívek csomagját biztosítja, amelyek az O(dim) memória műveletek, bijektív neurális rétegek és kvantum-relációs gráf feldolgozás támogatására vannak tervezve. Az architektúra hangsúlyt fektet a gyorsítótár lokalitásra, a lock-free párhuzamosságra a nagy áteresztőképességű folyamatokhoz, és a biztonságos memóriakezelésre az érzékeny modell súlyokhoz.

Mag allokátorok

A JAIDE számos allokációs stratégiát valósít meg a különböző életciklus és teljesítmény követelmények kezeléséhez, a rövid életű neurális aktivációktól a hosszú távú relációs gráf tárolásig.

Arena és ArenaAllocator

Az Arena egy rögzített méretű, szálbiztos lineáris allokátor, amely előre allokált puffert használ. Kötegelt műveletekre van optimalizálva, ahol az összes memória egyszerre visszanyerhető a reset() segítségével.

Az ArenaAllocator rugalmasabb, növekvő arenát biztosít, amely szükség szerint új puffereket allokál egy szülő allokátorból. Megvalósítja a szabványos Zig Allocator interfészt.

Slab és Pool allokátorok

- SlabAllocator: Nagy "slab"-okban kezeli a memóriát, kisebb darabokra osztva azokat a töredezettség csökkentéséhez a változó méretű allokációk során.
- PoolAllocator: Egységes méretű objektumokra optimalizált (pl. NSIR csomópontok). Rögzített méretű blokkok szabad listáját tartja fenn, O(1) allokációt és felszabadítást biztosítva.
- BuddyAllocator: Nagy összefüggő régiók kezelésére használt (mint a Tensor pufferek által igényeltek), kettő hatványán osztva és egyesítve a blokkokat a töredezettség és sebesség egyensúlyozásához.

Oldal és nyomkövető allokátorok

- PageAllocator: Alacsony szintű allokátor, amely közvetlenül az operációs rendszerrel kommunikál a MemoryConfig.PAGE_SIZE-hoz igazított memória allokálásához (16KB macOS ARM-on, 4KB egyébként).
- TrackingAllocator: Fejlesztés és profilozás során használt burkoló a memóriahasználat figyeléséhez, szivárgások észleléséhez és a globális MemoryStats feltöltéséhez.

Szinkronizáció és lock-free struktúrák

A ReasoningOrchestrator és a ChaosCoreKernel támogatásához a JAIDE számos szinkronizációs primitívet biztosít, amelyek minimalizálják a szál versengést.

SpinLock és ReadWriteLock

- SpinLock: Alacsony terhelésű zár, amelyet nagyon rövid kritikus szakaszokhoz használnak, ahol a kontextusváltás terhelése (std.Thread.Mutex-en keresztül) nem kívánatos.
- ReadWriteLock: Több egyidejű olvasót enged meg, de kizárólagos hozzáférést biztosít az íróknak, elengedhetetlen a SelfSimilarRelationalGraph-hoz, ahol a topológia olvasások gyakoriak, de a frissítések ritkák.

Lock-free sor és verem

A JAIDE nem blokkoló adatstruktúrákat valósít meg a Neurális Feldolgozó Réteg és a Mag Relációs Réteg közötti kommunikáció megkönnyítéséhez.

- LockFreeQueue: Több termelős, több fogyasztós sor, amelyet a DynamicTaskScheduler használ gráf műveletek elküldéséhez a következtetési ciklus blokkolása nélkül.
- LockFreeStack: Elsősorban a PoolAllocator szabad listáinak kezelésére használt, hogy nagy teljesítményű allokációt biztosítson több szálon keresztül.

Biztonság és globális nyomkövetés

EncryptedBlob

A modell súlyok és az érzékeny InferenceWitness adatok védelméhez az EncryptedBlob absztrakciót biztosít a nyugalomban titkosított memóriához, amelyet csak az aktív számítás során dekódolnak védett Arena szegmensekbe.

Biztonságos memória műveletek

A rendszer secureZeroMemory-t biztosít annak biztosítására, hogy az érzékeny adatok (mint a BigInt512 privát kulcsok vagy HomomorphicEncryption paraméterek) fizikailag törlődjenek a RAM-ból, nem csak szabadként jelölve. Mind az Arena, mind az ArenaAllocator támogatja a secureDeinit és secureReset metódusokat.

Globális MemoryStats

A JAIDE globális MemoryStats struktúrát tart fenn a rendszer állapotának valós idejű nyomon követéséhez. Ezt a PowerGatingController és a FractalLPU használja terheléselosztási döntésekhez.

| Mérőszám | Leírás |
| :--- | :--- |
| allocated_bytes | Az összes aktív allokátor által jelenleg tartott bájtok összege. |
| peak_usage | A legmagasabb rögzített memóriafogyasztás az indítás óta. |
| fragmentation_ratio | A buddy/slab allokátor hatékonyságának mértéke. |
| page_faults | A TrackingAllocator-on keresztül figyelt teljesítményhangoláshoz. |

---

2.3 I/O ÉS MODELL PERZISZTENCIA

Ez a szakasz részletezi a JAIDE rendszer mag I/O primitívjeit és az egységes bináris modell formátumot, amelyet hosszú távú tároláshoz és terjesztéshez használnak. A rendszer a memória-leképezésen keresztüli nagy teljesítményű adathozzáférést helyezi előtérbe, és kriptográfiai ellenőrző összegekkel és atomi írási műveletekkel biztosítja az adatok integritását.

Mag I/O primitívek

A JAIDE alacsony szintű I/O segédprogramok készletét valósítja meg, amelyek nagy áteresztőképességű neurális és relációs adatfeldolgozásra vannak tervezve.

MMAP (Memória leképezés)

Az MMAP struktúra magas szintű interfészt biztosít a memória-leképezett fájlhozzáféréshez, SHARED és PRIVATE leképezési módokat egyaránt támogatva. A std.posix.mmap-et használja a fájlok folyamat címterébe való leképezéséhez, lehetővé téve az O(1) hozzáférést a nagy modell súlyokhoz explicit olvasási/írási rendszerhívások nélkül minden egyes műveletnél.

Főbb jellemzők:

- Szálbiztonság: A hozzáférés std.Thread.Mutex-szel védett.
- Automatikus méretezés: Automatikusan igazítja a fájlméreteket az IoConfig.PAGE_SIZE-hoz (4KB).
- Erőforrás nyomkövetés: Nyomon követi a last_read puffert a memória életciklus kezeléséhez a szekvenciális olvasások során.

DurableWriter és atomi műveletek

Az adatok integritásának biztosítása érdekében a JAIDE DurableWriter-t alkalmaz, amely egy std.io.BufferedWriter-t burkol annak biztosítására, hogy az írások ki legyenek ürítve és szinkronizálva legyenek a fizikai médiával. Az atomicWrite függvény "írás-majd-átnevezés" mintát biztosít: az adatokat egy ideiglenes fájlba írja (.tmp utótaggal), és a std.fs.Dir.rename-t használja a célfájl cseréjéhez csak sikeres kiürítés után, megakadályozva az adatsérülést áramkimaradás vagy összeomlás esetén.

Pufferelt I/O

- BufferedReader: Egy std.io.BufferedReader burkoló alapértelmezett 8KB BUFFER_SIZE-zal a hatékony szekvenciális olvasáshoz.
- BufferedWriter: Egy kísérő az írásokhoz, biztosítva, hogy a kis írási műveletek kötegbe kerüljenek a fájlrendszer elérése előtt.

JAIDE40 bináris modell formátum

A JAIDE40 formátum az összes modell komponens egységes tárolója, beleértve az RSF neurális vermet, a Ranker-t, a Tokenizálót (MGT) és a Tanult Beágyazásokat.

Fájlstruktúra

Egy JAIDE40 fájl fejlécből, JSON metaadat blokkból és szerializált komponensek sorozatából áll, amelyek mindegyikét SHA-256 ellenőrző összeg védi.

| Eltolás | Komponens | Típus | Leírás |
| :--- | :--- | :--- | :--- |
| 0 | Mágikus fejléc | [8]u8 | JAIDE40\0 konstans |
| 8 | Verzió | u32 | Formátum verzió (Jelenlegi: 1) |
| 12 | Metaadat hossz | u32 | A JSON metaadat blokk hossza |
| 16 | Metaadat | JSON | Modell neve, dimenziók és rétegszámok |
| ... | Komponensek | Bináris | Szerializált RSF, Ranker, MGT és Beágyazások |
| EOF - 32 | Ellenőrző összeg | [32]u8 | SHA-256 hash az összes megelőző adatról |

Szerializációs logika

A ModelFormat.save függvény orchestrálja a szerializálást:

1. Fejléc: Írja a mágikus bájtokat és a verziót.
2. Metaadat: Szerializálja a ModelMetadata struktúrát JSON-ba, beleértve az rsf_layers és mgt_vocab_size paramétereket.
3. Komponens blokkok: Minden komponens (RSF, Ranker, MGT, Beágyazás) hossz-előtagolt blobként kerül írásra.
4. Integritás: A teljes adatfolyam egy Sha256 hashelőn megy keresztül az írás során a végső lábléc ellenőrző összeg generálásához.

Komponens formátumok

- LearnedEmbedding (JEMB): 0x4A454D42 mágikus számot használ. Tárolja a vocab_size-t, dim-et és a nyers f32 súlyokat.
- RSF: Az RSF.save-en keresztül szerializálva, amely végigiterál a rétegeken és menti a súly tenzorokat.

NSIR gráf perzisztencia

A SelfSimilarRelationalGraph (NSIR) speciális perzisztencia mechanizmust igényel a gráf topológia és a kvantum állapot adatok kezeléséhez.

Csomópont és él szerializáció

A gráf a belső nodes és edges gyűjteményeken való iterálással kerül tárolásra.

- Csomópontok: Minden csomópont tárolja a Qubit állapotát és a fractal_dimension-t.
- Élek: Az élek tartalmazzák az EdgeQuality-t (pl. entangled, coherent, fractal) és a súly tenzorokat.

Determinisztikus hashelés

Az NSIR gráf computeTopologyHash függvényt alkalmaz, amely SHA-256 kivonatot generál a gráf struktúrájából. Ez a hash annak ellenőrzésére szolgál, hogy a lemezről betöltött relációs állapot megfelel-e az érvelési orchestrátor által várt konfigurációnak. A hash első két bájtját a VPU quantumVectorOps hívás használja a θ és φ szögek deriválására a gráf-beágyazás vektorokra.

Integráció a ModelFormat-tal

Míg a neurális súlyok a JAIDE40 tárolóban vannak tárolva, az NSIR gráf exportálható tenzorként a modell fájlba való felvételhez, vagy külön tárolható a következtetés során végzett dinamikus gráf frissítésekhez. A tréner checkpoint fájlja emellett tartalmazza a saveCheckpoint útján a csomópontok Qubit állapotát, az élek EdgeQuality-jét és súlyait a checkpoint_version=7 formátumban.

---

3 NEURÁLIS FELDOLGOZÓ RÉTEG (RSF)

A Neurális Feldolgozó Réteg a JAIDE elsődleges számítási motorja, amely felelős a nagy dimenziós vektor transzformációkért és a jellemzőkivonásért. A Visszafordítható Szórt Folyam (RSF) architektúrára épül, amely bijektív neurális hálózati paradigma, amely biztosítja az információ megőrzését és lehetővé teszi a hatékony memóriakezelést a tanítás során.

Cél és hatókör

Az RSF réteg hídként működik a nyers bemeneti beágyazások és a Mag Relációs Réteg között. A hagyományos disszipáló neurális hálózatokkal ellentétben az RSF visszafordítható csatoló rétegeket alkalmaz, lehetővé téve a bemenetek pontos rekonstrukcióját a kimenetekből. Ez a tulajdonság kritikus a rendszer "kvantum-relációs" megismeréséhez, ahol az állapotátmenetek integritásának megőrzése kiemelkedő fontosságú.

Az RSF verem három elsődleges alkomponensből áll:

1. RSF Modell Tároló: Több transzformációs réteg orchestrálását kezeli.
2. OFTB (Ortogonális Fraktál Transzformációs Blokk): Nagy entrópiájú keverést biztosít az osztott adatútvonalak között.
3. SFD Optimalizáló: Adaptív optimalizáló, amely kifejezetten a visszafordítható folyamok spektrális tulajdonságaira van hangolva; GradientFlowController-rel egészül ki a beágyazás gradiensek normavágásához.

Mag komponensek

Visszafordítható Szórt Folyam Processzor (RSF)

Az RSF modell tároló LayerCore példányok vermét kezeli. Minden réteg affin csatolási mechanizmust valósít meg, ahol a bemenet két félre osztódik. Az egyik fél változatlan marad, miközben paraméterezte a másik fél transzformációját (skála S és fordítás T).

- Szálbiztonság: Thread.RwLock-on keresztül kezelve a LayerCore-ban.
- Szerializáció: v4 bináris formátumot használ CRC32 integritás ellenőrzésekkel.
- GPU gyorsítás: A súlyok az accel interfészen keresztül szinkronizálódnak a hardver gyorsítókkal.

Ortogonális Fraktál Transzformációs Blokk (OFTB)

Az OFTB "pillangó" stílusú keverési transzformációt biztosít. Biztosítja, hogy az osztott tenzor mindkét feléből származó információ diffundáljon a következő csatoló réteg előtt. Rögzített FRACTAL_SCALE-t használ, amely körülbelül 0.7071, az egységvariancia fenntartásához.

- Teljesítmény: SIMD-vektorizált forwardInPlace és backwardInPlace rutinokat valósít meg.
- Invertálhatóság: A transzformáció tökéletesen visszafordítható, lehetővé téve a backwardInPlace függvény számára az eredeti bemenet visszanyerését a gradiens számításhoz.

Tokenizálás és beágyazások

Mielőtt belépne az RSF verembe, az adatokat a Multi-Gram Tokenizáló (MGT) dolgozza fel és a LearnedEmbedding segítségével folytonos térbe képezi le.

- MGT: Morfológiai dekompozíciót és szódarab tartalékot kezel.
- LearnedEmbedding: Nagy sebességű kereséseket végez és SGD-t kezel impulzussal a beágyazás frissítésekhez.

SFD Optimalizáló és GradientFlowController

A Spektrális Fisher Diagonalizáló (SFD) a speciális optimalizáló, amelyet az RSF verem tanítására használnak. Tartalmazza:

- SophiaSOAP: Másodrendű optimalizálás K-FAC előkondicionálással.
- Vegyes pontosság: FP4-től FP32-ig terjedő tanítás támogatása a B200 TMEM hardver kihasználásához.
- GradientFlowController: initWithConfig(GradientFlowConfig) útján inicializálható, amely rendelkezik gradient_clip_norm, use_normalized_gradient_flow és spectral_power_iterations mezőkkel. A trainer propagateEmbeddingGradients útvonalán ez a controller vezérli a manuális L2-norma vágást a beágyazás gradiens vektorokra.

Adatfolyam összefoglalója

| Fázis | Entitás | Művelet | Fájl hivatkozás |
| :--- | :--- | :--- | :--- |
| Bemenet | MGT | Morfológiai dekompozíció | src/processor/rsf.zig |
| Keverés | OFTB | SIMD pillangó transzformáció | src/processor/oftb.zig |
| Csatolás | LayerCore | Affin skála/eltolás (S, T) homogén koordinátákban | src/processor/rsf.zig |
| Optimalizálás | GradientFlowController | Normavágás + normalizált gradiens folyam | src/optimizer/sfd.zig |
| Tárolás | SAVE_VERSION | CRC32-validált v5 I/O | src/processor/rsf.zig |

---

3.1 RSF: VISSZAFORDÍTHATÓ SZÓRT FOLYAM PROCESSZOR

A Visszafordítható Szórt Folyam (RSF) processzor a JAIDE architektúra elsődleges neurális transzformációs motorja. Bijektív neurális vermet valósít meg affin csatoló rétegeken alapulva, biztosítva, hogy a hálózaton átmenő minden előre irányuló menetnek matematikailag pontos inverze legyen. Ez a tulajdonság lehetővé teszi az O(1) memória komplexitást a mélységhez képest a visszaterjesztés során, mivel a közbenső aktivációk rekonstruálódnak ahelyett, hogy tárolnák őket.

1. RSFLayer: Affin csatolás és Exp-vágás

Az RSFLayer az RSF verem alapvető építőköve. A bemeneti tenzort két félre osztva, az egyik félre nem-lineáris transzformációt alkalmazva a másik feltételezésével, majd az OFTB-n keresztül keverve őket működik.

Affin csatolási mechanizmus

A réteg a következő transzformációt valósítja meg:

1. Osztás: Az x bemenet x1-re és x2-re osztódik.
2. Skála és fordítás: Az x2 transzformálódik s = exp(clip(x1 Ws + bs)) és t = x1 Wt + bt segítségével.
3. Kombinálás: y2 = x2 ⊙ s + t, míg y1 = x1 változatlan marad.
4. Keverés: A kimenetek az OFTB.forwardInPlace-en keresztül mennek a keresztdimenziós információáramlás biztosításához.

A LayerCore struktúra kezeli a súlymátrixokat (Ws, Wt) ezekhez a transzformációkhoz. Az eltolások (bias) homogén koordináták révén be vannak olvasztva a súlymátrixokba: minden súlymátrix alakja [dim × (dim+1)], ahol az utolsó oszlop tárolja az abszorbeált eltolást. Így nincsenek külön eltolás-tenzorok, a rétegenkénti paraméterszám változatlan (dim² + dim = dim × (dim+1)). Exp-vágást alkalmaz (clip_min és clip_max által meghatározva) a numerikus instabilitás megelőzéséhez az exponenciális skálázási tényezőben.

| Komponens | Kód entitás | Leírás |
| :--- | :--- | :--- |
| Skála súlyok | s_weight | Ws tenzor a skálázási komponenshez. |
| Fordítás súlyok | t_weight | Wt tenzor a fordítási komponenshez. |
| Vágási tartomány | clip_min/clip_max | A log-skála kimenet határai az inf értékek megelőzéséhez. |
| Szálbiztonság | rwlock | std.Thread.RwLock a szinkronizált súly frissítésekhez. |

2. RSF Modell Tároló és Orchestráció

Az RSF struktúra RSFLayer példányok sorozatának tárolójaként szolgál. Orchestrálja az előre, inverz és visszafelé irányuló meneteket a teljes vermen keresztül.

Végrehajtási folyam

- Előre irányuló menet: Végigiterál a 0...N rétegeken, affin csatolást és OFTB keverést alkalmazva.
- Inverz menet: Végigiterál az N...0 rétegeken fordítva, OFTB.backwardInPlace-t alkalmazva, majd az inverz affin transzformációt: x2 = (y2 - t) ⊙ exp(-s).
- Visszafelé irányuló menet: A visszafordíthatóságot kihasználva számítja a gradienseket az aktivációk tárolása nélkül. Rekonstruálja minden réteg bemenetét az inverz menet segítségével a gradiens számítási fázis során.

3. Handle/Core Regiszter és Szálbiztonság

A nagy párhuzamosságú következtetés és tanítás támogatásához az RSF handle-alapú regiszter rendszert valósít meg. A LayerCore tartalmazza a tényleges Tensor adatokat és egy std.Thread.RwLock-ot.

- Súly szinkronizálás: GPU-n futtatáskor a súlyok az RSFAccelerator interfészen keresztül szinkronizálódnak.
- Párhuzamos hozzáférés: Az rwlock lehetővé teszi több szál számára az előre irányuló menetek végrehajtását (olvasási zár), miközben blokkolja az SFD optimalizáló frissítéseit (írási zár) számára.

- Memóriabiztonság: A LayerCore dedikált Allocator-t használ és támogatja az initOwned-et az explicit életciklus-kezeléshez.

4. Bináris szerializáció (v4) és CRC32

Az RSF rendszer robusztus bináris formátumot alkalmaz a modell perzisztenciájához, amelyet a SAVE_VERSION = 5 azonosít. A szerializáció biztosítja az adatok integritását különböző hardver architektúrákon.

Szerializációs elrendezés

A formátum szigorú sorrendet követ:

1. Fejléc: Mágikus bájtok és SAVE_VERSION.
2. Metaadat: dim, num_layers, clip_min, clip_max.
3. Réteg adatok: Minden réteghez az s_weight és t_weight tenzorok kerülnek írásra, mindkettő [dim × (dim+1)] alakban, ahol az utolsó oszlopok az abszorbeált eltolásokat tartalmazzák.
4. Integritás: CRC32 lábléc validálja a teljes fájlt a betöltéskor.

5. SFD optimalizáló csatolás

Az SFD-vel való integráció fúzionált kerneleket alkalmaz:

1. TMEM elrendezés: A statisztikák (m, v) csempézve vannak a helyi 128KB TMEM bankokba való illeszkedéshez.
2. Fúzionált kernelek: A Fisher frissítés és a paraméter kivonás egyetlen kernelbe van fúzionálva az HBM-be való visszautazások minimalizálásához.

Segédprogramok

A modul számos matematikai primitívet biztosít a sztochasztikus becsléshez:

- fillRademacher: Tenzort tölt fel {-1, 1} értékekkel a Hutchinson nyom becsléshez.
- fillRandomNormal: Box-Muller transzformációt alkalmaz Gauss zaj generálásához.
- erfApprox: A hibafüggvény gyors numerikus közelítése a valószínűségi modellezéshez.

---

4 MAG RELÁCIÓS RÉTEG

A Mag Relációs Réteg a JAIDE rendszer kognitív motorját képviseli. Míg a Neurális Feldolgozó Réteg (RSF) nagy dimenziós vektor transzformációkat kezel, a Relációs Réteg strukturált, szimbolikus és kvantum-inspirált keretrendszert biztosít az érveléshez, az oksági ellenőrzéshez és a hosszú távú memória integrációhoz. A neurális aktivációkat explicit relációs gráf struktúrába képezi le, lehetővé téve az O(d) szelektív figyelmet és a determinisztikus logikai végrehajtást.

Kognitív architektúra áttekintés

Az alrendszer áthidalja a nyers neurális tenzorok és a szimbolikus logika közötti szakadékot egy hierarchikus érvelési verem segítségével, amelyet a ReasoningOrchestrator kezel. Ez az orchestrátor koordinálja a jelfolyamot a gráf alapú memória (NSIR), az oksági validációs folyamat (CREV), a vektorprocesszor (VPU), a fraktál dinamika (FNDS), a formális invariáns ellenőrzés (FormalVerificationEngine), a biztonsági bizonyítások (SecurityProofEngine), a kvantum feladat orchestráció (QuantumTaskAdapter) és a relációs futtatókörnyezet (ZRuntime) között.

NSIR: Önhasonló Relációs Gráf

A SelfSimilarRelationalGraph (SSRG), vagyis az NSIR, az elsődleges adatstruktúra a tokenek közötti kapcsolatok tárolásához. A szabványos figyelemmechanizmusokkal ellentétben, amelyek O(n^2 * d) skálázódnak, az NSIR gráf ritka, szelektív reprezentációt tart fenn, ahol az élek kvantum tulajdonságokkal rendelkeznek, mint a szuperpozíció, összefonódott és fraktál. Kvantum kapu alkalmazásokat (Hadamard, CNOT) közvetlenül a gráf csomópontokon támogat a komplex valószínűségi függőségek szimulálásához.

ReasoningOrchestrator és ESSO

A ReasoningOrchestrator háromszintű érvelési hierarchiát valósít meg: helyi, globális és meta. Az EntangledStochasticSymmetryOptimizer-t (ESSO) alkalmazza a gráf topológia finomításához. Az ESSO szimulált hűtést és szimmetria alapú perturbációkat alkalmaz a relációs állapot energiájának minimalizálásához, biztosítva a legkoherensebb logikai struktúra fenntartását a következtetés során.

ChaosCoreKernel és CAS

A ChaosCoreKernel végrehajtási környezetet biztosít a nemlineáris dinamikához és a kaotikus perturbációkhoz, amelyeket a CREV folyamatban alkalmaznak. Integrálódik a ContentAddressableStorage-val (CAS) a hatékony adatdeduplikációhoz és a MemoryBlock állapotkezeléshez, amely nyomon követi, hogy a memória szabad, allokált vagy összefonódott-e.

CREV folyamat és ZRuntime

A CREVPipeline (Oksági Érvelés és Ellenőrzés) RelationalTriplet struktúrákat (alany-állítmány-tárgy) von ki a neurális adatokból és validálja azokat a meglévő tudással szemben. Ezeket a validált műveleteket a ZRuntime hajtja végre, egy relációs végrehajtó motor, amely a változókat kvantum-összekapcsolt entitásokként (ZVariable) kezeli és minden műveletet determinisztikus ExecutionHistoryEntry-ben rögzít. A CREVPipeline.processTextStream közvetlenül meghívódik minden tanítási buildKnowledgeGraph hívásban és minden inferencia handleInference/handleBatchInference kérésben.

Jelterjedés és FNDS

Az információ az NSIR gráfon keresztül a SignalPropagationEngine segítségével utazik, amely aktivációs hullámokat és gráf konvolúciókat szimulál. A trainerben a signal_engine nem opcionális mező, minden runCoreRelationalPass ciklusban propagateStep meghívódik. A hierarchikus adatszervezést az FNDSManager (Fraktál Neurális Dinamikus Rendszer) kezeli, amely FractalTree-t alkalmaz az önhasonló struktúrák fenntartásához az absztrakció különböző skáláin. Az FNDSManager minden tanítási lépésben createTree(6, 4)-et fut a tokenlistákhoz és createTree(4, 3)-at a buildKnowledgeGraph útvonalon; minden inferencia kérésben createTree(4, 3)-at fut az input tenzor bájtjaira és mintaindexet készít inference_patterns néven.

VPU: Vektorprocesszor

A VPU (VectorProcessingUnit) SIMD vektor típusokat (F32x4, F32x8, F64x2, F64x4, I32x4, I32x8) és kvantum-inspirált gráf beágyazási műveleteket biztosít. A trainer runCoreRelationalPass útvonalán és az inferencia handleInference/handleBatchInference útvonalán:

1. computeGraphEmbeddings(graph) F64x4 vektorokat állít elő a gráf csomópontokból.
2. quantumVectorOps(vectors, theta, phi) kvantum operátorokat alkalmaz a vektorokra, ahol a szögek a tanításnál a topology_hash első két bájtjából, az inferenciánál a request_count % 314 és % 628 értékből származnak.
3. computeSimilarityMatrix(vectors) hasonlósági mátrixot számít, amelynek első cellája a tanításnál közvetlenül modulálja a learning_rate-et (felső határ 0.1).

FormalVerificationEngine

A FormalVerificationEngine invariáns nyilvántartást, Hoare-logika ellenőrzést és tétel bizonyítást biztosít. Az inferencia útvonalán a verifyGraph(graph) hívódik meg, amely strukturális invariánsokat ellenőriz az NSIR gráfon. Az inferencia szerverben egy heap-allokált *FormalVerificationEngine mező tárolódik, amelyet a loadModel inicializál és a deinit szabadít fel.

SecurityProofEngine

A SecurityProofEngine Bell-LaPadula, Biba, hozzáférés vezérlés és nem-interferencia biztonsági bizonyításokat biztosít. Az inferencia útvonalán a proveInformationFlowSecurity(graph) hívódik meg, amely az NSIR gráf éleinek információáramlási biztonságát validálja. Az inferencia szerverben heap-allokált *SecurityProofEngine mező tárolódik.

QuantumTaskAdapter

A QuantumTaskAdapter kvantum szubgráfokat azonosít az NSIR gráfban, majd azokat a lokális állapotvektor szimulátorra (32 qubit határ) vagy IBM hardverre irányítja. Az inferencia útvonalán háromfázisú ciklust futtat:

1. identifyQuantumSubgraphs() — ArrayList(QuantumSubgraph)-ot ad vissza.
2. Minden szubgráfra executeQuantumTask(subgraph) — QuantumTaskResult-ot ad vissza success mezővel.
3. Ha task_result.success igaz, applyResultsToGraph(subgraph, &task_result) frissíti a gráfot.

Meglepetés memória és Temporális gráf

A SurpriseMemoryManager online tanulást valósít meg azáltal, hogy azonosítja a magas "meglepetés" értékű tokeneket (Jaccard-disszimilaritás segítségével) és hosszú távú tárolóba rögzíti azokat. Ezeket a változásokat idővel a TemporalGraph követi nyomon, amely NodeVersion és EdgeVersion pillanatképeket tart fenn, lehetővé téve a rendszer számára, hogy bármely nanoszekundum időbélyegnél lekérdezze tudásának állapotát.

Rendszer integrációs térkép

A következő leírás bemutatja, hogyan lépnek kölcsönhatásba a mag relációs komponensek az alapul szolgáló hardverrel és a neurális veremmel.

Az RSF neurális verem (rsf.zig:LayerCore) a relációs feldolgozáshoz (CPU/R-GPU/VPU) csatlakozik, ahol a ReasoningOrchestrator, ZRuntime, SelfSimilarRelationalGraph, VPU, FNDSManager, CREVPipeline, FormalVerificationEngine, SecurityProofEngine, QuantumTaskAdapter, SignalPropagationEngine és RelationalGraphProcessingUnit (r_gpu.zig) találhatók. A SelfSimilarRelationalGraph a TemporalGraph-hoz és a ContentAddressableStorage-hoz kapcsolódik a perzisztencia és memória területén.

---

4.1 NSIR: ÖNHASONLÓ RELÁCIÓS GRÁF

Az Önhasonló Relációs Gráf (SSRG), amelyet az NSIR (Non-Sequential Information Retrieval) keretrendszeren belül valósítanak meg, a JAIDE rendszer elsődleges kognitív adatstruktúrájaként szolgál. Áthidalja a diszkrét szimbolikus relációk és a folytonos kvantum-valószínűségi állapotok közötti szakadékot, lehetővé téve a rendszer számára, hogy komplex, fraktál kapcsolatokat képviseljen, amelyek idővel fejlődnek.

Mag adatprimitívek

1. A Qubit primitív

A gráf minden csomópontja tartalmaz egy Qubit struktúrát, amely a kvantum állapotát képviseli a számítási bázisban. std.math.Complex(f64)-et alkalmaz a nagy pontosságú valószínűségi amplitúdókhoz.

- Inicializálás: A Qubitek |0> vagy |1> bázisra inicializálódnak, vagy specifikus amplitúdókon keresztül, amelyek automatikusan normalizálódnak.
- Normalizálás: A normalizeInPlace függvény biztosítja, hogy a négyzetes norma <psi|psi> = 1.0 legyen. Ha a norma NaN vagy végtelen, alapértelmezés szerint |0> bázisra áll vissza.
- Mérési valószínűség: A prob0() és prob1() kiszámítja az egyik bázisállapotra való összeomlás valószínűségét.

2. EdgeQuality enum

A csomópontok közötti kapcsolatot az EdgeQuality enum minősíti, amely meghatározza, hogyan terjednek a jelek a gráfon keresztül:

| Enum érték | Leírás |
| :--- | :--- |
| superposition | A kapcsolat több potenciális állapotban létezik. |
| entangled | A célcsomópont állapota a forráscsomóponttól függ. |
| coherent | Stabil, fázis-igazított kapcsolat. |
| collapsed | Meghatározott, klasszikus kapcsolat. |
| fractal | Önhasonló kapcsolat, amely skálákon ismétlődik. |

Adatstruktúra implementáció

Csomópont és él életciklus

A Node és Edge struktúrák saját memóriájukat egy megadott std.mem.Allocator segítségével kezelik.

- Csomópont: Egyedi azonosítót, nyers adatbájtokat, Qubit-et, fázist (f64) és StringHashMap-et tartalmaz tetszőleges metaadatokhoz.
- Él: Forrás és cél azonosítót köt össze. Tartalmaz quantum_correlation-t (Complex f64) és fractal_dimension-t (f64) a gráf bejárás és energia számítások befolyásolásához.

SelfSimilarRelationalGraph

A fő tároló SelfSimilarRelationalGraph szálbiztos környezetet biztosít a gráf manipulációhoz std.Thread.Mutex segítségével.

| Függvény | Cél |
| :--- | :--- |
| addNode | Létrehoz és regisztrál egy új csomópontot. |
| addEdge | Összeköt két meglévő csomópontot; error.NodeNotFound-ot ad vissza, ha az azonosítók hiányoznak. |
| applyHadamard | Szuperpozícióba helyezi a csomópont Qubit-jét. |
| entangleNodes | Beállítja az EdgeQuality.entangled-et és frissíti a quantum_correlation-t. |
| encodeInformation | Bájtsorozatot rögzít a gráfban és frissíti a topology_hash-t. |

Fejlett gráf műveletek

Determinisztikus topológia hashelés

A gráf integritásának ellenőrzéséhez és a Content-Addressable Storage (CAS) támogatásához a gráf calculateTopologyHash-t valósít meg. Ez a függvény SHA-256 hash-t generál a teljes gráf struktúrájából.

1. Végigiterál az összes csomóponton, azonosító szerint rendezve a determinizmus biztosításához.
2. Hash-eli a csomópont azonosítókat és aktuális Qubit amplitúdóikat.
3. Végigiterál az összes élen, hash-elve a forrás/cél párokat és súlyokat.

A topology_hash első bájtja (hash[0]) és második bájtja (hash[1]) közvetlenül a VPU quantumVectorOps hívás θ és φ szögeit deriválja: theta = hash[0]/255.0 · π és phi = hash[1]/255.0 · π a runCoreRelationalPass útvonalon.

Tenzor export/import

Az SSRG integrálódik a core_tensor rendszerrel, lehetővé téve a gráf állapot feldolgozását neurális rétegek (RSF) által.

- Export: Az exportToTensor szerializálja a csomópont fázisokat és qubit valószínűségeket egy core_tensor.Tensor objektumba.
- Import: Az importFromTensor frissíti a gráf belső állapotait a neurális feldolgozás kimenete alapján, megkönnyítve a neurális-relációs hidat.

Memória és allokátor integráció

Az SSRG nagy teljesítményű környezetekre van tervezve és integrálódik a core_memory allokátorokkal. Kifejezetten StringHashMap-et alkalmaz az O(1) csomópont kereséshez azonosító alapján.

Amikor a deinit() meghívódik a gráfon, mély tisztítást végez:

1. Végigiterál a csomópont térképen, meghívva a deinit()-et minden csomóponton a metaadatok és azonosító karakterláncok felszabadításához.
2. Végigiterál az él listán, felszabadítva az allokált forrás/cél karakterláncokat.
3. Törli az összes belső ArrayList és HashMap struktúrát.

---

4.2 REASONINGORCHESTRATOR ÉS ESSO

A ReasoningOrchestrator a JAIDE Mag Relációs Réteg központi végrehajtója, amely felelős a SelfSimilarRelationalGraph (NSIR) alacsony energiájú állapot felé való hajtásáért hierarchikus érvelésen keresztül. Integrálja az Entangled Stochastic Symmetry Optimizer-t (ESSO) a strukturális invariánsok észleléséhez és a ChaosCoreKernel-t alkalmazza az állapot relaxációhoz. Ez a rendszer áthidalja a diszkrét relációs logika és a folytonos neurális moduláció közötti szakadékot.

Hierarchikus érvelési rendszer

Az orchestrátor három különböző hierarchikus szinten működik, amelyeket a ThoughtLevel enum definiál. Minden szint a gráf topológia különböző granularitásait célozza:

| Szint | Hatókör | Cél |
| :--- | :--- | :--- |
| local | Csomópont-szomszédságok | Azonnali relációs konzisztencia és helyi qubit igazítás. |
| global | Teljes gráf topológia | Nagy léptékű kapcsolódási minták és klaszter képzés. |
| meta | Érvelési előzmények | Magának az érvelési folyamatnak az értékelése és minták újraalkalmazása. |

Érvelési fázis életciklus

Minden érvelési munkamenet ReasoningPhase blokkokra van osztva. Egy fázis beágyazott ciklusokon (inner_iterations és outer_iterations) keresztül hajtódik végre, amíg el nem éri a target_energy-t vagy a rendszer nem teljesíti a hasConverged kritériumokat. A trainer runCoreRelationalPass és a szerver handleInference minden ciklusa runHierarchicalReasoning(50)-et hív, ami 50 belső iterációt jelent.

Energia számítás

A rendszer "haladását" egy többkomponensű energia függvény méri. Az orchestrátor megpróbálja minimalizálni ezt az értéket a gráf leglogikusabb vagy legstabilabb konfigurációjának megtalálásához.

1. Strukturális energia: Az él súlyokból és a gráf topológiából származtatva.
2. Kvantum energia: Méri a csomópontokon belüli qubitek koherenciáját és összefonódási entrópiáját.
3. Fázis energia: Egy temporális komponens, amely nyomon követi az aktuális érvelési pálya stabilitását.

Az energia frissítések az updateEnergy segítségével kerülnek rögzítésre, amely előzményt tart fenn a konvergencia delták kiszámításához.

ESSO: Összefonódott Sztochasztikus Szimmetria Optimalizáló

Az EntangledStochasticSymmetryOptimizer (ESSO) az orchestrátor által használt elsődleges optimalizálási motor. Szimmetriákat (tükrözések, rotációk, eltolások) azonosít a gráfon belül az információ tömörítéséhez és a konvergencia gyorsításához.

Szimmetria észlelés

Az ESSO SymmetryGroup-ot alkalmaz a minták kategorizálásához:

- Rotációs: rotation_90, rotation_180, rotation_270 és custom_rotation.
- Tükrözési: Tengelyen való tükrözés, amelyet a SymmetryTransform definiál.
- Eltolási: Eltolás-invariancia a relációs téren.

Optimalizálási ciklus

Az ESSO sztochasztikus keresést végez az optimális SymmetryTransform paraméterekért. Transzformációkat alkalmaz a csomópont koordinátákra és qubit állapotokra, mérve a "Szimmetria Hibát". Ha magas fokú szimmetria kerül megtalálásra (pl. 4-es rendű rotáció), az orchestrátor ezt a gráf "újraegyensúlyozásához" használja, hatékonyan propagálva a frissítéseket az egyik csomópontból az összes szimmetrikus párjára.

Állapotkezelés: Pillanatkép és visszagörgetés

A nemlineáris optimalizálás "káoszának" kezeléséhez a ReasoningOrchestrator robusztus pillanatkép mechanizmust valósít meg.

- Pillanatkép: A nagy entrópiájú műveletek (mint a chaosRelaxation) előtt az orchestrátor klónozza az aktuális SelfSimilarRelationalGraph-ot és a hozzá tartozó QuantumState-et.
- Visszagörgetés: Ha egy érvelési fázis energia divergenciához vezet (a veszteség/energia tájban "robbanás"), az orchestrátor visszagörgetést indít az utolsó ismert stabil pillanatképre.
- Fraktál újraegyensúlyozás: Ha a gráf túl ritkává vagy túl sűrűvé válik, az orchestrátor újraegyensúlyozási menetet indít a FractalTree (FNDS) segítségével az O(log N) keresési komplexitás fenntartásához.

Integráció: Relációs tér a neurális térbe

A ReasoningOrchestrator végső kimenete Modulációs Tényezők halmaza. Ezek lebegőpontos tenzorok, amelyek a végső gráf energiából és szimmetria sűrűségből származnak. Ezek a tényezők visszakerülnek az RSF (Visszafordítható Szórt Folyam) rétegekhez a neurális súlyok modulálásához a következő következtetési menetben. Emellett a VPU computeSimilarityMatrix első cellája (koherencia) közvetlenül modulálja a learning_rate-et a trainerben.

Főbb implementációs részletek

Orchestrátor statisztikák

A rendszer saját teljesítményét OrchestratorStatistics segítségével követi nyomon:

- total_inner_loops: Összes iteráció az összes fázison.
- best_energy_achieved: A munkamenet során talált globális minimális energia.
- patterns_discovered: Rögzített egyedi SymmetryPattern azonosítók száma.

Konvergencia logika

A konvergenciát az energia relatív változása határozza meg:
delta = |aktuális - előző| / max(|előző|, 1.0)

Ha delta < convergence_threshold, a fázis leáll.

---

4.3 CHAOSCOREKERNEL ÉS TARTALOM-CÍMEZHETŐ TÁROLÁS

A ChaosCoreKernel a JAIDE mag relációs réteg nagy teljesítményű futtatókörnyezeti motorja. Kezeli a relációs gráfok végrehajtását egy elosztott memória modell orchestrálásával, amely Tartalom-Címezhető Tárolásra (CAS), állapotgép-vezérelt memória életciklusra és dinamikus feladatütemezőre épül, amely az adat-mag affinitásra optimalizál.

Tartalom-Címezhető Tárolás (CAS)

A ChaosCoreKernel Tartalom-Címezhető Tárolási mechanizmust alkalmaz az adatdeduplikáció és integritás biztosításához a relációs gráfon. Minden adatdarab egy MemoryBlock-ban tárolódik, amelyet a tartalom hash-e azonosít, nem egy illékony memória cím.

Implementációs részletek:

- Blokk azonosítás: A blokkok 16 bájtos block_id és 16 bájtos content_hash segítségével azonosítódnak.
- Deduplikáció: A ContentAddressableStorage struktúra content_hash-ből block_id-be való leképezést tart fenn. Új memória allokálása előtt a kernel ellenőrzi, hogy a hash már létezik-e a meglévő MemoryBlock újrafelhasználásához.
- Tárolási térkép: Az elsődleges tárolást std.HashMap kezeli egyedi BlockIdContext segítségével a MemoryBlock objektumok hatékony kereséséhez.

MemoryBlock állapotgép

A ChaosCoreKernel memóriája nem csupán "allokált" vagy "szabad". A MemoryBlockState enum által definiált állapotgépet követi a kvantum-relációs funkciók, mint az összefonódás és a hardver szintű migráció támogatásához.

| Állapot | Leírás |
| :--- | :--- |
| free | A blokk visszanyerésre elérhető. |
| allocated | Szabványos aktív memória blokk, amely érvényes adatokat tartalmaz. |
| entangled | A blokk logikailag más blokkokhoz van kapcsolva; a változások propagálódhatnak. |
| migrating | A blokk jelenleg feldolgozó magok között mozog az affinitás optimalizálásához. |

Minden MemoryBlock saját metaadatait követi nyomon a DataFlowAnalyzer és DynamicTaskScheduler segítésére:

- Affinitás: Az affinity_core tárolja annak a magnak az azonosítóját, ahol az adatokhoz leggyakrabban hozzáférnek.
- Hozzáférés követés: Az access_count és last_access_time minden olvasás/íráskor frissül az LRU kiürítési és migrációs logika tájékoztatásához.
- Összefonódás: Egy BlockIdSet nyomon követi az ezzel összefonódott más blokkok azonosítóit, megkönnyítve a relációs propagációt.

Dinamikus feladatütemezés és affinitás

A DynamicTaskScheduler TaskDescriptor objektumok prioritási sorát kezeli. Egyensúlyozza a számítási terhelést a magok között, miközben minimalizálja az adatmozgást az "adat-mag affinitás" tiszteletben tartásával.

Feladat végrehajtási logika:

1. Prioritási sor: A feladatok ArrayList-ben tárolódnak és priority és inference_priority szerint rendezve.
2. Affinitás leképezés: A DataFlowAnalyzer nyomon követi, hogy melyik magok melyik block_id-hez férnek hozzá.
3. Migráció: Ha a ChaosCoreKernel terhelési egyensúlyhiányt észlel (meghaladva a LOAD_HIGH_THRESHOLD-ot), rebalanceLoad()-ot indít, amely frissíti a blokkok affinity_core-ját és feladatokat mozgat az alulhasznált magokra.

Terheléselosztási konstansok:

- OPTIMIZATION_THRESHOLD: (0.6) Minimális nyereség a blokk migráció indításához.
- BALANCE_INTERVAL_CYCLES: (100) A terheléselosztó végrehajtásának gyakorisága.

executeGraphOnKernel interfész

Az executeGraphOnKernel függvény az elsődleges belépési pont komplex NSIR (Önhasonló Relációs Gráf) műveletek futtatásához a kernelen.

Végrehajtási folyamat:

1. Gráf elemzés: A kernel fogad egy SelfSimilarRelationalGraph-ot.
2. Feladat generálás: A csomópontok és élek TaskDescriptor egységekké konvertálódnak.
3. Függőség feloldás: Az adatfüggőségek CAS block_id-kre kerülnek leképezve a data_dependencies.append() segítségével.
4. Párhuzamos végrehajtás: A DynamicTaskScheduler feladatokat küld a RelationalGraphProcessingUnit-hoz (R-GPU) vagy helyi CPU szálakhoz a ChaosCoreConfig alapján.

---

4.4 CREV FOLYAMAT ÉS ZRUNTIME

A CREV (Oksági Érvelés és Ellenőrzés) folyamat és a ZRuntime végrehajtási motor alkotják a JAIDE rendszer mag kognitív feldolgozási rétegét. Míg a Neurális Feldolgozó Réteg (RSF) nagy dimenziós vektor transzformációkat kezel, a CREV/ZRuntime verem diszkrét relációs kivonást, oksági validációt és kvantum-relációs változó végrehajtást kezel.

1. CREV Folyamat

A CREV folyamat felelős a strukturálatlan természetes nyelv strukturált relációs hármasokká való átalakításáért és oksági konzisztenciájuk ellenőrzéséért a SelfSimilarRelationalGraph-on belül. A processTextStream a folyamat elsődleges belépési pontja, és minden tanítási buildKnowledgeGraph és minden inferencia handleInference/handleBatchInference hívás közvetlenül meghívja.

1.1 Kivonási szakaszok

A folyamat az ExtractionStage enum által definiált diszkrét szakaszok sorozatán keresztül működik:

| Szakasz | Leírás |
| :--- | :--- |
| tokenization | Kezdeti morfológiai és szó szintű szegmentálás. |
| triplet_extraction | Minta alapú Alany-Reláció-Tárgy (SRO) struktúrák azonosítása. |
| validation | Oksági lánc ellenőrzés és megbízhatósági pontozás. |
| integration | Validált hármasok összevonása az NSIR gráfba. |
| indexing | Relációs indexek frissítése a visszakereséshez. |

1.2 Hármas azonosság és hashelés

Az adatok integritásának és deduplikációjának biztosítása érdekében a CREV két hashelési stratégiát alkalmaz:

1. Azonosság hashelés: Sha256-ot alkalmaz a subject, relation és object mezőkön egy relációs tény egyedi azonosítójának generálásához.
2. Mező hashelés: Tartalmazza a confidence-t és extraction_time-ot a kivonás specifikus példányainak nyomon követéséhez.

2. ZRuntime Végrehajtási Motor

A ZRuntime a relációs logika végrehajtási környezete. Kezeli a ZVariable entitások életciklusát, amelyek szimbolikus változókat kvantum-relációs állapotokra képezik le.

2.1 ZVariable életciklus

Egy ZVariable egy SelfSimilarRelationalGraph-ot és egy RelationalQuantumLogic példányt foglal magában.

- Hozzárendelés (assign): Szimbolikus értéket köt a változóhoz, rögzítve azt a HistoryEntry naplóban.
- Reláció (relateTo): Élt hoz létre az aktuális változó és egy célváltozó között a gráfon belül.
- Mérés (measure): Összeomlasztja a változó kvantum állapotát egy diszkrét értékre, a RelationalQuantumLogic motort alkalmazva.

A trainer runCoreRelationalPass minden ciklusban létrehoz egy változót train_<global_step> néven; az inferencia handleInference minden kérésben létrehoz egy inf_<request_count> változót.

2.2 Relációs műveletek és kvantum kapuk

A relációs kifejezések elemzésre kerülnek és közvetlenül kvantum kapu műveletekre képezik le. A ZRuntime számos magas szintű operátort támogat:

| Relációs op | Kvantum leképezés | Implementáció |
| :--- | :--- | :--- |
| AND | Többvezérelt fázis | z_runtime.zig |
| OR | Szuperpozíció / Hadamard | z_runtime.zig |
| XOR | CNOT / Pauli-X | z_runtime.zig |
| ENTANGLE | Bell állapot létrehozás | z_runtime.zig |

2.3 Végrehajtási előzmények és auditálás

A ZRuntime-on belül végrehajtott minden művelet rögzítésre kerül egy ExecutionHistoryEntry-ben. Ez lehetővé teszi az érvelési folyamat teljes auditálhatóságát, beleértve a megcélzott változó, az elvégzett művelet és az eredmény kvantum állapot rögzítését.

3. VPU és FNDS integráció

A VPU (VectorProcessingUnit) és az FNDSManager kiegészíti a CREV/ZRuntime magot. A VPU a gráf topológiából F64x4 vektorokat származtat, kvantum operátorokat alkalmaz rájuk és hasonlósági mátrixot számol. Az FNDSManager fraktál fákat épít a tokenlistákból és mintaindexeket regisztrál PatternLocation objektumokkal.

VPU vektor típusok

| Típus | Leírás |
| :--- | :--- |
| F32x4, F32x8 | Egyszeres pontosságú SIMD vektorok |
| F64x2, F64x4 | Kétszeres pontosságú SIMD vektorok |
| I32x4, I32x8 | Egész SIMD vektorok |

FNDS mag műveletek

| Művelet | Leírás |
| :--- | :--- |
| createTree(max_depth, branching_factor) | Fraktál fát hoz létre és visszaadja a [32]u8 tree_id-t |
| insertIntoTree(tree_id, node_id, data, level) | Adatot szúr be egy fába a megadott szinten, !bool visszatérési értékkel |
| createIndex(index_id) | Új önhasonló indexet hoz létre |
| addPatternToIndex(index_id, pattern, location) | Mintát regisztrál egy indexbe egy PatternLocation-nel |

PatternLocation

A PatternLocation.init(allocator, tree_id, level, node_id, offset, length, confidence) allokálva jön létre; a node_id belső duplikátumként tárolódik. A confidence [0.0, 1.0] tartományba esik különben InvalidConfidence hibát ad. Deinit útján felszabadul.

---

4.5 KVANTUM ALRENDSZER ÉS ELLENŐRIZHETŐ KÖVETKEZTETÉS

A JAIDE rendszer teljes kvantum verméért, beleértve a helyi szimulátort, az IBM Quantum hardver integrációt és a formális ellenőrzést. Ez a réteg biztosítja, hogy a hibrid kvantum-neurális következtetések megbízhatóak és matematikailag helyesen kerültek végrehajtásra az érzékeny modell paraméterek felfedése nélkül.

Kvantum Logika és Hardver Interfész

A JAIDE átfogó kvantum logikai kapu készletet valósít meg, amely túlmutat a szabványos qubiteken és relációs műveleteket is tartalmaz. A LogicGate enum mind a szabványos kapukat (Hadamard, CNOT, Toffoli), mind a JAIDE-specifikus relációs primitíveket definiálja, mint a RELATIONAL_AND és a FRACTAL_TRANSFORM.

A QuantumState struktúra kezeli a komplex amplitúdókat és fázis információkat ezekhez a műveletekhez, segédprogramokat biztosítva a normalizáláshoz és a valószínűség számításhoz. A fizikai végrehajtáshoz az IBMQuantumClient kezeli a QuantumCircuit életciklusát, az OpenQASM 3.0 szerializálástól az IBM hardver családokon (HERON, EAGLE vagy FALCON) való benyújtásig.

Főbb komponensek:

- RelationalQuantumLogic: Kapu alkalmazásokat orchestrál az NSIR gráfon.
- IBMQuantumClient: Kezeli a backend kalibrációs adatokat (T1/T2 idők) és a feladat sorba állítást.
- Hibrid Optimalizáló: Paraméter-eltolás gradienseket alkalmaz a kvantum-klasszikus paraméterek hangolásához.
- QuantumTaskAdapter: identifyQuantumSubgraphs → executeQuantumTask → applyResultsToGraph ciklus, amely minden inferencia handleInference/handleBatchInference kérésben lefut.

Nulla-Tudás Ellenőrzési Rendszer

A hibrid kvantum-neurális következtetések biztonságának és helyességének biztosítása érdekében a JAIDE Nulla-Tudás (ZK) ellenőrzési réteget alkalmaz. Ez a rendszer, amelynek középpontjában a ZKInferenceProver és a VerifiedInferenceEngine áll, Groth16 bizonyítékokat generál a bn128 görbe segítségével.

A rendszer circom eszközláncot alkalmaz egy inference_trace.circom áramkör fordításához, amely validálja a következtetési folyamat Poseidon-láncát. Ez lehetővé teszi a JAIDE számára, hogy bizonyítsa, hogy egy specifikus kimenetet egy specifikus modell és bemenet generált, anélkül, hogy felfedné az alapul szolgáló Tensor súlyokat vagy az NSIR gráf topológiát. A build rendszer -Dzk=true kapcsolóval automatikusan lefordítja az áramkört és végrehajtja a Groth16 megbízható beállítást.

Főbb jellemzők:

- Differenciális Adatvédelem: Laplace/Gauss zaj injektálása az adathalmaz adatvédelmének védelméhez.
- Rögzített Pontos Skálázás: Az InferenceWitness kezeli a JAIDE Fixed32_32 aritmetikája és a ZK-barát prímtestek közötti konverziót.
- Biztonságos Aggregáció: Merkle-fa alapú bizonyíték aggregációt tesz lehetővé a nagy áteresztőképességű köteg ellenőrzéshez.

Formális invariáns ellenőrzés

A FormalVerificationEngine futásidejű invariáns ellenőrzést biztosít. A verifyGraph(graph) az inferencia kérésekben az NSIR gráf strukturális invariánsait validálja InvariantType (pl. MEMORY_SAFETY, COHERENCE) és ProofRule (pl. MODUS_PONENS, INDUCTION) alapján. A build rendszer -Dverify=true kapcsolóval a src/verification/oftb.lean Lean4 bizonyításokat is lefordítja lake build útján, amelyek statikusan validálják az OFTB split_at művelet tulajdonságait.

Biztonsági bizonyítás

A SecurityProofEngine Bell-LaPadula szigorúan növekvő biztonsági osztályokkal, Biba integritási osztályokkal, hozzáférés vezérlési mátrixszal és nem-interferencia bisimulációval biztosítja a rendszer bizonyítható biztonságát. A proveInformationFlowSecurity(graph) az inferencia kérésekben az NSIR gráf éleinek engedélyezett információáramlási irányát validálja SecurityLabel-ek és FlowEdge-ek alapján.

Integrációs logika

A ZRuntime végrehajtási motorként szolgál, amely összeköti ezeket a komponenseket. ExecutionAction parancsokat dolgoz fel, mint a quantum_circuit vagy az entangle_variables, és azokat a helyi RelationalQuantumLogic szimulátorhoz vagy az IBMQuantumClient-hez irányítja. Az eredmények ezután a VerifiedInferenceEngine, a FormalVerificationEngine, a SecurityProofEngine és a QuantumTaskAdapter együttes csővezetékén keresztül kerülnek feldolgozásra a végső, kriptográfiailag biztosított válasz generálásához.

| Jellemző | Kód entitás | Fájl |
| :--- | :--- | :--- |
| Relációs kapuk | LogicGate.RELATIONAL_XOR | src/core_relational/quantum_logic.zig |
| Állapotkezelés | QuantumState | src/core_relational/quantum_logic.zig |
| Végrehajtási motor | ZRuntime | src/core_relational/z_runtime.zig |
| Hardver híd | IBMQuantumClient | src/core_relational/quantum_hardware.zig |
| Kvantum feladat | QuantumTaskAdapter | src/core_relational/quantum_task_adapter.zig |
| ZK Bizonyítás | ZKInferenceProver | src/core_relational/zk_verification.zig |
| Formális ellenőrzés | FormalVerificationEngine | src/core_relational/formal_verification.zig |
| Biztonsági bizonyítás | SecurityProofEngine | src/core_relational/security_proofs.zig |

---

5.1 KVANTUM LOGIKA ÉS IBM HARDVER INTERFÉSZ

A Kvantum Logika és IBM Hardver Interfész hidat biztosít a JAIDE rendszer relációs kognitív struktúrái és a fizikai kvantum számítás között. Magában foglalja a RelationalQuantumLogic motort a helyi szimulációhoz, az IBMQuantumClient-et a valós hardveren való végrehajtáshoz és a QuantumTaskAdapter-t az NSIR gráf szubgráfjainak kvantum feldolgozásához.

RelationalQuantumLogic Motor

A RelationalQuantumLogic motor felelős a relációs műveletek kvantum kapukra való leképezéséért és a kvantum állapotok életciklusának kezeléséért. Szabványos kvantum primitívek és speciális relációs kapuk készletét biztosítja.

Kapu műveletek

A rendszer átfogó logikai kapu készletet definiál a LogicGate enumban. Ezek tartalmazzák:

- Szabványos kapuk: HADAMARD, PAULI_X/Y/Z, PHASE, CNOT, TOFFOLI.
- Relációs kapuk: RELATIONAL_AND, RELATIONAL_OR, RELATIONAL_NOT, RELATIONAL_XOR.
- Speciális kapuk: FRACTAL_TRANSFORM az önhasonló állapot keveréshez.

A motor a kapukat qubit követelményeik szerint osztályozza és támogatja az egykubites műveleteket a többkubites összefonódó műveletekkel szemben.

Kvantum állapot reprezentáció

A kvantum állapotokat a QuantumState struktúra képviseli, amely nyomon követi:

- Amplitúdók: Komplex számok 2 elemű tömbje, amely az állapot vektort képviseli.
- Fázis: A qubit globális fázisa.
- Összefonódási fok: Skaláris érték, amely a más csomópontokkal való korrelációs erősséget képviseli.

QuantumTaskAdapter és szubgráf kivonás

A QuantumTaskAdapter felelős azon NSIR gráf-részek azonosításáért, amelyek elég kicsik ahhoz, hogy állapotvektor szimulátorra vagy IBM hardverre képezhetők. Az inferencia útvonalán az adapter háromfázisú ciklust futtat:

1. identifyQuantumSubgraphs() → ArrayList(QuantumSubgraph)
2. Minden szubgráfra executeQuantumTask(subgraph) → QuantumTaskResult
3. Ha task_result.success igaz, applyResultsToGraph(subgraph, &task_result) frissíti a NSIR gráfot.

A szubgráfok életciklusát a szerver kezeli: az adapter által visszaadott ArrayList minden elemének deinit-je meghívódik, majd az ArrayList maga is felszabadul.

IBM Quantum Hardver Interfész

Az IBMQuantumClient kezeli az IBM Quantum Platformmal való kommunikációt REST API-n keresztül.

Backend családok és kalibráció

A rendszer több IBM hardver családot támogat, előre definiált specifikációkkal és hibaprofillal az IBMBackendSpecs-ben:

- HERON: 133 qubit, T1 kb. 350 mikroszekundum.
- EAGLE: 127 qubit, T1 kb. 200 mikroszekundum.
- FALCON: 27 qubit, T1 kb. 100 mikroszekundum.

Az IBMBackendCalibrationData struktúra valós idejű telemetriát tárol, beleértve a T1/T2 időket, leolvasási hibákat és kapu hibákat a kiválasztott backend minden qubitjéhez.

Feladat benyújtás és OpenQASM

A kliens kezeli a kvantum feladat teljes életciklusát:

1. Szerializáció: Az áramkörök OpenQASM 3.0 karakterláncokká konvertálódnak.
2. Benyújtás: A submitJobWithBackend POST kérést hajt végre az IBM Cloud API-hoz.
3. Lekérdezés: Az eredmények a getJobResult segítségével kerülnek visszanyerésre a visszaadott feladat azonosító alapján.

Szimuláció és hibrid optimalizálás

Állapotvektor szimulátor zajmodellezéssel

Ha a use_real_backend hamis, a QuantumTaskAdapter a local_simulator-t alkalmazza. Ez a szimulátor megvalósítja:

- Zajmodellezés: Az IBMDocumentedBackendSpecs kalibrációs adatait alkalmazza a dekoherencia (T1/T2) és kapu hűtlenségek szimulálásához.
- Korlátok: A szimuláció 32 qubitre van korlátozva (SIMULATOR_QUBITS).

A QuantumTaskAdapter deinitje meghívja a self.local_simulator.deinit()-et.

Kvantum-Klasszikus Hibrid Optimalizáló

A rendszer hibrid algoritmusokat (VQE, QAOA) támogat egy QuantumClassicalHybridOptimizer-en keresztül.

- Paraméter-eltolás gradiensek: Gradienseket számít a kvantum áramkör paraméterek eltolásával (pl. rotációs szögek) az objektív függvény optimalizálásához klasszikus hardveren.
- Konfiguráció: Az alapértelmezések 0.1-es tanulási rátát és 10^-6 toleranciát tartalmaznak.

Főbb konstansok összefoglalója

| Paraméter | Érték | Leírás |
| :--- | :--- | :--- |
| HERON_QUBITS | 133 | Max qubitek Heron osztályú hardverhez |
| SIMULATOR_QUBITS | 32 | Max qubitek helyi állapotvektor szimulációhoz |
| HARDWARE_MAX_SHOTS | 100 000 | Maximális mintavételi lövések áramkörenként |
| POLL_INTERVAL_MS | 100 | Lekérdezési frekvencia a feladat eredményekhez |

---

5.2 NULLA-TUDÁS ELLENŐRZÉSI RENDSZER

A JAIDE Nulla-Tudás (ZK) Ellenőrzési Rendszere mechanizmust biztosít az ellenőrizhető következtetéshez, biztosítva, hogy a neurális hálózati számítások és a relációs gráf átmenetek helyesen kerültek végrehajtásra anélkül, hogy felfednék az alapul szolgáló modell súlyokat vagy az érzékeny bemeneti adatokat. A Groth16 bizonyítási rendszert alkalmazza a bn128 elliptikus görbe felett, egyedi Circom-alapú eszközláncot alkalmazva az áramkör generáláshoz és snarkjs-t a bizonyíték orchestráláshoz.

Rendszer architektúra

A ZK rendszer egy magas szintű Zig interfészre (ZKInferenceProver) és egy alacsony szintű R1CS áramkör definícióra (inference_trace.circom) van osztva. Az architektúra "kötelezd el-majd-bizonyítsd" mintát követ, ahol a bemenetek és kimenetek Blake3 vagy Poseidon hash-ekkel kerülnek elkötelezésre, és a bizonyíték validálja az ezen elkötelezések közötti átmenetet.

Build integráció

A build.zig -Dzk=true kapcsolóval három rendszerparancsot fűz össze:

1. circom src/zk/inference_trace.circom --r1cs --wasm --sym -o src/zk/
2. snarkjs groth16 setup src/zk/inference_trace.r1cs pot12_final.ptau src/zk/inference_trace.zkey
3. snarkjs zkey export verificationkey src/zk/inference_trace.zkey src/zk/verification_key.json

A zk build lépés a snarkjs_vkey step-től függ, tehát a zig build zk parancs a teljes ZK pipeline végrehajtja.

CircomProver és eszközlánc integráció

A CircomProver osztály hídként működik a Zig futtatókörnyezet és a snarkjs/circom eszközlánc között. Kezeli az áramkör fordítást, a megbízható beállítást (Groth16) és a tanú generálást.

Főbb függvények:

- compileCircuit(): circom folyamatot indít az R1CS és WASM artifaktumok generálásához.
- generateWitness(): A lefordított WASM-t és node-ot alkalmazza a tanú kiszámításához a bemeneti jelekből.
- prove(): Végrehajtja az snarkjs groth16 prove-t egy ZKProofBundle létrehozásához, amely tartalmazza a Groth16Proof-ot és a PublicSignals-t.

Adatstruktúrák

| Struktúra | Cél |
| :--- | :--- |
| ZKCircuitConfig | Definiálja a .wasm, .zkey útvonalakat és a pontossági paramétereket (alapértelmezett 64 bites). |
| Groth16Proof | Magában foglalja a G1 és G2 pontokat (pi_a, pi_b, pi_c) a bn128-hoz. |
| PublicSignals | i256 értékek gyűjteménye, amelyek az áramkör nyilvános bemeneteit/kimeneteit képviselik. |

Következtetési nyom áramkör (inference_trace.circom)

A ZK rendszer mag logikája az inference_trace.circom-ban található. Rögzített pontos aritmetikát és specifikus neurális rétegeket valósít meg ZK-barát módon.

Poseidon láncolás

Mivel a szabványos hash-ek, mint az SHA-256, drágák az R1CS-ben, a JAIDE PoseidonChain(n)-t alkalmaz az állapot elkötelezésekhez. A bemeneteket 6-os darabokban dolgozza fel, Poseidon hash függvényeken láncolva azokat egyetlen mezőelem kimenet előállításához.

RSF réteg ellenőrzés

Az RSFLayerComputation(dim) sablon tükrözi az RSFLayer-t a neurális veremben. Validálja:

1. Osztás: Az x bemeneti vektor x1-re és x2-re osztódik.
2. Affin csatolás: y2 = x2 ⊙ exp(S(x1)) + T(x1).
3. Rögzített pontos skálázás: Mivel a Circom véges testekben dolgozik, a floatok FIXED_POINT_SCALE (10^6) segítségével skálázódnak.
4. Taylor közelítés: Az exp függvény köbös Taylor sorral közelítendő: 1 + x + 0.5x^2 + 0.166667x^3.

Tartomány és tagság bizonyítékok

- RangeProof(bits): Biztosítja, hogy egy érték [min, max] tartományban legyen Num2Bits dekompozíció és Pedersen elkötelezések segítségével minden bithez.
- VerifyMerkleProof(depth): Szabványos Merkle fa útvonal validálást valósít meg Poseidon(2) hashelők és Mux1 segítségével az útvonal index váltáshoz.

Adatvédelem és ellenőrzési logika

A rendszer Differenciális Adatvédelmet és Biztonságos Aggregációt tartalmaz az egyéni adatpontok védelmére az ellenőrzési folyamat során.

Differenciális adatvédelem

A ZKInferenceProver zajt alkalmaz a következtetési nyomra az (ε, δ)-differenciális adatvédelem teljesítéséhez.

- Laplace zaj: SecureRng segítségével generálva és a nyilvános jelekbe injektálva a pontos értékek elhomályosításához.
- Gauss zaj: Magasabb dimenziós aggregációkhoz alkalmazva.

Biztonságos aggregáció

Az elosztott következtetéshez a rendszer támogatja a bizonyítékok aggregálását több résztvevőtől.

- SecureAggregation: Biztosítja, hogy az aggregált eredmény helyes legyen az egyéni hozzájárulások felfedése nélkül.
- Blake3 elkötelezés: Nagy sebességű bemenet/kimenet integritás ellenőrzéshez alkalmazva a drágább ZK bizonyíték generálása előtt.

Hibakezelés

A ZKProofError enum definiálja az ellenőrzési folyamat meghibásodási módjait, beleértve a CircomCompilationFailed, WitnessGenerationFailed és SnarkjsNotFound hibákat. Ezek a hibák a VerifiedInferenceEngine-en keresztül propagálódnak annak biztosítására, hogy az ellenőrizetlen eredmények soha ne kerüljenek érvényesként kezelésre magas integritású módokban.

---

6 KÖVETKEZTETÉSI SZERVER ÉS VISSZAKERESÉS

A Következtetési Szerver és Visszakeresési réteg az elsődleges interfészként szolgál a külső fogyasztók számára a JAIDE rendszerrel való interakcióhoz. Orchestrálja az átmenetet a nyers szöveges bemenetektől a nagy dimenziós neurális reprezentációkig, a relációs gráf érvelésig és végül a token generálásig. Ez a réteg kezeli a HTTP kapcsolatok életciklusát, érvényesíti a biztonsági és sebességkorlátozásokat, és speciális indexelési struktúrákat (SSI) és rangsorolási algoritmusokat alkalmaz a kontextus és koherencia fenntartásához a következtetés során.

Kiszolgálási architektúra áttekintés

Az InferenceServer egy többszálú HTTP motor, amelyet nagy áteresztőképességű token generálásra terveztek. ThreadPool-t alkalmaz az egyidejű kapcsolatok kezeléséhez, inference_mutex-szel védve a szálbiztos hozzáférés biztosítása érdekében az alapul szolgáló modell súlyokhoz és állapothoz.

A szerver szabványos RESTful végpontokat valósít meg, beleértve a /v1/health-et a monitorozáshoz és a /v1/inference-t az egyszeri kérés feldolgozáshoz. Nagy sűrűségű munkaterhelésekhez a /v1/batch_inference végpont lehetővé teszi több prompt párhuzamos feldolgozását.

A következtetési folyamat

Amikor egy kérés érkezik, a szerver komplex folyamatot hajt végre, amely áthidalja a diszkrét tokenek és a Mag Relációs Réteg közötti szakadékot. Minden inferencia kérés az alábbi 15 lépéses csővezetéken halad át:

| Fázis | Komponens | Művelet |
| :--- | :--- | :--- |
| 1. Belépés | RateLimiter | IP/API kulcs validálása max_requests_per_minute ellen. |
| 2. Tokenizálás | MGT | Nyers szöveg szódarab egységekké konvertálása. |
| 3. Beágyazás | LearnedEmbedding | Token azonosítók sűrű f32 vektorokká alakítása. |
| 4. Neurális folyam | RSFLayer | Beágyazások átadása Visszafordítható Szórt Folyam rétegeken. |
| 5. NSIR kódolás | SelfSimilarRelationalGraph | Tenzor bájtjainak encodeInformation útján gráfba integrálása. |
| 6. FractalLPU | FractalLPU | Csomópont hash → tile leképezés és balanceAllTiles. |
| 7. R-GPU | RelationalGraphProcessingUnit | Gráf elosztás fizikai magokra. |
| 8. Érvelés | ReasoningOrchestrator | Hierarchikus érvelés indítása 50 belső iterációval. |
| 9. Meglepetés memória | SurpriseMemoryManager | storeWithSurprise Jaccard-alapú CAS küszöbökkel. |
| 10. Temporális gráf | TemporalGraph | addNodeAtTime és advanceTime nanoszekundum pontossággal. |
| 11. Verifier | VerifiedInferenceEngine | performVerifiedInference bemeneti/kimeneti pufferrel (opcionális JAIDE_VERIFY=1). |
| 12. Signal Engine | SignalPropagationEngine | propagateStep a rebindált gráfon. |
| 13. ZRuntime | ZRuntime | inf_<request_count> változó létrehozása. |
| 14. VPU | VPU | computeGraphEmbeddings → quantumVectorOps → computeSimilarityMatrix. |
| 15. FNDS | FNDSManager | createTree → insertIntoTree → createIndex → addPatternToIndex. |
| 16. CREV | CREVPipeline | processTextStream a bemeneti szövegen. |
| 17. Formális | FormalVerificationEngine | verifyGraph az NSIR gráfon. |
| 18. Biztonság | SecurityProofEngine | proveInformationFlowSecurity az NSIR gráfon. |
| 19. Kvantum | QuantumTaskAdapter | identifyQuantumSubgraphs → executeQuantumTask → applyResultsToGraph. |
| 20. Boost | boostAboveMean | Átlag feletti aktivációk enyhe kiemelése (1.05×). |
| 21. Visszakeresés | SSI és Ranker | Szegmentált Szekvencia Index lekérdezése releváns kontextushoz. |
| 22. Generálás | Auto-regressziós ciklus | Token generálási ciklus végrehajtása; max_new_tokens határig. |

Minden komponens minden kérésben ténylegesen részt vesz — nincs olyan mező, amely null-ként marad utólagos aktiválásra várva. A loadModel útvonalán a szerver mind a VPU, FNDSManager, CREVPipeline, FormalVerificationEngine, SecurityProofEngine és QuantumTaskAdapter mezőket inicializálja; a deinit útvonalán mind felszabadul.

Visszakeresés és kontextus rangsorolás

A JAIDE Szegmentált Szekvencia Indexet (SSI) alkalmaz a neurális állapotok és relációs hármasok kereshető előzményének fenntartásához. Az SSI hierarchikus hash faként van strukturálva, lehetővé téve a retrieveTopK hasonlósági kereséseket, amelyek tájékoztatják a Ranker-t. A Ranker n-gram csökkentési súlyozást és Jaccard hasonlóságot alkalmaz a potenciális következő tokenek pontozásához, biztosítva, hogy a generált kimenet a megadott kontextusban és a modell belső memóriájában maradjon.

Adatfolyam: Szövegtől a relációs állapotig

Az InferenceServer a CPU-kötött API logika és a GPU/LPU-kötött neurális számítások koordinátoraként működik. A VerifiedInferenceEngine, FormalVerificationEngine, SecurityProofEngine és QuantumTaskAdapter együtt biztosítják, hogy a felhasználónak visszaadott eredmények kriptográfiailag konzisztensek, invariánsan helyesek, információáramlás-biztonságosak és kvantum szubgráf-frissítettek legyenek a modell állapotával.

---

6.1 HTTP KÖVETKEZTETÉSI SZERVER

Az InferenceServer a JAIDE kiszolgálási rétegének elsődleges belépési pontja, nagy teljesítményű HTTP interfészt biztosítva mind az egyszeri, mind a köteg következtetéshez. Orchestrálja a komplex átmenetet a természetes nyelvi bemenetektől az RSF neurális vermen, a Mag Relációs Rétegen, a formális ellenőrzésen, a biztonsági bizonyításokon és a kvantum feladat adapteren keresztül.

Szerver architektúra

Az InferenceServer dedikált ThreadPool-ra épülő többszálú, aszinkron architektúrán alapul a kapcsolat életciklusok kezeléséhez a fő eseményhurok blokkolása nélkül.

Főbb komponensek:

- RateLimiter: Csúszóablak algoritmust valósít meg az IP-cím szerinti kérések nyomon követéséhez.
- Következtetési Mutex: Egy globális Thread.Mutex biztosítja a szálbiztos hozzáférést az alapul szolgáló modell súlyokhoz és a SelfSimilarRelationalGraph-hoz az előre irányuló menet során.
- Kapcsolat életciklus: Minden bejövő kapcsolatot a fő szál fogad és a poolhoz irányít, ahol API kulcsokra kerül validálásra (ha a require_api_key engedélyezve van).

Mag mezők

Az InferenceServer struktúra a következő komponens mezőket tárolja, amelyek mind ténylegesen aktívak minden kérésben:

- model, ssi, ranker, embedding — alap neurális/index verem.
- nsir_graph, chaos_kernel, esso, surprise_memory, temporal_graph — mag relációs komponensek.
- verifier — VerifiedInferenceEngine JAIDE_VERIFY=1 esetén.
- signal_engine — SignalPropagationEngine a graph+flow_analyzer-hez kötve.
- z_runtime — kvantum-relációs változó motor.
- fractal_lpu, r_gpu — hardver gyorsítás.
- vpu — VectorProcessingUnit (opcionális, catch null-nel inicializálva).
- fnds_manager — FNDSManager (opcionális).
- crev_pipeline — CREVPipeline a chaos_kernel-hez kötve.
- formal_verifier — heap-allokált *FormalVerificationEngine.
- security_engine — heap-allokált *SecurityProofEngine.
- quantum_adapter — QuantumTaskAdapter a nsir_graph-ra kötve.

Minden mező vagy loadModel útján inicializálódik, vagy null marad, de sohasem marad utólagos aktiválásra várva. A deinit fordított sorrendben mindent tisztít, beleértve a heap-allokált formal_verifier és security_engine felszabadítását destroy útján.

API végpontok

A szerver három elsődleges REST végpontot tesz elérhetővé:

| Végpont | Módszer | Leírás |
| :--- | :--- | :--- |
| /v1/health | GET | Visszaadja a szerver állapotát, üzemidejét és modell betöltési állapotát. |
| /v1/inference | POST | Szabványos egyszeri kérés következtetés. InferenceRequest JSON-t vár. |
| /v1/batch_inference | POST | Több prompt párhuzamos feldolgozása ServerConfig.batch_size-ig. |

Mind a /v1/inference, mind a /v1/batch_inference útvonal minden kérésben végigfut a teljes 22 lépéses csővezetéken, beleértve a VPU, FNDS, CREV, formális ellenőrzés, biztonsági bizonyítás és kvantum feladat komponenseket.

A következtetési folyamat

A szerver magja a handleInference és handleBatchInference függvények, amelyek végrehajtják a teljes transzformációt a tokenektől a relációs érvelésig és vissza a generált szövegig.

Adatfolyam szakaszok:

1. Tokenizálás: Az MGT (Multi-Gram Tokenizáló) a bemeneti szöveget token azonosítók sorozatává konvertálja.
2. Beágyazás: A tokenek nagy dimenziós térbe kerülnek vetítve a LearnedEmbedding segítségével.
3. RSF előre irányuló menet: Az RSFLayer verem visszafordítható affin csatolást és OFTB keverést hajt végre.
4. NSIR kódolás: A neurális állapot kódolódik a SelfSimilarRelationalGraph-ba (NSIR).
5. FractalLPU / R-GPU: A gráf csomópontok hardver tile-okra képződnek le, majd fizikai magokra elosztódnak.
6. Érvelés orchestrálás: A ReasoningOrchestrator futtatja a háromfázisú érvelési ciklust (helyi, globális, meta).
7. Meglepetés memória: Az újszerű minták a SurpriseMemoryManager-be kerülnek rögzítésre hosszú távú megőrzésre.
8. Temporális pillanatkép: A TemporalGraph nanoszekundum időbélyeggel rögzíti a csomópont állapotokat.
9. Ellenőrzött következtetés: Ha JAIDE_VERIFY=1, a VerifiedInferenceEngine performVerifiedInference hívódik.
10. Signal propagation: SignalPropagationEngine.propagateStep egy lépést hajt végre.
11. ZRuntime: Változó létrehozás inf_<request_count> névvel.
12. VPU gráf beágyazás: F64x4 vektorok, quantum vector ops és hasonlósági mátrix.
13. FNDS mintaindex: createTree, insertIntoTree, createIndex, addPatternToIndex.
14. CREV pipeline: processTextStream a bemeneti szövegen.
15. Formális invariáns ellenőrzés: verifyGraph az NSIR gráfon.
16. Biztonsági bizonyítás: proveInformationFlowSecurity az NSIR gráfon.
17. Kvantum feladat: identifyQuantumSubgraphs → executeQuantumTask → applyResultsToGraph.
18. boostAboveMean: Átlag feletti aktivációk kiemelése.
19. Token generálás: Ranker által vezérelt auto-regressziós ciklus.
20. Válasz szerializáció: JSON serializáció és HTTP válasz.

Ellenőrzött következtetés integráció

A szerver "Ellenőrzött" módot támogat a VerifiedInferenceEngine-en keresztül. Ha engedélyezve van, a szerver Nulla-Tudás (ZK) bizonyítékot generál a következtetés végrehajtásáról.

- Elkötelezés: A bemeneti tokenek és modell súlyok Blake3 segítségével kerülnek hash-elve egy elkötelezés létrehozásához.
- Nyom rögzítés: A ReasoningOrchestrator minden művelete rögzítésre kerül egy InferenceWitness-be.
- Bizonyíték generálás: A kérés befejezésekor a VerifiedInferenceEngine a CircomProver-t alkalmazza egy Groth16 bizonyíték generálásához, amely igazolja, hogy a kimenet helyesen lett levezetva az elkötelezett bemenetből és modellből.

Ezen kívül a FormalVerificationEngine invariáns ellenőrzést, a SecurityProofEngine információáramlás biztonsági bizonyítást és a QuantumTaskAdapter kvantum szubgráf frissítést végez minden kérésben, függetlenül a JAIDE_VERIFY beállítástól.

Szerver konfiguráció és inicializálás

A szerver a ServerConfig struktúrán keresztül kerül konfigurálásra. Az inference_server_main.zig fő belépési pontján keresztül inicializálható és indítható.

Konfigurációs paraméterek:

- batch_size: Meghatározza az egyidejűleg feldolgozható szekvenciák maximális számát az RSF veremben.
- esso_initial_temp, esso_cooling_rate, esso_max_iterations: Szabályozzák az EntangledStochasticSymmetryOptimizer viselkedését az érvelési fázis során.
- require_api_key: Logikai jelző a hitelesítés érvényesítéséhez.
- max_request_size_bytes, request_timeout_ms, keep_alive_timeout_ms: HTTP szintű biztonsági korlátok.

Fő végrehajtás

A main függvény kezeli a parancssori argumentum elemzést, a környezeti változó felülírásokat (pl. JAIDE_MODEL_PATH, JAIDE_API_KEY, JAIDE_VERIFY, JAIDE_REASONING_CYCLES) és a kecses leállítást.

---

6.2 SSI INDEX ÉS RANKER

A Szegmentált Szekvencia Index (SSI) és a Ranker alrendszerek biztosítják a JAIDE következtetési folyamat alapvető visszakeresési és pontozási infrastruktúráját. Az SSI hierarchikus hash fát valósít meg a token szegmensek hatékony tárolásához és integritás-ellenőrzött visszakereséséhez, míg a Ranker többtényezős pontozást biztosít n-gram csökkentéssel, MinHash-alapú Jaccard hasonlósággal és diverzitási metrikákkal.

SSI: Szegmentált Szekvencia Index

Az SSI egy hierarchikus hash fa struktúra, amelyet token hash-ek alapján Segment adatok tárolására és visszakeresésére terveztek. Tartalom-címezhető indexként működik Merkle-stílusú integritás ellenőrzésekkel és automatikus egyensúlyozással.

Adatstruktúrák és hierarchia

Az index Node objektumok fájává van szervezve, ahol minden csomópont lehet ág (gyermekeket tartalmaz) vagy levél (szegmenseket és ütközési láncokat tartalmaz).

- Segment: Token sorozatot képvisel kapcsolódó metaadatokkal, beleértve egy globális position-t, score-t és anchor_hash-t.
- Node: Tartalmaz egy hash-t, amely a részfájának állapotát képviseli, children listát (ágakhoz) és segment-et vagy collision_chain-t (levelekhez).
- CollisionNode: Láncolt lista struktúra a levél csomópontokon belüli hash ütközések kezeléséhez.

SSI implementációs logika

Az SSI 6-os bucket_width-et alkalmaz, ami 64 gyermeket eredményez ág csomópontonként. A hash integritást a refreshHash tartja fenn, amely egy csomópont hash-ét a gyermekei (ágakhoz) vagy szegmensei (levelekhez) alapján számítja.

| Jellemző | Implementációs részlet |
| :--- | :--- |
| Hash algoritmus | Egyedi mixHash 0x9E3779B185EBCA87 konstanssal |
| Integritás | Merkle-stílusú rekurzív hashelés computeBranchHash-en keresztül |
| Ütközés kezelés | Láncolt lista collision_chain levél csomópontokban |
| Keresés | retrieveTopK hasonlósági keresés szegmens pontszámok alapján |

Ranker: Szekvencia pontozás és visszakeresés

A Ranker felelős a token szekvenciák relevanciájának és minőségének értékeléséért. N-gram súlyok, Lokalitás-Érzékeny Hashelés (LSH) és diverzitási pontozás kombinációját alkalmazza normalizált pontszám előállításához.

Pontozási komponensek

A Ranker több súlyozott tényezőn keresztül számítja a pontszámokat a RankerConfig által kezelt konfiguráció alapján. A topKHeap és rankCandidatesWithQuery függvények a szerver token generálási ciklusából hívódnak minden generációs lépésben.

---

7 HARDVER GYORSÍTÁS

A JAIDE hardver gyorsítási rétege több szintből áll: Futhark-alapú C/CUDA/OpenCL kernelek a neurális műveletekhez, FractalLPU a fraktál csempézéshez és R-GPU a gráf elosztáshoz, valamint Haskell-alapú RTL modulok a hardver szintű memória arbiter, ranker mag és SSI keresés szimulációjához.

7.1 FUTHARK KERNELEK

A Futhark egy funkcionális tömb-nyelv, amely C-re, OpenCL-re vagy CUDA-ra fordul le. A JAIDE ezt a nyelvet használja a nagy teljesítményű neurális kernelek generálásához.

Kernel források

- src/hw/accel/futhark_kernels.fut — CPU kernelek (futhark c segítségével fordítva).
- src/hw/accel/main.fut — GPU kernelek (futhark opencl segítségével fordítva -Dgpu=true esetén).

Build integráció

A build.zig futhark_cpu_step és futhark_gpu_step lépéseket definiál, amelyek automatikusan meghívják a futhark parancsot. Minden Zig futtatható fájl, amely linkeli a generált C fájlokat, függést hordoz ezekre a lépésekre.

Zig kötések

A src/hw/accel/futhark_bindings.zig extern deklarációkat biztosít a Futhark által generált C API-hoz, beleértve a struct_futhark_f16_2d, struct_futhark_f16_3d típusokat és a futhark_entry_batch_forward, futhark_entry_batch_oftb_forward, futhark_entry_batch_gradients_full stb. függvényeket.

7.2 FRACTAL LPU ÉS R-GPU

A FractalLPU (Local Processing Unit) fraktál-arányos csempézést biztosít a gráf csomópontokhoz, míg a RelationalGraphProcessingUnit (R-GPU) a csomópontokat fizikai magokra osztja el.

FractalLPU

A mapNode(node_hash, weight) függvény egy csomópont hash-ét egy tile-hoz rendeli, a balanceAllTiles pedig újraegyensúlyozza a tile terhelést. Az inferencia szerver handleInference minden kérésben végigmegy az NSIR csomópontokon és leképezi őket.

RelationalGraphProcessingUnit

A distributeGraph(graph) a gráfot fizikai magokra osztja el aszinkron NoC (Network on Chip) segítségével. A trainer runCoreRelationalPass és a szerver handleInference/handleBatchInference minden ciklusban meghívja.

7.3 HASKELL RTL MODULOK

Ez a szakasz dokumentálja a Clash-ben (egy Haskell-alapú HDL fordító) megvalósított Register Transfer Level (RTL) hardver modulokat. Ezek a modulok speciális hardver gyorsítást biztosítanak a memória arbitrációhoz, rangsoroláshoz és index kereséshez a JAIDE rendszeren belül. A build.zig -Drtl=true kapcsolóval GHC-vel megosztott könyvtárba (librtl_sim.so) fordítja ezeket a modulokat, majd összefordítja a jaide-rtl-sim Zig futtatható fájlt, amely önmagában futtatható RTL szimulációt biztosít.

jaide-rtl-sim futtatható fájl

Az src/hw/rtl/rtl_sim_main.zig biztosítja a Zig-alapú RTL szimulátort. Használat: jaide-rtl-sim <cycles> <banks> <requests_per_cycle>. Kimenete tartalmazza:

- Memory arbiter statisztikái: grant ratio, avg/max latency, avg bank pressure.
- RankerCore statisztikái: top ranker score, median ranker score.
- SSISearch statisztikái: SSI probes, SSI hits, SSI hit ratio.

MemoryArbiter

A MemoryArbiter modul rögzített prioritású arbitrációs Véges Állapotgépet (FSM) valósít meg a megosztott memória erőforráshoz való egyidejű hozzáférés kezeléséhez több hardver kliens számára. Kölcsönösen kizárólagos hozzáférést biztosít a megosztott memória erőforráshoz, miközben kezeli a kérés-válasz ciklusokat.

Implementációs részletek

Az arbitrátor Mealy gépként van megvalósítva az arbiterT átmeneti függvény segítségével. 4 klienst (NumClients) kezel és rögzített 4 ciklusos (ServiceCycles) kiszolgálási ablakot érvényesít megadott kérésenként.

Állapotok és átmenetek

Az arbitrátor két elsődleges állapotban működik az ArbiterState-ben definiálva:

- ArbIdle: Az arbitrátor átvizsgálja a clientReqs-t a findIndex segítségével az első aktív kérés azonosításához. Ha talál, ArbServing-re vált és hozzáférést biztosít a specifikus ClientID4-nek.
- ArbServing: Az arbitrátor fenntartja az aktuális kapcsolatot a ServiceCycles által meghatározott időtartamig. Növeli a belső számlálót, amíg el nem éri a határt, majd visszatér az ArbIdle-hoz.

Adatfolyam és demultiplexálás

Az arbitrátor egyetlen MemRequest-et ad ki a memória vezérlőnek és MemResponse jelek vektorát vissza a klienseknek. A válaszok a filterResp függvény segítségével kerülnek demultiplexálásra, amely biztosítja, hogy a válasz csak a respClient azonosítóval egyező kliensnek legyen látható.

SSI keresési logika állapottáblázat

| Állapot | Átmeneti feltétel | Művelet |
| :--- | :--- | :--- |
| Idle | Just SearchRequest | Átmenet Fetching-re rootAddr segítségével. |
| Fetching | Just TreeNode | Átmenet Comparing-ra vagy rekurzió. |
| Comparing | key == nodeKey | Leállítás SearchResult(found=True) eredménnyel. |
| Comparing | key < nodeKey | Átmenet Fetching(leftChild)-re. |
| Comparing | key > nodeKey | Átmenet Fetching(rightChild)-re. |

RankerCore

A RankerCore egy hardver gyorsító, amelyet a visszakeresés során a szegmensek pontszámainak kiszámítására terveztek. Pozíció-torzított rangsorolási algoritmust valósít meg, amely az eredményeket az eredeti szekvencia pozíciójuk alapján súlyozza.

Pontozási logika

A mag finalScore-t számít a baseScore és egy számított bias kombinálásával.

- Pozíció torzítás: A computePositionBias segítségével számítva, amely reciprok skálázást alkalmaz: positionBiasScale / (position + 1).
- Skálázási tényező: A positionBiasScale 1000-re van rögzítve.

Állapotkezelés

A RankerState nyomon követi a stateCounter-t (az azonos hash-re vonatkozó szekvenciális lekérdezések észleléséhez) és a lastScore-t. Ha egy új RankRequest megegyezik a lastQuery-vel, a belső rang számláló növekszik; egyébként 1-re áll vissza.

SSISearch

Az SSISearch modul egy hardver motor a Szegmentált Szekvencia Index (SSI) fák bejárásához. Nagy sebességű HashKey64 kulcsok keresését végzi a memóriában tárolt fa struktúrán belül.

Keresési FSM

A motor háromállapotú FSM-et valósít meg a SearchState által definiálva:

1. Idle: SearchRequest-re vár, amely tartalmaz egy searchKey-t és egy rootAddr-t.
2. Fetching: Memória kérést ad ki egy TreeNode-hoz egy specifikus NodeAddr32-nél.
3. Comparing: Miután egy TreeNode megérkezik, a checkNode összehasonlítja a searchKey-t a nodeKey-vel. Ezután dönt, hogy leállítja (megtalálva/nem találva) vagy visszatér Fetching-re a leftChild vagy rightChild esetén.

Korlátok és biztonság:

- Max mélység: A rosszul formált fákban való végtelen ciklusok megelőzéséhez a keresés depthExceeded eredménnyel leáll, ha a currentDepth eléri a MaxSearchDepthConfig-ot (64).
- Null mutatók: A motor kifejezetten ellenőrzi a nullAddr-t (0) a gyermekek lekérésének megkísérlése előtt.

---

8 ELOSZTOTT TANÍTÁS

Az elosztott tanítást a JAIDE-ban egy több GPU-s orchestrációs réteg kezeli, amely NCCL-t (NVIDIA Kollektív Kommunikációs Könyvtár) alkalmaz a nagy teljesítményű kommunikációhoz és Futhark-ot a gyorsított kernel végrehajtáshoz. A rendszer egy-rang-per-eszköz modellt követ, ahol több folyamat szinkronizálja a gradienseket és a köteg statisztikákat a nagy léptékű RSF modellek tanításához.

Rendszer architektúra

Az elosztott tanítási verem áthidalja a magas szintű tanítási logikát az alacsony szintű GPU hardver kezeléssel. A DistributedTrainerFuthark orchestrálja a tanítási ciklust, míg a GPUCoordinator kezeli az alapul szolgáló NCCL kommunikátorokat és CUDA streameket. A Modal felhő telepítéshez a rendszer két útvonalat biztosít: Python-orchestrált modal_distributed_train.py szkriptet és Zig-natív jaide-distributed-futhark --deploy módot.

Elosztott tréner

A DistributedTrainerFuthark a több GPU-s tanítás központi struktúrája. Integrálja az MGT tokenizálót, az RSFAccelerator-t és a teljes Mag Relációs Réteget (NSIR, CREV, ReasoningOrchestrator, VPU, FNDS, SignalPropagationEngine) egy egységes tanítási interfész biztosításához.

Fő felelősségek:

- Inicializálás: Rang-tudatos környezetek beállítása, ahol minden tréner példány ismeri a world_size-t és a rank-ot.
- Folyamat végrehajtás: A trainStepFuthark futtatása, amely kezeli a tokenizálást, a beágyazás kereséseket és az előre/visszafelé irányuló meneteket Futhark kerneleken keresztül.
- Relációs integráció: A runCoreRelationalPass periodikus futtatása a neurális frissítések szinkronizálásához a SelfSimilarRelationalGraph-gal, VPU-val, FNDSManager-rel és minden aktív komponenssel.
- Ellenőrzőpont: Verzionált modell állapotok mentése és betöltése a klaszteren keresztül.

TrainerComponents és initWithComponents

A tréner két konstruktort biztosít:

- initWithConfig(allocator, coordinator, model_dim, num_layers, local_batch_size, config): Egy alapértelmezett angol szókincs alapú MGT tokenizert épít, ideiglenes SignalPropagationEngine-t hoz létre, és meghívja az initWithComponents-t.
- initWithComponents(allocator, coordinator, model_dim, num_layers, local_batch_size, config, components): A fő konstruktor. A TrainerComponents struktúra {tokenizer: MGT, signal_engine: SignalPropagationEngine, embedding_accel: ?EmbeddingAccelerator} tagokat tartalmaz. A konstruktor a signal_engine-t automatikusan rebindálja a végső nsir_graph és crev_kernel.flow_analyzer referenciákra, tehát nincs szükség post-construction aktiválásra (postInit el van távolítva).

Ez a tervezés biztosítja, hogy egyetlen mező sem marad null-ként utólagos aktiválásra várva. A signal_engine mező nem opcionális, hanem mindig egy érvényes SignalPropagationEngine-re mutat.

GPU Koordinátor és NCCL

A GPUCoordinator alacsony szintű kötéseket biztosít az NVIDIA hardveréhez és kommunikációs primitíveihez. Absztrahálja az NCCL és CUDA stream kezelés komplexitását egy tiszta Zig interfészbe.

Főbb jellemzők:

- Eszköz kezelés: Automatikusan leképezi a rangokat a helyi GPU-kra cudaSetDevice segítségével.
- Kollektív műveletek: Szabványos elosztott primitívek megvalósítása, beleértve az allReduce, broadcast, allGather és reduceScatter műveleteket.
- Szinkronizálás: Egy barrier megvalósítás egy dummy allReduce-on keresztül egy dedikált barrier_buffer-en.
- Memória életciklus: Eszköz memória allokáció (cudaMalloc) és gazdagép-eszköz átvitelek kezelése egy elosztott rang kontextusában.

Felhő telepítés

Míg a JAIDE helyi klasztereken futhat, Modal felhő telepítésre van optimalizálva. Ez lehetővé teszi a DistributedTrainerFuthark gyors skálázását nagy teljesítményű B200/B300 példányokon. Két telepítési útvonal áll rendelkezésre:

1. Python-orchestrált: modal run scripts/modal_distributed_train.py — Modal konténerbe indított több-rangú alfolyamat. Minden rang egy jaide-distributed-futhark alfolyamatot futtat.
2. Zig-natív: jaide-distributed-futhark --deploy <model_path> <dataset_path> — Modal API-hoz irányított közvetlen POST kérés a ModalGPUClient (src/distributed/modal_gpu.zig) segítségével. MODAL_API_TOKEN környezeti változó szükséges. 30 másodpercenként lekérdezi a feladat állapotát completed/failed állapotig.

---

8.1 ELOSZTOTT TRÉNER

A DistributedTrainerFuthark a JAIDE rendszer több GPU-s tanításának elsődleges orchestrátora. Integrálja a Futhark-gyorsított RSF neurális vermet a Mag Relációs Réteggel, kezelve az adatpárhuzamosságot, a gradiens szinkronizálást NCCL-en keresztül és a nagy léptékű adathalmazok nagy teljesítményű I/O-ját.

1. Inicializálás és konfiguráció

A tréner initWithComponents útján inicializálódik, amely beállítja a szükséges komponenseket mind a neurális, mind a relációs feldolgozáshoz. Szigorú validálást érvényesít a modell dimenziókon (amelyeknek párosnak kell lenniük az RSF csatoló rétegekhez), a rang/világ méret paramétereken és a hiperparamétereken.

Komponens összetétel

A tréner az alábbi kritikus alrendszereket aggregálja, mind kötelező (nem opcionális) mezőként:

- MGT szókincs: Angol tokenek és morfológiai dekompozíciós szabályok alapkészletével inicializálva, vagy külsőleg átadott tokenizerrel.
- RSFAccelerator: Kezeli a Futhark GPU kontextust és a többrétegű RSF súlyokat, actual_model_dim-mel, amely nem lehet kisebb, mint a tokenizer.next_token_id.
- LearnedEmbedding: 50000 vocab_size-zel és a beágyazás gradiens propagációhoz.
- EmbeddingAccelerator: Opcionális, catch null-nel biztosított; alapértelmezetten az RSFAccelerator ctx-jével inicializálódik.
- Relációs verem: Tartalmazza a CREVPipeline-t, a ChaosCoreKernel-t (heap-allokált), a SelfSimilarRelationalGraph-ot, az EntangledStochasticSymmetryOptimizer-t, a SurpriseMemoryManager-t, a TemporalGraph-ot és a ReasoningOrchestrator-t (per-pass létrehozva).
- SignalPropagationEngine: Nem opcionális mező; automatikusan rebindálva a nsir_graph és crev_kernel.flow_analyzer referenciáira.
- ZRuntime: Heap-allokált *ZRuntime a kvantum-relációs változó kezeléshez.
- RelationalGraphProcessingUnit: catch null-nel biztosított.
- FNDSManager: Kötelező mező.
- VPU: Kötelező mező.
- GPUCoordinator: Kezeli a rang-specifikus eszköz hozzárendeléseket és az NCCL kollektív műveleteket.

Nincs postInit hívás. Minden mező a konstruktoron belül kötődik és aktiválódik.

2. Elosztott adathalmaz betöltés

A rendszer rang-tudatos JSONL betöltőt alkalmaz az adatpárhuzamosság megvalósításához. Minden rang kiszámítja az adathalmaz saját szeletét a GPU-k közötti átfedés elkerülése érdekében.

Rang particionálási logika

1. Minta számlálás: A teljes mintaszám a JAIDE_TOTAL_SAMPLES-ből vagy fájl átvizsgálással kerül lekérésre.
2. Index számítás: Minden rang meghatározza a start_valid_index-ét és a samples_per_rank-ját a világ mérete és a saját rang azonosítója alapján.
3. JSONL elemzés: Az extractDatasetText függvény JSON objektumokat elemez, kifejezetten a "text" kulcsot keresve.

Adathalmaz particionálási táblázat

| Paraméter | Leírás |
| :--- | :--- |
| base_per_rank | total_samples / world_size |
| remainder | total_samples % world_size |
| start_valid_index | Az aktuális rang eltolása a globális adathalmazban |

3. Tanítási folyamat

A trainStepFuthark függvény valósítja meg a mag tanítási ciklust, amely áthidalja a természetes nyelvi tokenek és a GPU-gyorsított tenzor műveletek közötti szakadékot.

Adatfolyam: tokenizálás, beágyazás, kernelek

1. Tokenizálás: A bemeneti szöveg token azonosítókká konvertálódik az MGT.encode segítségével.
2. Beágyazás keresés: A token azonosítók sűrű vektorokra képeződnek le a LearnedEmbedding rétegben.
3. Futhark előre irányuló menet: A beágyazások FutharkArray3DF16-ként kerülnek feltöltésre a GPU-ra és az RSF rétegeken keresztül kerülnek feldolgozásra.
4. Futhark visszafelé menet és gradiens: propagateEmbeddingGradients a Futhark-ból származó gradienst átvezeti a beágyazás visszaterjesztésén, a GradientFlowController.initWithConfig(.{ .gradient_clip_norm = 1.0, .use_normalized_gradient_flow = true, .spectral_power_iterations = 5 }) beállítással, manuális L2-norma vágással.
5. Relációs integráció: A runCoreRelationalPass meghívódik minden lépésben a következő sorrendben:
   1. nsir_graph.encodeInformation a bemeneti tokenek bájtjaira.
   2. VPU: computeGraphEmbeddings → quantumVectorOps (topology_hash-ből származó szögekkel) → computeSimilarityMatrix → learning_rate moduláció.
   3. FNDSManager: createTree(6, 4) → insertIntoTree minden sample-re → createIndex → addPatternToIndex PatternLocation-nel.
   4. R-GPU distributeGraph.
   5. ReasoningOrchestrator runHierarchicalReasoning(50).
   6. SurpriseMemoryManager storeWithSurprise.
   7. TemporalGraph addNodeAtTime + advanceTime.
   8. SignalPropagationEngine propagateStep (a rebindált gráfon).
   9. ZRuntime createVariable train_<global_step> néven.

A buildKnowledgeGraph útvonal a CREVPipeline processTextStream mellett FNDSManager createTree(4, 3) + insertIntoTree hívást is végrehajt kg_<text_hash> node azonosítóval.

Gradiens szinkronizálás

Egy korszak vagy köteg szekvencia végén a tréner allReduceFloat32Values, allReduceFloat32Max és allReduceFloat16 kombinációját hajtja végre a súly delták szinkronizálásához a klaszteren keresztül. Egyetlen rang esetén (world_size <= 1) csak lokális frissítés történik.

4. Ellenőrzőpont kezelés

A tréner verzionált ellenőrzőpontokat támogat a tanítás folytonosságának és a modell perzisztenciájának biztosítása érdekében.

Mentés/betöltés mechanizmus

- Verzió követés: A TrainerConfig meghatároz egy checkpoint_version-t (jelenleg 7) a kompatibilitás fenntartásához.
- Szerializáció: Az RSF súlyok, beágyazási mátrixok és az NSIR gráf állapota bináris formátumba kerülnek szerializálva; a fájl atomi rename-mel írja felül a célt.
- 0-ás rang felelőssége: Csak a gyökér rang (0-ás rang) végzi a tényleges fájl I/O-t az ellenőrzőpontokhoz az írási versengés elkerülése érdekében, amelyet egy synchronize barrier követ a többi ranghoz.
- Signal engine rebind: A loadCheckpoint után a signal_engine automatikusan újra rebindálódik a betöltött nsir_graph és crev_kernel.flow_analyzer referenciáira.

---

8.2 GPU KOORDINÁTOR ÉS NCCL

A GPU Koordinátor a JAIDE rendszer több GPU-s elosztott tanításának központi kezelő entitása. Egy-rang-per-eszköz modellt valósít meg, kezelve a CUDA eszközök, memória allokációk és nagy teljesítményű kollektív kommunikációk életciklusát NCCL (NVIDIA Kollektív Kommunikációs Könyvtár) kötéseken keresztül.

Architektúra és eszközkezelés

A GPUCoordinator struktúra kezeli a folyamat rangjának az elosztott "világban" és a hozzárendelt fizikai GPU-nak a kapcsolatát. Biztosítja, hogy minden folyamat egy specifikus CUDA eszközhöz legyen rögzítve a cudaSetDevice segítségével a rangja alapján.

Adatfolyam: Inicializálás

1. Eszköz hozzárendelés: A koordinátor meghatározza a device_id-t a rang és a helyi eszközszám modulójának kiszámításával.
2. NCCL inicializálás: NCCL kommunikátort (ncclComm) inicializál az összes rangon megosztott egyedi azonosító segítségével. A rang 0 generálja az azonosítót és fájlba írja, majd a többi rang beolvassa a JAIDE_NCCL_ID_PATH útvonalról.
3. Stream létrehozás: Dedikált CUDA stream kerül létrehozásra az aszinkron kollektív műveletek számára a gazdagép oldali végrehajtás blokkolásának elkerülése érdekében.
4. Barrier beállítás: Egy kis 4 bájtos puffer kerül allokálásra az eszközön a barrierek megkönnyítéséhez dummy kollektív műveletek segítségével.

Eszköz memória kezelés

A koordinátor egyszerűsített interfészt biztosít az eszközön tárolt memória kezeléséhez, a nyers CUDA mutatókat Zig-barát absztrakciókba burkolva.

| Függvény | Cél | Implementációs részlet |
| :--- | :--- | :--- |
| allocDeviceMemory | Bájtokat allokál az aktuális GPU-n. | Meghívja az nccl.cudaMalloc-ot. |
| freeDeviceMemory | Felszabadítja a GPU memóriát. | Meghívja az nccl.cudaFree-t. |
| copyHostToDevice | Adatokat visz át a gazdagépről az eszközre. | cudaMemcpyHostToDevice-t alkalmaz. |
| copyDeviceToHost | Adatokat visz át az eszközről a gazdagépre. | cudaMemcpyDeviceToHost-ot alkalmaz. |

Kollektív műveletek

Az elosztott tanítás magja az NCCL kollektívákon alapul. A GPUCoordinator ezeket aszinkron műveletekként teszi elérhetővé, amelyek a belső cuda_stream-en hajtódnak végre.

Támogatott kollektívák

- allReduce: Adatokat kombinál az összes rangból egy redukciós operátor (Sum, Max stb.) segítségével és az eredményt visszaosztja az összes ranghoz.
- broadcast: Puffert másol egy gyökér rangból az összes többi ranghoz.
- allGather: Adatokat gyűjt az összes rangból és az összesített tömböt osztja el az összes ranghoz.
- reduceScatter: Redukciót hajt végre, majd az eredményt szétszórja a rangok között.
- barrier: Egy allReduce végrehajtásával valósul meg a belső barrier_buffer-en. Ez biztosítja, hogy az összes rang elérte ugyanazt a végrehajtási pontot.
- allReduceFloat16Avg: Átlagolt f16 redukció a súly deltákhoz.
- allReduceFloat32Max: Max redukció f32 értékekhez (max_seq_len szinkronizáláshoz).

NCCL kötések

A rendszer az NCCL megosztott könyvtárral egy vékony Zig burkolón keresztül kommunikál az nccl_bindings.zig fájlban. Ez a fájl definiálja a szükséges C-ABI típusokat és extern függvényeket.

- Eredménykódok: Az ncclResult_t enum leképezi az NCCL visszatérési kódokat, mint az ncclSuccess és az ncclUnhandledCudaError.
- Adattípusok: Zig/Futhark típusokat képez le NCCL típusokra, mint az ncclFloat32 vagy az ncclBfloat16.
- Redukciós operátorok: Definiál olyan műveleteket, mint az ncclSum, ncclProd és ncclMax.

Modal integráció

Felhő léptékű tanításhoz a ModalGPUClient és a kapcsolódó Python szkriptek orchestrálják az elosztott bináris telepítését.

- Erőforrás specifikáció: A tanítási feladatok csúcskategóriás hardverre vannak konfigurálva, kifejezetten B200 vagy B300 GPU-kat kérve.
- Környezet beállítás: A Modal image az nvidia/cuda:12.8.1-devel-ubuntu24.04 alapján épül és tartalmazza a szükséges libnccl2 és libnccl-dev könyvtárakat.
- Feladat telepítés: A deployTrainingJob függvény szerializálja a tanítási paramétereket (gpu, gpu_count, image, model_path, dataset_path, batch_size, epochs) és POST kérést hajt végre az https://api.modal.com/v1/functions/deploy végpontra Bearer <MODAL_API_TOKEN> autorizációval.
- Feladat lekérdezés: A getJobStatus 30 másodpercenként GET kérést hajt végre az https://api.modal.com/v1/functions/<job_id>/status végpontra a completed vagy failed állapotig.

---

9 BIZTONSÁG, ELLENŐRZÉS ÉS VÉDELEM

A JAIDE rendszer többrétegű biztonsági és helyességi architektúrát tartalmaz, amelyet a modell következtetés integritásának, a tanítási adatok adatvédelmének és az alapvető algoritmusok matematikai megalapozottságának biztosítására terveztek. Ez az alrendszer áthidalja az alacsony szintű memória biztonsági primitíveket a magas szintű kriptográfiai bizonyítékokkal és formális ellenőrzéssel.

Rendszer biztonsági és védelmi áttekintés

A biztonsági architektúra öt fő területre épül, amelyek mind aktívan részt vesznek minden inferencia kérésben:

- Formális ellenőrzés: src/verification/oftb.lean (statikus Lean4 bizonyítások, -Dverify=true) és src/core_relational/formal_verification.zig (futásidejű FormalVerificationEngine.verifyGraph minden inferencia kérésen).
- Biztonsági bizonyítás: src/core_relational/security_proofs.zig (SecurityProofEngine.proveInformationFlowSecurity minden inferencia kérésen).
- Nulla-Tudás bizonyítékok: VerifiedInferenceEngine és ZKInferenceProver, opcionális JAIDE_VERIFY=1 esetén.
- Adathalmaz adatvédelem: HomomorphicEncryption és DatasetFingerprint.
- Memória biztonság: safeIntCast, SecureRng, secureZeroBytes, constantTimeCompare.

Ellenőrzött Következtetési Motor

A VerifiedInferenceEngine "kötelezd el-majd-bizonyítsd" életciklust biztosít a modell végrehajtáshoz. Biztosítja, hogy az RSF (Visszafordítható Szórt Folyam) verem által generált kimenet egy specifikus bemenet és modell állapot determinisztikus eredménye, anélkül, hogy felfedné a belső súlyokat.

Főbb jellemzők:

- Elkötelezési sémák: Blake3-at alkalmaz a bemeneti/kimeneti elkötelezésekhez.
- Nyom rögzítés: Működési nyomot rögzít a következtetés során a ProofOfCorrectness segítségével.
- Skálázható ellenőrzés: BatchVerifier-t és ProofAggregator-t valósít meg Merkle fák segítségével több következtetés egyidejű ellenőrzéséhez.

Formális ellenőrzés motor

A FormalVerificationEngine futásidejű bizonyíték ellenőrzőt valósít meg a gráf invariánsokhoz. InvariantType-ot (pl. MEMORY_SAFETY, COHERENCE) és ProofRule-t (pl. MODUS_PONENS, INDUCTION) definiál a SelfSimilarRelationalGraph állapotának validálásához. A verifyGraph(graph) minden inferencia kérésben meghívódik és a következő elemekre épül:

- Proposition: Atomic, negation, binary (AND, OR, IMPLIES), quantified (FORALL, EXISTS), Hoare triple.
- FormalProof: Sorozatos ProofStep, mindegyik egy ProofRule alkalmazásával.
- Invariant: InvariantRegistry a gráfra vonatkozó tulajdonságokhoz.
- HoareLogicVerifier: Preciózus/Poszt-feltételes ellenőrzés.
- TheoremProver: Bizonyítási fa (ProofTreeNode) unifikációval és rezolúcióval.

Biztonsági bizonyítás motor

A SecurityProofEngine az alábbi bizonyítási módokat biztosítja, minden inferencia kérésben aktívan:

- Bell-LaPadula: Nem-olvasás-fel, nem-írás-le SecurityLabel szigorúan növekvő láncokkal.
- Biba: Nem-olvasás-le, nem-írás-fel integritási osztályokkal.
- Non-interference: BisimulationRelation-al bizonyítja, hogy alacsony szintű megfigyelés nem szivárogtat magas szintű információt.
- Access control: AccessControlMatrix, AccessRule, SeparationOfDutiesConstraint.
- Kriptográfiai: HashChain, MerkleTree, CommitmentScheme.

A proveInformationFlowSecurity(graph) egy SecurityProof-ot állít elő SecurityProofStep sorozatból.

Adathalmaz adatvédelem és elhomályosítás

A JAIDE kriptográfiai elhomályosítás és statisztikai adatvédelmi intézkedések kombinációján keresztül védi az érzékeny tanítási adatokat. A HomomorphicEncryption modul a Paillier kriptoszisztémát valósítja meg, lehetővé téve korlátozott aritmetikai műveleteket titkosított adatokon.

Formális ellenőrzés és biztonsági primitívek

A JAIDE megbízhatóságának alapja biztonsági primitívek készlete, amelyek megakadályozzák a szoftver általános sebezhetőségeit, mint az egész szám túlcsordulások és a mutató helytelen igazítása.

Biztonsági segédprogramok:

- safeIntCast: Validálja az előjelet és a bit szélességet az IntegerOverflow és IntegerUnderflow megelőzéséhez.
- safePtrCast: Biztosítja, hogy a mutatók nem null értékűek és helyesen igazítottak a célhoz.

SecureRng

A SecureRng struktúra hibrid megközelítést valósít meg az entrópiához. Az std.crypto.random rendszer által biztosított kriptográfiai véletlenszerűséget keveri egy Lineáris Kongruenciális Generátor (LCG) tartalék állapottal a magas minőségű véletlenszerűség biztosítása érdekében még nagy versengés esetén vagy korlátozott entrópia forrásokkal rendelkező környezetekben is.

Kriptográfiai primitívek

Az érzékeny adatkezeléshez a JAIDE biztosítja:

- secureZeroBytes: Biztosítja, hogy a memória törlésre kerüljön anélkül, hogy a fordító optimalizálná el.
- constantTimeCompare: Megakadályozza az időzítési támadásokat azáltal, hogy bájt puffereket rögzített számú ciklusban hasonlít össze.

Lean4 formális bizonyítékok

A JAIDE formális ellenőrzést alkalmaz a legkritikusabb algoritmusok helyességének bizonyítására, kifejezetten a neurális rétegben alkalmazott Ortogonális Fraktál Transzformációs Blokkhoz (OFTB). Az src/verification/oftb.lean fájl Lean4 tételeket tartalmaz, amelyek validálják a split_at művelet tulajdonságait. A -Dverify=true kapcsolóval a build.zig meghívja a lake build parancsot az src/verification/ könyvtárban, és a test-all célra is függőségként kerül.

BigInt512 aritmetika

A homomorf titkosításhoz és nagy léptékű koordináta rendszerekhez a JAIDE BigInt512 aritmetikát valósít meg a safety.zig fájlban. Ez tartalmaz konstans idejű összehasonlítást és biztonságos nullázást annak biztosítására, hogy a nagy egész szám műveletek ne szivárogtatnak ki oldalsó csatorna információkat.

QuantumTaskAdapter integráció

A biztonsági rétegbe illeszkedik a QuantumTaskAdapter is, amely minden inferencia kérésben végrehajtja a kvantum szubgráf ciklust. A local_simulator zajmodellezéssel dolgozik, és az applyResultsToGraph csak akkor módosítja a gráfot, ha a task_result.success igaz — ez megelőzi a hibás kvantum eredmények szivárgását a gráfba.

---

10 TESZTELÉS ÉS BENCHMARKING

A JAIDE kódbázis átfogó teljesítmény benchmark és stressz teszt csomagot tartalmaz, amelyet a mag matematikai és relációs alrendszerek hatékonyságának és helyességének validálására terveztek. Ez az infrastruktúra biztosítja, hogy az optimalizálások - mint a SIMD vektorizáció, a többszálú mátrixszorzás és a lock-free referenciaszámlálás - stabil és teljesítő maradjanak az architektúrális változások során.

Magas szintű teszt architektúra

A tesztelési infrastruktúra négy elsődleges kategóriára van osztva:

1. Teljesítmény benchmarkok: Dedikált futtatható fájlok, amelyek mérik az áteresztőképességet (GFLOPS, elemek/mp) a kritikus útvonalakon, mint az RSF és a Tenzor műveletek.
2. Stressz tesztek: Nagy párhuzamossági környezetek, amelyek versenyhelyzeteket keresnek a memóriakezelésben és a referenciaszámlálásban.
3. Egységtesztek: Build rendszerbe integrált tesztek az alrendszerek logikájának validálásához, mint az NSIR, CREV, VPU, FNDS, formális ellenőrzés, biztonsági bizonyítás, kvantum feladat és jelterjedés.
4. C API test: A src/tests/c_api_test.c fájl az ABI szintű ellenőrzést végzi (int64 roundtrip, double roundtrip, hash determinizmus, memória allokáció) a jaide-c-api-test futtatható fájlon keresztül.

A benchmark csomag egy központi függőségi modulra támaszkodik, az src/_bench_deps.zig-re, amely belső névtereket (rsf, core_tensor, sfd) tesz elérhetővé a tesztelési futtatók számára.

Benchmark csomag

A teljesítmény csomag értékeli a rendszer neurális és matematikai primitíveinek számítási korlátait.

- RSF áteresztőképesség: A bench_rsf méri az elemek-per-másodperc feldolgozást a Visszafordítható Szórt Folyam modell előre és visszafelé irányuló menetei során. Ellenőrzi a verem matematikai invertálhatóságát is.
- Lineáris algebra: A bench_matmul benchmarkol a csempézett, gyorsítótár-barát mátrixszorzást (i-p-j ciklus sorrend) változó mátrix méreteken (128-tól 1024-ig), GFLOPS-ban jelenti a teljesítményt.
- SIMD műveletek: A bench_tensor_ops az elemenként végzett sávszélesség kihasználásra összpontosít a fill, add és mul műveleteknél nagy folytonos memória blokkokra (4M elem), GB/s-ban jelenti az eredményeket.
- Optimalizálási sebesség: A bench_sfd a Spektrális Fisher Diagonalizáló FP4 kvantálását és spektrális normalizálását benchmarkolja.

---

10.1 TELJESÍTMÉNY BENCHMARKOK

RSF áteresztőképesség (bench_rsf)

A bench_rsf segédprogram értékeli a Visszafordítható Szórt Folyam neurális verem teljesítményét. Több méret variánson iterál, mérve az előre és visszafelé irányuló menet késleltetést és áteresztőképességet.

- Előre irányuló menet: Az affin csatolást és az OFTB keverést méri.
- Visszafelé irányuló menet: A visszafordíthatóság alapú gradiens rekonstrukciót méri.
- Invertálhatóság: Ellenőrzi a forward→inverse roundtripet 1e-4 tűréssel a numerikus stabilitás biztosításához.

Mátrixszorzás (bench_matmul)

A bench_matmul segédprogram értékeli a csempézett, gyorsítótár-barát i-p-j mátrixszorzás implementáció teljesítményét a Tensor osztályban.

Teljesítmény mérőszámok

A benchmark 128, 256, 512 és 1024 méretű négyzetes mátrixokon iterál. Minden mérethez kiszámítja:

- Teljes idő: Kumulatív idő 100 iterációhoz.
- Iterációnkénti: Átlagos késleltetés matmul hívásonként.
- Áteresztőképesség (GFLOPS): 2.0 × N^3 × iterációk / másodpercek képlettel számítva.

Tenzor elemenként végzett műveletek (bench_tensor_ops)

Ez a benchmark a SIMD-vektorizált elemenként végzett műveletekre összpontosít nagy folytonos memória blokkokra (4M elem). Méri a Tensor implementáció memória sávszélesség kihasználását.

Értékelt műveletek

| Függvény | Leírás |
| :--- | :--- |
| benchFill | Méri a t.fill(val) sebességét és GB/s sávszélességét. |
| benchAdd | Méri az a.add(&b) elemenként végzett összeadást. |
| benchMul | Méri az a.mul(&b) elemenként végzett szorzást. |

SFD optimalizáló primitívek (bench_sfd)

A bench_sfd benchmark a Spektrális Fisher Diagonalizáló által alkalmazott specifikus matematikai kerneleket célozza, kifejezetten az FP4 kvantálást és a Spektrális Normalizálást.

FP4 kvantálás

A benchmark teszteli a quantizeFP4 logikát, amely értékeket vág [-6.0, 6.0] tartományra és diszkrét 4 bites lebegőpontos reprezentációra képezi le azokat. 1M értéket dolgoz fel 100 iteráción keresztül az elemenkénti nanoszekundum meghatározásához.

Spektrális normalizálás

Értékeli a SpectralNormalizer.normalizeWeights függvényt. A benchmark összehasonlítja:

1. Teljes hatványiterációk: 20 iteráció a nagy pontosságú szinguláris érték becsléshez.
2. Ritka hatványiterációk: 5 iteráció a tanítás során végzett gyors közelítéshez.

---

10.2 STRESSZ TESZTEK ÉS EGYSÉGTESZTEK

A JAIDE tesztelési infrastruktúra biztosítja a rendszer matematikai helyességét, memória biztonságát és párhuzamos stabilitását. Ez az oldal részletezi a párhuzamos referenciaszámlálás speciális stressz tesztjeit és a Zig build rendszerben definiált egységtesztek csomagját a mag relációs és neurális komponensekhez.

1. Stressz teszt: stress-refcount

A stress-refcount build lépés a src/tests/stress_tensor_refcount.zig fájlt futtatja a Tensor referenciaszámlálási mechanizmus szálbiztonságának validálásához. Mivel a JAIDE Másolás-íráskor (CoW) szemantikára és megosztott memóriára támaszkodik több szálon keresztül (pl. matmul vagy elosztott tanítás során), a retain() és release() atomi integritása kritikus.

Implementációs részletek

A teszt több szálat indít, amelyek egyidejűleg véletlenszerű referencia műveleteket hajtanak végre egy megosztott Tensor objektum készleten.

- Szinkronizálás: Egy std.atomic.Value(usize) barrier biztosítja, hogy az összes szál egyidejűleg kezdje el a műveleteket a versengés maximalizálásához.
- Munkaterhelés: Minden threadWorker konfigurálható számú műveletet hajt végre (ops_per_thread). A műveletek tartalmazzák az egyszeres retain-eket, dupla retain-eket és több tenzoros retain-eket a komplex adatfolyamok szimulálásához.
- Ellenőrzés: Miután az összes szál csatlakozik, a teszt ellenőrzi, hogy minden tenzor végső referenciaszámlálója pontosan 1-re tért vissza (az eredeti tulajdonosi referencia).

Referenciaszámlálás stressz teszt adatfolyam

| Rendszer fogalom | Kód entitás |
| :--- | :--- |
| Párhuzamos munkás | threadWorker |
| Atomi barrier | std.atomic.Value(usize) |
| Referencia növelés | Tensor.retain() |
| Referencia csökkentés | Tensor.release() |
| Biztonsági ellenőrzés | getRefcount |

2. Build rendszer egységtesztek

A JAIDE a Zig build rendszert alkalmazza moduláris teszt lépések definiálásához. Ezek egyenként vagy összesítve futtathatók a test-all lépésen keresztül.

2.1 Mag relációs tesztek

Ezek a tesztek validálják az NSIR (Önhasonló Relációs Gráf) és az érvelési folyamatok integritását.

| Teszt lépés | Célmodul | Validálási hatókör |
| :--- | :--- | :--- |
| test-nsir | nsir_core.zig | Csomópont/él létrehozás, kvantum kapu alkalmazás és topológia hashelés. |
| test-reasoning | reasoning_orchestrator.zig | Energia számítás, állapot pillanatképek és ESSO szimmetria észlelés. |
| test-crev | crev_pipeline.zig | Oksági érvelés, hármas kivonás és validálási láncok. |
| test-temporal | temporal_graph.zig | QuantumState pillanatképek és idősor gráf evolúció. |
| test-surprise | surprise_memory.zig | Jaccard-disszimilaritás szűrés és CAS elkötelezési küszöbök. |
| test-vpu | vpu.zig | SIMD vektor típusok, gráf beágyazás, kvantum vektor ops, hasonlósági mátrix. |
| test-fnds | fnds.zig | Fraktál fa, önhasonló index, mintakeresés, PatternLocation életciklus. |
| test-formal | formal_verification.zig | Invariáns, Hoare-triplet, tétel bizonyítás, unifikáció. |
| test-security | security_proofs.zig | Bell-LaPadula, Biba, nem-interferencia, hozzáférés vezérlés. |
| test-quantum-adapter | quantum_task_adapter.zig | Szubgráf azonosítás, feladat végrehajtás, eredmény visszaírás. |
| test-signal | signal_propagation.zig | Jelterjedés, aktivációs nyomkövetés, inferencia hookok. |

2.2 Neurális és memória tesztek

Ezek validálják az alapvető matematikai és memóriakezelési primitíveket.

- test-tensor: Validálja a Tensor alak/lépés elrendezést, a SIMD-vektorizált elemenként végzett műveleteket és a bináris szerializációs formátumot.
- test-memory: Validálja a speciális allokátorokat, beleértve az ArenaAllocator-t, SlabAllocator-t és BuddyAllocator-t a töredezettség és teljesítmény szempontjából.
- test-rsf: Validálja az RSFLayer affin csatolást (skála S és fordítás T) és az előre/inverz menet visszafordíthatóságát.
- test-oftb: Validálja az Ortogonális Fraktál Transzformációs Blokk pillangó stílusú keverési transzformációit.
- test-embedding: Validálja a LearnedEmbedding előre/visszafelé menet és gradiens propagációt.

2.3 C API és stressz

- test-c-api: Lefordítja és futtatja a src/tests/c_api_test.c fájlt, amely int64/double roundtrip-ot, ABI elrendezést, hash determinizmust és malloc/memset ellenőrzést végez.
- stress-refcount: A fentebb ismertetett Tensor referenciaszámláló stressz teszt.

3. Teszt végrehajtás és konfiguráció

Tesztek futtatása

A tesztek a zig build paranccsal hajthatók végre. A felhasználók specifikus alrendszereket vagy a teljes csomagot célozhatják:

zig build test-all

zig build test-tensor
zig build test-nsir
zig build test-vpu
zig build test-fnds
zig build test-formal
zig build test-security
zig build test-quantum-adapter
zig build test-signal
zig build test-c-api
zig build stress-refcount

zig build test-rsf -Dgpu=true
zig build test-all -Dverify=true

Optimalizálási statisztikák

A relációs optimalizálási tesztek során (pl. ESSO) a rendszer OptimizationStatistics-t követ nyomon a sztochasztikus folyamatok helyes konvergenciájának biztosítása érdekében.

Optimalizálási mérőszámok követése:

- iterations_completed
- moves_accepted
- best_energy
- temperature

4. Hibakezelés a tesztekben

A teszt csomag szabványosított C-kompatibilis hibakódok készletét alkalmazza a c_api-ban definiálva, biztosítva, hogy a mag relációs réteg meghibásodásai nagy granularitással kerüljenek jelentésre.

| Hibakód | Jelentés |
| :--- | :--- |
| JAIDE_ERROR_ALLOCATION | Memória meghibásodás a speciális allokátorokban. |
| JAIDE_ERROR_NODE_NOT_FOUND | NSIR gráf keresési meghibásodás. |
| JAIDE_ERROR_MATH_ERROR | Túlcsordulás vagy alulcsordulás a neurális/kvantum műveletekben. |
| JAIDE_ERROR_THREADING | Mutex versengés vagy atomi meghibásodás. |
| JAIDE_ERROR_INVALID_STATE | Utólagos aktiválás nélküli mező elérése (a jelenlegi kódbázisban nem fordulhat elő, mert minden mező konstruktorban aktív). |

---

11 SZÓJEGYZÉK

Ez az oldal technikai definíciókat és kód-specifikus mutatókat biztosít a JAIDE rendszer architektúrális komponenseihez, matematikai primitívjeihez és kognitív fogalmaihoz.

1. Architektúrális paradigmák

5. gyök architektúra

A JAIDE alapvető paradigmája, amely a Perceptron, CNN, RNN és Transformer után következik. A Visszafordítható Szórt Folyam (RSF) segítségével valósul meg, amely a bijektivitást és az O(dim) memória komplexitást helyezi előtérbe.

RSF (Visszafordítható Szórt Folyam)

Kereszt-affin csatoló rétegekből és determinisztikus szórt permutációkból álló neurális architektúra. Minden réteg bijektív, lehetővé téve az aktivációk pontos inverz rekonstrukcióját a visszafelé irányuló menet során aktiváció gyorsítótár nélkül.

- Implementáció: LayerCore az src/processor/rsf.zig fájlban.
- Matematikai forma:
  - Előre: y1 = x1 ⊙ exp(clip(Ws · x2 + bs))
  - Inverz: x2 = y2 - Wt · y1 - bt

Mag Relációs Réteg

A JAIDE kognitív alrendszere, amely magas szintű érvelést, gráf alapú tudásreprezentációt, vektorprocesszort, fraktál indexelést, formális ellenőrzést, biztonsági bizonyításokat és kvantum-inspirált optimalizálást kezel. Minden komponense minden tanítási lépésben és minden inferencia kérésben aktív; nincs olyan mező, amely null-ként inicializálódik utólagos aktiválásra várva.

2. Neurális tér és kód entitás leképezés

Az RSF feldolgozási folyamat:

A felhasználói prompt (karakterlánc) a MorphoGraphTokenizer (mgt.zig) segítségével tokenizálódik, majd a LearnedEmbedding (learned_embedding.zig) beágyazásokat végez, az RSF modell (rsf.zig) feldolgozza, az OFTB (oftb.zig) szórást/gyűjtést végez, majd az inverseInPlace() aktiváció rekonstrukciót hajt végre. A GradientFlowController (sfd.zig) initWithConfig útján normavágást biztosít a beágyazás visszaterjesztéskor.

3. Mag terminológia táblázat

| Kifejezés | Definíció | Kód mutató |
| :--- | :--- | :--- |
| NSIR (SSRG) | Önhasonló Relációs Gráf. Egy gráf, ahol az élek kvantum-inspirált korrelációkat képviselnek a tokenek között. | src/core_relational/nsir_core.zig |
| EdgeQuality | Enum, amely meghatározza egy gráf él állapotát: szuperpozíció, összefonódott, koherens, összeomlott vagy fraktál. | src/core_relational/nsir_core.zig |
| OFTB | Ortogonális Fraktál Transzformációs Blokk. Paraméter nélküli Haar-wavelet alapú keverési réteg O(1) memóriával. | src/processor/rsf.zig |
| SFD | Spektrális Fisher Diagonalizáló. Másodrendű optimalizáló, amely Fisher információs mátrix átló becslést alkalmaz. | src/optimizer/sfd.zig |
| GradientFlowController | Gradiens folyam stabilizáló, initWithConfig(GradientFlowConfig)-al inicializálva; manuális L2-norma vágást biztosít a beágyazás gradiensekre. | src/optimizer/sfd.zig |
| SSI | Önhasonló Index. Pozíció-megőrző külső memória struktúra, amely O(log n) visszakeresést tesz lehetővé. | src/index/ssi.zig |
| ESSO | Összefonódott Sztochasztikus Szimmetria Optimalizáló. Gráf topológiát optimalizál szimulált hűtéssel a szimmetriákon. | src/core_relational/reasoning_orchestrator.zig |
| Qubit | Komplex értékű primitív (Complex(f64)), amelyet a csomópont állapotok reprezentálásához alkalmaznak az NSIR gráfban. | src/core_relational/nsir_core.zig |
| ThoughtLevel | Hierarchikus érvelési fázisok: helyi (token szintű), globális (kontextus szintű) és meta (rendszer szintű). | src/core_relational/reasoning_orchestrator.zig |
| VPU | Vektorprocesszor. F64x4 gráf beágyazások, kvantum vektor operátorok és hasonlósági mátrix számítás. | src/core_relational/vpu.zig |
| FNDSManager | Fraktál Neurális Dinamikus Rendszer kezelő. Fraktál fák, önhasonló indexek és PatternLocation regisztráció. | src/core_relational/fnds.zig |
| PatternLocation | Fraktál fába szúrt minta lokalizációja: tree_id, level, node_id, offset, length, confidence. | src/core_relational/fnds.zig |
| CREVPipeline | Oksági Érvelés és Ellenőrzés folyamat. processTextStream a bemeneti szöveg oksági kivonásához. | src/core_relational/crev_pipeline.zig |
| FormalVerificationEngine | Invariáns nyilvántartás, Hoare-logika, tétel bizonyítás. verifyGraph az inferencia kérésekben. | src/core_relational/formal_verification.zig |
| SecurityProofEngine | Bell-LaPadula, Biba, nem-interferencia, hozzáférés vezérlés. proveInformationFlowSecurity az inferencia kérésekben. | src/core_relational/security_proofs.zig |
| QuantumTaskAdapter | Kvantum szubgráf azonosítás, végrehajtás és eredmény visszaírás. | src/core_relational/quantum_task_adapter.zig |
| SignalPropagationEngine | Nem opcionális mező a trainerben; jelterjedés minden runCoreRelationalPass ciklusban. | src/core_relational/signal_propagation.zig |
| TrainerComponents | {tokenizer, signal_engine, embedding_accel} — átadható a DistributedTrainerFuthark.initWithComponents-nek. | src/distributed/distributed_trainer_futhark.zig |
| ModalGPUClient | HTTP klient a Modal API-hoz, deployTrainingJob és getJobStatus. | src/distributed/modal_gpu.zig |

4. Alrendszer specifikus fogalmak

Memóriakezelési primitívek

- MemoryBlockState: Meghatározza egy memória blokk életciklusát: szabad, allokált, összefonódott vagy migrálódó.
- PinnedMemory: cudaHostAlloc segítségével allokált memória a nagy sebességű gazdagép-eszköz átvitelek megkönnyítéséhez.

Kriptográfia és ellenőrzés

- HomomorphicEncryption: A Paillier kriptoszisztéma implementációja additív homomorf műveletekhez érzékeny adathalmazokon.
- ZKProofBundle: Tároló a Groth16 bizonyítékokhoz, nyilvános jelekhez és ellenőrzési állapothoz a nulla-tudás következtetéshez.
- Groth16Proof: pi_a (G1), pi_b (G2), pi_c (G1) pontok a bn128 görbén.

Hardver gyorsítás

- WeightKind: Súlytípusok felsorolása (pl. weights_s, weights_t, velocity_s), amelyeket a Futhark/CUDA gyorsítói interfész alkalmaz.
- FutharkContext: Kezeli a Futhark GPU futtatókörnyezet életciklusát, beleértve az eszköz kiválasztást és a parancs szinkronizálást.
- futhark_cpu_step / futhark_gpu_step: build.zig rendszerparancs lépések, amelyek meghívják a futhark c és futhark opencl parancsokat.
- librtl_sim.so: Haskell modulokból (MemoryArbiter, RankerCore, SSISearch) GHC-vel fordított megosztott könyvtár -Drtl=true esetén.
- jaide-rtl-sim: Zig-alapú RTL szimulátor futtatható fájl, cycles/banks/requests_per_cycle argumentumokkal.
- jaide-c-api-test: C-alapú ABI smoketest futtatható fájl.

5. Build opciók összefoglalója

| Opció | Alapértelmezett | Aktivált artifaktumok |
| :--- | :--- | :--- |
| -Dgpu=true | false | jaide-distributed-futhark + CUDA/NCCL linkelés + futhark_gpu_step |
| -Dzk=true | false | circom + snarkjs pipeline (R1CS, zkey, verification_key.json) |
| -Dverify=true | false | lake build az src/verification/-ben; test-all függősége |
| -Drtl=true | false | GHC librtl_sim.so + jaide-rtl-sim |

6. Környezeti változók

| Változó | Cél |
| :--- | :--- |
| JAIDE_API_KEY | Kötelező, ha require_api_key=true; a Bearer token értéke az inferencia szerverhez. |
| JAIDE_MODEL_PATH | A modell fájl útvonala az inferencia szerverhez. |
| JAIDE_VERIFY | Ha "1", engedélyezi a VerifiedInferenceEngine-t. |
| JAIDE_REASONING_CYCLES | Felülbírálja az alapértelmezett 50 belső iterációs számot. |
| JAIDE_MODEL_DIM, JAIDE_LAYERS, JAIDE_BATCH_SIZE | Modell dimenzió, réteg szám és köteg méret a trainerhez. |
| JAIDE_EPOCHS, JAIDE_LEARNING_RATE | Tanítási hiperparaméterek. |
| JAIDE_DATASET | JSONL dataset útvonala. |
| JAIDE_TOTAL_SAMPLES, JAIDE_MAX_SAMPLES | Dataset partícionálás. |
| JAIDE_MAX_SEQ_LEN | Maximális szekvencia hossz per batch (alapértelmezett 256). |
| JAIDE_NCCL_ID_PATH | NCCL egyedi azonosító megosztás útvonala. |
| WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT | Elosztott tanítás rangkonfigurációja. |
| MODAL_API_TOKEN | A jaide-distributed-futhark --deploy módhoz szükséges Modal API token. |
