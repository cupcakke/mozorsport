# JAIDE

JAIDE is a foundational language model built upon the 5th root architecture paradigm. Unlike the Perceptron, CNN, RNN, and Transformer paradigms, it employs a Reversible Scattered Flow (RSF) stack, where every layer is bijective and invertible. The base operation of the layer is a bijective, invertible cross-affine coupling (with scale and translation components, deterministic scattered permutations), rather than an affine transformation of the form σ(W·x + b). This provides O(dim) memory complexity during backpropagation, as activations can be reconstructed on the fly instead of being stored.

Built upon the reversible neural backbone is a Core Relational Layer, which employs quantum-inspired relational graphs and fractal dynamics for reasoning. The entire stack (VPU, FNDSManager, CREVPipeline, FormalVerificationEngine, SecurityProofEngine, QuantumTaskAdapter) directly participates in every training step and every inference request. No single component is initialized as null waiting for post-activation.

## Neural core: RSF

The core of JAIDE's neural processing is the RSFLayer (src/processor/rsf.zig), consisting of cross-affine coupling layers and deterministic scattered permutations. In contrast to Transformers' O(L · S · d) memory scaling, JAIDE maintains a fixed memory footprint regardless of depth L.

The complete RSF stack runs through on CPU (forward + backward + optimizer), the test suite is green, the benchmarks execute successfully. The backward pass does not store activations, but reconstructs them with the invertible primitive. The forward→inverse roundtrip passes with a 1e-4 tolerance, which verifies the layer's bijectivity. Detailed data can be found in the BENCHMARKS.md file.

The affine coupling mechanism of the RSFLayer works as follows: the input is split into x1 and x2, x2 is transformed using s = exp(clip(x1·Ws + bs)) and t = x1·Wt + bt, resulting in y2 = x2⊙s + t, while y1 = x1 remains unchanged. Outputs pass through OFTB.forwardInPlace. Biases are fused into the weight matrices via homogeneous coordinates: the shape of each weight matrix is [dim × (dim+1)], with the last column storing the absorbed bias. The layer's parameter count is thus dim² + dim.

The main neural components: the RSF layer implements forwardInPlace and inverseInPlace operations, the OFTB (Orthogonal Fractal Transformation Block) is a parameter-less, deterministic Haar-wavelet scatter/gather layer with a fixed FRACTAL_SCALE (approx. 0.7071), the SFD (Spectral Fisher Diagonalizer) is a second-order optimizer that approximates the diagonal Fisher information matrix with spectral clipping, and the GradientFlowController initializes via the initWithConfig(.{ .gradient_clip_norm, .use_normalized_gradient_flow, .spectral_power_iterations }) call and applies manual L2-norm clipping to the embedding gradients.

Binary serialization uses the SAVE_VERSION = 5 format with CRC32 integrity checking. Per-layer data: dim, num_layers, clip_min, clip_max metadata, followed by s_weight and t_weight tensors for each layer in [dim × (dim+1)] shape.

## Core Relational Layer

JAIDE maintains a Self-Similar Relational Graph (SSRG / NSIR) in the src/core_relational/nsir_core.zig file, which stores relationships between tokens as a sparse graph with EdgeQuality states (superposition, entangled, coherent, collapsed, fractal). Every node in the graph contains a Qubit structure with std.math.Complex(f64) amplitudes, supports Hadamard and CNOT gates, and generates an SHA-256 hash from the structure using the calculateTopologyHash function. The first two bytes of the topology_hash (hash[0], hash[1]) derive the θ and φ angles for VPU's quantumVectorOps: theta = hash[0]/255.0 · π and phi = hash[1]/255.0 · π.

The ReasoningOrchestrator (src/core_relational/reasoning_orchestrator.zig) manages a three-level hierarchy: local (node-neighborhoods), global (complete graph topology), and meta (reasoning history). The ESSO (EntangledStochasticSymmetryOptimizer) refines the graph topology with simulated annealing and symmetry perturbations. The runHierarchicalReasoning executes 50 inner iterations. Convergence: delta = |current - previous| / max(|previous|, 1.0), and if delta < convergence_threshold, the phase stops.

Through processTextStream, the CREVPipeline extracts RelationalTriplet structures and performs causal validation across discrete stages (tokenization, triplet_extraction, validation, integration, indexing). The ZRuntime (src/core_relational/z_runtime.zig) handles quantum-relational variables, where the trainer's runCoreRelationalPass creates a train_<global_step> variable, and the inference's handleInference creates an inf_<request_count> variable. Relational operators (AND, OR, XOR, ENTANGLE) map directly to quantum gates.

The VPU (src/core_relational/vpu.zig) provides SIMD vector types (F32x4, F32x8, F64x2, F64x4, I32x4, I32x8), and in the runCoreRelationalPass path, computeGraphEmbeddings produces F64x4 vectors, quantumVectorOps applies quantum operators, and computeSimilarityMatrix calculates a similarity matrix whose first cell directly modulates the learning_rate in training (upper bound 0.1). On the inference path, the θ and φ angles are derived from the request_count % 314 and % 628 values.

The FNDSManager (src/core_relational/fnds.zig) manages fractal trees and self-similar indices. The createTree(max_depth, branching_factor) returns a [32]u8 tree_id, insertIntoTree(node_id, data, level) returns !bool. In every training step, the system runs createTree(6, 4) for token lists and createTree(4, 3) on the buildKnowledgeGraph path. In every inference request, createTree(4, 3) runs on the input tensor bytes and a pattern index is created under the name inference_patterns. PatternLocation.init(allocator, tree_id, level, node_id, offset, length, confidence) is created via allocation; the confidence falls within the [0.0, 1.0] range, otherwise it returns an InvalidConfidence error.

The FormalVerificationEngine (src/core_relational/formal_verification.zig) checks structural invariants via verifyGraph based on InvariantType (MEMORY_SAFETY, COHERENCE) and ProofRule (MODUS_PONENS, INDUCTION). In the inference server, a heap-allocated *FormalVerificationEngine field is stored. The SecurityProofEngine (src/core_relational/security_proofs.zig) constructs Bell-LaPadula, Biba, access control, and non-interference bisimulation proofs via proveInformationFlowSecurity. The QuantumTaskAdapter (src/core_relational/quantum_task_adapter.zig) runs a three-phase cycle: identifyQuantumSubgraphs → executeQuantumTask → applyResultsToGraph, where applyResultsToGraph modifies the graph only upon task_result.success.

The ChaosCoreKernel implements Content-Addressable Storage (CAS) with 16-byte block_id and 16-byte content_hash identification, a MemoryBlockState state machine (free, allocated, entangled, migrating), and a DynamicTaskScheduler. Load balancing operates with the OPTIMIZATION_THRESHOLD = 0.6 and BALANCE_INTERVAL_CYCLES = 100 constants.

## Subsystem overview

File locations of the entire stack: the numerical core under src/core/tensor.zig and src/core/memory.zig, the neural stack under src/processor/rsf.zig and src/processor/oftb.zig, the optimizer under src/optimizer/sfd.zig, the relational layer under src/core_relational/ (nsir_core, reasoning_orchestrator, vpu, fnds, crev_pipeline, formal_verification, security_proofs, quantum_task_adapter, signal_propagation, z_runtime), distributed training under src/distributed/, hardware RTL under src/hw/rtl/, serving under src/index/ssi.zig and src/api/inference_server.zig, the C API under src/core_relational/c_api.zig, the ZK circuit under src/zk/inference_trace.circom, and the formal Lean4 proofs under src/verification/oftb.lean.

## Build system

The build system is managed by build.zig with four flags. The gpu (default false) enables CUDA acceleration and jaide-distributed-futhark compilation. The zk (false) runs the Circom and snarkjs pipeline, exporting verification_key.json. The verify (false) executes the Lean4 lake build and is a dependency of test-all. The rtl (false) compiles Haskell modules into a librtl_sim.so shared library and produces the jaide-rtl-sim executable. These propagate as gpu_acceleration, zk_enabled, verify_enabled, and rtl_enabled fields.

The toolchain requirements: Zig 0.14.0, Futhark for the futhark c and futhark opencl commands, system C compiler, optionally CUDA Toolkit (cuda, cudart, nvrtc, nccl at /usr/local/cuda/lib64 paths, when -Dgpu=true), optionally Circom + snarkjs (when -Dzk=true, pot12_final.ptau based Groth16 setup), optionally Lake + Lean4 (when -Dverify=true, src/verification/ directory), optionally GHC (when -Drtl=true).

Futhark kernel generation happens in two steps: futhark_cpu_step runs the futhark c --library src/hw/accel/futhark_kernels.fut -o src/hw/accel/futhark_kernels command, and every Zig executable that links futhark_kernels.c carries a step.dependOn(&futhark_cpu_step.step) dependency. The futhark_gpu_step runs the futhark opencl --library src/hw/accel/main.fut -o src/hw/accel/main_gpu command, and jaide-distributed-futhark also depends on this when -Dgpu=true.

The ZK pipeline runs three chained commands: circom src/zk/inference_trace.circom --r1cs --wasm --sym -o src/zk/, then snarkjs groth16 setup src/zk/inference_trace.r1cs pot12_final.ptau src/zk/inference_trace.zkey, and finally snarkjs zkey export verificationkey src/zk/inference_trace.zkey src/zk/verification_key.json. The Lean4 verification runs in the src/verification/ working directory via lake build. Haskell compilation uses the ghc -O2 -dynamic -shared -fPIC src/hw/rtl/MemoryArbiter.hs src/hw/rtl/RankerCore.hs src/hw/rtl/SSISearch.hs -o src/hw/rtl/librtl_sim.so command.

The primary executables: jaide-inference-server (src/inference_server_main.zig, HTTP interface, with futhark_cpu_step dependency), jaide-distributed-futhark (src/main_distributed_futhark.zig, CUDA/NCCL linking, --deploy <model_path> <dataset_path> command with ModalGPUClient), jaide-rtl-sim (cycles banks requests_per_cycle arguments), jaide-c-api-test (C ABI smoketest).

Benchmarking is accessible from the src/_bench_deps.zig file with rsf, core_tensor, and sfd namespaces. Unit tests can be run with the test-tensor, test-memory, test-rsf, test-oftb, test-embedding, test-nsir, test-reasoning, test-crev, test-surprise, test-temporal, test-vpu, test-fnds, test-formal, test-security, test-quantum-adapter, test-signal, stress-refcount, test-c-api, and test-all (which also invokes the lake build when -Dverify=true) commands.

## Architecture

JAIDE consists of two domains: the Neural Processing Layer (RSF) and the Core Relational Layer. The RSF layers are bijective, thus the backward pass reconstructs activations from the outputs, the memory load is O(dim), independent of the number of L layers.

The neural-relational bridge works as follows: the RSF forward pass output is a tensor byte sequence that runs through nsir_graph.encodeInformation and updates the topology_hash. The VPU computeGraphEmbeddings produces F64x4 vectors, the quantumVectorOps operates with angles derived from the first two bytes of the topology_hash, and the first cell of computeSimilarityMatrix modulates the learning_rate in training. The FNDSManager creates a fractal tree, inserts tokens, builds a pattern index, and registers PatternLocations. The ReasoningOrchestrator runs hierarchical reasoning with 50 internal cycles. The SurpriseMemoryManager records high-surprise-value patterns with CAS thresholds based on Jaccard-dissimilarity. The TemporalGraph works with nanosecond-precision addNodeAtTime and advanceTime calls. The SignalPropagationEngine executes propagateStep, and the ZRuntime creates a variable for the global step. On the inference path, the CREVPipeline processTextStream, FormalVerificationEngine verifyGraph, SecurityProofEngine proveInformationFlowSecurity, and the full cycle of the QuantumTaskAdapter are added to this.

The ten active relational components (NSIR, ReasoningOrchestrator, ChaosCoreKernel, CREVPipeline, ZRuntime, SignalPropagationEngine, VPU, FNDSManager, FormalVerificationEngine, SecurityProofEngine, QuantumTaskAdapter, SurpriseMemoryManager, TemporalGraph, R-GPU) participate in every cycle of both inference and training. In the initWithComponents path of DistributedTrainerFuthark, every field is bound within the constructor; the signal_engine is not optional, but always points to a valid SignalPropagationEngine.

## Numerical core

The Tensor structure (src/core/tensor.zig) contains an allocator, a data buffer, and a Shape structure. The Shape stores metadata, dimensions, and strides, the system supports broadcasting for compatible dimensions. Tensors are aligned to 32-byte boundaries for AVX/SIMD support, the default vector width is 8 (f32 elements). The Copy-on-Write mechanism employs atomic reference counting: retain() increments the counter and sets the cow flag, release() decrements and frees at zero, and ensureWritable() checks the cow flag and makes a fresh copy before mutation.

Matmul orchestrates multiplication, validates dimensions, and selects the optimal path. MatmulComptime uses inline loops for small, fixed dimensions (M, K, N). The system also provides determinant, inverse, and transposition for the Jacobian calculations of the RSF layers. The binary format (JAIDE40) operates with a magic header (4 bytes), rank (u32), dimensions (N u64), strides (N u64), and a data buffer.

Among the memory allocators, Arena is a fixed-size thread-safe linear allocator with a reset() function, ArenaAllocator is flexible and growing with the std Allocator interface, SlabAllocator splits large slabs into smaller chunks, PoolAllocator provides O(1) allocation for uniformly sized objects, BuddyAllocator operates on powers of two, PageAllocator communicates with the operating system (16KB on macOS ARM, 4KB otherwise), and TrackingAllocator monitors memory usage.

Synchronization primitives: SpinLock for a low-overhead lock, ReadWriteLock with multiple readers and exclusive writers, LockFreeQueue as a multi-producer multi-consumer queue for the DynamicTaskScheduler, LockFreeStack for the free lists of the PoolAllocator. The global MemoryStats includes the allocated_bytes, peak_usage, fragmentation_ratio, and page_faults fields. EncryptedBlob provides an abstraction for data encrypted at rest in memory, secureZeroMemory ensures that sensitive data is physically erased. Arena and ArenaAllocator support the secureDeinit and secureReset methods.

I/O primitives: MMAP memory-mapped file access with SHARED and PRIVATE modes, based on std.posix.mmap, protected by std.Thread.Mutex, with automatic resizing to IoConfig.PAGE_SIZE (4KB). DurableWriter employs a "write-then-rename" pattern to prevent data corruption: it writes to a temporary file, then swaps the target with std.fs.Dir.rename. BufferedReader and BufferedWriter operate with an 8KB default BUFFER_SIZE.

The JAIDE40 format: 8-byte magic header, 4-byte version (1), 4-byte metadata length, JSON metadata, component blocks (RSF, Ranker, MGT, Embedding), and at the end a 32-byte SHA-256 checksum. ModelFormat.save serializes the ModelMetadata (rsf_layers, mgt_vocab_size) into JSON after the magic bytes and version, then writes the components in length-prefixed blobs, and passes the data stream through an Sha256 hasher. LearnedEmbedding (JEMB) stores the 0x4A454D42 magic number, vocab_size, dim, and f32 weights. The persistence of the NSIR graph generates an SHA-256 hash with the computeTopologyHash function, and saves Qubit states, EdgeQuality, and weight tensors in the checkpoint_version=7 format.

## Hardware acceleration

The Futhark kernels are located in the src/hw/accel/futhark_kernels.fut (CPU, futhark c) and src/hw/accel/main.fut (GPU, futhark opencl) files. src/hw/accel/futhark_bindings.zig provides extern declarations for the struct_futhark_f16_2d, struct_futhark_f16_3d types, and the futhark_entry_batch_forward, futhark_entry_batch_oftb_forward, futhark_entry_batch_gradients_full functions.

FractalLPU's mapNode(node_hash, weight) function maps node hashes to tiles, and balanceAllTiles rebalances the load. In every request, the inference server handleInference iterates through the NSIR nodes. The RelationalGraphProcessingUnit distributeGraph function distributes the graph to physical cores via an asynchronous NoC.

The Haskell RTL modules are written in Clash HDL. MemoryArbiter is a fixed-priority Mealy machine with 4 clients (NumClients) and a 4-cycle ServiceCycles window, with two states (ArbIdle, ArbServing). RankerCore performs position-biased scoring with a positionBiasScale = 1000 constant and 1/(position+1) reciprocal scaling. SSISearch operates with a tri-state FSM (Idle, Fetching, Comparing), with a MaxSearchDepthConfig = 64 depth limit. The output of jaide-rtl-sim: memory arbiter statistics (grant ratio, avg/max latency, bank pressure), RankerCore statistics (top, median score), SSISearch statistics (probes, hits, hit ratio).

## Distributed training

DistributedTrainerFuthark (src/distributed/distributed_trainer_futhark.zig) initializes via initWithComponents. The TrainerComponents structure contains {tokenizer: MGT, signal_engine: SignalPropagationEngine, embedding_accel: ?EmbeddingAccelerator} members. The constructor automatically rebinds the signal_engine to the nsir_graph and crev_kernel.flow_analyzer references. Model dimensions must be even for the RSF coupling layers. The trainer contains the MGT vocabulary, RSFAccelerator, LearnedEmbedding with 50000 vocab_size, CREVPipeline, heap-allocated ChaosCoreKernel, NSIR, ESSO, SurpriseMemoryManager, TemporalGraph, per-pass ReasoningOrchestrator, non-optional SignalPropagationEngine, heap-allocated ZRuntime, catch null R-GPU, mandatory FNDSManager and VPU, as well as GPUCoordinator. There is no postInit call.

The trainStepFuthark progresses through tokenization, embedding lookup, Futhark forward and backward passes, and gradient propagation steps. propagateEmbeddingGradients operates with the GradientFlowController.initWithConfig(.{ .gradient_clip_norm = 1.0, .use_normalized_gradient_flow = true, .spectral_power_iterations = 5 }) setting, with manual L2-norm clipping. The runCoreRelationalPass executes sequentially in every step: nsir_graph.encodeInformation, VPU (computeGraphEmbeddings, quantumVectorOps from topology_hash, computeSimilarityMatrix, learning_rate modulation), FNDSManager (createTree(6, 4), insertIntoTree, createIndex, addPatternToIndex), R-GPU distributeGraph, ReasoningOrchestrator runHierarchicalReasoning(50), SurpriseMemoryManager storeWithSurprise, TemporalGraph addNodeAtTime and advanceTime, SignalPropagationEngine propagateStep, ZRuntime createVariable named train_<global_step>. The buildKnowledgeGraph path executes an FNDSManager createTree(4, 3) + insertIntoTree call with node identifier kg_<text_hash> alongside CREVPipeline processTextStream.

Gradient synchronization uses a combination of allReduceFloat32Values, allReduceFloat32Max, and allReduceFloat16. In the case of a single rank (world_size <= 1), only a local update occurs. Checkpoints are written in the checkpoint_version = 7 format, with atomic rename, written only by rank 0, followed by a synchronize barrier. After loadCheckpoint, the signal_engine automatically rebinds itself.

GPUCoordinator (src/distributed/gpu_coordinator.zig) implements a one-rank-per-device model. Initialization: device_id from modulo of rank and local device count, NCCL communicator (ncclComm) initialization with a unique ID (rank 0 generates and writes it to the JAIDE_NCCL_ID_PATH path), dedicated CUDA stream, 4-byte barrier_buffer. Device memory management works with cudaMalloc, cudaFree, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost calls. Collective operations: allReduce (Sum, Max), broadcast, allGather, reduceScatter, barrier (dummy allReduce on the barrier_buffer), allReduceFloat16Avg, allReduceFloat32Max.

Modal deployment works via two paths. The Python-orchestrated modal run scripts/modal_distributed_train.py launches multi-rank subprocesses into a container. The Zig-native jaide-distributed-futhark --deploy <model_path> <dataset_path> makes a POST request to the https://api.modal.com/v1/functions/deploy endpoint using ModalGPUClient (src/distributed/modal_gpu.zig) with Bearer <MODAL_API_TOKEN> authorization, the Modal image is based on nvidia/cuda:12.8.1-devel-ubuntu24.04 with libnccl2 and libnccl-dev libraries. getJobStatus queries the https://api.modal.com/v1/functions/<job_id>/status endpoint every 30 seconds until completed/failed status.

## Inference server

The InferenceServer (src/inference_server_main.zig) is a multi-threaded HTTP engine with a ThreadPool and inference_mutex. RateLimiter is sliding-window IP-based, require_api_key is a configuration flag. The fields of the server: model, ssi, ranker, embedding, nsir_graph, chaos_kernel, esso, surprise_memory, temporal_graph, verifier (when JAIDE_VERIFY=1), signal_engine, z_runtime, fractal_lpu, r_gpu, vpu (catch null), fnds_manager, crev_pipeline, formal_verifier (heap-allocated), security_engine (heap-allocated), quantum_adapter. Every field is initialized via loadModel, the deinit frees everything in reverse order.

The three REST endpoints: /v1/health (GET), /v1/inference (POST, InferenceRequest JSON), /v1/batch_inference (POST, up to ServerConfig.batch_size). The handleInference and handleBatchInference go through a 22-step pipeline: RateLimiter, MGT tokenization, LearnedEmbedding, RSFLayer, NSIR encodeInformation, FractalLPU mapNode and balanceAllTiles, R-GPU distributeGraph, ReasoningOrchestrator (50 internal iterations), SurpriseMemoryManager storeWithSurprise, TemporalGraph addNodeAtTime and advanceTime, VerifiedInferenceEngine performVerifiedInference (optional JAIDE_VERIFY=1), SignalPropagationEngine propagateStep, ZRuntime inf_<request_count> variable, VPU computeGraphEmbeddings/quantumVectorOps/computeSimilarityMatrix, FNDSManager createTree/insertIntoTree/createIndex/addPatternToIndex, CREVPipeline processTextStream, FormalVerificationEngine verifyGraph, SecurityProofEngine proveInformationFlowSecurity, QuantumTaskAdapter identifyQuantumSubgraphs/executeQuantumTask/applyResultsToGraph, boostAboveMean (1.05×), SSI retrieveTopK, Ranker scoring, auto-regressive token generation up to max_new_tokens limit.

The configuration: batch_size, esso_initial_temp, esso_cooling_rate, esso_max_iterations, require_api_key, max_request_size_bytes, request_timeout_ms, keep_alive_timeout_ms. The environment variables: JAIDE_MODEL_PATH, JAIDE_API_KEY, JAIDE_VERIFY, JAIDE_REASONING_CYCLES.

The SSI (src/index/ssi.zig) is a hierarchical hash tree, bucket_width of 6 (64 children per branch), mixHash with the 0x9E3779B185EBCA87 constant, Merkle-style integrity via computeBranchHash, with a collision_chain linked list. retrieveTopK performs similarity search based on segment scores. The Ranker applies n-gram weights, LSH (Locality-Sensitive Hashing), Jaccard similarity, and diversity scoring, using the topKHeap and rankCandidatesWithQuery functions.

## Quantum and ZK

The RelationalQuantumLogic engine (src/core_relational/quantum_logic.zig) defines HADAMARD, PAULI_X/Y/Z, PHASE, CNOT, TOFFOLI standard gates in the LogicGate enum, as well as RELATIONAL_AND, RELATIONAL_OR, RELATIONAL_NOT, RELATIONAL_XOR, and FRACTAL_TRANSFORM relational gates. QuantumState tracks complex amplitudes, phase, and entanglement degree.

The IBMQuantumClient (src/core_relational/quantum_hardware.zig) works with OpenQASM 3.0 serialization and supports three backend families in IBMBackendSpecs: HERON (133 qubits, T1 approx. 350 µs), EAGLE (127 qubits, T1 approx. 200 µs), FALCON (27 qubits, T1 approx. 100 µs). IBMBackendCalibrationData stores T1/T2 times, readout errors, and gate errors. submitJobWithBackend executes a POST request, getJobResult queries based on job ID. QuantumClassicalHybridOptimizer applies parameter-shift gradients, default learning rate 0.1 and tolerance 10⁻⁶. The local simulator is limited to 32 qubits (SIMULATOR_QUBITS), HARDWARE_MAX_SHOTS = 100 000, POLL_INTERVAL_MS = 100.

The ZK system employs Groth16 on the bn128 curve. CircomProver (src/core_relational/zk_verification.zig) provides the compileCircuit(), generateWitness(), and prove() functions, where prove generates a ZKProofBundle with Groth16Proof (pi_a G1, pi_b G2, pi_c G1) and PublicSignals i256 values. inference_trace.circom employs PoseidonChain(n) in chunks of 6, RSFLayerComputation(dim) validates the split, affine coupling (y2 = x2 ⊙ exp(S(x1)) + T(x1)), with FIXED_POINT_SCALE = 10⁶ fixed-point scaling and a cubic Taylor approximation for the exp function (1 + x + 0.5x² + 0.166667x³). RangeProof(bits) works with Num2Bits decomposition and Pedersen commitments, VerifyMerkleProof(depth) with Poseidon(2) hashers. Differential privacy injects Laplace and Gaussian noise with SecureRng. The ZKProofError enum: CircomCompilationFailed, WitnessGenerationFailed, SnarkjsNotFound.

## Security and Verification

The VerifiedInferenceEngine (src/core_relational/zk_verification.zig) employs a Blake3 commitment, records a trace in InferenceWitness of all operations of the ReasoningOrchestrator, and generates a Groth16 proof with the CircomProver. BatchVerifier and ProofAggregator verify multiple inferences simultaneously using Merkle trees.

The FormalVerificationEngine consists of Proposition (atomic, negation, binary, quantified, Hoare triple), FormalProof (ProofStep sequence with ProofRule application), InvariantRegistry, HoareLogicVerifier, and TheoremProver components. With the -Dverify=true flag, the src/verification/oftb.lean Lean4 proofs are also compiled via lake build, which validate the properties of the OFTB split_at operation.

The SecurityProofEngine provides Bell-LaPadula (no-read-up, no-write-down), Biba (no-read-down, no-write-up), AccessControlMatrix, SeparationOfDutiesConstraint, and Non-interference bisimulation with HashChain, MerkleTree, and CommitmentScheme cryptographic primitives. proveInformationFlowSecurity produces a SecurityProof from a SecurityProofStep sequence.

The security primitives: safeIntCast to prevent IntegerOverflow and IntegerUnderflow, safePtrCast to check null and alignment, SecureRng a hybrid entropy source (std.crypto.random + LCG fallback), secureZeroBytes against compiler optimizations, constantTimeCompare against timing attacks. BigInt512 (src/core_relational/safety.zig) provides constant-time comparison and secure zeroing. HomomorphicEncryption implements the Paillier cryptosystem for additive homomorphic operations.

## Testing and benchmarking

The benchmark suite is accessible from the src/_bench_deps.zig file with rsf, core_tensor, sfd namespaces, with futhark_cpu_step dependency. bench_rsf measures the forward and backward pass throughput of the RSF stack, checking the forward→inverse roundtrip with a 1e-4 tolerance. bench_matmul measures GFLOPS (2.0 × N³ × iterations / seconds) on square matrices of sizes 128, 256, 512, and 1024. bench_tensor_ops measures GB/s on contiguous blocks of 4M elements (fill, add, mul). bench_sfd benchmarks FP4 quantization (clip [-6.0, 6.0], 1M values, 100 iterations) and spectral normalization (20 full vs. 5 sparse power iterations).

The stress-refcount test (src/tests/stress_tensor_refcount.zig) launches threads with std.atomic.Value(usize) barrier synchronization, performing single and double retain/release operations on shared Tensor objects, and finally verifies that every reference counter returned exactly to 1.

The complete test suite runs with the test-all step, which includes all test-* steps, stress-refcount, test-c-api (int64/double roundtrip, ABI layout, hash determinism, malloc/memset), and when -Dverify=true, the lake build.

The C API error codes: JAIDE_ERROR_ALLOCATION (memory failure), JAIDE_ERROR_NODE_NOT_FOUND (NSIR lookup error), JAIDE_ERROR_MATH_ERROR (overflow), JAIDE_ERROR_THREADING (mutex contention), JAIDE_ERROR_INVALID_STATE (field without post-activation, which cannot occur in the current codebase).

## Glossary

The 5th root architecture is the paradigm succeeding Perceptron, CNN, RNN, and Transformer, with RSF. RSF (Reversible Scattered Flow) consists of cross-affine coupling layers and deterministic scattered permutations, forward: y1 = x1 ⊙ exp(clip(Ws · x2 + bs)), inverse: x2 = y2 - Wt · y1 - bt.

NSIR (SSRG) is the Self-Similar Relational Graph in src/core_relational/nsir_core.zig. The EdgeQuality enum: superposition, entangled, coherent, collapsed, fractal. OFTB (Orthogonal Fractal Transformation Block) is a Haar-wavelet based mixing layer with O(1) memory in src/processor/rsf.zig. SFD (Spectral Fisher Diagonalizer) is a second-order optimizer in src/optimizer/sfd.zig. The GradientFlowController initializes with initWithConfig(GradientFlowConfig), providing manual L2-norm clipping for embedding gradients. SSI (Self-Similar Index) is a position-preserving external memory structure with O(log n) retrieval in src/index/ssi.zig. ESSO (Entangled Stochastic Symmetry Optimizer) optimizes graph topology using simulated annealing in src/core_relational/reasoning_orchestrator.zig. Qubit is a Complex(f64) primitive. ThoughtLevel: local, global, meta. The VPU calculates F64x4 graph embeddings, quantum vector operators, and similarity matrix in src/core_relational/vpu.zig. FNDSManager handles fractal trees, self-similar indices, and PatternLocation registration in src/core_relational/fnds.zig. PatternLocation.init(allocator, tree_id, level, node_id, offset, length, confidence) is created via allocation. CREVPipeline works via processTextStream in src/core_relational/crev_pipeline.zig. FormalVerificationEngine via verifyGraph in src/core_relational/formal_verification.zig. SecurityProofEngine via proveInformationFlowSecurity in src/core_relational/security_proofs.zig. QuantumTaskAdapter in src/core_relational/quantum_task_adapter.zig. SignalPropagationEngine is a non-optional field, signal propagation in every runCoreRelationalPass cycle in src/core_relational/signal_propagation.zig. TrainerComponents {tokenizer, signal_engine, embedding_accel} in src/distributed/distributed_trainer_futhark.zig. ModalGPUClient is an HTTP client to the Modal API, deployTrainingJob and getJobStatus in src/distributed/modal_gpu.zig.

MemoryBlockState: free, allocated, entangled, migrating. PinnedMemory allocated with cudaHostAlloc. HomomorphicEncryption implements the Paillier cryptosystem. ZKProofBundle stores Groth16 proofs, public signals, and verification state. Groth16Proof contains pi_a (G1), pi_b (G2), pi_c (G1) points on the bn128 curve.

WeightKind: weights_s, weights_t, velocity_s. FutharkContext manages the lifecycle of the Futhark GPU runtime. futhark_cpu_step and futhark_gpu_step are build.zig system command steps. librtl_sim.so is a shared library compiled from Haskell modules with GHC. jaide-rtl-sim runs with cycles/banks/requests_per_cycle arguments. jaide-c-api-test is the C ABI smoketest.

Environment variables: JAIDE_API_KEY (Bearer token, if require_api_key=true), JAIDE_MODEL_PATH, JAIDE_VERIFY (if "1" VerifiedInferenceEngine is active), JAIDE_REASONING_CYCLES (overrides the 50 internal iterations), JAIDE_MODEL_DIM, JAIDE_LAYERS, JAIDE_BATCH_SIZE, JAIDE_EPOCHS, JAIDE_LEARNING_RATE, JAIDE_DATASET (JSONL), JAIDE_TOTAL_SAMPLES, JAIDE_MAX_SAMPLES, JAIDE_MAX_SEQ_LEN (default 256), JAIDE_NCCL_ID_PATH, WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT, MODAL_API_TOKEN (for --deploy mode).


JAIDE CPU Benchmark
Intel Xeon 6 "Granite Rapids" (P-core) · 8 vCPU · x86_64 · AVX-512 + AMX

Matrix Multiplication

Config: 100 iteráció / méret, 10 warmup

Méret	CF ms/iter	CF GFLOPS	SIMD+T ms/iter	SIMD+T GFLOPS
128 × 128	2.45	1.72	2.35	1.78
256 × 256	4.21	7.96	3.55	9.45
512 × 512	8.42	31.89	8.55	31.40
1024 × 1024	19.78	108.55	19.78	108.55
CF = Cache-friendly · SIMD+T = SIMD+Tiled

Tensor Element-wise Operations

Config: 16 MB tensor (4 194 304 × float32), 500 iteráció, 50 warmup

Művelet	ns / elem	Sávszélesség
fill	0.04 ns	90.52 GB/s
add	0.10 ns	40.52 GB/s
mul	0.10 ns	40.40 GB/s

SFD Optimizations

Teszt	Eredmény
FP4 kvantálás (1M elem) — ns/elem	0.32 ns
FP4 kvantálás (1M elem) — throughput	3.08 × 10⁹ elem/s
SpectralNorm 20 power iter — ms/iter	7.86 ms
SpectralNorm 5 power iter — ms/iter	2.70 ms
Sparse speedup	2.91×


RSF Forward / Backward Pass

Config: dim=512, layers=12, batch=64, 200 iteráció, 20 warmup

Fázis	ms / iter	Throughput
Forward	312.58 ms	209 664 elem/s
Backward	594.57 ms	110 225 elem/s
Backward / Forward arány	1.90×	—

