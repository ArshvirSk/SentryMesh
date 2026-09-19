Product Requirements Document: SentryMesh
Ensemble ML-Based IoT Intrusion Detection with Cloud–Fog Offloading

1. Overview
   SentryMesh is an intrusion detection system for IoT networks that splits machine learning work across two tiers: a cloud layer that trains and selects the best-performing ensemble model from historical traffic data, and a fog layer that runs the selected lightweight model for real-time attack classification close to the devices generating the traffic.

The design follows the cloud–fog offloading pattern described in "Detecting IoT Attacks Using an Ensemble Machine Learning Model" (Tomer & Sharma, Future Internet, 2022): expensive model search happens where compute is abundant (cloud), and only the cheapest, most accurate resulting model is deployed where latency matters (fog).

2. Problem Statement
   IoT devices are cheap to compromise and expensive to protect. They lack the CPU, memory, and power budget to run deep learning-grade detectors themselves, but attacks (DoS, probing, credential/authentication attacks) need to be caught in near real time — sending every packet to the cloud for a verdict introduces latency that defeats the purpose. Existing approaches either put too much intelligence at the edge (accuracy suffers) or too much in the cloud (latency suffers). There isn't a straightforward reference implementation that operationalizes the "search in the cloud, serve at the edge" split for a small team to build on.

3. Goals
   Build a working pipeline that trains and benchmarks multiple base-classifier + ensemble-method combinations on labeled network traffic data.
   Automatically select the combination with the best accuracy/latency trade-off.
   Deploy the selected model to a fog-tier service that classifies traffic in well under 2 seconds per batch, using under 10% of a constrained-node's CPU.
   Demonstrate the full loop — cloud training → model export → fog inference → results dashboard — end to end on a public dataset (NSL-KDD, or a more current equivalent).
   Non-Goals
   Building custom IoT hardware or sensors (traffic data is assumed to be collected already, e.g., via a gateway or simulated).
   Deep learning / neural architectures — this project deliberately stays in the classical ML ensemble space to keep the fog-tier footprint small.
   Production-grade multi-tenant deployment; this is a working prototype/demo-grade system.
4. Users
   User Need
   Network/security engineer Wants attacks flagged fast, without provisioning heavy compute at every edge node
   ML/infra engineer (builder) Wants a repeatable pipeline to retrain and redeploy as traffic patterns drift
   Evaluator / reviewer (e.g., academic panel, hiring manager) Wants to see measurable accuracy, latency, and resource numbers, not just a working demo
5. System Architecture
   Three tiers, mirroring the source paper:

Thing layer — IoT devices/sensors generating traffic (simulated from the dataset for v1; live capture is a stretch goal).
Fog layer — a lightweight service (can run on a Raspberry Pi–class device or a resource-capped container) that loads the exported "best model" and classifies incoming traffic in real time.
Cloud layer — training environment that ingests historical/labeled traffic, tries every base-classifier × ensemble-method combination, scores each on accuracy/precision/recall/ROC/execution time, and exports the winner as a portable artifact (e.g., ONNX or a pickled scikit-learn pipeline) for the fog layer to consume.
[Thing/IoT traffic] → [Fog: real-time inference, lightweight] ⇄ [Cloud: model search + training, heavy] 6. ML Approach
Base classifiers: Decision Tree, Random Forest, K-Nearest Neighbors, Logistic Regression, Naïve Bayes.
Ensemble methods: Voting and Stacking, applied over combinations of the base classifiers.
Selection loop: train every combination, score on accuracy, precision, recall, F-measure, kappa, ROC-AUC, and execution time; select the combination with the best accuracy/latency trade-off (not simply the highest accuracy — the source paper explicitly trades a small accuracy loss for a large execution-time win).
Dataset: NSL-KDD for v1 (41 features, DoS/Probe/U2R/R2L attack categories), with an 80:20 train/validation split and k-fold cross-validation in the cloud stage. Stretch goal: swap in a more recent IoT-specific dataset (e.g., CICIoT2023) once the pipeline is validated.
No feature reduction/extraction step in v1, consistent with the source paper's approach of keeping the pipeline simple and fast. 7. Key Features (v1 scope)
Training pipeline (cloud): ingest dataset → preprocess (encode categorical fields, normalize) → generate all base-classifier combinations → run each through voting and stacking → log all metrics.
Model selection: ranking logic that picks the deployable model based on a configurable weighting of accuracy vs. execution time.
Model export: serialize the winning pipeline in a format the fog service can load without the full training dependencies.
Fog inference service: lightweight API/process that accepts a traffic record (or batch) and returns a classification (benign / DoS / Probe / U2R / R2L) with a confidence score, and logs its own latency and CPU usage.
Results dashboard: simple view showing per-model comparison (accuracy, ROC, execution time) from the cloud stage, and live classification stats from the fog stage. 8. Non-Functional Requirements
Fog-tier inference: < 2s per batch, < 10% CPU on a constrained node.
Cloud-tier training: must complete a full sweep of all base-classifier × ensemble-method combinations in a single run, with results reproducible from a fixed random seed.
Model artifact portability: the exported model must load and run on the fog service without requiring the full cloud-side training stack installed. 9. Success Metrics
Selected model achieves ≥ 99% accuracy and ≥ 0.99 ROC-AUC on held-out data (benchmark set by the source paper's Model 8: KNN + Naïve Bayes + Decision Tree with voting).
Fog-tier execution time ≤ 1.5s per inference batch.
Fog-tier CPU usage stays under 10%.
End-to-end demo: traffic in → classification out, with the cloud→fog handoff working without manual intervention. 10. Milestones
Phase Deliverable
1 Dataset ingestion + preprocessing pipeline working
2 Full base-classifier × ensemble sweep running in cloud stage, metrics logged
3 Model selection logic + export format finalized
4 Fog inference service running the exported model, latency/CPU instrumented
5 Dashboard + end-to-end demo
6 (stretch) Swap in live/simulated real-time traffic instead of static test split; try a more current IoT dataset 11. Risks / Open Questions
Dataset staleness: NSL-KDD is from 2009-era attack patterns; results may not generalize to modern IoT attack traffic without validating against a newer dataset.
Fog hardware target undefined: actual latency/CPU numbers depend heavily on what "fog node" means in practice (Raspberry Pi? containerized limit?) — needs to be pinned down before NFRs can be validated.
Model drift: no retraining trigger/schedule defined yet — traffic patterns will shift over time and the cloud stage needs a cadence for re-running the sweep.
Class imbalance: U2R and R2L attack classes are typically rare in NSL-KDD; per-class recall (not just aggregate accuracy) should be tracked to avoid a model that's blind to rare-but-severe attacks.
Reference: Tomer, V.; Sharma, S. "Detecting IoT Attacks Using an Ensemble Machine Learning Model." Future Internet 2022, 14, 102.
