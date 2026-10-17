# El Yazid TEBBAA

### Data & AI Engineer

Building reliable ML systems, evaluation pipelines, distributed applications, and developer tooling.

[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--5499--8589-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-5499-8589)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-El_Yazid_Tebbaa-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/el-yazid-tebbaa-71a27434a/)

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=20&duration=2600&pause=700&color=58A6FF&center=true&vCenter=true&width=900&height=48&lines=GPU+engineering+%C2%B7+~13B+parameters+%C2%B7+one+Tesla+T4;35%25+less+simulation+compute+%C2%B7+built+scientific+SDK;209%2C924+job+postings+matched+with+explainable+ranking;%2B4.07+pp+recall+at+flat+accuracy+on+70%2C692+records;%2B16.95+dB+from+seeded+impulse+noise" alt="GPU engineering: roughly 13B parameters on one Tesla T4; research engineering: 35% less simulation compute and a scientific SDK; image denoising: 16.95 dB PSNR gain on seeded impulse noise" />

![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![C++](https://img.shields.io/badge/c%2B%2B-%2300599C.svg?style=for-the-badge&logo=cplusplus&logoColor=white)
![C#](https://img.shields.io/badge/C%23-%23239120.svg?style=for-the-badge&logo=dotnet&logoColor=white)
![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=openjdk&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-%234479A1.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23fff.svg?style=for-the-badge&logo=opencv&logoColor=black)
![FastAPI](https://img.shields.io/badge/fastapi-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

I work on the evaluation side of machine learning: whether a metric measures what it claims
to, whether a result survives a change of preprocessing, and what a model costs to run on
hardware you actually have. Each project below is built around an experiment rather than a
demo, and reports the number that settles it.


## Featured collaboration: Noesis / EigenBrains

An autonomous scientific-discovery laboratory, built for the [Hack-Nation 7th Global AI Hackathon](https://github.com/Ismailea4/Agentic-Scientific-Discovery-EigenBrains). I designed three of its layers.

**The evidence runtime.** A result is only worth as much as a stranger's ability to reconstruct it, so I built the lab's findings on an append-only research ledger over content-addressed bundles, reached through contract-checked RPC from typed Python, Rust and Julia SDKs, with checkpoint and resume for long runs. Corruption checks fail closed: a damaged bundle halts the run rather than quietly degrading it, and any recorded finding reproduces byte for byte.

**Decision economics.** Compute is the lab's scarce resource, so I designed the planner to buy information rather than samples: expected-information-gain planning under an explicit budget, redundancy-aware portfolios across competing hypotheses, downside-risk analysis, and selective escalation from 6 to 12 to 24 seeds that continues only while a preregistered verdict stays inconclusive. Across five replicates and seven hypotheses this saved a mean of **35.15%** of simulation compute against always running the full-size design (efficiency ratio **1.54x**, 95% bootstrap CI **[1.28, 1.90]**), while agreeing with a 48-seed reference on 28 of 35 decisions, where the full-size design it replaced agreed on 29 of 35.

**The statistical layer.** I chose methods that survive clustering and multiplicity rather than ones that flatter a result: paired and stratified cluster bootstraps, AUROC and Brier analysis, Wilson intervals, exact McNemar tests, Hodges–Lehmann estimates, Wilcoxon tests with Holm correction, and CVaR, measured over dynamic evolutionary-search benchmarks with entropy telemetry.

**The design decision I would defend first** is sealing a cross-language study before running it. A promising development signal (ΔAUROC **+0.1393**) disappeared on untouched landscapes (**−0.0059**). I published the null result with its protocol hash and an independently verifiable evidence bundle instead of reporting the development number.

[Acceleration study](https://github.com/Ismailea4/Agentic-Scientific-Discovery-EigenBrains/tree/main/discolab/results/escalation) | [Sealed result](https://github.com/Ismailea4/Agentic-Scientific-Discovery-EigenBrains/blob/main/docs/SEALED_ENTROPY_EARLY_WARNING_RESULT.md) | [Cross-language SDK](https://github.com/Ismailea4/Agentic-Scientific-Discovery-EigenBrains/tree/main/sdk)

---

## Selected work

### [Adversarial testing of a medical LLM](https://github.com/coderyaz856/Adversarial-Testing-of-MedGemma)

A three-model loop that searches for prompt prefixes which degrade a medical model's answers.
**Mistral-7B-Instruct** (4-bit NF4) generates candidate prefixes, **MedGemma-4B-IT** answers under
them, and **Gemma-2-2B-IT** scores each answer as strict JSON on a 0–10 scale. An **evolutionary
hill-climbing** loop keeps the top-K prefixes as elites and mutates them, stopping early once a
candidate crosses threshold.

Roughly **13B parameters** across three models share **one Tesla T4**, and most of the engineering is in
making that hold. The runtime's preinstalled PyTorch is removed and rebuilt against the CUDA 12.1
wheel index with the pip cache purged between steps, so the resolver cannot reuse the incompatible
wheels it already had. The compatibility boundary was then isolated to a single package and pinned
as **`numpy<2.0`** rather than freezing the environment to an old snapshot, which keeps it
reproducible without surrendering unrelated upgrades. Two runtime faults were traced to source rather than worked
around: TorchDynamo's compilation path is disabled before any import, and the target runs greedy
decoding, which also makes every run reproducible.

### [LDA vs NMF under a controlled preprocessing sweep](https://github.com/coderyaz856/Topic-Modeling-Comparing-Optimum-Aided-LDA-and-Non-negative-Matrix-Factorization)

8,000 AG News articles, with the topic count chosen by a **$C_v$ coherence sweep** over
$k \in \{2, 4, 6, 8, 10, 12\}$ rather than assumed: **$k = 6$ at coherence $0.5327$**. Five preprocessing variants
were scored on vocabulary retention and TF-IDF entropy, with KMeans silhouette as a check;
53 words held stable across three or more variants.

The finding worth reporting is the disagreement: the two models assign only **27.7%** of
documents to corresponding topics. LDA partitions evenly (normalised entropy 0.991, Gini 0.101)
while NMF concentrates **53.1%** of documents in one topic yet yields the more nameable themes.
They are **complementary, not interchangeable**, which is the opposite of how the two are usually
substituted for one another.

### [Explainable CV-to-job matching over 209,924 postings](https://github.com/coderyaz856/NLP-for-Job-Recommendation-Data-Engineering-Case-)

**Nine weighted signals** (must-have skill coverage, MiniLM sentence embeddings, TF-IDF, LDA topic
similarity, seniority and years, CEFR language gating, certifications, soft skills, and salary
and work-mode constraints) are renormalised to one and reported per signal, so every ranking
arrives with its matched skills and its gaps.

Semantic similarity is **capped rather than summed**: below 25% required-skill coverage the score
cannot exceed 0.35, and below 50% it cannot exceed 0.55, so no amount of embedding similarity
rescues a candidate missing the essentials. Embedding and TF-IDF artifacts are cached on row
count, columns and file mtime; the Streamlit layer caches on a SHA-256 of the CV text plus the
weight vector.

### [Bandit-selected operating point for a diabetes screen](https://github.com/coderyaz856/Epsilon-Bandit-Aided-Diabetes-MLP-Prediction)

70,692 BRFSS records, reduced to 11 predictors by correlation filtering, classified by a
deliberately small MLP of **105 parameters**, trained on CPU. Ten decision thresholds act as bandit
arms with Beta posteriors updated from validation accuracy; **Thompson sampling** with a randomised
near-greedy tie-break selected **0.456**. The test set was touched **once**.

| vs. default threshold 0.5 | Recall | Precision | Accuracy |
| --- | --- | --- | --- |
| $\Delta$ at 0.456 | **+4.07 pp** | −1.57 pp | −0.14 pp |

For a screening task that trade is the point: 1.6 points of precision buys 4.1 points of recall
at flat accuracy.

### [Filter search for image denoising](https://github.com/coderyaz856/Image-denoising---Python)

A greedy search over classical OpenCV filters, scored on PSNR, SSIM and MSE against a clean
reference, chaining a filter only when it improves the score. On seeded Gaussian noise
($\sigma = 25$), **bilateral filtering** $\times 3$ reaches **35.53 dB / SSIM 0.9570**, a gain of
**+15.10 dB**. On 8% salt-and-pepper noise, **median filtering** $\times 2$ reaches
**33.15 dB / SSIM 0.9866**, a gain of **+16.95 dB**.

The ranking inverts with the corruption model, and the search finds it without receiving a
noise-specific filter rule. The committed generator retains the seeds, exact metrics, baselines,
selected chains, and library versions.

### [Distributed smart city platform](https://github.com/coderyaz856/distributed-smart-city-platform)

Java 11. Simulated sensor readings flow through Kafka into four domain services that persist to
embedded H2, evaluate thresholds, raise alerts and push live updates to a browser dashboard.
**Five communication mechanisms** coexist in one application: Kafka, Java RMI, raw TCP sockets
behind a ten-thread pool, JAX-RS REST on Grizzly, and WebSocket.

The details that matter under load: an **idempotent producer** with `acks=all`, auto-commit disabled
with an explicit `commitSync()` after each batch, and read endpoints that hash their payload to
return `304 Not Modified` when nothing has changed.

---

## Publication

**A Modular DevSecOps Framework for Vulnerability Detection, Risk Scoring, and LLM-Augmented Remediation in CI/CD Environments**<br>
S. Sadki, A. Lamgari, E. Y. Tebbaa · CloudTech'25 · *Lecture Notes in Networks and Systems*, vol. 1941, pp. 213–227 · Springer, 2026<br>
[10.1007/978-3-032-23844-3_23](https://doi.org/10.1007/978-3-032-23844-3_23)
