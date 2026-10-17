# El Yazid TEBBAA

### Data & AI Engineer

Building reliable ML systems, evaluation pipelines, distributed applications, and developer tooling.

[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--5499--8589-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-5499-8589)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-El_Yazid_Tebbaa-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/el-yazid-tebbaa-71a27434a/)

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=22&duration=2600&pause=700&color=58A6FF&center=true&vCenter=true&width=760&height=46&lines=Applied+mathematics+%C2%B7+efficient+software+%C2%B7+machine+learning;209%2C924+job+postings+matched+with+explainable+ranking;%2B4.07+pp+recall+at+flat+accuracy+on+70%2C692+records;%2B16.05+dB+recovered+from+a+noisy+input" alt="Machine learning, applied mathematics, distributed systems" />

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
![Apache Kafka](https://img.shields.io/badge/apachekafka-%23231F20.svg?style=for-the-badge&logo=apachekafka&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

I work on the evaluation side of machine learning: whether a metric measures what it claims
to, whether a result survives a change of preprocessing, and what a model costs to run on
hardware you actually have. Each project below is built around an experiment rather than a
demo, and reports the number that settles it.

---

## Selected work

### [Adversarial testing of a medical LLM](https://github.com/coderyaz856/Adversarial-Testing-of-MedGemma)

A three-model loop that searches for prompt prefixes which degrade a medical model's answers.
Mistral-7B-Instruct (4-bit NF4) generates candidate prefixes, MedGemma-4B-IT answers under
them, and Gemma-2-2B-IT scores each answer as strict JSON on a 0–10 scale. An evolutionary
hill-climbing loop keeps the top-K prefixes as elites and mutates them, stopping early once a
candidate crosses threshold.

Three models at two precisions fit in a single Tesla T4 session, and the target is queried with
greedy decoding so any run reproduces exactly.

### [LDA vs NMF under a controlled preprocessing sweep](https://github.com/coderyaz856/Topic-Modeling-Comparing-Optimum-Aided-LDA-and-Non-negative-Matrix-Factorization)

8,000 AG News articles, with the topic count chosen by a C_v coherence sweep over
k ∈ {2,4,6,8,10,12} rather than assumed — k = 6 at coherence 0.5327. Five preprocessing variants
were scored on vocabulary retention and TF-IDF entropy, with KMeans silhouette as a check;
53 words held stable across three or more variants.

The finding worth reporting is the disagreement: the two models assign only **27.7%** of
documents to corresponding topics. LDA partitions evenly (normalised entropy 0.991, Gini 0.101)
while NMF concentrates 53.1% of documents in one topic yet yields the more nameable themes.
They are complementary, not interchangeable — the opposite of how the two are usually
substituted for one another.

### [Explainable CV-to-job matching over 209,924 postings](https://github.com/coderyaz856/NLP-for-Job-Recommendation-Data-Engineering-Case-)

Nine weighted signals — must-have skill coverage, MiniLM sentence embeddings, TF-IDF, LDA topic
similarity, seniority and years, CEFR language gating, certifications, soft skills, and salary
and work-mode constraints — renormalised to one and reported per signal, so every ranking
arrives with its matched skills and its gaps.

Semantic similarity is capped rather than summed: below 25% required-skill coverage the score
cannot exceed 0.35, and below 50% it cannot exceed 0.55, so no amount of embedding similarity
rescues a candidate missing the essentials. Embedding and TF-IDF artifacts are cached on row
count, columns and file mtime; the Streamlit layer caches on a SHA-256 of the CV text plus the
weight vector.

### [Bandit-selected operating point for a diabetes screen](https://github.com/coderyaz856/Epsilon-Bandit-Aided-Diabetes-MLP-Prediction)

70,692 BRFSS records, reduced to 11 predictors by correlation filtering, classified by a
deliberately small MLP — 105 parameters, trained on CPU. Ten decision thresholds act as bandit
arms with Beta posteriors updated from validation accuracy; Thompson sampling with a randomised
near-greedy tie-break selected 0.456. The test set was touched once.

| vs. default threshold 0.5 | Recall | Precision | Accuracy |
| --- | --- | --- | --- |
| Δ at 0.456 | **+4.07 pp** | −1.57 pp | −0.14 pp |

For a screening task that trade is the point: 1.6 points of precision buys 4.1 points of recall
at flat accuracy.

### [Filter search for image denoising](https://github.com/coderyaz856/Image-denoising---Python)

A greedy search over classical OpenCV filters, scored on PSNR, SSIM and MSE against a clean
reference, chaining a filter only when it improves the score. On Gaussian noise (σ = 25) it
chains bilateral filtering ×4 to reach **36.65 dB / SSIM 0.985**, a gain of +16.05 dB over the
noisy input. On 8% salt-and-pepper it switches to the median filter (31.20 dB / SSIM 0.9835),
where bilateral filtering is among the worst performers.

The ranking inverts exactly as the theory predicts, and the search finds it without being told
the noise type.

### [Distributed smart city platform](https://github.com/coderyaz856/distributed-smart-city-platform)

Java 11. Simulated sensor readings flow through Kafka into four domain services that persist to
embedded H2, evaluate thresholds, raise alerts and push live updates to a browser dashboard.
Five communication mechanisms coexist in one application — Kafka, Java RMI, raw TCP sockets
behind a ten-thread pool, JAX-RS REST on Grizzly, and WebSocket.

The details that matter under load: an idempotent producer with `acks=all`, auto-commit disabled
with an explicit `commitSync()` after each batch, and read endpoints that hash their payload to
return `304 Not Modified` when nothing has changed.

---

## Publication

**A Modular DevSecOps Framework for Vulnerability Detection, Risk Scoring, and LLM-Augmented Remediation in CI/CD Environments**<br>
S. Sadki, A. Lamgari, E. Y. Tebbaa · CloudTech'25 · *Lecture Notes in Networks and Systems*, vol. 1941, pp. 213–227 · Springer, 2026<br>
[10.1007/978-3-032-23844-3_23](https://doi.org/10.1007/978-3-032-23844-3_23)
