## Hi, I'm Yaz 👋

I'm an AI and software engineer from Morocco. I work mostly on machine learning and applied maths, and I spend a fair amount of time getting models to run on the hardware I actually have rather than the hardware I'd like.

The part I enjoy most is the checking: does this metric measure what it claims to, and does the result hold up when I change the preprocessing?

### Areas of interest

Machine Learning · Reinforcement Learning · Computer Vision · Natural Language Processing · Applied Mathematics · Efficient Inference

### Projects

- **[Adversarial Testing of MedGemma](https://github.com/coderyaz856/Adversarial-Testing-of-MedGemma)**, stress-tests a medical model with adversarial prompt prefixes written by a second model, while a third scores the answers. All three fit in one GPU session thanks to 4-bit quantisation.
- **[Topic Modeling: LDA vs NMF](https://github.com/coderyaz856/Topic-Modeling-Comparing-Optimum-Aided-LDA-and-Non-negative-Matrix-Factorization)**, runs both models on the same corpus and compares them. The topic count comes from a coherence sweep, and I tried five preprocessing variants first to see which words survive all of them.
- **[NLP for Job Recommendation](https://github.com/coderyaz856/NLP-for-Job-Recommendation-Data-Engineering-Case-)**, matches a CV against roughly 210,000 job postings using keyword, topic and embedding scores. Every result comes with the reasons behind it and the gaps it found.
- **[Bandit-Optimised Diabetes Classifier](https://github.com/coderyaz856/Epsilon-Bandit-Aided-Diabetes-MLP-Prediction)**, an MLP whose decision threshold is picked by a multi-armed bandit. Accuracy barely moves, but recall gains about 4 points, which is the trade you want for screening.
- **[Image Denoising](https://github.com/coderyaz856/Image-denoising---Python)**, searches over classical filters and keeps whichever one best restores a noisy image, measured with PSNR and SSIM against the clean original. Bilateral wins on Gaussian noise, median on salt-and-pepper.
- **[Distributed Smart City Platform](https://github.com/coderyaz856/distributed-smart-city-platform)**, a Java system where the same services are reachable over REST, RMI, raw sockets and WebSocket, with an embedded database behind them and a live dashboard on top.

### Publication

**A Modular DevSecOps Framework for Vulnerability Detection, Risk Scoring, and LLM-Augmented Remediation in CI/CD Environments**
S. Sadki, A. Lamgari, E. Y. Tebbaa. CloudTech'25, Lecture Notes in Networks and Systems 1941, pp. 213–227, Springer, 2026. [doi:10.1007/978-3-032-23844-3_23](https://doi.org/10.1007/978-3-032-23844-3_23)

### Technologies

Python · C · C++ · C# · Java · SQL · PyTorch · scikit-learn · FastAPI
