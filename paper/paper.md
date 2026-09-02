---
title: 'Forest: A data analysis library for raw high-throughput digital phenotyping data'
tags:
  - high-throughput
  - digital phenotyping
  - smartphone
authors:
  - name: Jukka-Pekka Onnela^[Principal Investigator and corresponding author.]
    orcid: 0000-0001-6613-8668
    affiliation: "1"
  - name: Zachary Clement
    orcid: 0000-0003-2279-5265
    affiliation: "1, 7"
  - name: Hassan Y. Dawood
    orcid: 0000-0002-2190-5146
    affiliation: "1"
  - name: Georgios Efstathiadis
    orcid: 0009-0006-2278-1882
    affiliation: "1, 3"
  - name: Patrick Emedom-Nnamdi
    orcid: 0000-0003-4442-924X
    affiliation: "1, 3"
  - name: Emily J. Huang
    orcid: 0000-0003-1964-5231
    affiliation: "1, 6"
  - name: Marta Karas
    orcid: 0000-0001-5889-3970
    affiliation: "1, 2"
  - name: Gang Liu
    orcid: 0000-0003-3544-363X
    affiliation: "1, 8"
  - name: Max Melnikas
    orcid: 0009-0005-4327-4495
    affiliation: "1"
  - name: Nellie Ponarul
    orcid: 0009-0003-1279-3757
    affiliation: "1, 9"
  - name: Marcin Straczkiewicz
    orcid: 0000-0002-8703-4451
    affiliation: "1, 4"
  - name: Ilya Sytchev
    orcid: 0009-0003-0647-5613
    affiliation: "1"
  - name: Anna L. Beukenhorst
    orcid: 0000-0002-1765-4890
    affiliation: "1, 5"
affiliations:
 - name: Department of Biostatistics, Harvard T.H. Chan School of Public Health, Harvard University, Boston, MA, USA
   index: 1
 - name: Takeda Development Center Americas, Cambridge, MA, USA
   index: 2
 - name: Olira, 11201 USA, New York City, NY, USA
   index: 3
 - name: Department of Measurement and Electronics, AGH University of Krakow, Krakow, Poland
   index: 4
 - name: Leyden Labs, Leiden, The Netherlands
   index: 5
 - name: Department of Statistical Sciences, Wake Forest University, Winston-Salem, North Carolina, USA
   index: 6
 - name: CVS Health
   index: 7
 - name: Google LLC, Mountain View, CA, USA
   index: 8
 - name: Peal Health, New York, NY, USA
   index: 9
date: 28 May 2025
bibliography: paper.bib
---

# Summary
Forest is a data analysis library designed for raw, high-throughput digital phenotyping data. It is intended to integrate directly with its companion software—the Beiwe platform—which enables large-scale data collection from both Android and iOS smartphones. Beiwe supports digital phenotyping by gathering both active data (such as surveys and audio samples) and passive data (such as accelerometer data) from participants' smartphones. Most smartphone applications depend on software development kits (SDKs) that generate behavioral summary measures using closed, proprietary algorithms that lack external validation. As a result, these applications often fall short of scientific standards for reproducibility and may force researchers to adjust their scientific questions based on the constraints of available data summaries. By contrast, Beiwe collects raw sensor and phone usage data, with collection parameters that can be customized to match the needs of any scientific study.

The analysis of such raw data however presents significant challenges, requiring dedicated statistical methods and specialized software. This is the need that Forest addresses. While the quantitative methods necessary for analyzing raw digital phenotyping data have been developed and published elsewhere [@barnett2020inferring; @liu2021bidirectional; @straczkiewicz2023walking; @huang2022movelets], Forest provides robust, accessible software implementations of these methods in Python. Each method within Forest is named after a tree—for example, Jasmine implements sparse online Gaussian process imputation for missing GPS trajectories [@liu2021bidirectional]. We expect Forest to continue growing as additional methods become available.

# Statement of need

**Background.** The phenotype of an organism comprises a collection of traits, such as enzyme activity, hormone levels, and behavior. Increasingly, researchers advocate for a more substantial role for large-scale phenotyping as a route to advances in the biomedical sciences [@houle2010phenomics; @delude2015deep; @bilder2009phenomics; @robinson2012deep]. Of the many types of phenotypes, social, behavioral, and cognitive phenotypes are particularly challenging to study due to their temporal and contextual dependencies. Traditionally, various surveys and assessments are used to ascertain these phenotypes, but these are cross-sectional, subjective, and often burdensome. The ubiquity and capabilities of smartphones—when coupled with appropriate data analytic techniques—can help overcome these limitations. We coined the term digital phenotyping to refer to the “moment-by-moment quantification of the individual-level human phenotype in situ using data from personal digital devices, in particular smartphones” [@onnela2016harnessing; @torous2016new]. Onnela Lab began developing Beiwe, an open source high-throughput smartphone-based digital phenotyping platform that collects active and passive data, with support from an NIH Director’s New Innovator Award, in 2013. The platform consists of iOS and Android front-end applications and an AWS-based back-end infrastructure with a web-based study management portal for data processing and secure storage. This approach not only allows for more objective measurement of known phenotypes but can also give rise to entirely new phenotypes.

**Limitations of traditional assessment.** Social and behavioral phenotypes have traditionally been ascertained using either participant-administered or investigator-administered surveys and assessments in research settings and patient-reported or clinician-reported equivalents in clinical settings. For example, the Amyotrophic Lateral Sclerosis Functional Rating Scale - Revised (ALSFRS-R) includes 12 items (questions), each scored on a 0 (no function) to 4 (full function) scale, and has been used both for diagnosing patients and for tracking disease progression [@cedarbaum1999alsfrsr]. In observational studies and clinical trials, it may be administered every six weeks with smaller within-subject standard deviation when administered by the participant themselves rather than when administered by a clinician [@berry2019design]. In order to eliminate recall bias, some of the items in ALSFRS-R can potentially be measured objectively in real-world settings. For example, two items in the ALSFRS-R relate to physical activity: walking (Item 8) and climbing stairs (Item 9). Both of these can be estimated using smartphone accelerometer and gyroscope data [@straczkiewicz2021systematic].

An important development in this field has been the introduction of software development kits (SDKs) for smartphones, such as Apple’s ResearchKit and Google’s ResearchStack, which have facilitated the creation of software for these devices. However, the use of prepackaged software restricts the types of data that can be collected, thereby limiting the research questions that can be investigated and the data analyses that can be performed [@onnela2021opportunities]. For example, Apple’s ResearchKit does not support background sensor data collection [@researchkit]; Apple's HealthKit does support background data collection for selected sensors only [@sensorkit]; and the Core Motion framework allows the collection of raw accelerometer data in the background, but only for up to 12 hours at a time [@cmsensor]. The algorithms underlying HealthKit metrics, such as step count [@sensorkit_stepcount], are proprietary and subject to change without notice. The use of closed algorithms, which may be updated at any time, makes it hard or impossible to compare data collected at different times or across different SDKs.

Most investigators currently rely on data from commercially maintained SDKs as described above. The small number of investigators who use raw data often apply general-purpose statistical methods not designed for this type of data, or ad hoc methods whose statistical properties are not well understood. Having statisticians, data scientists, and machine learning experts develop appropriate methods and implement them in accessible software like Forest is highly beneficial. This is expected to improve both the quality of statistical analyses and the scientific evidence generated.

# State of the Field

Forest operates within a growing ecosystem of tools for mobile health and digital phenotyping. Existing approaches broadly fall into two categories: (1) smartphone software development kits (SDKs) such as Apple’s ResearchKit and Google’s ResearchStack, and (2) general-purpose data science tools (e.g., Python and R libraries for time series and signal processing). SDK-based approaches facilitate data collection but typically provide precomputed behavioral summary measures derived from proprietary algorithms that are not externally validated and may change over time. This limits reproducibility and constrains downstream scientific analysis.

In contrast, general-purpose tools offer flexibility but lack domain-specific methods for handling high-frequency, irregular, and missing smartphone sensor data. As a result, researchers often implement ad hoc pipelines, leading to inconsistent methodologies and unclear statistical properties. To our knowledge, there are few open-source libraries that provide standardized, statistically grounded methods specifically for raw digital phenotyping data collected from smartphones.

Forest addresses this gap by providing an open-source, domain-specific analysis library designed explicitly for raw digital phenotyping data. Rather than duplicating existing SDK functionality, Forest complements data collection platforms like Beiwe by focusing on statistically principled transformation of raw data into interpretable features. The decision to develop a new library reflects the need for tightly integrated methods that account for the unique structure, scale, and missingness patterns of smartphone-derived behavioral data.

Several open-source tools now analyze data of this kind. Cortex is a data processing pipeline optimized for the mindLAMP apps, deriving clinical features, quality metrics and visualizations [@burns2024cortex; @vaidyam2022lamp]. RAPIDS is a reproducible workflow standardizing preprocessing and feature extraction across Android and iOS sensing applications as well as Fitbit and Empatica devices [@vega2021rapids]; it also ports the mobility measures of [@barnett2020inferring], a method Forest implements as well. That overlap is real, and these tools complement Forest rather than compete with it. Forest differs in scope and provenance: it is maintained by the authors of the methods it implements, it spans mobility, accelerometry, communication logs, survey processing and data simulation rather than a single domain, and it tracks the methodological literature as it develops, providing the sparse online Gaussian process approach of [@liu2021bidirectional] alongside the earlier method. Forest exists as a separate library because these methods evolve with the research that produces them, and each new method can be added as an additional tree.

# Software Design

Forest is designed as a modular analysis library organized around independent processing units referred to as “trees,” each implementing a specific methodological pipeline (e.g., mobility inference [@barnett2020inferring; @liu2021bidirectional], accelerometry processing [@straczkiewicz2023walking; @huang2022movelets], communication analysis). This architecture reflects a trade-off between extensibility and cohesion: individual trees can be developed and validated independently while adhering to shared data structures and conventions provided by the Poplar utility layer.

A key design decision is the separation between data collection (handled by the Beiwe platform) and data analysis (handled by Forest). This improves reproducibility by ensuring that raw data remain unchanged and that all derived features are generated through transparent, version-controlled methods. Forest operates directly on raw data files rather than requiring intermediate proprietary formats, enabling full auditability of the analysis pipeline.

The library is implemented in Python to leverage its scientific computing ecosystem and accessibility to researchers. Design choices also prioritize scalability and robustness to missing data, which is pervasive in smartphone sensing. Methods such as Jasmine explicitly model missingness using probabilistic approaches rather than relying on simple interpolation [@barnett2020inferring; @liu2021bidirectional].

# Research Impact Statement

Forest has been developed in conjunction with the Beiwe platform, which has been used in observational studies and clinical research settings across multiple domains, including psychiatry, neurology, and oncology. The analytical methods implemented in Forest—such as mobility trajectory inference from GPS data and accelerometer-based estimation of physical activity—are grounded in peer-reviewed statistical research [@barnett2020inferring; @liu2021bidirectional; @straczkiewicz2023walking; @huang2022movelets] and have been applied in published studies examining behavioral and clinical outcomes in oncology and surgical recovery [@panda2020smartphone; @panda2020using; @panda2021smartphone; @wright2018hope], neurology [@beukenhorst2021smartphone], spine and rehabilitation medicine [@cote2019digital; @mercier2020digital], and psychiatry [@barnett2018relapse; @staples2017comparison; @torous2018characterizing; @fortgang2020increase; @pelligrini2021estimating].

By providing standardized, open-source implementations of these methods, Forest enables reproducible analysis of high-throughput digital phenotyping data, addressing a key limitation of existing approaches that rely on proprietary or ad hoc analytical pipelines. The library supports large-scale datasets and facilitates the generation of consistent summary metrics across studies, improving comparability and enabling downstream statistical analysis.

Forest also serves as a foundation for ongoing methodological development. Its modular design allows new analytical methods to be incorporated as additional trees, supporting the continued evolution of digital phenotyping research. In addition, the inclusion of synthetic data generation tools enables testing, benchmarking, and validation of analytical pipelines, further supporting reproducible and transparent research practices.

# Acknowledgements

The Principal Investigator, Jukka-Pekka Onnela, is extremely grateful for his NIH Director’s New Innovator Award in 2013 (DP2MH103909) for enabling the crystallization of the concept of digital phenotyping and the construction of the Beiwe platform. He is also grateful to the members of the Onnela Lab.

# AI Usage Disclosure

Generative AI tools were used in a very limited capacity to assist with minor editing and refinement of the manuscript. The authors wrote the paper in full, and all content, including any AI-assisted edits, was carefully reviewed and verified for accuracy, consistency, and alignment with the underlying methods and software implementation.

# References
