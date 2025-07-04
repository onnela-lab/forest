---
title: 'Forest: A data analysis library for raw high-throughput digital phenotyping data'
tags:
  - high-throughput
  - digital phenotyping
  - smartphone
authors:
  - name: Jukka-Pekka Onnela^[Principal Investigator and corresponding author.]
    orcid: 0000-0001-6613-8668
    affiliation: 1
  - name: Josh Barback
    orcid: ______________
    affiliation: 1
  - name: Zachary Clement
    orcid: 0000-0003-2279-5265
    affiliation: 1,7
  - name: Hassan Dawood
    orcid: 0000-0002-2190-5146
    affiliation: 1
  - name: Georgios Efstathiadis
    orcid: 0009-0006-2278-1882
    affiliation: 1,3
  - name: Patrick Emedom-Nnamdi
    orcid: 0000-0003-4442-924X
    affiliation: 1,3
  - name: Emily J. Huang
    orcid: 0000-0003-1964-5231
    affiliation: 1,6
  - name: Marta Karas
    orcid: 0000-0001-5889-3970
    affiliation: 1,2
  - name: Gang Liu
    orcid: 0000-0003-3544-363X
    affiliation: 1,8
  - name: Max Melnikas
    orcid: 0009-0005-4327-4495
    affiliation: 1
  - name: Nellie Ponarul
    orcid: 0009-0003-1279-3757
    affiliation: 1,9
  - name: Marcin Straczkiewicz
    orcid: 0000-0002-8703-4451
    affiliation: 1,4
  - name: Ilya Sytchev
    orcid: 0009-0003-0647-5613
    affiliation: 1
  - name: Anna Beukenhorst
    orcid: 0000-0002-1765-4890
    affiliation: 1,5
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
Forest is a data analysis library designed for raw, high-throughput digital phenotyping data. It is intended to integrate directly with its companion software—-the Beiwe platform—-which enables large-scale data collection from both Android and iOS smartphones. Beiwe supports digital phenotyping by gathering both active data (such as surveys and audio samples) and passive data (such as accelerometer data) from participants' smartphones. Most smartphone applications depend on software development kits (SDKs) that generate behavioral summary measures using closed, proprietary algorithms that lack external validated. As a result, these applications often fall short of scientific standards for reproducibility and may force researchers to adjust their scientific questions based on the constraints of available data summaries. By contrast, Beiwe collects raw sensor and phone usage data, with collection parameters that can be customized to match the needs of any scientific study.

The analysis of such raw data however presents significant challenges, requiring dedicated statistical methods and specialized software. This is the need that Forest addresses. While the quantitative methods necessary for analyzing raw digital phenotyping data have been developed and published elsewhere, Forest provides robust, accessible software implementations of these methods in Python. Each method within Forest is named after a tree—for example, Jasmine implements sparse online Gaussian process imputation for missing GPS trajectories. We expect Forest to continue growing as additional methods become available.

# Statement of need

**Background.** The phenotype of an organism comprises a collection of traits, such as enzyme activity, hormone levels, and behavior. Increasingly, researchers advocate for a more substantial role for large-scale phenotyping as a route to advances in the biomedical sciences [@houle2010phenomics; @delude2015deep; @bilder2009phenomics; @robinson2012deep]. Of the many types of phenotypes, social, behavioral, and cognitive phenotypes are particularly challenging to study due to their temporal and contextual dependencies. Traditionally, various surveys and assessments are used to ascertain these phenotypes, but these are cross-sectional, subjective, and often burdensome. The ubiquity and capabilities of smartphones—-when coupled with appropriate data analytic techniques—-can help overcome these limitations. We coined the term digital phenotyping to refer to the “moment-by-moment quantification of the individual-level human phenotype in situ using data from personal digital devices, in particular smartphones” [@onnela2016harnessing; @torous2016new]. Beiwe was developed specifically for use in smartphone-based digital phenotyping research. This approach not only allows for more objective measurement of known phenotypes but can also give rise to entirely new phenotypes.

**State of the field.** Social and behavioral phenotypes have traditionally been ascertained using either participant-administered or investigator-administered surveys and assessments in research settings and patient-reported or clinician-reported equivalents in clinical settings. For example, the Amyotrophic Lateral Sclerosis Functional Rating Scale - Revised (ALSFRS-R) includes 12 items (questions), each scored on a 0 (no function) to 4 (full function) scale, and has been used both for diagnosing patients and for tracking disease progression [@mora2017edaravone]. In observational studies and clinical trials, it may be administered every six weeks with smaller within-subject standard deviations than when administered by a clinician [@berry2019design]. In order to eliminate recall bias, some of the items in ALSFRS-R can potentially be measured objectively in real-world settings. For example, two items in the ALSFRS-R relate to physical activity: walking (Item 8) and climbing stairs (Item 9). Both of these can be estimated using smartphone accelerometer and gyroscope data [@straczkiewicz2021systematic].

An important development in this field has been the introduction of software development kits (SDKs) for smartphones, such as Apple’s ResearchKit and Google’s ResearchStack, which have facilitated the creation of software for these devices. However, the use of prepackaged software restricts the types of data that can be collected, thereby limiting the research questions that can be investigated and the data analyses that can be performed [@onnela2021opportunities]. For example, Apple’s ResearchKit does not support background sensor data collection [@researchkit]; Apple's HealthKit does support background data collection for selected sensors only [@sensorkit]; and the Core Motion framework allows the collection of raw accelerometer data in the background, but only for up to 12 hours at a time [@cmsensor]. The algorithms underlying HealthKit metrics, such as step count [@sensorkit_stepcount], are proprietary and subject to change without notice. The use of closed algorithms, which may be updated at any time, makes it hard or impossible to compare data collected at different times or across different SDKs.

To our knowledge, Forest is the only existing library for the analysis of raw, high-throughput smartphone data. Most investigators currently rely on data from commercially maintained SDKs as described above. The small number of investigators who use raw data often apply general-purpose statistical methods not designed for this type of data, or ad hoc methods whose statistical properties are not well understood. Having statisticians, data scientists, and machine learning experts develop appropriate methods and implement them in accessible software like Forest is highly beneficial. This is expected to improve both the quality of statistical analyses and the scientific evidence generated.

**Workflow.** The Forest library picks up where the Beiwe platform leaves off. While Beiwe is used in health research settings for the collection of both active and passive data, Forest is used to process this raw data, which typically ranges from 100 MB to 1 GB per participant-month, and convert it into interpretable summary statistics, such as estimated daily step count. Users can specify the time granularity of these summaries as needed; in some contexts, for example, hourly step counts may be more relevant than daily counts if the timing of the activity is of interest. These summary statistics can then serve as outcomes or predictors in further analyses. The chosen time granularity will often influence the statistical methods employed. For example, generalized linear mixed models might be used to examine the longitudinal association between daily mood and daily step count, accounting for within-person correlation over time. By contrast, if the same analysis is performed using hourly time windows, methods from functional data analysis may be more appropriate. Notably, some of the methods within Forest have also been validated for use with data from wearable devices other than smartphones—for example, the Jasmin method can impute data from any wearable GPS tracker, and the Oak method can estimate step counts or gait properties from smartwatch accelerometer data.

**Supported methods.** Currently, there are XXX trees in production in the Forest library, and this number is expected to grow. A brief overview of each included method (tree) is provided below.

**Jasmine** is used to convert raw GPS data—comprising time, latitude, longitude, altitude, and accuracy—into a more abstract data type consisting of “flights” (periods of linear movement) and “pauses” (periods of non-movement). To avoid distortion associated with planar projection, these objects are specified in spherical coordinates with the origin at the center of the Earth, so that flights correspond to short arcs along the Earth’s surface. When movement patterns are represented in terms of flights and pauses, the resulting sequence is called a mobility trajectory. Missingness is common in mobility trajectories; in some cases, data gaps are intentional by design (for example, GPS sampling may alternate between on and off cycles to conserve battery), while in other cases, missingness may result from technical limitations (such as loss of GPS signal when a person is indoors). Jasmine addresses missingness using a resampling method first introduced by Barnett & Onnela (CITE) and later refined by Liu & Onnela (CITE). The latter approach, which is based on sparse online Gaussian processes, offers several advantages over the original method, including preservation of the original spherical geometry of the problem without resorting to planar projection and the capacity for bidirectional imputation—advancing simultaneously both forward and backward in time. In addition to generating imputed mobility trajectories, Jasmine also computes various mobility summary statistics. Users can specify several input parameters, such as the threshold distance used to define a pause, the number of basis vectors for flights and pauses in the kernel function, and the variance parameter in the sparse online Gaussian process.

**Oak** is used to convert raw time-stamped triaxial accelerometer data (time, $a_x$, $a_y$, $a_z$) into summary statistics that capture properties of gait, cadence, and step counts. Some of these summaries, such as those characterizing gait, are *intensive* or density-like measures, while others, like daily step count, are *extensive* or volume-like measures. This distinction is important when considering how to address missing data in general and device non-wear in particular in subsequent analyses. Let us assume that a person's physical activity properties do not change over the course of the day (e.g., gait does not become slower in the evening), which corresponds to the mathematical assumption that the accelerometer data-generating process is stationary within a day (though it may or may not be stationary across days). Most studies of physical activity must address device non-wear. For intensive quantities under these assumptions, non-wear reduces the precision of estimates somewhat but does not introduce bias. In contrast, for extensive quantities, non-wear can result in significant bias. Using step count as an example, the estimand of interest may be the (true but unobserved) daily step count, while the observed value is a conditional step count, the number of steps recorded while the device is worn. Depending on the scientific question, study design, and planned statistical analyses, some reweighting, adjustment, or imputation may be needed for extensive measures. Currently, summary statistics may be computed on a daily or hourly basis. Examples of summary statistics include walking time and the number of steps taken.

**Willow** is used to process Android communication logs into summary statistics that capture phone-based measures of sociability and communicativity. Communication logs on the device contain the actual phone numbers of communication partners, which poses a clear privacy risk if shared in their raw form. To address this concern, communication logs collected via the Beiwe platform employ a one-way hash function to convert phone numbers into surrogate keys. This process occurs directly on the phone before the data are uploaded to the system back-end. Because this mapping is not reversible, it is impossible to recover the original phone numbers from the surrogate keys. Since each phone number is mapped to a unique key, it is possible during analysis to distinguish, for example, one phone call each to five different individuals from five calls to the same individual. Communication logs encompass phone calls, text messages, and multimedia messages. No audio from calls and no message content is collected by Beiwe; however, metadata such as call duration (recorded in seconds) and text message length (number of characters) are available. Willow uses these modified, privacy-preserving communication logs to derive metrics of interest. Examples of summary statistics include the total number of unique incoming callers (people who called the subject), the total number of unique outgoing callees (people whom the subject called), and the total number of characters received in a text message.

**Sycamore** is used to generate survey data summary statistics and provide preprocessed data elements for surveys conducted on the Beiwe platform. The methods are designed to operate on the `survey_timings`, `survey_answers`, and `audio_recordings` data streams generated by Beiwe. For optimal data processing, both the `survey_timings` and `survey_answers` streams are required. The `survey_timings stream` is the preferred source, as it contains precise information about when a user responded to each survey question. However, since survey files are not always uploaded to the Beiwe server, the `survey_answers` stream serves as a backup. The `survey_answers` stream contains only the survey responses and the timestamp of the survey’s final submission; therefore, it should not be used alone for survey processing. The `audio_recordings` stream may also be included in survey summaries. While Sycamore does not process the audio content itself, it can generate summaries that include submission frequencies and survey durations for audio surveys. Examples of summary statistics produced by Sycamore include the survey ID, the number of submitted survey responses, and the average time between survey delivery and submission for complete surveys. Because the implementation of surveys and audio surveys is specific to the Beiwe platform, the features of this tree are intended for use with Beiwe-generated survey and audio files.


**Poplar** is used to implement a range of common functions for data preparation, primarily to support other trees, such as time zone conversion and data reading or writing.

**Bonsai** is used to generate synthetic communication log data and synthetic GPS data, both of which are intended for testing, prototyping, and the development of statistical and machine learning methods.

------------------------

<!-- OLD: **Data elements.** The Beiwe platform can collect both active data (subject input required) and passive data (subject input not required). Currently supported active data types for both Android and iOS are text surveys and audio diary entries and their associated metadata. Passive data can be further divided into two groups: phone sensor data and phone logs. Beiwe collects raw sensor data and raw phone logs, which is absolutely crucial in scientific settings, yet this point remains underappreciated. Relying on generic SDKs for data collection is convenient but for many reasons ineffective: the data generated by different SDKs are not comparable; the algorithms used to generate data are proprietary and hence unsuitable for reproducible research; new data summaries cannot be implemented post data collection; the composition of the metrics collected by SDKs changes in time, making it difficult or impossible to make comparisons across subjects when they are enrolled at different points in time; and data cannot be pooled across studies for later re-analyses or meta-analyses. Currently supported passive data types are the following: accelerometer, gyroscope, magnetometer, GPS, call and text message logs on Android devices (metadata only, no content), proximity,  device motion, reachability, Wi-Fi, Bluetooth, and power state. For each sensor, such as GPS, data collection alternates between an on-cycle (data collected) and an off-cycle (data not collected); logs are collected without sampling if their collection is specified. The investigators specify what data is collected based on the scientific question: for example, a study on mobility might choose to collect accelerometer and gyroscope data used in human activity recognition. The text that appears within the application is also customizable for each study. Data streams that contain identifiers, such as phone numbers in communication (call and text message) logs, are anonymized on the device; the "fuzzy" GPS feature if enabled adds randomly generated noise to the GPS coordinates on the device. Finally, study meta settings are also customizable, and include items such as frequency of uploading data files to the back-end (typically 1 hour) and duration before auto logout from the application (typically 10 minutes). -->

<!-- OLD: **Privacy and security.** All Beiwe data are encrypted while stored on the phone awaiting upload and while in transit, and they are re-encrypted for storage on the study server while at rest. More specifically, during study registration the platform provides the smartphone app with the public half of a 2048-bit RSA encryption key. With this key the device can encrypt data, but only the server, which has the private key, can decrypt it. Thus, the Beiwe application cannot read its own data that it stores temporarily, and therefore there is no way for a user (or anyone else) to export the data. The RSA key is used to encrypt symmetric Advanced Encryption Standard (AES) keys for bulk encryption. These keys are generated as needed by the app and must be decrypted by the study server before data recovery. Data received by the cloud server is re-encrypted with the study master key provided and then stored on the cloud. Some of the collected data contain identifiers: communication logs on Android devices contain phone numbers, and Wi-Fi and Bluetooth scans contain media access control (MAC) address. If the study is configured to collect these data, the identifiers in them are anonymized on the phone, and only anonymized versions of the data are uploaded to the back-end server. Briefly, the Beiwe front-end application generates a unique cryptographic code, called a salt, during the Beiwe registration process, and then uses the salt to encrypt phone numbers and other similar identifiers. The salt never gets uploaded to the server and is known only to the phone for this purpose. Using the industry standard SHA-256 (Secure Hash Algorithm) and PBKDF2 (Password-Based Key Derivation Function 2) algorithms, an identifier is transformed into an 88-character anonymized string that can then be used in data analysis. -->

<!-- OLD: **Use cases.** At the time of writing, Beiwe is or has been used in tens of scientific studies on three continents across various fields, and there are likely several additional studies we are not aware of. Smartphone-based digital phenotyping is potentially very promising in behavioral and mental health [@onnela2016harnessing], and new research tools like Beiwe are especially needed in psychiatry [@torous2016new], where in the context of schizophrenia it has been used to predict patient relapse [@barnett2018relapse], compare passive and active estimates of sleep [@staples2017comparison], and characterize the clinical relevance of digital phenotyping data quality [@torous2018characterizing]. The platform has also been used to assess depressive symptoms in a transdiagnostic cohort [@pelligrini2021estimating] and to capture suicidal thinking during the COVID-19 pandemic [@fortgang2020increase]. There is an increasing amount of research on the use of Beiwe in neurological disorders, such as in the quantification of ALS progression [@berry2019design] and behavioral changes in people with ALS during the COVID-19 pandemic [@beukenhorst2021smartphone]. The platform has been used in the context of cancer to assess postoperative physical activity among patients undergoing cancer surgery [@panda2021smartphone], to capture novel recovery metrics after cancer surgery [@panda2020using], to enhance recovery assessment after breast cancer surgery [@panda2020smartphone], and to enhance cancer care [@wright2018hope]. Digital phenotyping and Beiwe have also been applied to quantifying mobility and quality of life of spine patients [@cote2019digital] and to study psychosocial well-being of individuals after spinal cord injury [@mercier2020digital]. -->


# Acknowledgements

The Principal Investigator, Jukka-Pekka Onnela, is extremely grateful for his NIH Director’s New Innovator Award in 2013 (DP2MH103909) for enabling the crystallization of the concept of digital phenotyping and the construction of the Beiwe platform. He is also grateful to the members of the Onnela Lab.

# References
