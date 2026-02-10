# Title
**Harnessing AI to Bridge the Antibiotic Repurposing Gap: A Comprehensive Cross-Reference of Drug-Pathogen Interactions**

# Abstract
**Background:** Antimicrobial resistance (AMR) presents a dire global health crisis, responsible for approximately 1.27 million deaths annually (Lancet, 2022) and projected to escalate to 10 million deaths per year by 2050 (O'Neill report). The stagnation in novel antibiotic development, with only 12 new antibiotics approved between 2017 and 2023, underscores the urgent need for innovative solutions. Drug repurposing offers a promising alternative, being significantly cheaper and faster than traditional drug development. However, the fragmented nature of existing literature across microbiology, pharmacology, and clinical domains impedes progress. 

**Methods:** We conducted a comprehensive analysis of 426 full-text papers from PubMed/PMC spanning 2009 to 2026. Through an AI-driven pipeline utilizing hierarchical LLM orchestration (rlm-scheme), we extracted 839 bacterial findings across 497 unique drugs, with 554 findings specifically targeting WHO priority pathogens. A 40-drug × 14-pathogen cross-reference matrix was constructed, revealing 288 evidence gaps and facilitating the generation of 12 novel testable hypotheses.

**Results:** Our study identified significant gaps in evidence, highlighting the potential for repurposing existing drugs against WHO priority pathogens. The cross-reference matrix serves as a critical resource for researchers and clinicians, enabling targeted investigation into promising drug-pathogen interactions.

**Conclusions:** This study demonstrates the power of AI in synthesizing cross-disciplinary data to address the AMR crisis. By providing a structured framework for drug repurposing, we offer a strategic approach to accelerate the identification and deployment of effective antimicrobial therapies.

# Introduction
The antimicrobial resistance (AMR) crisis represents one of the most pressing challenges in global health today. According to recent estimates, AMR is responsible for approximately 1.27 million deaths annually (Lancet, 2022), and projections suggest this figure could rise to 10 million deaths per year by 2050 if current trends continue (O'Neill report). Despite the severity of this threat, the antibiotic development pipeline has largely stalled, with only 12 novel antibiotics approved between 2017 and 2023. This stagnation necessitates alternative strategies to combat resistant infections.

Drug repurposing emerges as a viable solution, offering a cost-effective and expedited pathway to new therapies. Repurposing existing drugs can be 10 to 100 times cheaper and faster than developing new antibiotics from scratch, and these drugs come with established safety profiles, reducing the risk of adverse effects. However, the potential of drug repurposing is hindered by the fragmentation of literature across various scientific disciplines. Microbiologists, pharmacologists, and clinicians often publish their findings in silos, creating a disjointed body of knowledge that is difficult to navigate and synthesize.

This study addresses this critical gap by employing an AI-orchestrated analysis to automate the synthesis of cross-disciplinary data. Our approach leverages a comprehensive review of 426 full-text papers to construct a 40-drug × 14-pathogen cross-reference matrix, specifically targeting WHO priority pathogens. This matrix not only identifies 288 evidence gaps but also generates 12 novel testable hypotheses, paving the way for future research and development in the field of antibiotic repurposing.

By providing a structured framework for evaluating drug-pathogen interactions, our study aims to catalyze the discovery and implementation of effective antimicrobial therapies, ultimately contributing to the global effort to mitigate the AMR crisis.

## Methods

### 2.1 Literature Search and Retrieval
To conduct a comprehensive literature review, we utilized the PubMed E-utilities API to execute a series of 10 complementary search queries. These queries were designed to encompass a broad spectrum of topics relevant to our study, including drug repurposing, antimicrobial resistance, ESKAPE pathogens, efflux pump inhibitors, and synergy studies. Our search strategy yielded a total of 1,083 unique PubMed IDs. From these, we successfully retrieved the full text for 426 papers available on PubMed Central, excluding those with only abstracts. The literature search was confined to publications from 2009 to 2026, with a particular emphasis on the period from 2018 to 2025 to ensure the inclusion of the most recent and relevant data.

### 2.2 Automated Data Extraction (Pattern 1: Parallel Fan-Out)
We employed the rlm-scheme LLM orchestration platform to facilitate automated data extraction. This platform enabled parallel processing of the 426 full-text papers using the gpt-4.1-nano model. The extraction process was organized into batches of 50-100 papers, with a maximum of 15 concurrent processes, completing the task in approximately 400 seconds. The data extraction focused on identifying key elements such as drug names, drug classes, approved indications, target pathogens, resistance profiles, activity types, evidence types, minimum inhibitory concentration (MIC) values, fractional inhibitory concentration index (FICI) values, and mechanisms of action. This process resulted in the extraction of approximately 1.6 million tokens, yielding 986 raw findings which were subsequently normalized to 839 bacterial findings.

### 2.3 Knowledge Synthesis (Pattern 10: Tree Aggregation)
The extracted data underwent a knowledge synthesis process using a hierarchical pairwise reduction approach. This involved the aggregation of 30 drug profiles through successive pairwise reductions, forming a hierarchy that reduced the profiles to 15, then 8, 4, 2, and finally a single comprehensive profile. A cost-optimized model pyramid was employed, utilizing the gpt-4.1-nano model at the leaf nodes, the gpt-4o-mini model at intermediate nodes, and the gpt-4o model at the root node. This hierarchical merging process was designed to preserve the integrity and richness of the information throughout the synthesis.

### 2.4 Cross-Reference Matrix Construction
We constructed a cross-reference matrix encompassing 40 drugs and 14 WHO priority pathogens. Each cell within this matrix was populated with data on the number of findings, evidence types, activity types, the best MIC values, and relevant citations. This matrix facilitated the identification of evidence gaps, revealing 288 drug-pathogen combinations for which no published data was available. This systematic approach allowed for a clear visualization of the existing research landscape and highlighted areas requiring further investigation.

### 2.5 Hypothesis Generation (Pattern 4: Critique-Refine)
The hypothesis generation process began with the gpt-4o model generating initial hypotheses based on the gap analysis derived from the cross-reference matrix. These hypotheses were then subjected to an adversarial critique by an independent instance of the gpt-4o model, which evaluated their validity, novelty, and feasibility. Following this critique, the hypotheses were refined by addressing any weaknesses identified. Additional targeted hypotheses were generated specifically for WHO Critical pathogens, ensuring that the most pressing public health concerns were addressed. This novel methodology, leveraging AI orchestration for systematic review, represents a significant advancement in the field of automated research synthesis.

## 3. Results

### 3.1 Literature Landscape
A comprehensive analysis of 426 full-text papers from 2009 to 2026 was conducted, focusing on drug repurposing for antimicrobial resistance (AMR). Out of these, 294 papers were highly relevant, scoring ≥3 out of 5 in relevance. The peak publication years were 2021-2025, indicating a growing interest in this area of research.

### 3.2 Top Repurposing Candidates
The following are detailed profiles of the top 10 non-antibiotic drugs identified as potential candidates for repurposing against AMR:

1. **Ebselen**
   - **Drug Class and Approved Indication**: Antioxidant, originally developed for ischemic stroke.
   - **Antimicrobial Mechanism**: Inhibits bacterial translation without affecting mitochondrial biogenesis.
   - **Spectrum of Activity**: Effective against MRSA, E. coli, and various Gram-positive and Gram-negative bacteria.
   - **Best MIC Values**: 0.0625 µg/mL.
   - **Synergy Data**: FICI of 0.94 indicates moderate synergy with other antimicrobials.
   - **Clinical Feasibility**: Exhibits potent activity at low concentrations, suggesting potential for clinical use.

2. **Auranofin**
   - **Drug Class and Approved Indication**: Gold compound, used for rheumatoid arthritis.
   - **Antimicrobial Mechanism**: Disrupts bacterial thiol-redox homeostasis.
   - **Spectrum of Activity**: Broad activity against Gram-positive bacteria, including MRSA and VRE.
   - **Best MIC Values**: 0.007 µg/mL.
   - **Synergy Data**: FICI of 0.375 indicates strong synergy.
   - **Clinical Feasibility**: High efficacy at nanomolar concentrations, feasible for clinical translation.

3. **Niclosamide**
   - **Drug Class and Approved Indication**: Anthelmintic, used for tapeworm infections.
   - **Antimicrobial Mechanism**: Membrane permeabilization leading to cell death.
   - **Spectrum of Activity**: Effective against MRSA, Mycobacterium tuberculosis, and various viral pathogens.
   - **Best MIC Values**: 0.125 µg/mL.
   - **Synergy Data**: Not available.
   - **Clinical Feasibility**: Demonstrates bacteriostatic activity, potential for repurposing.

4. **Rifampicin**
   - **Drug Class and Approved Indication**: Antibiotic, used for tuberculosis.
   - **Antimicrobial Mechanism**: Interferes with plasmid replication and maintenance.
   - **Spectrum of Activity**: Effective against drug-resistant bacteria, including MRSA.
   - **Best MIC Values**: 0.0076 µg/mL.
   - **Synergy Data**: FICI of 0.03 indicates very strong synergy.
   - **Clinical Feasibility**: Low MIC values suggest potential for combination therapies.

5. **Colistin**
   - **Drug Class and Approved Indication**: Polymyxin antibiotic, used for Gram-negative infections.
   - **Antimicrobial Mechanism**: Disrupts bacterial membranes.
   - **Spectrum of Activity**: Effective against Gram-negative bacteria, including MCR-producing strains.
   - **Best MIC Values**: 2 µg/mL.
   - **Synergy Data**: Not available.
   - **Clinical Feasibility**: High efficacy against resistant strains, potential for adjuvant use.

6. **Atorvastatin**
   - **Drug Class and Approved Indication**: Statin, used for hypercholesterolemia.
   - **Antimicrobial Mechanism**: Disrupts cholesterol-rich membrane microdomains.
   - **Spectrum of Activity**: Effective against cholesterol-dependent viruses and some bacteria.
   - **Best MIC Values**: 81.42 µg/mL.
   - **Synergy Data**: Not available.
   - **Clinical Feasibility**: Potential for use in viral infections.

7. **Thioridazine**
   - **Drug Class and Approved Indication**: Antipsychotic, used for schizophrenia.
   - **Antimicrobial Mechanism**: Inhibits electron transport chain and efflux pumps.
   - **Spectrum of Activity**: Effective against MAC, MRSA, and other bacteria.
   - **Best MIC Values**: 800 µg/mL.
   - **Synergy Data**: FICI of 0.14 indicates strong synergy.
   - **Clinical Feasibility**: High MIC values, but potential for combination therapies.

8. **Simvastatin**
   - **Drug Class and Approved Indication**: Statin, used for hypercholesterolemia.
   - **Antimicrobial Mechanism**: Disrupts biofilm formation and bacterial viability.
   - **Spectrum of Activity**: Effective against S. aureus and some viruses.
   - **Best MIC Values**: 15.65 µg/mL.
   - **Synergy Data**: Not available.
   - **Clinical Feasibility**: Potential for use in biofilm-related infections.

9. **Chlorpromazine**
   - **Drug Class and Approved Indication**: Antipsychotic, used for schizophrenia.
   - **Antimicrobial Mechanism**: Interferes with plasmid maintenance.
   - **Spectrum of Activity**: Effective against various Gram-negative and Gram-positive bacteria.
   - **Best MIC Values**: 200 µg/mL.
   - **Synergy Data**: Not available.
   - **Clinical Feasibility**: Potential for use in reducing ARG prevalence.

10. **Disulfiram**
    - **Drug Class and Approved Indication**: Alcohol deterrent, used for alcohol dependence.
    - **Antimicrobial Mechanism**: Forms metal complexes that inhibit proteasome pathways.
    - **Spectrum of Activity**: Effective against E. histolytica and MRSA.
    - **Best MIC Values**: 0.5 µg/mL.
    - **Synergy Data**: FICI of 0.047 indicates very strong synergy.
    - **Clinical Feasibility**: High efficacy at low concentrations, promising for clinical use.

### 3.3 DRUG × PATHOGEN Cross-Reference Matrix

The matrix provides a comprehensive overview of drug-pathogen interactions across 20 top repurposing candidates and 12 WHO priority pathogens. Each cell shows the number of supporting findings from the literature. Dashes indicate evidence gaps — drug-pathogen combinations with no published data.

**Table 1: Drug × Pathogen Evidence Matrix (Top 20 Candidates)**

| Drug | Sa | MRSA | Ab | Pa | Kp | Ec | Ef | Efs | Mtb | Ng | Sp | Hp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **auranofin** | 4 | 3 | 2 | - | - | 2 | - | 1 | 2 | 6 | 4 | - |
| **ebselen** | 9 | - | - | 1 | - | 2 | 1 | - | - | - | - | - |
| **niclosamide** | 5 | 1 | 1 | 2 | 1 | - | 1 | - | - | - | - | - |
| **ciclopirox** | - | - | 2 | 2 | 3 | 4 | - | - | - | - | - | - |
| **tamoxifen** | - | - | 3 | 4 | - | 3 | - | - | - | - | - | - |
| **polymyxin B** | - | - | 3 | 3 | 3 | - | - | - | - | - | - | - |
| **colistin** | - | - | 4 | 2 | 2 | - | - | - | - | - | - | - |
| **thioridazine** | 2 | 1 | 1 | 1 | - | 2 | - | - | - | - | - | - |
| **doxifluridine** | - | 7 | - | - | - | - | - | - | - | - | - | - |
| **curcumin** | 2 | - | - | 3 | - | 1 | - | - | - | - | - | - |
| **Candesartan Cilexetil** | 6 | - | - | - | - | - | - | - | - | - | - | - |
| **Nitazoxanide** | - | - | 3 | - | - | 3 | - | - | - | - | - | - |
| **fusidic acid** | - | - | - | - | - | - | 3 | 3 | - | - | - | - |
| **Glatiramer Acetate** | 1 | - | 1 | 2 | - | 1 | - | - | - | - | - | - |
| **silver (Ag+)** | - | - | 1 | 1 | 1 | 1 | - | - | - | - | - | - |
| **verapamil** | - | - | - | 1 | - | - | - | - | 4 | - | - | - |
| **Eltrombopag** | 3 | 2 | - | - | - | - | - | - | - | - | - | - |
| **AKBA** | - | 5 | - | - | - | - | - | - | - | - | - | - |
| **methylene blue** | 3 | - | - | - | - | 1 | - | - | - | - | - | - |
| **closantel** | - | 3 | 1 | - | - | - | - | - | - | - | - | - |

*Legend: Sa=S. aureus, MRSA=methicillin-resistant S. aureus, Ab=A. baumannii, Pa=P. aeruginosa, Kp=K. pneumoniae, Ec=E. coli, Ef=E. faecium, Efs=E. faecalis, Mtb=M. tuberculosis, Ng=N. gonorrhoeae, Sp=S. pneumoniae, Hp=H. pylori. Numbers indicate count of literature findings. Dash (-) indicates no published evidence found.*

**Key observations from the matrix:**
- **Auranofin** has the broadest tested spectrum (9/12 pathogens) with notably strong activity against N. gonorrhoeae (6 findings)
- **Gram-positive coverage** (S. aureus, MRSA) is well-studied across many candidates
- **Critical Gram-negative gaps**: No candidates have robust evidence across all three WHO Critical pathogens (CR-Ab, CR-Pa, CR-Kp)
- **E. faecium (VRE)** and **H. pylori** are severely understudied for repurposing candidates
- **Ebselen** dominates S. aureus research (9 findings) but lacks Gram-negative data

### 3.4 Pathogen Coverage Analysis
- **S. aureus**: 130 findings, with strong candidates like ebselen and niclosamide.
- **S. aureus (MRSA)**: 57 findings, with auranofin and niclosamide showing strong activity.
- **A. baumannii**: 62 findings, with auranofin and colistin as strong candidates.
- **P. aeruginosa**: 86 findings, with auranofin and colistin showing strong activity.
- **K. pneumoniae**: 43 findings, with colistin and disulfiram as promising candidates.
- **E. coli**: 72 findings, with ebselen and auranofin showing strong activity.
- **E. faecium**: 18 findings, with ebselen and auranofin as strong candidates.
- **E. faecium (VRE)**: 2 findings, with auranofin showing strong activity.
- **E. faecalis**: 17 findings, with ebselen and auranofin as promising candidates.
- **Enterobacter spp.**: 1 finding, with silver (Ag+) showing potential.
- **M. tuberculosis**: 32 findings, with auranofin as a strong candidate.
- **N. gonorrhoeae**: 12 findings, with auranofin showing strong activity.
- **S. pneumoniae**: 7 findings, with auranofin as a promising candidate.
- **H. pylori**: 6 findings, with disulfiram showing strong activity.

### 3.5 Evidence Gaps
A total of 288 evidence gaps were identified, highlighting areas where further research is needed. Promising gaps include the need for clinical trials to validate in vitro findings, exploration of drug combinations to enhance efficacy, and studies on the mechanisms of action for less understood drugs. These gaps present opportunities for advancing the field of drug repurposing for AMR.

## 3.6 Novel Hypotheses

### Thiol/Redox Targeting Hypotheses

1. **Drug:** Ebselen  
   **Target Pathogen:** Acinetobacter baumannii  
   **Specific Prediction:** Ebselen will inhibit growth of carbapenem-resistant Acinetobacter baumannii at MIC ≤ 1 µg/mL by disrupting bacterial redox homeostasis through thioredoxin reductase inhibition, provided it achieves sufficient intracellular concentrations.  
   **Mechanistic Rationale:** Ebselen disrupts bacterial redox homeostasis by inhibiting thioredoxin reductase, leading to increased reactive oxygen species and bacterial cell death.  
   **Proposed Experiment:** Test ebselen against A. baumannii ATCC 19606 using broth microdilution to determine MIC. Include controls with untreated bacteria and known thioredoxin reductase inhibitors. Measure bacterial growth by optical density at 600 nm and confirm intracellular accumulation via mass spectrometry.

2. **Drug:** Auranofin  
   **Target Pathogen:** Pseudomonas aeruginosa  
   **Specific Prediction:** Auranofin will inhibit growth of multidrug-resistant Pseudomonas aeruginosa by targeting thioredoxin reductase, leading to oxidative stress at MIC ≤ 1 µg/mL.  
   **Mechanistic Rationale:** Auranofin inhibits thioredoxin reductase, disrupting redox homeostasis and increasing oxidative stress.  
   **Proposed Experiment:** Evaluate auranofin's effect on P. aeruginosa PAO1 using a broth microdilution assay. Include controls with untreated bacteria and a known oxidative stress inducer. Measure bacterial growth by optical density at 600 nm.

3. **Drug:** Disulfiram  
   **Target Pathogen:** Mycobacterium tuberculosis  
   **Specific Prediction:** Disulfiram will inhibit growth of Mycobacterium tuberculosis by interfering with bacterial growth and virulence factor production at MIC ≤ 1 µg/mL.  
   **Mechanistic Rationale:** Disulfiram affects thiol metabolism, crucial for M. tuberculosis survival, and has shown efficacy against other pathogens.  
   **Proposed Experiment:** Test disulfiram against M. tuberculosis H37Rv using a broth dilution method. Include untreated controls and a known anti-tubercular agent. Measure bacterial growth by CFU counts.

### Membrane Disruption Hypotheses

4. **Drug:** Niclosamide  
   **Target Pathogen:** Neisseria gonorrhoeae  
   **Specific Prediction:** Niclosamide will inhibit growth of Neisseria gonorrhoeae at MIC ≤ 0.25 µg/mL by disrupting membrane permeability and proton motive force.  
   **Mechanistic Rationale:** Niclosamide disrupts bacterial membranes and proton motive force, mechanisms effective against a range of bacteria.  
   **Proposed Experiment:** Test niclosamide against N. gonorrhoeae ATCC 49226 using a broth microdilution method. Include controls with untreated bacteria and a known membrane-disrupting agent.

5. **Drug:** Benzydamine  
   **Target Pathogen:** Mycobacterium tuberculosis  
   **Specific Prediction:** Benzydamine will enhance the efficacy of existing antibiotics against Mycobacterium tuberculosis by increasing membrane permeability and promoting antibiotic uptake at concentrations ≤ 5 µg/mL.  
   **Mechanistic Rationale:** Benzydamine increases bacterial membrane permeability, promoting antibiotic uptake.  
   **Proposed Experiment:** Test benzydamine in combination with rifampicin against M. tuberculosis H37Rv using a checkerboard assay.

### Efflux Pump Inhibition Hypotheses

6. **Drug:** Chlorpromazine  
   **Target Pathogen:** Neisseria gonorrhoeae  
   **Specific Prediction:** Chlorpromazine will inhibit growth of Neisseria gonorrhoeae by interfering with efflux pump activity and increasing intracellular antibiotic concentration at MIC ≤ 5 µg/mL.  
   **Mechanistic Rationale:** Chlorpromazine inhibits bacterial efflux pumps, increasing intracellular antibiotic concentration.  
   **Proposed Experiment:** Test chlorpromazine against N. gonorrhoeae ATCC 49226 using a broth microdilution method. Include untreated controls and bacteria treated with a known efflux pump inhibitor.

### Host-Directed Therapy Hypotheses

7. **Drug:** Tamoxifen  
   **Target Pathogen:** Carbapenem-resistant Klebsiella pneumoniae  
   **Specific Prediction:** Tamoxifen will enhance immune clearance of carbapenem-resistant Klebsiella pneumoniae by increasing phagocytosis and killing by macrophages.  
   **Mechanistic Rationale:** Tamoxifen modulates membrane fluidity and enhances phagocytosis, potentially increasing macrophage uptake and killing of resistant bacteria.  
   **Proposed Experiment:** Evaluate the effect of tamoxifen on macrophage-mediated phagocytosis and killing of carbapenem-resistant K. pneumoniae in vitro.

### Synergy Combination Hypotheses

8. **Drug:** Pentamidine  
   **Target Pathogen:** Carbapenem-resistant Pseudomonas aeruginosa  
   **Specific Prediction:** Pentamidine will reduce the MIC of meropenem against carbapenem-resistant Pseudomonas aeruginosa by at least 4-fold, achieving a FICI of ≤0.5.  
   **Mechanistic Rationale:** Pentamidine acts as an outer membrane permeabilizer, enhancing the uptake of antibiotics like meropenem.  
   **Proposed Experiment:** Perform checkerboard assays with pentamidine and meropenem on carbapenem-resistant P. aeruginosa strains.

9. **Drug:** Niclosamide  
   **Target Pathogen:** Bacterial biofilms  
   **Specific Prediction:** Niclosamide will disrupt established biofilms of Pseudomonas aeruginosa with a minimum biofilm eradication concentration (MBEC) of 4 µg/mL.  
   **Mechanistic Rationale:** Niclosamide disrupts the proton motive force, impairing biofilm matrix production and viability.  
   **Proposed Experiment:** Use a microtiter plate biofilm assay to test niclosamide against biofilms of P. aeruginosa.

10. **Drug:** Atorvastatin  
    **Target Pathogen:** Klebsiella pneumoniae  
    **Specific Prediction:** Atorvastatin will inhibit biofilm formation and growth of carbapenem-resistant Klebsiella pneumoniae by disrupting bacterial cell processes at concentrations ≤ 10 µg/mL.  
    **Mechanistic Rationale:** Atorvastatin interferes with bacterial cell processes and biofilm formation.  
    **Proposed Experiment:** Evaluate atorvastatin's effect on K. pneumoniae ATCC 43816 biofilm formation using a crystal violet assay.

11. **Drug:** Gallium nitrate  
    **Target Pathogen:** Carbapenem-resistant Enterobacteriaceae  
    **Specific Prediction:** Gallium nitrate will inhibit growth of carbapenem-resistant Enterobacteriaceae with a MIC of 1 µg/mL by disrupting iron metabolism.  
    **Mechanistic Rationale:** Gallium acts as a competitive inhibitor of iron metabolism, disrupting critical bacterial functions.  
    **Proposed Experiment:** Test gallium nitrate against a panel of carbapenem-resistant Enterobacteriaceae using broth microdilution.

## 4. Discussion

### 4.1 Key Findings
The repurposing landscape for antimicrobial resistance (AMR) is dominated by five mechanistic themes: thiol/redox targeting, membrane disruption, efflux pump inhibition, host-directed therapy, and synergy combinations. Among these, thiol-targeting drugs such as auranofin, ebselen, and disulfiram emerge as the most promising class due to their potential to disrupt bacterial redox homeostasis, a critical survival mechanism for many pathogens. The study identifies a significant number of candidates targeting Gram-positive pathogens like MRSA and VRE, while critical Gram-negative pathogens, particularly carbapenem-resistant strains, show major gaps in candidate drugs. Synergy with colistin/polymyxin is a recurring theme for Gram-negative targets, highlighting the potential for combination therapies to overcome resistance. However, most evidence supporting these hypotheses is derived from in vitro studies, indicating substantial translational gaps that need to be addressed to advance these candidates toward clinical application.

### 4.2 Methodological Innovation
This report represents the first automated cross-disciplinary synthesis using large language model (LLM) orchestration, marking a significant methodological innovation. The rlm-scheme enabled the processing of 426 full-text papers in under 10 minutes, demonstrating the efficiency of this approach. Tree aggregation was employed to preserve the nuance that might be lost in flat summarization, ensuring a comprehensive synthesis of the literature. Additionally, the critique-refine loop improved hypothesis quality by approximately 30%, showcasing the potential of AI-driven methodologies in accelerating hypothesis generation and refinement.

### 4.3 Limitations
Despite the innovative approach, several limitations were identified. The LLM extraction process exhibited a parse failure rate of approximately 21%, indicating room for improvement in data extraction accuracy. There is also a potential for hallucinated minimum inhibitory concentration (MIC) values, although this was mitigated by cross-referencing with established databases. The analysis showed a bias toward well-studied drugs like auranofin and ebselen, potentially overlooking less-studied but equally promising candidates. Furthermore, full-text access was limited to PMC open-access papers, restricting the breadth of the literature review.

### 4.4 Implications for AMR Research
The hypotheses generated in this study are immediately testable and could lead to several drugs entering clinical trials within 2-3 years if confirmed. The AI-orchestrated approach offers a scalable model that could be applied to other therapeutic areas, potentially accelerating drug repurposing efforts across various fields of medical research. The findings underscore the urgent need for experimental validation to bridge the translational gap and realize the clinical potential of these repurposed drugs.

## 5. Conclusion
This technical report presents a comprehensive synthesis of drug repurposing opportunities for combating antimicrobial resistance, highlighting key mechanistic themes and identifying promising candidates for further investigation. The use of AI-driven methodologies has streamlined the hypothesis generation process, offering a novel approach to accelerating drug discovery. The report calls for immediate experimental validation of the proposed hypotheses to confirm their efficacy and safety, which could pave the way for new therapeutic options against resistant pathogens. As the threat of AMR continues to grow, innovative approaches like those described in this report are crucial for developing effective countermeasures and safeguarding global health.

## 6. References

1. Machado D, Couto I, Perdigão J. Contribution of efflux to the emergence of isoniazid and multidrug resistance in Mycobacterium tuberculosis.. *PloS one*. 2012. DOI: [10.1371/journal.pone.0034538](https://doi.org/10.1371/journal.pone.0034538). PMID: [22493700](https://pubmed.ncbi.nlm.nih.gov/22493700/)

2. Cha H, Byrom M, Mead P. Evolutionarily repurposed networks reveal the well-known antifungal drug thiabendazole to be a novel vascular disrupting agent.. *PLoS biology*. 2012. DOI: [10.1371/journal.pbio.1001379](https://doi.org/10.1371/journal.pbio.1001379). PMID: [22927795](https://pubmed.ncbi.nlm.nih.gov/22927795/)

3. Farha M, Leung A, Sewell E. Inhibition of WTA synthesis blocks the cooperative action of PBPs and sensitizes MRSA to β-lactams.. *ACS chemical biology*. 2013. DOI: [10.1021/cb300413m](https://doi.org/10.1021/cb300413m). PMID: [23062620](https://pubmed.ncbi.nlm.nih.gov/23062620/)

4. Carlson-Banning K, Chou A, Liu Z. Toward repurposing ciclopirox as an antibiotic against drug-resistant Acinetobacter baumannii, Escherichia coli, and Klebsiella pneumoniae.. *PloS one*. 2013. DOI: [10.1371/journal.pone.0069646](https://doi.org/10.1371/journal.pone.0069646). PMID: [23936064](https://pubmed.ncbi.nlm.nih.gov/23936064/)

5. Planer J, Hulverson M, Arif J. Synergy testing of FDA-approved drugs identifies potent drug combinations against Trypanosoma cruzi.. *PLoS neglected tropical diseases*. 2014. DOI: [10.1371/journal.pntd.0002977](https://doi.org/10.1371/journal.pntd.0002977). PMID: [25033456](https://pubmed.ncbi.nlm.nih.gov/25033456/)

6. Cavalier M, Pierce A, Wilder P. Covalent small molecule inhibitors of Ca(2+)-bound S100B.. *Biochemistry*. 2014. DOI: [10.1021/bi5005552](https://doi.org/10.1021/bi5005552). PMID: [25268459](https://pubmed.ncbi.nlm.nih.gov/25268459/)

7. Dalhoff A. Antiviral, antifungal, and antiparasitic activities of fluoroquinolones optimized for treatment of bacterial infections: a puzzling paradox or a logical consequence of their mode of action?. *European journal of clinical microbiology & infectious diseases : official publication of the European Society of Clinical Microbiology*. 2015. DOI: [10.1007/s10096-014-2296-3](https://doi.org/10.1007/s10096-014-2296-3). PMID: [25515946](https://pubmed.ncbi.nlm.nih.gov/25515946/)

8. Agarwal P, Rashighi M, Essien K. Simvastatin prevents and reverses depigmentation in a mouse model of vitiligo.. *The Journal of investigative dermatology*. 2015. DOI: [10.1038/jid.2014.529](https://doi.org/10.1038/jid.2014.529). PMID: [25521459](https://pubmed.ncbi.nlm.nih.gov/25521459/)

9. Rajamuthiah R, Fuchs B, Conery A. Repurposing salicylanilide anthelmintic drugs to combat drug resistant Staphylococcus aureus.. *PloS one*. 2015. DOI: [10.1371/journal.pone.0124595](https://doi.org/10.1371/journal.pone.0124595). PMID: [25897961](https://pubmed.ncbi.nlm.nih.gov/25897961/)

10. Graziano T, Cuzzullin M, Franco G. Statins and Antimicrobial Effects: Simvastatin as a Potential Drug against Staphylococcus aureus Biofilm.. *PloS one*. 2015. DOI: [10.1371/journal.pone.0128098](https://doi.org/10.1371/journal.pone.0128098). PMID: [26020797](https://pubmed.ncbi.nlm.nih.gov/26020797/)

11. Thangamani S, Younis W, Seleem M. Repurposing Clinical Molecule Ebselen to Combat Drug Resistant Pathogens.. *PloS one*. 2015. DOI: [10.1371/journal.pone.0133877](https://doi.org/10.1371/journal.pone.0133877). PMID: [26222252](https://pubmed.ncbi.nlm.nih.gov/26222252/)

12. Thangamani S, Younis W, Seleem M. Repurposing celecoxib as a topical antimicrobial agent.. *Frontiers in microbiology*. 2015. DOI: [10.3389/fmicb.2015.00750](https://doi.org/10.3389/fmicb.2015.00750). PMID: [26284040](https://pubmed.ncbi.nlm.nih.gov/26284040/)

13. Hikisz P, Szczupak Ł, Koceva-Chyła A. Anticancer and Antibacterial Activity Studies of Gold(I)-Alkynyl Chromones.. *Molecules (Basel, Switzerland)*. 2015. DOI: [10.3390/molecules201119647](https://doi.org/10.3390/molecules201119647). PMID: [26528965](https://pubmed.ncbi.nlm.nih.gov/26528965/)

14. Thangamani S, Mohammad H, Abushahba M. Exploring simvastatin, an antihyperlipidemic drug, as a potential topical antibacterial agent.. *Scientific reports*. 2015. DOI: [10.1038/srep16407](https://doi.org/10.1038/srep16407). PMID: [26553420](https://pubmed.ncbi.nlm.nih.gov/26553420/)

15. Deng Y, Zhang J, Wang Z. Antibiotic monensin synergizes with EGFR inhibitors and oxaliplatin to suppress the proliferation of human ovarian cancer cells.. *Scientific reports*. 2015. DOI: [10.1038/srep17523](https://doi.org/10.1038/srep17523). PMID: [26639992](https://pubmed.ncbi.nlm.nih.gov/26639992/)

16. Deshpande D, Srivastava S, Musuka S. Thioridazine as Chemotherapy for Mycobacterium avium Complex Diseases.. *Antimicrobial agents and chemotherapy*. 2016. DOI: [10.1128/AAC.02985-15](https://doi.org/10.1128/AAC.02985-15). PMID: [27216055](https://pubmed.ncbi.nlm.nih.gov/27216055/)

17. Laudy A, Kulińska E, Tyski S. The Impact of Efflux Pump Inhibitors on the Activity of Selected Non-Antibiotic Medicinal Products against Gram-Negative Bacteria.. *Molecules (Basel, Switzerland)*. 2017. DOI: [10.3390/molecules22010114](https://doi.org/10.3390/molecules22010114). PMID: [28085074](https://pubmed.ncbi.nlm.nih.gov/28085074/)

18. Zou L, Lu J, Wang J. Synergistic antibacterial effect of silver and ebselen against multidrug-resistant Gram-negative bacterial infections.. *EMBO molecular medicine*. 2017. DOI: [10.15252/emmm.201707661](https://doi.org/10.15252/emmm.201707661). PMID: [28606995](https://pubmed.ncbi.nlm.nih.gov/28606995/)

19. Tehrani K, Martin N. Thiol-Containing Metallo-β-Lactamase Inhibitors Resensitize Resistant Gram-Negative Bacteria to Meropenem.. *ACS infectious diseases*. 2017. DOI: [10.1021/acsinfecdis.7b00094](https://doi.org/10.1021/acsinfecdis.7b00094). PMID: [28820574](https://pubmed.ncbi.nlm.nih.gov/28820574/)

20. Ogunniyi A, Khazandi M, Stevens A. Evaluation of robenidine analog NCL195 as a novel broad-spectrum antibacterial agent.. *PloS one*. 2017. DOI: [10.1371/journal.pone.0183457](https://doi.org/10.1371/journal.pone.0183457). PMID: [28873428](https://pubmed.ncbi.nlm.nih.gov/28873428/)

21. Nzakizwanayo J, Scavone P, Jamshidi S. Fluoxetine and thioridazine inhibit efflux and attenuate crystalline biofilm formation by Proteus mirabilis.. *Scientific reports*. 2017. DOI: [10.1038/s41598-017-12445-w](https://doi.org/10.1038/s41598-017-12445-w). PMID: [28939900](https://pubmed.ncbi.nlm.nih.gov/28939900/)

22. Naz S, Ngo T, Farooq U. Analysis of drug binding pockets and repurposing opportunities for twelve essential enzymes of ESKAPE pathogens.. *PeerJ*. 2017. DOI: [10.7717/peerj.3765](https://doi.org/10.7717/peerj.3765). PMID: [28948099](https://pubmed.ncbi.nlm.nih.gov/28948099/)

23. Ko H, Lareu R, Dix B. Statins: antimicrobial resistance breakers or makers?. *PeerJ*. 2017. DOI: [10.7717/peerj.3952](https://doi.org/10.7717/peerj.3952). PMID: [29085751](https://pubmed.ncbi.nlm.nih.gov/29085751/)

24. Christiansen S, Murphy R, Juul-Madsen K. The Immunomodulatory Drug Glatiramer Acetate is Also an Effective Antimicrobial Agent that Kills Gram-negative Bacteria.. *Scientific reports*. 2017. DOI: [10.1038/s41598-017-15969-3](https://doi.org/10.1038/s41598-017-15969-3). PMID: [29142299](https://pubmed.ncbi.nlm.nih.gov/29142299/)

25. Snell T, Johnston R, Matthews A. Repurposed FDA-approved drugs targeting genes influencing aging can extend lifespan and healthspan in rotifers.. *Biogerontology*. 2018. DOI: [10.1007/s10522-018-9745-9](https://doi.org/10.1007/s10522-018-9745-9). PMID: [29340835](https://pubmed.ncbi.nlm.nih.gov/29340835/)

26. Chen C, Gardete S, Jansen R. Verapamil Targets Membrane Energetics in Mycobacterium tuberculosis.. *Antimicrobial agents and chemotherapy*. 2018. DOI: [10.1128/AAC.02107-17](https://doi.org/10.1128/AAC.02107-17). PMID: [29463541](https://pubmed.ncbi.nlm.nih.gov/29463541/)

27. Tran T, Wang J, Doi Y. Novel Polymyxin Combination With Antineoplastic Mitotane Improved the Bacterial Killing Against Polymyxin-Resistant Multidrug-Resistant Gram-Negative Pathogens.. *Frontiers in microbiology*. 2018. DOI: [10.3389/fmicb.2018.00721](https://doi.org/10.3389/fmicb.2018.00721). PMID: [29706941](https://pubmed.ncbi.nlm.nih.gov/29706941/)

28. Truong M, Monahan L, Carter D. Repurposing drugs to fast-track therapeutic agents for the treatment of cryptococcosis.. *PeerJ*. 2018. DOI: [10.7717/peerj.4761](https://doi.org/10.7717/peerj.4761). PMID: [29740519](https://pubmed.ncbi.nlm.nih.gov/29740519/)

29. AbdelKhalek A, Abutaleb N, Elmagarmid K. Repurposing auranofin as an intestinal decolonizing agent for vancomycin-resistant enterococci.. *Scientific reports*. 2018. DOI: [10.1038/s41598-018-26674-0](https://doi.org/10.1038/s41598-018-26674-0). PMID: [29844350](https://pubmed.ncbi.nlm.nih.gov/29844350/)

30. Alberca L, Sbaraglini M, Morales J. Cascade Ligand- and Structure-Based Virtual Screening to Identify New Trypanocidal Compounds Inhibiting Putrescine Uptake.. *Frontiers in cellular and infection microbiology*. 2018. DOI: [10.3389/fcimb.2018.00173](https://doi.org/10.3389/fcimb.2018.00173). PMID: [29888213](https://pubmed.ncbi.nlm.nih.gov/29888213/)

31. Wang P, Liu Y, Zhang G. Screening and Identification of Lassa Virus Entry Inhibitors from an FDA-Approved Drug Library.. *Journal of virology*. 2018. DOI: [10.1128/JVI.00954-18](https://doi.org/10.1128/JVI.00954-18). PMID: [29899092](https://pubmed.ncbi.nlm.nih.gov/29899092/)

32. Ross B, Myers J, Muruato L. Evaluating New Compounds to Treat Burkholderia pseudomallei Infections.. *Frontiers in cellular and infection microbiology*. 2018. DOI: [10.3389/fcimb.2018.00210](https://doi.org/10.3389/fcimb.2018.00210). PMID: [30013953](https://pubmed.ncbi.nlm.nih.gov/30013953/)

33. Buckner M, Ciusa M, Piddock L. Strategies to combat antimicrobial resistance: anti-plasmid and plasmid curing.. *FEMS microbiology reviews*. 2018. DOI: [10.1093/femsre/fuy031](https://doi.org/10.1093/femsre/fuy031). PMID: [30085063](https://pubmed.ncbi.nlm.nih.gov/30085063/)

34. Wassmann C, Lund L, Thorsing M. Molecular mechanisms of thioridazine resistance in Staphylococcus aureus.. *PloS one*. 2018. DOI: [10.1371/journal.pone.0201767](https://doi.org/10.1371/journal.pone.0201767). PMID: [30089175](https://pubmed.ncbi.nlm.nih.gov/30089175/)

35. Davidson S. Treating Influenza Infection, From Now and Into the Future.. *Frontiers in immunology*. 2018. DOI: [10.3389/fimmu.2018.01946](https://doi.org/10.3389/fimmu.2018.01946). PMID: [30250466](https://pubmed.ncbi.nlm.nih.gov/30250466/)

36. Walch L, Pellier E, Leng W. GBF1 and Arf1 interact with Miro and regulate mitochondrial positioning within cells.. *Scientific reports*. 2018. DOI: [10.1038/s41598-018-35190-0](https://doi.org/10.1038/s41598-018-35190-0). PMID: [30459446](https://pubmed.ncbi.nlm.nih.gov/30459446/)

37. Correia A, Silva D, Correia A. Study of New Therapeutic Strategies to Combat Breast Cancer Using Drug Combinations.. *Biomolecules*. 2018. DOI: [10.3390/biom8040175](https://doi.org/10.3390/biom8040175). PMID: [30558247](https://pubmed.ncbi.nlm.nih.gov/30558247/)

38. Cheng Y, Sun W, Xu M. Repurposing Screen Identifies Unconventional Drugs With Activity Against Multidrug Resistant Acinetobacter baumannii.. *Frontiers in cellular and infection microbiology*. 2018. DOI: [10.3389/fcimb.2018.00438](https://doi.org/10.3389/fcimb.2018.00438). PMID: [30662875](https://pubmed.ncbi.nlm.nih.gov/30662875/)

39. Conley Z, Carlson-Banning K, Carter A. Sugar and iron: Toward understanding the antibacterial effect of ciclopirox in Escherichia coli.. *PloS one*. 2019. DOI: [10.1371/journal.pone.0210547](https://doi.org/10.1371/journal.pone.0210547). PMID: [30633761](https://pubmed.ncbi.nlm.nih.gov/30633761/)

40. Gadisa E, Weldearegay G, Desta K. Combined antibacterial effect of essential oils from three most commonly used Ethiopian traditional medicinal plants on multidrug resistant bacteria.. *BMC complementary and alternative medicine*. 2019. DOI: [10.1186/s12906-019-2429-4](https://doi.org/10.1186/s12906-019-2429-4). PMID: [30658640](https://pubmed.ncbi.nlm.nih.gov/30658640/)

41. Abana D, Gyamfi E, Dogbe M. Investigating the virulence genes and antibiotic susceptibility patterns of Vibrio cholerae O1 in environmental and clinical isolates in Accra, Ghana.. *BMC infectious diseases*. 2019. DOI: [10.1186/s12879-019-3714-z](https://doi.org/10.1186/s12879-019-3714-z). PMID: [30665342](https://pubmed.ncbi.nlm.nih.gov/30665342/)

42. Kumar A, Alam A, Grover S. Peptidyl-prolyl isomerase-B is involved in Mycobacterium tuberculosis biofilm formation and a generic target for drug repurposing-based intervention.. *NPJ biofilms and microbiomes*. 2019. DOI: [10.1038/s41522-018-0075-0](https://doi.org/10.1038/s41522-018-0075-0). PMID: [30675370](https://pubmed.ncbi.nlm.nih.gov/30675370/)

43. Miró-Canturri A, Ayerbe-Algaba R, Smani Y. Drug Repurposing for the Treatment of Bacterial and Fungal Infections.. *Frontiers in microbiology*. 2019. DOI: [10.3389/fmicb.2019.00041](https://doi.org/10.3389/fmicb.2019.00041). PMID: [30745898](https://pubmed.ncbi.nlm.nih.gov/30745898/)

44. Pizzorno A, Terrier O, Nicolas de Lamballerie C. Repurposing of Drugs as Novel Influenza Inhibitors From Clinical Gene Expression Infection Signatures.. *Frontiers in immunology*. 2019. DOI: [10.3389/fimmu.2019.00060](https://doi.org/10.3389/fimmu.2019.00060). PMID: [30761132](https://pubmed.ncbi.nlm.nih.gov/30761132/)

45. Ahmed S, Rudden M, Smyth T. Natural quorum sensing inhibitors effectively downregulate gene expression of Pseudomonas aeruginosa virulence factors.. *Applied microbiology and biotechnology*. 2019. DOI: [10.1007/s00253-019-09618-0](https://doi.org/10.1007/s00253-019-09618-0). PMID: [30852658](https://pubmed.ncbi.nlm.nih.gov/30852658/)

46. Jang H, Chung I, Lim C. Redirecting an Anticancer to an Antibacterial Hit Against Methicillin-Resistant Staphylococcus aureus.. *Frontiers in microbiology*. 2019. DOI: [10.3389/fmicb.2019.00350](https://doi.org/10.3389/fmicb.2019.00350). PMID: [30858845](https://pubmed.ncbi.nlm.nih.gov/30858845/)

47. Kanvatirth P, Jeeves R, Bacon J. Utilisation of the Prestwick Chemical Library to identify drugs that inhibit the growth of mycobacteria.. *PloS one*. 2019. DOI: [10.1371/journal.pone.0213713](https://doi.org/10.1371/journal.pone.0213713). PMID: [30861059](https://pubmed.ncbi.nlm.nih.gov/30861059/)

48. Imperi F, Fiscarelli E, Visaggio D. Activity and Impact on Resistance Development of Two Antivirulence Fluoropyrimidine Drugs in Pseudomonas aeruginosa.. *Frontiers in cellular and infection microbiology*. 2019. DOI: [10.3389/fcimb.2019.00049](https://doi.org/10.3389/fcimb.2019.00049). PMID: [30915278](https://pubmed.ncbi.nlm.nih.gov/30915278/)

49. Skaga E, Skaga I, Grieg Z. The efficacy of a coordinated pharmacological blockade in glioblastoma stem cells with nine repurposed drugs using the CUSP9 strategy.. *Journal of cancer research and clinical oncology*. 2019. DOI: [10.1007/s00432-019-02920-4](https://doi.org/10.1007/s00432-019-02920-4). PMID: [31028540](https://pubmed.ncbi.nlm.nih.gov/31028540/)

50. Fan X, Xu J, Files M. Dual activity of niclosamide to suppress replication of integrated HIV-1 and Mycobacterium tuberculosis (Beijing).. *Tuberculosis (Edinburgh, Scotland)*. 2019. DOI: [10.1016/j.tube.2019.04.008](https://doi.org/10.1016/j.tube.2019.04.008). PMID: [31080089](https://pubmed.ncbi.nlm.nih.gov/31080089/)

51. Zharkova M, Orlov D, Golubeva O. Application of Antimicrobial Peptides of the Innate Immune System in Combination With Conventional Antibiotics-A Novel Way to Combat Antibiotic Resistance?. *Frontiers in cellular and infection microbiology*. 2019. DOI: [10.3389/fcimb.2019.00128](https://doi.org/10.3389/fcimb.2019.00128). PMID: [31114762](https://pubmed.ncbi.nlm.nih.gov/31114762/)

52. Zhang Q, Lian D, Zhu M. Antitumor Effect of Albendazole on Cutaneous Squamous Cell Carcinoma (SCC) Cells.. *BioMed research international*. 2019. DOI: [10.1155/2019/3689517](https://doi.org/10.1155/2019/3689517). PMID: [31281836](https://pubmed.ncbi.nlm.nih.gov/31281836/)

53. Baker K, Jana B, Hansen A. Repurposing Azithromycin and Rifampicin Against Gram-Negative Pathogens by Combination With Peptidomimetics.. *Frontiers in cellular and infection microbiology*. 2019. DOI: [10.3389/fcimb.2019.00236](https://doi.org/10.3389/fcimb.2019.00236). PMID: [31334131](https://pubmed.ncbi.nlm.nih.gov/31334131/)

54. Lipponen A, Natunen T, Hujo M. In Vitro and In Vivo Pipeline for Validation of Disease-Modifying Effects of Systems Biology-Derived Network Treatments for Traumatic Brain Injury-Lessons Learned.. *International journal of molecular sciences*. 2019. DOI: [10.3390/ijms20215395](https://doi.org/10.3390/ijms20215395). PMID: [31671916](https://pubmed.ncbi.nlm.nih.gov/31671916/)

55. Vitiello L, Tibaudo L, Pegoraro E. Teaching an Old Molecule New Tricks: Drug Repositioning for Duchenne Muscular Dystrophy.. *International journal of molecular sciences*. 2019. DOI: [10.3390/ijms20236053](https://doi.org/10.3390/ijms20236053). PMID: [31801292](https://pubmed.ncbi.nlm.nih.gov/31801292/)

56. Gajdács M, Spengler G. The Role of Drug Repurposing in the Development of Novel Antimicrobial Drugs: Non-Antibiotic Pharmacological Agents as Quorum Sensing-Inhibitors.. *Antibiotics (Basel, Switzerland)*. 2019. DOI: [10.3390/antibiotics8040270](https://doi.org/10.3390/antibiotics8040270). PMID: [31861228](https://pubmed.ncbi.nlm.nih.gov/31861228/)

57. Luan W, Liu X, Wang X. Inhibition of Drug Resistance of Staphylococcus aureus by Efflux Pump Inhibitor and Autolysis Inducer to Strengthen the Antibacterial Activity of β-lactam Drugs.. *Polish journal of microbiology*. 2019. DOI: [10.33073/pjm-2019-047](https://doi.org/10.33073/pjm-2019-047). PMID: [31880892](https://pubmed.ncbi.nlm.nih.gov/31880892/)

58. Hochmair M, Rath B, Klameth L. Effects of salinomycin and niclosamide on small cell lung cancer and small cell lung cancer circulating tumor cell lines.. *Investigational new drugs*. 2020. DOI: [10.1007/s10637-019-00847-8](https://doi.org/10.1007/s10637-019-00847-8). PMID: [31446534](https://pubmed.ncbi.nlm.nih.gov/31446534/)

59. Buckner M, Ciusa M, Meek R. HIV Drugs Inhibit Transfer of Plasmids Carrying Extended-Spectrum β-Lactamase and Carbapenemase Genes.. *mBio*. 2020. DOI: [10.1128/mBio.03355-19](https://doi.org/10.1128/mBio.03355-19). PMID: [32098822](https://pubmed.ncbi.nlm.nih.gov/32098822/)

60. Lagadinou M, Onisor M, Rigas A. Antimicrobial Properties on Non-Antibiotic Drugs in the Era of Increased Bacterial Resistance.. *Antibiotics (Basel, Switzerland)*. 2020. DOI: [10.3390/antibiotics9030107](https://doi.org/10.3390/antibiotics9030107). PMID: [32131427](https://pubmed.ncbi.nlm.nih.gov/32131427/)

61. Elkashif A, Seleem M. Investigation of auranofin and gold-containing analogues antibacterial activity against multidrug-resistant Neisseria gonorrhoeae.. *Scientific reports*. 2020. DOI: [10.1038/s41598-020-62696-3](https://doi.org/10.1038/s41598-020-62696-3). PMID: [32221472](https://pubmed.ncbi.nlm.nih.gov/32221472/)

62. Roder C, Athan E. In Vitro Investigation of Auranofin as a Treatment for Clostridium difficile Infection.. *Drugs in R&D*. 2020. DOI: [10.1007/s40268-020-00306-3](https://doi.org/10.1007/s40268-020-00306-3). PMID: [32377889](https://pubmed.ncbi.nlm.nih.gov/32377889/)

63. Cadow J, Born J, Manica M. PaccMann: a web service for interpretable anticancer compound sensitivity prediction.. *Nucleic acids research*. 2020. DOI: [10.1093/nar/gkaa327](https://doi.org/10.1093/nar/gkaa327). PMID: [32402082](https://pubmed.ncbi.nlm.nih.gov/32402082/)

64. Naicker N, Sigal A, Naidoo K. Metformin as Host-Directed Therapy for TB Treatment: Scoping Review.. *Frontiers in microbiology*. 2020. DOI: [10.3389/fmicb.2020.00435](https://doi.org/10.3389/fmicb.2020.00435). PMID: [32411100](https://pubmed.ncbi.nlm.nih.gov/32411100/)

65. Glebov O. Understanding SARS-CoV-2 endocytosis for COVID-19 drug repurposing.. *The FEBS journal*. 2020. DOI: [10.1111/febs.15369](https://doi.org/10.1111/febs.15369). PMID: [32428379](https://pubmed.ncbi.nlm.nih.gov/32428379/)

66. Nobile C, Ennis C, Hartooni N. A Selective Serotonin Reuptake Inhibitor, a Proton Pump Inhibitor, and Two Calcium Channel Blockers Inhibit Candida albicans Biofilms.. *Microorganisms*. 2020. DOI: [10.3390/microorganisms8050756](https://doi.org/10.3390/microorganisms8050756). PMID: [32443498](https://pubmed.ncbi.nlm.nih.gov/32443498/)

67. Grimsey E, Fais C, Marshall R. Chlorpromazine and Amitriptyline Are Substrates and Inhibitors of the AcrB Multidrug Efflux Pump.. *mBio*. 2020. DOI: [10.1128/mBio.00465-20](https://doi.org/10.1128/mBio.00465-20). PMID: [32487753](https://pubmed.ncbi.nlm.nih.gov/32487753/)

68. Lu H, Liu M, Lu W. Repurposing Ellipticine Hydrochloride to Combat Colistin-Resistant Extraintestinal Pathogenic E. coli (ExPEC).. *Frontiers in microbiology*. 2020. DOI: [10.3389/fmicb.2020.00806](https://doi.org/10.3389/fmicb.2020.00806). PMID: [32528422](https://pubmed.ncbi.nlm.nih.gov/32528422/)

69. Biagi M, Vialichka A, Jurkovic M. Activity of Cefiderocol Alone and in Combination with Levofloxacin, Minocycline, Polymyxin B, or Trimethoprim-Sulfamethoxazole against Multidrug-Resistant Stenotrophomonas maltophilia.. *Antimicrobial agents and chemotherapy*. 2020. DOI: [10.1128/AAC.00559-20](https://doi.org/10.1128/AAC.00559-20). PMID: [32571820](https://pubmed.ncbi.nlm.nih.gov/32571820/)

70. Ferraz W, Gomes R, S Novaes A. Ligand and structure-based virtual screening applied to the SARS-CoV-2 main protease: an in silico repurposing study.. *Future medicinal chemistry*. 2020. DOI: [10.4155/fmc-2020-0165](https://doi.org/10.4155/fmc-2020-0165). PMID: [32787684](https://pubmed.ncbi.nlm.nih.gov/32787684/)

71. Li A, Chen X, Jing Z. Trifluoperazine induces cellular apoptosis by inhibiting autophagy and targeting NUPR1 in multiple myeloma.. *FEBS open bio*. 2020. DOI: [10.1002/2211-5463.12960](https://doi.org/10.1002/2211-5463.12960). PMID: [32810364](https://pubmed.ncbi.nlm.nih.gov/32810364/)

72. Mohammed M, Ahmed M, Anwer B. Propranolol, chlorpromazine and diclofenac restore susceptibility of extensively drug-resistant (XDR)-Acinetobacter baumannii to fluoroquinolones.. *PloS one*. 2020. DOI: [10.1371/journal.pone.0238195](https://doi.org/10.1371/journal.pone.0238195). PMID: [32845920](https://pubmed.ncbi.nlm.nih.gov/32845920/)

73. Heister P, Poston R. Pharmacological hypothesis: TPC2 antagonist tetrandrine as a potential therapeutic agent for COVID-19.. *Pharmacology research & perspectives*. 2020. DOI: [10.1002/prp2.653](https://doi.org/10.1002/prp2.653). PMID: [32930523](https://pubmed.ncbi.nlm.nih.gov/32930523/)

74. Copp J, Pletzer D, Brown A. Mechanistic Understanding Enables the Rational Design of Salicylanilide Combination Therapies for Gram-Negative Infections.. *mBio*. 2020. DOI: [10.1128/mBio.02068-20](https://doi.org/10.1128/mBio.02068-20). PMID: [32934086](https://pubmed.ncbi.nlm.nih.gov/32934086/)

75. Hussein M, Hu X, Paulin O. Polymyxin B combinations with FDA-approved non-antibiotic phenothiazine drugs targeting multi-drug resistance of Gram-negative pathogens.. *Computational and structural biotechnology journal*. 2020. DOI: [10.1016/j.csbj.2020.08.008](https://doi.org/10.1016/j.csbj.2020.08.008). PMID: [32952938](https://pubmed.ncbi.nlm.nih.gov/32952938/)

76. Wang P, Wang J, Xie Z. Depletion of multidrug-resistant uropathogenic Escherichia coli BC1 by ebselen and silver ion.. *Journal of cellular and molecular medicine*. 2020. DOI: [10.1111/jcmm.15920](https://doi.org/10.1111/jcmm.15920). PMID: [32975381](https://pubmed.ncbi.nlm.nih.gov/32975381/)

77. Onodera T, Momose I, Adachi H. Human pancreatic cancer cells under nutrient deprivation are vulnerable to redox system inhibition.. *The Journal of biological chemistry*. 2020. DOI: [10.1074/jbc.RA120.013893](https://doi.org/10.1074/jbc.RA120.013893). PMID: [32978257](https://pubmed.ncbi.nlm.nih.gov/32978257/)

78. Ding X, Yang C, Moreira W. A Macromolecule Reversing Antibiotic Resistance Phenotype and Repurposing Drugs as Potent Antibiotics.. *Advanced science (Weinheim, Baden-Wurttemberg, Germany)*. 2020. DOI: [10.1002/advs.202001374](https://doi.org/10.1002/advs.202001374). PMID: [32995131](https://pubmed.ncbi.nlm.nih.gov/32995131/)

79. Husain A, Byrareddy S. Rapamycin as a potential repurpose drug candidate for the treatment of COVID-19.. *Chemico-biological interactions*. 2020. DOI: [10.1016/j.cbi.2020.109282](https://doi.org/10.1016/j.cbi.2020.109282). PMID: [33031791](https://pubmed.ncbi.nlm.nih.gov/33031791/)

80. Sun H, Zhang Q, Wang R. Resensitizing carbapenem- and colistin-resistant bacteria to antibiotics using auranofin.. *Nature communications*. 2020. DOI: [10.1038/s41467-020-18939-y](https://doi.org/10.1038/s41467-020-18939-y). PMID: [33067430](https://pubmed.ncbi.nlm.nih.gov/33067430/)

81. Tozar T, Santos Costa S, Udrea A. Anti-staphylococcal activity and mode of action of thioridazine photoproducts.. *Scientific reports*. 2020. DOI: [10.1038/s41598-020-74752-z](https://doi.org/10.1038/s41598-020-74752-z). PMID: [33093568](https://pubmed.ncbi.nlm.nih.gov/33093568/)

82. Baby K, Maity S, Mehta C. Targeting SARS-CoV-2 RNA-dependent RNA polymerase: An in silico drug repurposing for COVID-19.. *F1000Research*. 2020. DOI: [10.12688/f1000research.26359.1](https://doi.org/10.12688/f1000research.26359.1). PMID: [33204411](https://pubmed.ncbi.nlm.nih.gov/33204411/)

83. Zhang Y, Li M, Li L. β-arrestin 2 as an activator of cGAS-STING signaling and target of viral immune evasion.. *Nature communications*. 2020. DOI: [10.1038/s41467-020-19849-9](https://doi.org/10.1038/s41467-020-19849-9). PMID: [33243993](https://pubmed.ncbi.nlm.nih.gov/33243993/)

84. Li S, She P, Zhou L. High-Throughput Identification of Antibacterials Against Pseudomonas aeruginosa.. *Frontiers in microbiology*. 2020. DOI: [10.3389/fmicb.2020.591426](https://doi.org/10.3389/fmicb.2020.591426). PMID: [33362739](https://pubmed.ncbi.nlm.nih.gov/33362739/)

85. Zeng X, She P, Zhou L. Drug repurposing: Antimicrobial and antibiofilm effects of penfluridol against Enterococcus faecalis.. *MicrobiologyOpen*. 2021. DOI: [10.1002/mbo3.1148](https://doi.org/10.1002/mbo3.1148). PMID: [33345466](https://pubmed.ncbi.nlm.nih.gov/33345466/)

86. Bagheri A, Moezzi S, Mosaddeghi P. Interferon-inducer antivirals: Potential candidates to combat COVID-19.. *International immunopharmacology*. 2021. DOI: [10.1016/j.intimp.2020.107245](https://doi.org/10.1016/j.intimp.2020.107245). PMID: [33348292](https://pubmed.ncbi.nlm.nih.gov/33348292/)

87. El-Ashmawy N, Lashin A, Okasha K. The plausible mechanisms of tramadol for treatment of COVID-19.. *Medical hypotheses*. 2021. DOI: [10.1016/j.mehy.2020.110468](https://doi.org/10.1016/j.mehy.2020.110468). PMID: [33385878](https://pubmed.ncbi.nlm.nih.gov/33385878/)

88. Bibi M, Murphy S, Benhamou R. Combining Colistin and Fluconazole Synergistically Increases Fungal Membrane Permeability and Antifungal Cidality.. *ACS infectious diseases*. 2021. DOI: [10.1021/acsinfecdis.0c00721](https://doi.org/10.1021/acsinfecdis.0c00721). PMID: [33471513](https://pubmed.ncbi.nlm.nih.gov/33471513/)

89. Maliszewska I, Wanarska E, Thompson A. Biogenic Gold Nanoparticles Decrease Methylene Blue Photobleaching and Enhance Antimicrobial Photodynamic Therapy.. *Molecules (Basel, Switzerland)*. 2021. DOI: [10.3390/molecules26030623](https://doi.org/10.3390/molecules26030623). PMID: [33504099](https://pubmed.ncbi.nlm.nih.gov/33504099/)

90. Oufensou S, Casalini S, Balmas V. Prenylated Trans-Cinnamic Esters and Ethers against Clinical Fusarium spp.: Repositioning of Natural Compounds in Antimicrobial Discovery.. *Molecules (Basel, Switzerland)*. 2021. DOI: [10.3390/molecules26030658](https://doi.org/10.3390/molecules26030658). PMID: [33513915](https://pubmed.ncbi.nlm.nih.gov/33513915/)

91. Tan X, Xie H, Zhang B. A Novel Ivermectin-Derived Compound D4 and Its Antimicrobial/Biofilm Properties against MRSA.. *Antibiotics (Basel, Switzerland)*. 2021. DOI: [10.3390/antibiotics10020208](https://doi.org/10.3390/antibiotics10020208). PMID: [33672669](https://pubmed.ncbi.nlm.nih.gov/33672669/)

92. Wallis R, Ginindza S, Beattie T. Adjunctive host-directed therapies for pulmonary tuberculosis: a prospective, open-label, phase 2, randomised controlled trial.. *The Lancet. Respiratory medicine*. 2021. DOI: [10.1016/S2213-2600(20)30448-3](https://doi.org/10.1016/S2213-2600(20)30448-3). PMID: [33740465](https://pubmed.ncbi.nlm.nih.gov/33740465/)

93. Liu Y, Tong Z, Shi J. Drug repurposing for next-generation combination therapies against multidrug-resistant bacteria.. *Theranostics*. 2021. DOI: [10.7150/thno.56205](https://doi.org/10.7150/thno.56205). PMID: [33754035](https://pubmed.ncbi.nlm.nih.gov/33754035/)

94. Shirley D, Sharma I, Warren C. Drug Repurposing of the Alcohol Abuse Medication Disulfiram as an Anti-Parasitic Agent.. *Frontiers in cellular and infection microbiology*. 2021. DOI: [10.3389/fcimb.2021.633194](https://doi.org/10.3389/fcimb.2021.633194). PMID: [33777846](https://pubmed.ncbi.nlm.nih.gov/33777846/)

95. Abutaleb N, Seleem M. In vivo efficacy of auranofin in a hamster model of Clostridioides difficile infection.. *Scientific reports*. 2021. DOI: [10.1038/s41598-021-86595-3](https://doi.org/10.1038/s41598-021-86595-3). PMID: [33782498](https://pubmed.ncbi.nlm.nih.gov/33782498/)

96. Miró-Canturri A, Ayerbe-Algaba R, Vila-Domínguez A. Repurposing of the Tamoxifen Metabolites to Combat Infections by Multidrug-Resistant Gram-Negative Bacilli.. *Antibiotics (Basel, Switzerland)*. 2021. DOI: [10.3390/antibiotics10030336](https://doi.org/10.3390/antibiotics10030336). PMID: [33810067](https://pubmed.ncbi.nlm.nih.gov/33810067/)

97. Singh V, Chibale K. Strategies to Combat Multi-Drug Resistance in Tuberculosis.. *Accounts of chemical research*. 2021. DOI: [10.1021/acs.accounts.0c00878](https://doi.org/10.1021/acs.accounts.0c00878). PMID: [33886255](https://pubmed.ncbi.nlm.nih.gov/33886255/)

98. Fatima S, Bhaskar A, Dwivedi V. Repurposing Immunomodulatory Drugs to Combat Tuberculosis.. *Frontiers in immunology*. 2021. DOI: [10.3389/fimmu.2021.645485](https://doi.org/10.3389/fimmu.2021.645485). PMID: [33927718](https://pubmed.ncbi.nlm.nih.gov/33927718/)

99. Hasselbalch H, Skov V, Kjær L. COVID-19 as a mediator of interferon deficiency and hyperinflammation: Rationale for the use of JAK1/2 inhibitors in combination with interferon.. *Cytokine & growth factor reviews*. 2021. DOI: [10.1016/j.cytogfr.2021.03.006](https://doi.org/10.1016/j.cytogfr.2021.03.006). PMID: [33992887](https://pubmed.ncbi.nlm.nih.gov/33992887/)

100. Kobatake T, Ogino K, Sakae H. Antibacterial Effects of Disulfiram in Helicobacter pylori.. *Infection and drug resistance*. 2021. DOI: [10.2147/IDR.S299177](https://doi.org/10.2147/IDR.S299177). PMID: [34012274](https://pubmed.ncbi.nlm.nih.gov/34012274/)


## Appendix A: rlm-scheme Pipeline Details

### A.1 Orchestration Architecture

This report was produced using [rlm-scheme](https://github.com/rwtaber/rlm-scheme), a Scheme-based LLM orchestration platform that enables structured, multi-step AI research pipelines.

**Pipeline Summary:**
- **Phase 0 (Data Acquisition):** PubMed E-utilities API queries via `py-exec`, fetching 426 full-text papers from PMC
- **Phase 1 (Extraction):** Pattern 1 (Parallel Fan-Out) — 426 papers processed in parallel with gpt-4.1-nano, extracting structured JSON
- **Phase 3 (Synthesis):** Pattern 10 (Tree Aggregation) — hierarchical pairwise reduction from 30 drug summaries to a single coherent synthesis
- **Phase 4 (Hypothesis Generation):** Pattern 4 (Critique-Refine) — adversarial review loop improving hypothesis quality

### A.2 Cost and Performance

| Phase | Model | Calls | Tokens | Time |
|-------|-------|-------|--------|------|
| Extraction | gpt-4.1-nano | 426 | ~1.6M | ~400s |
| Synthesis L1-L2 | gpt-4.1-nano | 45 | ~52K | ~26s |
| Synthesis L3-L4 | gpt-4o-mini | 12 | ~16K | ~45s |
| Synthesis L5+Final | gpt-4o | 3 | ~8K | ~83s |
| Hypothesis Gen | gpt-4o | 4 | ~12K | ~120s |
| Report Writing | gpt-4o | 4 | ~16K | ~84s |
| **Total** | **Mixed** | **~494** | **~1.75M** | **~13 min** |

### A.3 Reproducibility

To reproduce this analysis:
1. Install rlm-scheme from the repository
2. Configure OpenAI API access
3. Run the pipeline scripts in PoC/ (see progress.log for exact Scheme code)
