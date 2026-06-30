# Appendix of Paper “A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning”


## Appendix A. Demonstration Scenario

### Medical Consultation and Suggested Follow-up Question

| **Speaker**     | **Medical Consultation**                                                                 |
|------------------|------------------------------------------------------------------------------------------|
| *Dr. Smith*      | Can you describe the pain? Does it stay in one spot or spread to other areas?          |
| *Mr. Johnson*    | It starts in my lower back but spreads down my left buttock and leg.                   |
| *Dr. Smith*      | Do you feel any numbness, tingling, or weakness in your leg?                           |
| *Mr. Johnson*    | Sometimes my leg feels numb or tingles, especially after sitting for a long time.      |

---

**MedRAG-Suggested Follow-Up Question**

| **Speaker**     | **Follow-Up Question**                                                                  |
|------------------|------------------------------------------------------------------------------------------|
| *Dr. Smith*      | Does your pain worsen when standing or when sitting?                                   |
| *Mr. Johnson*    | Standing.                                                                               |

---

Lower back pain is a common complaint in chronic pain management, with multiple potential causes. Among them, lumbar spinal stenosis and sciatica share many overlapping manifestations, including radiating pain, numbness, and discomfort in the lower extremities. This similarity often makes differential diagnosis challenging, especially for less experienced physicians who must rely on precise questioning to uncover key distinguishing features.

Before beginning the consultation, Dr. Smith, a junior chronic pain specialist, initiates a session in MedRAG and activates real-time monitoring. MedRAG is pre-configured with three years of departmental EHR data. The system continuously transcribes her conversation with Mr. Johnson, who reports persistent lower back and leg pain. The recorded medical consultation is presented in the table above.

As the conversation progresses, MedRAG analyzes the dialogue, identifies missing critical information, and suggests a key diagnostic question to Dr. Smith. It then integrates the patient's response into the LLM context. MedRAG differentiates lumbar spinal stenosis from sciatica, noting that pain worsening while standing indicates spinal stenosis, whereas prolonged sitting aggravates sciatica. Based on this, the system generates a preliminary diagnosis, suggesting Mr. Johnson is more likely suffering from lumbar spinal stenosis.

## Appendix B. End-to-End Evaluation of Speech-to-Text Integration
To evaluate MedRAG’s robustness in real-world clinical settings, we conducted an end-to-end experiment focusing on its speech-to-text (STT) integration pipeline. This evaluation assessed the system’s ability to process spoken inputs and maintain diagnostic performance.

We utilized GPT-4 to transform structured EHRs into natural, multi-turn dialogues between doctors and patients. The generated dialogues covered various medical scenarios and included typical clinical terminology and expressions. These simulated dialogues were then converted to speech input, processed by the STT module, and fed into MedRAG's LLM component for final disease diagnosis accuracy evaluation. The results are shown in the following Table.


<div align="center"> <img src="./Fig/Modal_eval.png" alt="clustering" width="350"> </div >

The evaluation results revealed that the STT module successfully captured the majority of medical terminology while preserving some natural language variations and speech patterns. The LLM component demonstrated robust error tolerance, maintaining reasonable diagnostic accuracy despite potential transcription errors. Notably, the system maintained consistent diagnostic accuracy regardless of the input format (text vs. speech), suggesting that the LLM's strong language understanding capabilities effectively compensated for any STT-related errors. It's important to note that this evaluation focused solely on disease diagnosis accuracy, independent of the STT module's performance metrics. The results indicate that MedRAG maintains high diagnostic accuracy even when operating on speech-transcribed inputs, highlighting its resilience to minor transcription errors and its suitability for voice-based clinical interaction.

## Appendix C. Doctor Evaluation
We conducted a comparative human evaluation where four experienced doctors reviewed system UI, functionability and outputs from MedRAG and GPT-4o on EHR analysis and medical consultation analysis. 

The table below shows the detailed descriptions and corresponding questions of the five human factors criteria used for the doctor evaluation.

<div align="center"> <img src="./Fig/def.png" alt="clustering" width="950"> </div >

In this evaluation, we compare the performance of our MedRAG system with the state-of-the-art LLM GPT-4o. The comparative results are illustrated in the Figure below.
<div align="center"> <img src="./Fig/radar.png" alt="clustering" width="1000"> </div >

For the UI aspect, both systems demonstrated similar performance, with MedRAG showing a slight edge in Trust and Adoption Intention. This suggests that MedRAG’s interface design is perceived as marginally more intuitive and clinically usable. For the EHR analysis aspect, MedRAG clearly outperformed GPT-4o across all five criteria, particularly in Clinical Relevance and Recommendation. This reflects the tailored design for structured clinical data interpretation and higher reliability of MedRAG in decision support. For the conversation analysis aspect, MedRAG again achieved consistently higher scores, especially in Trust, Adoption Intention, and Recommendation.

