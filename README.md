# Mapping Large Language Model Predictions onto Human Data in Psycholinguistics Experiments: A Case Study on Forced Choice Tasks

This repository contains code and data for "Mapping Large Language Model Predictions onto Human Data in Psycholinguistics Experiments: A Case Study on Forced Choice Tasks" by Hening Wang, Sharon Grosch. The purpose of this project is originally intended as a course project for "LLMs: Implications for Linguistics, Cognitive Science and Society" taught by Michael Franke and Polina Tsvilodub in SS2023 at the University of Tübingen. The repository is adapted from another repository for ["A fine-grained comparison of pragmatic language understanding in humans and language models"](https://arxiv.org/abs/2212.06801) by Jennifer Hu, Sammy Floyd, Olessia Jouravlev, Evelina Fedorenko, and Edward Gibson. (TODO: Make clear which part (some scripts, human data etc.) is adapted, and which part (new prompts table, new model prediction etc.) is self-written.)

## Theoretical Background & Related Work & Ideas

### Our research ideas
The questions Hu et al. asked are related to the presence of pragmatic ability (more precisely, the ability to recover non-literal readings) of LLMs. If there are not any, which error do they produce (most of them are literal responses)? Finally, if the pattern of model predictions aligns with human data or not. But what they delivered are only qualitative effects. Their experiment setting with LLms is strictly speaking not a Psycholinguistic experiment (perhaps the original experiment is worth taking a look at), i.e. without theory-motivated predictions and explicit factor design providing further explainability (factors are 1. model structures; 2. pragmatic phenomena). Thus, we need to examine the pragmatic ability further with different link functions. Maybe the ability can be elicited with different promptings or measurements. We present here a first step towards a quantitive exploration.

**Questions:**
1. Why only text-davinci-002 is better than other models in all phenomena? (The role of RLHF?)
2. Why coherence is the only phenomenon that all models are good at? What linguistic features distinguish it from other phenomena?
3. The performance of other models for other phenomena is basically under at chance level. Could it be overcome with different link functions?
4. They report that models are bad at Humor, Irony, and Maxim. They provide a possible explanation that listeners in those tasks need to explicit reason about speakers' mental state representations. May adding an RSA-like module (or in general a module using fine-tuning that separates pragmatic reasoning from semantic computation, since they can capture the literal meaning quite well) could lead to improvement? 
5. It seems that the performance is correlated with the size of the parameters. 
6. The prompt they used to 
### Reading Hu et al. (2023)
**About why they add more binary options:** "However, prior empirical studies have primarily evaluated LMs based on a binary distinction between pragmatic and non-pragmatic responses, providing limited insights into models’ weaknesses. A model could fail to reach the target pragmatic interpretation in multiple ways – for example, by preferring a literal interpretation, or by preferring a non-literal interpretation that violates certain social norms." The value reveals itself when it comes to the results of Humor.

**Prompts they used:** zero-shot prompting 1. Task + 2. Scenario (i.e. stimuli human participants read) + 3. Options (HW: How is this zero-shot?); Scenario consists of 1. Cover story + 2. Utterance (critical linguistic stimuli) + 3. Question

**Models they tested:** Among others (GPT-2, Tk-Instruct, Flan-T5-xl, InstructGPT), Flan-T5 and InstructGPT (text-davinci-002, based on GPT3.5) achieve better results.

**Results they achieved:** Preliminary results show that models and humans are "sensitive" (HW: that's very arbitrary) to similar (HW: Here come link functions) linguistic cues. Models are bad at humor, irony and conversational maxims. (HW: May we should look at these tasks first? Because it's easy to achieve an improvement and such improvements are more interesting for practical reasons.) 

**Questions they asked:** 1. if models can recover pragmatic interpretation (qualitatively); 2. if not, what errors do they make (the role of distractors); 3. do models and humans use similar cues (HW: How to understand cues?)

**About similar cues:** 1. with or without a cover story; 2. with or without scrambling (against compositionality of nature language); 3. by item random effect

**Data analysis methods:** "For each item, we computed the human empirical distribution over answer choices, and compared it to **models' probability assigned to the answer tokens**"
## Programming 
**Models we could test:** InstructGPT (text-davinci-002, based on GPT3.5), Flan-T5-xxl, LLAMA-2-7B (free, but it takes some effort to build them locally), tkinstruct-3B

**Tasks:** Humor, Irony, Conversational Maxims (start with Conversational Maxims?)

**Measurements we could use:** 
1. Models can see all options
    1.1 Proportion of generated Responses (baseline, directly comparable to human data); 
    1.2 Surprisal of generated Responses
2. Models can only see the critical response (i.e. non-literal meaning)
    2.1 Ask models the categorical likelihood of that reading (proportion)
    2.2 Surprisal of embedded Responses
3. Models see nothing
    3.1 Ask the model for its interpretation and manually annotate them into categories

## Data Analysis

## Writing

## Time Plan

**Deadline: 01.09.2023, Countdown: 17 Days**

Today to u. 15 Days: Make a concrete plan about which phenomena we are going to investigate and start a scratch for scripts (i.e. set up methods)

Until 13 Days: First data.

Until 10 Days: First draft of manuscript.

Until 10 Days: Send a manuscript to Michael, and schedule a meeting with Michael

Until 7 Days: Meet with Michael

