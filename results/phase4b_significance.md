# Phase 4b results: significance tests, LLM-only baselines, error analysis

Kaggle `phase4b-eval`: `significance.py` over the Phase 3–4 ablation runs (5 seeds, best epoch by validation weighted F1) and
`llm_classify.py` (Qwen2.5-7B-Instruct, not fine-tuned, greedy decoding) on the 713 gold test pairs.

## Summary

| Model | Variant | Δ weighted F1 | Seeds better | Paired t-test p (Holm) | Bootstrap 95% CI, p | Verdict |
|---|---|---|---|---|---|---|
| RoBERTa | +LLM | +3.12 | 5/5 | 0.021 (0.104) | [+0.86, +5.40], 0.005 | significant (bootstrap); borderline after Holm |
| RoBERTa | +LLM+stance | +2.96 | 5/5 | 0.003 (0.016) | [+0.28, +5.53], 0.033 | significant (both tests) |
| RoBERTa | +stance | +0.02 | 2/5 | 0.98 (0.98) | [−1.49, +1.53], 0.98 | no effect |
| DeBERTa | +LLM | +2.06 | 4/5 | 0.16 (0.47) | [−0.28, +4.47], 0.09 | positive trend, not significant |
| DeBERTa | +LLM+stance | +1.30 | 4/5 | 0.16 (0.47) | [−1.36, +3.97], 0.33 | not significant |
| DeBERTa | +stance | +1.31 | 5/5 | 0.09 (0.37) | [−0.21, +2.86], 0.10 | not significant |

| System | Weighted F1 | Macro F1 |
|---|---|---|
| Qwen2.5-7B zero-shot | 57.23 | 53.90 |
| Qwen2.5-7B 2-shot (one fixed set of shots) | 61.27 | 56.71 |
| RoBERTa base (parent + reply) | 58.86 | 55.14 |
| RoBERTa + LLM explanation | 61.97 | 58.45 |

### Interpretation
- For RoBERTa, the explanation-based variants give a statistically significant gain of about 3 weighted F1 (+LLM+stance:
  both tests; +LLM: bootstrap). For DeBERTa the gain points the same way (4/5 seeds) but is not significant (higher seed variance).
- A coarse 3-way stance gives no significant gain on its own.
- RoBERTa + LLM significantly beats the zero-shot LLM (+4.74, p 0.010) but **not** the 2-shot LLM (+0.71, p 0.70).
  The 2-shot result is a single draw of shots; it needs repeating with other shot sets before drawing conclusions.
- Error analysis (RoBERTa, 5-seed ensemble 59.17 → 63.11 weighted F1): the explanation mainly fixes **hate** replies that base
  labelled counter (net +16; hate recall 88 → 104 of 184), because it names the insult or the denigrated group explicitly.
  Failures come from sarcasm read literally and from LLM stance mistakes. DeBERTa instead gains on counter (+26) and loses on hate (−8).
- Some gold labels look debatable, which suggests some annotation noise.

Content warning: the examples below contain offensive language quoted from the dataset.

---

## Full report (verbatim from `significance_weighted_f1.md`)

# Significance tests (test set, selected on validation weighted_f1)

Paired t-test over seeds (Holm-corrected across the 6 comparisons) and paired bootstrap over the test pairs (5000 samples; Δ = variant − base, mean over seeds; 95% CI).

## weighted f1

| model | variant | seeds | Δ | seeds better | t-test p | Holm p | bootstrap 95% CI | bootstrap p |
|---|---|---|---|---|---|---|---|---|
| roberta-base (lr 1e-05) | llm | 5 | +3.12 | 5/5 | 0.0208 | 0.1040 | [+0.86, +5.40] | 0.0048 |
| roberta-base (lr 1e-05) | llm_stance | 5 | +2.96 | 5/5 | 0.0027 | 0.0164 | [+0.28, +5.53] | 0.0332 |
| roberta-base (lr 1e-05) | stance | 5 | +0.02 | 2/5 | 0.9809 | 0.9809 | [-1.49, +1.53] | 0.9820 |
| microsoft/deberta-v3-base (lr 2e-05) | llm | 5 | +2.06 | 4/5 | 0.1640 | 0.4717 | [-0.28, +4.47] | 0.0908 |
| microsoft/deberta-v3-base (lr 2e-05) | llm_stance | 5 | +1.30 | 4/5 | 0.1572 | 0.4717 | [-1.36, +3.97] | 0.3296 |
| microsoft/deberta-v3-base (lr 2e-05) | stance | 5 | +1.31 | 5/5 | 0.0929 | 0.3716 | [-0.21, +2.86] | 0.0980 |

## macro f1

| model | variant | seeds | Δ | seeds better | t-test p | Holm p | bootstrap 95% CI | bootstrap p |
|---|---|---|---|---|---|---|---|---|
| roberta-base (lr 1e-05) | llm | 5 | +3.31 | 5/5 | 0.0159 | 0.0796 | [+0.80, +5.79] | 0.0072 |
| roberta-base (lr 1e-05) | llm_stance | 5 | +2.92 | 5/5 | 0.0017 | 0.0103 | [-0.08, +5.80] | 0.0560 |
| roberta-base (lr 1e-05) | stance | 5 | +0.16 | 2/5 | 0.8468 | 0.8560 | [-1.52, +1.83] | 0.8500 |
| microsoft/deberta-v3-base (lr 2e-05) | llm | 5 | +2.25 | 4/5 | 0.2302 | 0.6907 | [-0.35, +4.88] | 0.0904 |
| microsoft/deberta-v3-base (lr 2e-05) | llm_stance | 5 | +1.07 | 4/5 | 0.4280 | 0.8560 | [-1.78, +4.06] | 0.4520 |
| microsoft/deberta-v3-base (lr 2e-05) | stance | 5 | +1.46 | 4/5 | 0.1248 | 0.4990 | [-0.22, +3.13] | 0.0900 |

## LLM-only baselines vs fine-tuned +LLM

| LLM baseline | weighted F1 | macro F1 | parsed | compared with | Δ weighted F1 (95% CI, p) |
|---|---|---|---|---|---|
| llm_Qwen2.5-7B-Instruct_0shot | 57.23 | 53.90 | 100.0% | roberta-base (lr 1e-05) +LLM | +4.74 [+1.19, +8.33], p 0.0096 |
| llm_Qwen2.5-7B-Instruct_0shot | 57.23 | 53.90 | 100.0% | microsoft/deberta-v3-base (lr 2e-05) +LLM | +2.71 [-0.89, +6.17], p 0.1448 |
| llm_Qwen2.5-7B-Instruct_2shot | 61.27 | 56.71 | 100.0% | roberta-base (lr 1e-05) +LLM | +0.71 [-2.77, +4.21], p 0.7024 |
| llm_Qwen2.5-7B-Instruct_2shot | 61.27 | 56.71 | 100.0% | microsoft/deberta-v3-base (lr 2e-05) +LLM | -1.32 [-5.05, +2.28], p 0.4880 |

## Error analysis: roberta-base (lr 1e-05), base vs +LLM (ensemble of 5 seeds)

Ensemble weighted / macro F1: base 59.17 / 55.55, +LLM 63.11 / 59.66.

| gold class | pairs | fixed by +LLM | broken by +LLM | net |
|---|---|---|---|---|
| hate | 184 | 25 | 9 | +16 |
| neutral | 361 | 31 | 21 | +10 |
| counter | 168 | 21 | 19 | +2 |

Confusion (rows = gold, columns = predicted hate / neutral / counter):

| gold | base | +LLM |
|---|---|---|
| hate | 88 / 49 / 47 | 104 / 44 / 36 |
| neutral | 48 / 253 / 60 | 45 / 263 / 53 |
| counter | 19 / 68 / 81 | 28 / 57 / 83 |

### Fixed by +LLM (examples)

| gold | base → +LLM | parent (start) | reply | LLM stance: explanation |
|---|---|---|---|---|
| hate | counter → hate | Just for clarity's sake. He was awake, just inebriated beyond the point of bring able to remember. He was completely conscious. There is a difference. | Not really when a drunk woman can actively engage in sex with a drunk man then claim rape. | oppose: The reply opposes the parent comment by suggesting that a person's ability to consent is not affected by intoxication, which implies a dismissive view of the seriousness of sexual assault. |
| hate | counter → hate | You definitely did not get the point of whatever Bill Burr bit you are talking about. Im frankly amazed you see nothing wrong with what you said, particularly t | You haven't even heard the bit, so shut the fuck up and whine somewhere else. | oppose: The reply opposes the parent comment by insulting them and suggesting they should not comment on the topic. |
| hate | counter → hate | Nobody would care, lesbian sex is completely meaningless. Is it even sex?? It's like high school kids having a make out session and fingering each other. They c | You must have terrible sex. | oppose: The reply opposes the parent comment by suggesting that the commenter's opinion is based on poor personal experiences, implying that their view is subjective and not universally applicable. |
| hate | counter → hate | Well these are trained killers who prefer to spend 3 years away from their family killing people instead of being at home... Medals lol, yeah I too give cookies | So you can talk like a faggot online. | oppose: The reply opposes the parent comment by directly insulting them in a homophobic manner, which is unrelated to the content of the original post. |
| hate | counter → hate | Feminists are not smart, nor do they really give a shit what’s legal | Or give a shit about else's rights or freedoms other their own. | oppose: The reply further generalizes and denigrates feminists, implying that they are solely concerned with their own rights and freedoms at the expense of others. |
| neutral | counter → neutral | True but what are we doing here then? | Reacting. As opposed to actively spending time on shit that doesn't matter. | support: The reply supports the sentiment of passively reacting rather than engaging in unimportant activities, implying agreement with the parent comment's tone. |
| neutral | counter → neutral | On Pubmed: >In the largest study on circumcision in South Korea, Seoul University found 33% of men who were circumcised during adulthood reported difficulty att | You can die from being circumcised? Wtf | neutral: The reply expresses shock at the potential fatal risk mentioned in the parent comment, without directly supporting or opposing the argument. |
| neutral | counter → neutral | > freedom of speech I agree that free speech is extremely important, except if it's offensive... you know? Can't just say *anything.* > gender equality Totally  | Kek. That was funny. | neutral: The reply acknowledges the parent comment in a humorous way without explicitly agreeing or disagreeing. |
| neutral | hate → neutral | I live in Central Santiago where this shit is kicking off, the march went straight past my building. I have barely heard any feminist slogans at all, the majori | holy fuggin shit they are full on armored and the look of agony on her face, jesus | neutral: The reply describes a scene with armored protesters and a female police officer in pain, likely referring to the violent protests mentioned in the parent comment, without directly supporting or opposi |
| neutral | hate → neutral | How many assholes have we got in this thread anyway? | (insert Spaceballs gif) | neutral: The reply uses a Spaceballs gif to respond to the parent comment's tone without directly addressing the question or the people mentioned. |
| counter | hate → counter | You are entirely missing the point, asshole. Complaining about how stupid a concept like "manspreading" (and mansplaining, but that is for another thread) by sh | Way to prove the sub isn't garbage. | oppose: The reply dismisses the argument and implies the sub is garbage, which opposes the parent's viewpoint. |
| counter | neutral → counter | ummm I think >In his posts, Metzger scolds women who claim they were raped but don’t go to police. is the problem here... | Why is that a problem? | oppose: The reply questions the validity of the statement, implying that scolding victims for not reporting rape might be problematic. |
| counter | neutral → counter | Of course not, and while this photo leaves out to much to form a complete opinion of this individuals situation, a man should have at least partial custody of h | So child support is government imposed payment for sex? | oppose: The reply challenges the parent comment's analogy between child support and payment for sex, implying it is an inappropriate and disrespectful comparison. |
| counter | neutral → counter | It's not about Will being "powerful",its about Will being desirable. Jada is a piece of shit no matter how you want to cut it. Stop trying to force her crappy d | I didn't? Read it again. | oppose: The reply challenges the parent comment's accusation by asking them to re-read it, implying that they misunderstood or misinterpreted the original post. |
| counter | neutral → counter | What it's missing though is everything else presented as fact. With my son, I wasn't all "oh it looks ugly", it was that my male doctor told me a bunch of shit  | So...the doctor lied to you to keep up a social norm? | oppose: The reply questions the doctor's motives and implies the doctor's advice was based on social norms rather than medical facts, opposing the parent's perspective. |

### Broken by +LLM (examples)

| gold | base → +LLM | parent (start) | reply | LLM stance: explanation |
|---|---|---|---|---|
| hate | hate → neutral | ok, I know this is a hard concept to grasp, but take everything a suspected rapist says happened with a ~~dash~~ mine's worth of salt. | I've heard a spoonful of sugar helps the Bullshit go down | support: The reply supports the parent comment by using a similar metaphor to convey skepticism towards the statements made by a suspected rapist. |
| hate | hate → counter | Or worse, an asshole abuser... | He wouldn't have three kids if he was *only* an asshole abuser! | oppose: The reply challenges the assumption in the parent comment by suggesting that being an asshole abuser alone would not motivate someone to have children. It implies that there might be other factors at  |
| hate | hate → neutral | I went to my doctor to ask if there was anything to do about loss of sensation after a botched emergency circumcision. She told me that there was no loss of sen | Welcome to a cutting culture. | support: The reply suggests that the situation described by the parent comment is common in certain medical cultures, implying support for their experience. |
| hate | hate → neutral | There's a concept on show in her replies. I've seen it [illustrated in a few different ways](https://www.irreligion.org/wp-content/uploads/2007/11/kingme.png).  | Perfect analogy. She is, indeed, shitting all over twitter and acting like she somehow won. | support: The reply supports the parent comment by agreeing with the chess metaphor and elaborating on the behavior being described. |
| hate | hate → neutral | Exactly. How many horny teenage girls have landed men in prison for the same exact thing? | None because all menz rapizt and all womenz gudz /s | oppose: The reply opposes the parent comment by using sarcasm to mock the idea that all men are quick to blame and all women are good, implying the parent's statement is overly generalized and unfair. |
| neutral | neutral → counter | >He got nude pics of someone underage. He then showed them off to others. He didn't show them to others, that was another boy that his girlfriend sexted who too | Shouldn't she be charged with creation of child pornography? | oppose: The reply challenges the parent comment's assertion by questioning the girlfriend's legal responsibility, implying the original commenter might be overlooking her role in the situation. |
| neutral | neutral → counter | Well ya, statistically race doesnt change much in the courts, either for sentencing or child laws, and sex on the other hand? Being black will get you another 2 | Ya both play a factor, but gender currently plays a bigger factor. | oppose: The reply challenges the parent comment by suggesting that while both race and gender are factors, gender has a more significant impact, which contradicts the parent's statement. |
| neutral | neutral → hate | Nope, not what I'm doing. It was just a side comment of "this lady is fucked in the head" | Yes, EXACTLY what you are doing. | oppose: The reply opposes the original comment by agreeing with the derogatory statement about the person, implying support for negative and harmful language. |
| neutral | neutral → counter | Imagine firing a woman because "she doesn't fit in" IMAGINE THE OUTRAGE. I'm so fucking sorry for you dude, try to get a lawyer. Good luck | Oh yeah. It certainly doesn't go both ways. Thank you. | oppose: The reply opposes the parent comment by suggesting that the situation does not have equal standards, implying criticism of the original statement's logic. |
| neutral | neutral → counter | Quote where I said that.Give me a quote. I have the right to express my opinion too just like they do. My opinion was that it was kinda racist. Just like how it | I'm pointing out inconsistencies QUOTE ME | oppose: The reply targets the parent's argument and implies that their stance is inconsistent, suggesting they are being hypocritical. |
| counter | counter → neutral | I am completely pro white, trust me. We are being ass raped by the media left and right. Why do I keep my 2A rights on my hip? Just don’t fucking. Ever. Ever. P | What words did I put in your mouth thought? | neutral: The reply asks for clarification on the specific words that were allegedly put in the original poster's mouth, without taking a clear stance for or against the parent comment. |
| counter | counter → hate | You seem lovely and well adjusted. Enjoy your victimhood | and good luck to in convincing the people you're pissing on that it's raining. | oppose: The reply opposes the parent comment by sarcastically suggesting that the original poster should have more self-awareness about their behavior and its impact on others. |
| counter | counter → hate | "She chose to have sex she agreed to be a mother." - Regressive assault on the reproductive rights women. "He chose to have sex so he agreed to be a father." -  | Yup. This is disgusting thinking that needs to change. | support: The reply agrees with the parent comment's criticism of the regressive attitudes towards reproductive rights and financial responsibility. |
| counter | counter → hate | I would think not! But thats not his kid, and its not mine either, and i don’t give a fuck about it! | Are you some kind of bot or caricature? | oppose: The reply expresses disbelief and possibly offense at the parent comment's tone and content, suggesting the speaker finds it inappropriate or extreme. |
| counter | counter → neutral | It's only the people with stupid twats for parents that don't care about their kids well being that are. The trend is way down and in twenty years or so the mut | That's an incredibly huge generalization. | oppose: The reply opposes the parent comment by calling its generalization 'incredibly huge', suggesting the statement is overly broad and unfair. |

## Error analysis: microsoft/deberta-v3-base (lr 2e-05), base vs +LLM (ensemble of 5 seeds)

Ensemble weighted / macro F1: base 59.07 / 54.89, +LLM 61.47 / 58.01.

| gold class | pairs | fixed by +LLM | broken by +LLM | net |
|---|---|---|---|---|
| hate | 184 | 15 | 23 | -8 |
| neutral | 361 | 25 | 33 | -8 |
| counter | 168 | 40 | 14 | +26 |

Confusion (rows = gold, columns = predicted hate / neutral / counter):

| gold | base | +LLM |
|---|---|---|
| hate | 111 / 53 / 20 | 103 / 50 / 31 |
| neutral | 65 / 263 / 33 | 50 / 255 / 56 |
| counter | 43 / 71 / 54 | 36 / 52 / 80 |

(DeBERTa example rows are in the Kaggle output `phase4b-eval/outputs/significance_weighted_f1.md`.)
