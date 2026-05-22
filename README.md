# MBML-Skills

A C# implementation of the **"Assessing People's Skills"** example from
[Model-Based Machine Learning](http://www.mbmlbook.com/LearningSkills.html),
using [Infer.NET](https://dotnet.github.io/infer/) for Expectation Propagation inference.

---

## The Problem

Given a quiz with known structure — specifically, which skills each question tests — and
the responses of a group of people, can we infer which skills each person actually has?

This is harder than it looks. A person who answers a question incorrectly might lack
a required skill, or might have all the skills and simply made a mistake. A person who
answers correctly might genuinely know the material, or might have guessed. The
signal is noisy, and the skills themselves are never directly observed.

A natural approach is to treat skills as **latent binary variables** and reason
probabilistically over them. Rather than producing a single yes/no verdict per skill per
person, the model maintains a full probability distribution over each skill, updated by
the evidence from quiz responses. The result is a calibrated posterior: high confidence
when the data is informative, appropriately uncertain when it is not.

---

## The Model

### Generative story

The model is specified as a generative probabilistic program: a description of how data
could have been produced, from which we can infer backwards to the hidden causes.

Let $P$ be the number of persons, $Q$ the number of questions, and $S$ the number of skills.
Each question $q$ requires a known subset of skills $\mathcal{N}_q \subseteq \{1,\ldots,S\}$
(observed, provided as input).

**Step 1 — Sample each person's skills.**
Each person independently either has or lacks each skill, with equal prior probability:

$$\text{skill}_{p,s} \sim \text{Bernoulli}(0.5) \qquad p = 1,\ldots,P \quad s = 1,\ldots,S$$

**Step 2 — Sample each question's guessing probability.**
Each question has its own probability that someone without the required skills still
answers correctly by guessing. This varies by question and is itself uncertain, so it
gets a prior centred near 0.25 (consistent with a four-option multiple-choice format):

$$\text{probGuess}_q \sim \text{Beta}(2.5,\; 7.5) \qquad q = 1,\ldots,Q$$

The prior mean is $2.5 / (2.5 + 7.5) = 0.25$.

**Step 3 — Generate quiz responses.**
For each person $p$ and question $q$, first determine whether the person has *all* of
the required skills:

$$\text{hasSkills}_{p,q} = \bigwedge_{s \,\in\, \mathcal{N}_q} \text{skill}_{p,s}$$

Then sample whether they answer correctly. If they have all the skills they may still
make a mistake (fixed error rate of 10%); if they lack at least one skill they may still
guess correctly:

$$\text{isCorrect}_{p,q} \sim \begin{cases}
\text{Bernoulli}(0.9) & \text{if } \text{hasSkills}_{p,q} = \text{True} \\
\text{Bernoulli}(\text{probGuess}_q) & \text{if } \text{hasSkills}_{p,q} = \text{False}
\end{cases}$$

### Variables at a glance

| Symbol | Type | Role |
|---|---|---|
| $\text{skill}_{p,s}$ | `bool`, latent | Whether person $p$ has skill $s$ |
| $\text{hasSkills}_{p,q}$ | `bool`, deterministic | Whether person $p$ has all skills for question $q$ |
| $\text{probGuess}_q$ | `double`, latent | Probability of a correct guess on question $q$ |
| $\text{probNoMistake}$ | `double`, **fixed** at 0.9 | Probability of answering correctly when all skills are present |
| $\mathcal{N}_q$ | `int[]`, **observed** | Indices of skills required by question $q$ |
| $\text{isCorrect}_{p,q}$ | `bool`, **observed** | Whether person $p$ answered question $q$ correctly |

### Inference

Infer.NET runs **Expectation Propagation** to compute approximate posterior marginals
over the latent variables given the observed quiz responses.

**Per person, per skill** — a `Bernoulli` posterior:

$$P(\text{skill}_{p,s} = \text{True} \mid \text{data})$$

Values close to 1 indicate the model is confident the person has the skill; values close
to 0.5 indicate the responses were not informative enough to update the prior.

**Per question** — a `Beta` posterior over the guessing probability:

$$P(\text{probGuess}_q \mid \text{data})$$

Reported as the posterior mean. Questions where many unskilled people answered correctly
will have higher inferred guessing rates.

---

## Usage

Requires [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
dotnet build
```

**Generate synthetic data** (defaults: 5 skills, 10 questions, 50 persons):

```bash
dotnet run --project MBML-Skills -- generate
dotnet run --project MBML-Skills -- generate --skills 7 --questions 30 --persons 100 --output my-data
```

**Run inference on synthetic data:**

```bash
dotnet run --project MBML-Skills -- infer synthetic-data
```

**Run inference on the included real dataset** (22 persons, 48 questions, 7 skills):

```bash
dotnet run --project MBML-Skills -- infer MBML-Skills/data --real
```

---

## Data Format

### Synthetic format

Three headerless CSV files, all using `True`/`False` as cell values.

**`skills_required.csv`** — $Q$ rows, $S$ columns.
Row $q$, column $s$ is `True` if skill $s$ is required to answer question $q$ correctly.
Every row must have at least one `True` (every question must test at least one skill).

```
True,False,True,False,False
False,True,False,True,True
...
```

**`is_correct.csv`** — $P$ rows, $Q$ columns.
Row $p$, column $q$ is `True` if person $p$ answered question $q$ correctly.

```
True,False,True,True,False,True,...
False,True,False,True,True,False,...
...
```

**`person_skills.csv`** — $P$ rows, $S$ columns.
Row $p$, column $s$ is `True` if person $p$ has skill $s$ (ground truth).
This file is used only for display — it is not passed to the inference engine.

```
True,False,True,True,False
False,True,False,False,True
...
```

To run inference on your own data in this format, place the three files in a directory
and run:

```bash
dotnet run --project MBML-Skills -- infer <your-data-dir>
```

If you do not have ground-truth skill labels, supply a `person_skills.csv` where every
cell is `False`.

### Real data format

Two CSV files consumed by the `--real` flag. These match the format of the dataset
shipped in `MBML-Skills/data/`.

**`LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv`**

$Q$ rows, $S$ columns, no header. Same boolean-mask layout as the synthetic
`skills_required.csv` above.

**`LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv`**

Three kinds of rows, $8 + Q$ columns each:

| Row | Column 0 | Columns 1–7 | Columns 8+ |
|---|---|---|---|
| 0 (header) | `#` | `S1`…`S7` | `Q1`…`QN` |
| 1 (answers) | `ANS` | _(empty)_ | Correct answer index for each question |
| 2+ (persons) | Person ID | `True`/`False` self-assessed skill per skill | Integer answer choice per question |

The loader compares each person's answer choices against the correct answers to derive
`isCorrect`, and reads columns 1–7 as the ground-truth skill indicators for display.

---

## License

BSD. See [LICENSE](LICENSE).
