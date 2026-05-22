# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C# implementation of the "Assessing People's Skills" example from the [Model-Based Machine Learning (MBML) book](http://www.mbmlbook.com/LearningSkills.html), using the open-source [Infer.NET](https://dotnet.github.io/infer/) probabilistic programming framework.

## Build and Run

This is a .NET 10.0 console application using the SDK-style project format.

```bash
# Build
dotnet build

# Generate synthetic data (defaults: 5 skills, 10 questions, 50 persons → synthetic-data/)
dotnet run --project MBML-Skills -- generate
dotnet run --project MBML-Skills -- generate --skills 7 --questions 20 --persons 100 --output my-data

# Run inference on synthetic data
dotnet run --project MBML-Skills -- infer synthetic-data

# Run inference on the real dataset (pass the directory containing the two real CSVs)
dotnet run --project MBML-Skills -- infer MBML-Skills/data --real
```

## Architecture

All logic lives in `MBML-Skills/Program.cs`. `Main` dispatches on the first argument to one of two commands.

**Common internal representation** — both data paths produce the same three structures passed to `RunInference`:
- `skillsRequired[question][skill_index]` — jagged array of skill indices needed per question
- `isCorrect[person][question]` — bool matrix of whether each person answered correctly
- `personSkills[person][skill]` — ground-truth skill booleans (for display only, not part of the model)

**`generate` command** — samples synthetic data via `SampleSkillsNeededData`, `SamplePersonSkillsData`, and `SampleIsCorrectData`, then writes three CSVs (`skills_required.csv`, `is_correct.csv`, `person_skills.csv`) as boolean mask rows (`True`/`False`).

**`infer` command** — loads data via `LoadSyntheticData` or `LoadRealData` (flag `--real`), then calls `RunInference`.
- Synthetic loader: `File.ReadAllLines` + `string.Split` on the three simple CSVs above.
- Real data loader: `TextFieldParser` on the two original CSVs in `MBML-Skills/data/`; `GetTrueAnswersData`/`GetPersonAnswerData` parse columns 8+ as question answers, columns 1–7 as skill self-assessments.

**Infer.NET model** (in `RunInference`):
- **Latent**: `skill[person][skill]` ~ Bernoulli(0.5)
- **Observed**: `isCorrect[person][question]`
- For each (person, question): gather `relevantSkills` via `Variable.Subarray`, compute `hasSkills = Variable.AllTrue(relevantSkills)`, then Table factor — if `hasSkills`, answer is correct with `probNoMistake` = Bernoulli(0.9); otherwise with `probGuess[question]` ~ Beta(2.5, 7.5).
- Inference yields posterior `Bernoulli[][]` for skills and `Beta[]` for per-question guessing probabilities.

## Key Dependencies

- `Microsoft.ML.Probabilistic` (0.4.2504.701) — Infer.NET core: `Variable`, `Range`, `InferenceEngine`
- `Microsoft.ML.Probabilistic.Compiler` (0.4.2504.701) — required at runtime for model compilation
- `Microsoft.VisualBasic` — used for `TextFieldParser` CSV parsing
