//
// Implementation of the "Assessing People's Skills" example from "Model-based Machine Learning"
// Vladislavs Dovgalecs
//

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.VisualBasic.FileIO;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace MBMLSkills
{
    class MainClass
    {
        static void Main(string[] args)
        {
            if (args.Length == 0) { PrintUsage(); return; }
            switch (args[0])
            {
                case "generate": RunGenerate(args[1..]); break;
                case "infer":    RunInfer(args[1..]); break;
                default:         PrintUsage(); break;
            }
        }

        static void PrintUsage()
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("  generate [--skills N] [--questions N] [--persons N] [--output DIR]");
            Console.WriteLine("  infer <dir> [--real]");
            Console.WriteLine();
            Console.WriteLine("  generate  Sample synthetic data and write CSVs to DIR (default: synthetic-data).");
            Console.WriteLine("  infer     Run inference on data in <dir>.");
            Console.WriteLine("            --real  Read real data format instead of synthetic.");
        }

        // ─── Generate command ──────────────────────────────────────────────────────

        static void RunGenerate(string[] args)
        {
            int numSkills = 5, numQuestions = 10, numPersons = 50;
            string outputDir = "synthetic-data";

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--skills":    numSkills    = int.Parse(args[++i]); break;
                    case "--questions": numQuestions = int.Parse(args[++i]); break;
                    case "--persons":   numPersons   = int.Parse(args[++i]); break;
                    case "--output":    outputDir    = args[++i]; break;
                    default:
                        Console.Error.WriteLine($"Unknown option: {args[i]}");
                        PrintUsage();
                        return;
                }
            }

            Directory.CreateDirectory(outputDir);

            SampleSkillsNeededData(numSkills, numQuestions, out int[][] skillsNeeded);
            SamplePersonSkillsData(numSkills, numPersons, out bool[][] personSkills);
            SampleIsCorrectData(numPersons, numQuestions, numSkills, skillsNeeded, personSkills, out bool[][] isCorrect);

            WriteBoolMatrix(Path.Combine(outputDir, "skills_required.csv"), IndicesToMask(skillsNeeded, numSkills));
            WriteBoolMatrix(Path.Combine(outputDir, "is_correct.csv"), isCorrect);
            WriteBoolMatrix(Path.Combine(outputDir, "person_skills.csv"), personSkills);

            Console.WriteLine($"Generated {numPersons} persons, {numQuestions} questions, {numSkills} skills → {outputDir}/");
        }

        // Converts jagged skill-index arrays to a boolean mask matrix for CSV output.
        static bool[][] IndicesToMask(int[][] indices, int numSkills)
        {
            return indices.Select(row =>
            {
                bool[] mask = new bool[numSkills];
                foreach (int i in row) mask[i] = true;
                return mask;
            }).ToArray();
        }

        static void WriteBoolMatrix(string path, bool[][] matrix)
        {
            using var writer = new StreamWriter(path);
            foreach (var row in matrix)
                writer.WriteLine(string.Join(",", row));
        }

        // ─── Infer command ─────────────────────────────────────────────────────────

        static void RunInfer(string[] args)
        {
            if (args.Length == 0)
            {
                Console.Error.WriteLine("infer requires <dir>");
                PrintUsage();
                return;
            }

            string dir = args[0];
            bool realFormat = args.Contains("--real");

            var (skillsRequired, isCorrect, personSkills) = realFormat
                ? LoadRealData(dir)
                : LoadSyntheticData(dir);

            RunInference(skillsRequired, isCorrect, personSkills);
        }

        static void RunInference(int[][] skillsRequired, bool[][] isCorrect, bool[][] personSkills)
        {
            int numPersons   = isCorrect.Length;
            int numSkills    = personSkills[0].Length;
            int numQuestions = skillsRequired.Length;

            Range person    = new Range(numPersons).Named("persons");
            Range skills    = new Range(numSkills).Named("skills");
            Range questions = new Range(numQuestions).Named("questions");

            var skill = Variable.Array(Variable.Array<bool>(skills), person).Named("skillArray");
            skill[person][skills] = Variable.Bernoulli(0.5).ForEach(person).ForEach(skills);

            var questionSizesArray = Variable.Array<int>(questions).Named("questionSizesArray");
            questionSizesArray.ObservedValue = Util.ArrayInit(numQuestions, q => skillsRequired[q].Length);
            Range questionSizes = new Range(questionSizesArray[questions]).Named("questionSizes");

            var skillsNeeded = Variable.Array(Variable.Array<int>(questionSizes), questions).Named("skillsNeeded");
            skillsNeeded.ObservedValue = skillsRequired;
            skillsNeeded.SetValueRange(skills);

            var probGuess = Variable.Array<double>(questions).Named("probGuess");
            probGuess[questions] = Variable.Beta(2.5, 7.5).ForEach(questions);

            var probNoMistake = Variable.Bernoulli(0.9);

            var relevantSkills = Variable.Array(Variable.Array(Variable.Array<bool>(questionSizes), questions), person);
            var hasSkills      = Variable.Array(Variable.Array<bool>(questions), person);
            var isCorrectVar   = Variable.Array(Variable.Array<bool>(questions), person).Named("isCorrect");

            using (Variable.ForEach(person))
            {
                using (Variable.ForEach(questions))
                {
                    relevantSkills[person][questions] = Variable.Subarray<bool>(skill[person], skillsNeeded[questions]);
                    hasSkills[person][questions]      = Variable.AllTrue(relevantSkills[person][questions]);

                    using (Variable.If(hasSkills[person][questions]))
                        isCorrectVar[person][questions] = probNoMistake;
                    using (Variable.IfNot(hasSkills[person][questions]))
                        isCorrectVar[person][questions] = Variable.Bernoulli(probGuess[questions]);
                }
            }

            isCorrectVar.ObservedValue = isCorrect;

            var engine = new InferenceEngine();
            Bernoulli[][] skillMarginal     = engine.Infer<Bernoulli[][]>(skill);
            Beta[]        probGuessMarginal = engine.Infer<Beta[]>(probGuess);

            for (int p = 0; p < numPersons; p++)
            {
                Console.Write($"PERSON #{p + 1} inferred:    ");
                for (int j = 0; j < numSkills; j++)
                    Console.Write($"{skillMarginal[p][j].GetProbTrue():N3} ");
                Console.WriteLine();

                Console.Write($"           ground truth: ");
                for (int j = 0; j < numSkills; j++)
                    Console.Write($"{(personSkills[p][j] ? 1.0 : 0.0):N3} ");
                Console.WriteLine("\n");
            }

            for (int q = 0; q < numQuestions; q++)
                Console.WriteLine($"QUESTION #{q + 1} guessing probability: {probGuessMarginal[q].GetMean():N3}");
        }

        // ─── Synthetic data I/O ────────────────────────────────────────────────────

        static (int[][] skillsRequired, bool[][] isCorrect, bool[][] personSkills) LoadSyntheticData(string dir)
        {
            return (
                ReadSkillsRequired(Path.Combine(dir, "skills_required.csv")),
                ReadBoolMatrix(Path.Combine(dir, "is_correct.csv")),
                ReadBoolMatrix(Path.Combine(dir, "person_skills.csv"))
            );
        }

        // Reads a boolean-mask CSV back into jagged skill-index arrays.
        static int[][] ReadSkillsRequired(string path)
        {
            return File.ReadAllLines(path)
                .Select(line => line.Split(',')
                    .Select((v, i) => (v, i))
                    .Where(x => x.v == "True")
                    .Select(x => x.i)
                    .ToArray())
                .ToArray();
        }

        static bool[][] ReadBoolMatrix(string path)
        {
            return File.ReadAllLines(path)
                .Select(line => line.Split(',').Select(v => v == "True").ToArray())
                .ToArray();
        }

        // ─── Real data I/O ─────────────────────────────────────────────────────────

        static (int[][] skillsRequired, bool[][] isCorrect, bool[][] personSkills) LoadRealData(string dir)
        {
            string skillsQuestionsFile = Path.Combine(dir,
                "LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv");
            string rawResponsesFile = Path.Combine(dir,
                "LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv");

            var skillsQuestionsData = ReadCSV(skillsQuestionsFile);
            var rawResponsesData    = ReadCSV(rawResponsesFile);

            return (
                GetSkillsNeededData(skillsQuestionsData),
                BuildIsCorrect(GetTrueAnswersData(rawResponsesData), GetPersonAnswerData(rawResponsesData)),
                GetPersonSkillData(rawResponsesData)
            );
        }

        static List<string[]> ReadCSV(string path)
        {
            var list = new List<string[]>();
            using var parser = new TextFieldParser(path);
            parser.TextFieldType = FieldType.Delimited;
            parser.SetDelimiters(",");
            while (!parser.EndOfData)
                list.Add(parser.ReadFields()!);
            return list;
        }

        static int[] GetTrueAnswersData(List<string[]> list)
        {
            string[] fields = list[1];
            int numQuestions = fields.Length - 8;
            int[] trueAnswers = new int[numQuestions];
            for (int i = 8; i < fields.Length; i++)
                trueAnswers[i - 8] = int.Parse(fields[i]);
            return trueAnswers;
        }

        static int[][] GetPersonAnswerData(List<string[]> list)
        {
            int numPersons = list.Count - 2;
            int[][] personAnswers = new int[numPersons][];
            for (int i = 2; i < list.Count; i++)
            {
                string[] fields = list[i];
                int numQuestions = fields.Length - 8;
                personAnswers[i - 2] = new int[numQuestions];
                for (int j = 8; j < fields.Length; j++)
                    personAnswers[i - 2][j - 8] = int.Parse(fields[j]);
            }
            return personAnswers;
        }

        static bool[][] GetPersonSkillData(List<string[]> list)
        {
            int numPersons = list.Count - 2;
            bool[][] personSkills = new bool[numPersons][];
            for (int i = 2; i < list.Count; i++)
            {
                string[] fields = list[i];
                personSkills[i - 2] = new bool[7];
                for (int j = 1; j < 8; j++)
                    personSkills[i - 2][j - 1] = fields[j] == "True";
            }
            return personSkills;
        }

        static int[][] GetSkillsNeededData(List<string[]> list)
        {
            int numQuestions = list.Count;
            int[][] skillsNeeded = new int[numQuestions][];
            for (int i = 0; i < numQuestions; i++)
            {
                string[] values = list[i];
                skillsNeeded[i] = new int[values.Count(v => v == "True")];
                int pos = 0;
                for (int j = 0; j < values.Length; j++)
                    if (values[j] == "True")
                        skillsNeeded[i][pos++] = j;
            }
            return skillsNeeded;
        }

        static bool[][] BuildIsCorrect(int[] trueAnswers, int[][] personAnswers)
        {
            int numPersons   = personAnswers.Length;
            int numQuestions = trueAnswers.Length;
            bool[][] isCorrect = new bool[numPersons][];
            for (int p = 0; p < numPersons; p++)
            {
                isCorrect[p] = new bool[numQuestions];
                for (int q = 0; q < numQuestions; q++)
                    isCorrect[p][q] = trueAnswers[q] == personAnswers[p][q];
            }
            return isCorrect;
        }

        // ─── Synthetic data sampling ───────────────────────────────────────────────

        static void SampleSkillsNeededData(int numSkills, int numQuestions, out int[][] skillsNeeded)
        {
            skillsNeeded = new int[numQuestions][];
            var coin = new Bernoulli(0.5);
            var rng  = new Random();
            for (int q = 0; q < numQuestions; q++)
            {
                var skillsList = new List<int>();
                for (int s = 0; s < numSkills; s++)
                    if (coin.Sample()) skillsList.Add(s);
                // every question must require at least one skill
                if (skillsList.Count == 0)
                    skillsList.Add(rng.Next(numSkills));
                skillsNeeded[q] = skillsList.ToArray();
            }
        }

        static void SamplePersonSkillsData(int numSkills, int numPersons, out bool[][] personSkills)
        {
            personSkills = new bool[numPersons][];
            var coin = new Bernoulli(0.5);
            for (int p = 0; p < numPersons; p++)
            {
                personSkills[p] = new bool[numSkills];
                for (int s = 0; s < numSkills; s++)
                    personSkills[p][s] = coin.Sample();
            }
        }

        static void SampleIsCorrectData(int numPersons, int numQuestions, int numSkills,
                                        int[][] skillsNeeded, bool[][] personSkills,
                                        out bool[][] isCorrect)
        {
            var hasSkillsCoin    = new Bernoulli(0.9);
            var noSkillsCoin     = new Bernoulli(0.1);
            isCorrect = new bool[numPersons][];
            for (int p = 0; p < numPersons; p++)
            {
                isCorrect[p] = new bool[numQuestions];
                for (int q = 0; q < numQuestions; q++)
                {
                    bool hasAll = skillsNeeded[q].All(s => personSkills[p][s]);
                    isCorrect[p][q] = hasAll ? hasSkillsCoin.Sample() : noSkillsCoin.Sample();
                }
            }
        }
    }
}
