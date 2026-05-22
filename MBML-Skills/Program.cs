//
// Implementation of the "Assessing People's Skills" example from "Model-based Machine Learning"
// Vladislavs Dovgalecs
//

using System;
using System.Linq;

namespace MBMLSkills;

class Program
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
        Console.WriteLine("  generate [--skills N] [--questions N] [--persons N] [--output DIR] [--seed N]");
        Console.WriteLine("  infer <dir> [--real]");
        Console.WriteLine();
        Console.WriteLine("  generate  Sample synthetic data and write CSVs to DIR (default: synthetic-data).");
        Console.WriteLine("  infer     Run inference on data in <dir>.");
        Console.WriteLine("            --real  Read real data format instead of synthetic.");
    }

    static void RunGenerate(string[] args)
    {
        int numSkills = 5, numQuestions = 10, numPersons = 50;
        string outputDir = "synthetic-data";
        int? seed = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--skills":    numSkills    = int.Parse(args[++i]); break;
                case "--questions": numQuestions = int.Parse(args[++i]); break;
                case "--persons":   numPersons   = int.Parse(args[++i]); break;
                case "--output":    outputDir    = args[++i]; break;
                case "--seed":      seed         = int.Parse(args[++i]); break;
                default:
                    Console.Error.WriteLine($"Unknown option: {args[i]}");
                    PrintUsage();
                    return;
            }
        }

        var data = SyntheticDataGenerator.Generate(numSkills, numQuestions, numPersons, seed);
        DataIO.Write(outputDir, data);
        Console.WriteLine($"Generated {numPersons} persons, {numQuestions} questions, {numSkills} skills → {outputDir}/");
    }

    static void RunInfer(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("infer requires <dir>");
            PrintUsage();
            return;
        }

        string dir = args[0];
        var data   = args.Contains("--real") ? DataIO.ReadReal(dir) : DataIO.ReadSynthetic(dir);
        var result = SkillsModel.Run(data);
        PrintResult(result, data);
    }

    static void PrintResult(InferenceResult result, SkillsDataset data)
    {
        for (int p = 0; p < data.NumPersons; p++)
        {
            Console.Write($"PERSON #{p + 1} inferred:    ");
            for (int j = 0; j < data.NumSkills; j++)
                Console.Write($"{result.SkillMarginals[p][j].GetProbTrue():N3} ");
            Console.WriteLine();

            if (data.PersonSkills is not null)
            {
                Console.Write($"           ground truth: ");
                for (int j = 0; j < data.NumSkills; j++)
                    Console.Write($"{(data.PersonSkills[p][j] ? 1.0 : 0.0):N3} ");
                Console.WriteLine();
            }

            Console.WriteLine();
        }

        for (int q = 0; q < data.NumQuestions; q++)
            Console.WriteLine($"QUESTION #{q + 1} guessing probability: {result.ProbGuessMarginals[q].GetMean():N3}");
    }
}
